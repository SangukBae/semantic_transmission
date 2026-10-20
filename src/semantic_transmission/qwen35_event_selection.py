"""Opt-in source-only all-frame change screening and Qwen event-window selection.

The CPU detector proposes inspection windows, never transmitted keys. Qwen can
select an internal frame even when a window's endpoints show the same state.
Thresholds are engineering defaults, not fitted to the evaluation reference.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import time

from . import qwen35_semantic_selection as baseline

POLICY = 'qwen35_all_frame_events_v1'
PROMPT = '''Select useful NEW visible states for faithful semantic video reconstruction.
The ACCEPTED REFERENCE is already retained. Inspect the chronological WINDOW,
including its internal frames, even when its two endpoints look the same.
Select a minimal set of clear representative source frames showing meaningful
entry/exit, occlusion, appearance, object pose/action phase or position changes,
scene cuts, or camera movement revealing substantially different content.
Preserve a brief intermediate event that disappears before the last image.
Do not select tiny jitter, blur, lighting noise, every small motion increment,
or repeated equivalent states. Crops magnify the same source images; they are
not extra times. Distinguish camera motion from object motion. Use only visible
evidence, never infer hidden events. No target key count. Zero keys is valid.
Return ONLY compact JSON:
{"decision":"resolved|uncertain","keys":[SOURCE_IDS],
 "events":[{"frame":SOURCE_ID,"reason":"short visible difference"}]}.
events must correspond exactly to keys, in chronological order. Each reason
must have at most 18 words. Choose only selectable WINDOW IDs, never the accepted
reference. If visual evidence cannot be resolved, return uncertain with empty
keys and events. An uncertain answer is not permission to skip the window.'''


@dataclass(frozen=True)
class DetectorConfig:
    width: int = 288
    tile: int = 16
    local_floor: float = 12.0
    local_strong: float = 32.0
    edge_floor: float = 24.0
    motion_floor: float = 1.5
    motion_strong: float = 5.0
    histogram_cut: float = .35
    history: int = 32
    mad_scale: float = 3.0
    cumulative_lag: int = 4
    stride: int = 4
    max_window_gaps: int = 8


def pair_signals(previous, current, cfg=DetectorConfig()):
    """Robust camera alignment at reduced resolution, residuals at native size."""
    import cv2
    import numpy as np
    if previous.shape != current.shape or previous.ndim != 3:
        raise ValueError('expected equally sized BGR source frames')
    h, w = current.shape[:2]
    gray0, gray1 = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (previous, current)]
    size = (min(w, cfg.width), max(1, round(h * min(w, cfg.width) / w)))
    a, b = [cv2.resize(x, size, interpolation=cv2.INTER_AREA) for x in (gray0, gray1)]
    pts = cv2.goodFeaturesToTrack(a, maxCorners=160, qualityLevel=.015, minDistance=7)
    matrix = None; residual_motion = 0.; tracking_lost = 0.; unreliable = False
    if pts is not None and len(pts) >= 8:
        nxt, ok, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None, winSize=(21, 21), maxLevel=3)
        if nxt is not None:
            back, rok, _ = cv2.calcOpticalFlowPyrLK(b, a, nxt, None, winSize=(21, 21), maxLevel=3)
            good = ok.ravel().astype(bool) & rok.ravel().astype(bool)
            good &= np.linalg.norm(back - pts, axis=2).ravel() < 1.5
            tracking_lost = float(1 - good.mean())
            x, y = pts[good, 0], nxt[good, 0]
            if len(x) >= 6:
                matrix, inliers = cv2.estimateAffinePartial2D(x, y, method=cv2.RANSAC,
                    ransacReprojThreshold=2., maxIters=500, confidence=.99)
                if matrix is not None:
                    scale = float(np.linalg.norm(matrix[0, :2]))
                    if not .85 <= scale <= 1.18 or float(inliers.mean()) < .4:
                        matrix = None
                if matrix is not None:
                    delta = y - (x @ matrix[:, :2].T + matrix[:, 2])
                    residual_motion = float(np.percentile(np.linalg.norm(delta, axis=1), 90))
            unreliable = matrix is None
        else:
            unreliable = True; tracking_lost = 1.
    if matrix is None:
        matrix = np.array([[1., 0., 0.], [0., 1., 0.]])
    # Convert the reduced-coordinate transform to native coordinates.
    scaling = np.diag([size[0]/w, size[1]/h, 1.])
    full = (np.linalg.inv(scaling) @ np.vstack([matrix, [0, 0, 1]]) @ scaling)[:2]
    aligned = cv2.warpAffine(gray0, full, (w, h))
    valid = cv2.warpAffine(np.ones((h, w), np.uint8), full, (w, h), flags=cv2.INTER_NEAREST)
    valid = cv2.erode(valid, np.ones((3, 3), np.uint8)).astype(bool)
    diff = cv2.absdiff(aligned, gray1).astype(float)
    # Remove a uniform exposure offset, but preserve changes in a small region.
    if valid.any():
        offset = np.median(gray1.astype(float)[valid] - aligned.astype(float)[valid])
        diff = np.abs(gray1.astype(float) - aligned.astype(float) - offset)
    edges = cv2.absdiff(cv2.Laplacian(aligned, cv2.CV_32F), cv2.Laplacian(gray1, cv2.CV_32F))
    scores = []
    for y in range(0, h, cfg.tile):
        for x in range(0, w, cfg.tile):
            mask = valid[y:y+cfg.tile, x:x+cfg.tile]
            if mask.mean() < .8:
                continue
            patch = diff[y:y+cfg.tile, x:x+cfg.tile][mask]
            edge = edges[y:y+cfg.tile, x:x+cfg.tile][mask]
            scores.append((float(patch.mean()), float(edge.mean()), x, y))
    best = max(scores, default=(0., 0., 0, 0))
    x, y = best[2:]
    hist = [cv2.calcHist([im], [0], None, [32], [0, 256]).ravel() / im.size for im in (a, b)]
    return dict(local=best[0], edge=max((s[1] for s in scores), default=0.),
        motion=residual_motion, tracking_lost=tracking_lost,
        alignment_unreliable=unreliable, histogram=float(np.abs(hist[0]-hist[1]).sum()/2),
        newly_visible_fraction=float(1-valid.mean()),
        roi=[max(0,x-24), max(0,y-24), min(w,x+cfg.tile+24), min(h,y+cfg.tile+24)],
        camera_transform=full.tolist())


def detect(frame_paths, cfg=DetectorConfig(), progress=None):
    import cv2
    import numpy as np
    if len(frame_paths) < 2:
        raise ValueError('at least two frames required')
    cv2.setNumThreads(2); cv2.setRNGSeed(20261019)
    rows, images = [], []
    for i, path in enumerate(frame_paths):
        im = cv2.imread(str(path))
        if im is None:
            raise ValueError(f'unreadable source: {path}')
        images.append(im)
        if i == 0:
            continue
        value = pair_signals(images[-2], im, cfg)
        reasons = []
        for field, floor in [('local', cfg.local_floor), ('edge', cfg.edge_floor), ('motion', cfg.motion_floor)]:
            past = [r[field] for r in rows[-cfg.history:]]
            center = float(np.median(past)) if past else 0.
            mad = float(np.median(np.abs(np.array(past)-center))) if past else 0.
            threshold = max(floor, center + cfg.mad_scale * 1.4826 * mad)
            if value[field] > threshold:
                reasons.append(field)
        if value['local'] >= cfg.local_strong or value['motion'] >= cfg.motion_strong:
            reasons.append('strong_single_transition')
        if value['histogram'] >= cfg.histogram_cut:
            reasons.append('cut_or_exposure')
        if value['alignment_unreliable'] and value['local'] >= cfg.local_floor:
            reasons.append('alignment_uncertain')
        if value['newly_visible_fraction'] > .08:
            reasons.append('newly_exposed_region')
        cumulative = None
        if i % cfg.cumulative_lag == 0 and len(images) > cfg.cumulative_lag:
            cumulative = pair_signals(images[-cfg.cumulative_lag-1], im, cfg)
            if cumulative['local'] >= cfg.local_floor or cumulative['motion'] >= cfg.motion_floor:
                reasons.append('accumulated_change')
        rows.append(dict(frame=i, **value, reasons=reasons, triggered=bool(reasons), cumulative=cumulative))
        images = images[-cfg.cumulative_lag-1:]
        if progress:
            progress(i, len(frame_paths)-1)
    return dict(policy=POLICY, config=asdict(cfg), frames=len(frame_paths), transitions=len(rows), rows=rows)


def windows(scan):
    """Expand a triggered transition to both sides, merge, then bound image count."""
    cfg = DetectorConfig(**scan['config']); n = scan['frames']
    marked = set()
    for row in scan['rows']:
        if row['triggered']:
            # Include before AND after a flash/event; never require persistence.
            for i in range(max(1,row['frame']-1), min(n,row['frame']+2)):
                marked.add((i-1)//cfg.stride)
    out = []
    for cell in sorted(marked):
        start, end = cell*cfg.stride, min(n-1, (cell+1)*cfg.stride)
        if out and out[-1][1] == start and end-out[-1][0] <= cfg.max_window_gaps:
            out[-1][1] = end
        else:
            out.append([start,end])
    return out


def validate_event(value, allowed):
    if not isinstance(value, dict) or value.get('decision') not in ('resolved','uncertain'):
        raise ValueError('decision must be resolved or uncertain')
    keys, events = value.get('keys'), value.get('events')
    if (not isinstance(keys,list) or any(type(k) is not int for k in keys)
            or keys != sorted(set(keys)) or not set(keys) <= set(allowed)):
        raise ValueError(f'keys must be unique ordered selectable source IDs: {allowed}')
    if not isinstance(events,list) or len(events) != len(keys):
        raise ValueError('one event reason per selected key required')
    for key, event in zip(keys,events):
        if (not isinstance(event,dict) or type(event.get('frame')) is not int or event['frame'] != key
                or not isinstance(event.get('reason'),str) or not 1 <= len(event['reason'].split()) <= 18):
            raise ValueError('each event needs its selected source ID and a 1..18 word visible reason')
    if value['decision'] == 'uncertain' and (keys or events):
        raise ValueError('uncertain must have empty keys and events')
    return value


def select(frames, session, scan, *, stride=4, progress=None):
    if scan['frames'] != frames or scan['config']['stride'] != stride:
        raise ValueError('detector and selection schedule differ')
    event_windows = {a:b for a,b in windows(scan)}
    keys, trace = [0], []; cursor = 0; started = time.monotonic()
    while cursor < frames-1:
        reference = keys[-1]
        if cursor in event_windows:
            end = event_windows[cursor]; allowed = list(range(max(cursor,reference+1),end+1))
            ids = sorted({reference,*range(cursor,end+1)})
            peak = max((r for r in scan['rows'] if cursor < r['frame'] <= end), key=lambda r:r['local'])
            crop_ids = sorted({max(cursor,peak['frame']-1),peak['frame'],min(end,peak['frame']+1)})
            regions = [dict(frame=i,box=peak['roi']) for i in crop_ids]
            prompt = PROMPT + f'\nACCEPTED REFERENCE={reference}; SELECTABLE WINDOW IDS={allowed}.'
            value = session.call_event(ids,prompt,lambda v:validate_event(v,allowed),regions)
            if value['decision'] == 'uncertain':
                raise RuntimeError(f'event window {cursor}..{end} uncertain; records preserved, no silent skip')
            keys.extend(value['keys'])
            trace.append(dict(mode='event',reference=reference,start=cursor,end=end,result=value,regions=regions))
        else:
            end = min(frames-1,cursor+stride)
            prompt = baseline.PROMPT.replace('REF,CAND',f'{reference},{end}')
            prompt += f'\nACCEPTED REFERENCE={reference}; CANDIDATE={end}.'
            validator = lambda v:baseline.validate_decision(v,reference,end)
            value = session.call('semantic_select',[reference,end],prompt,validator,256)
            if value['decision'] == 'uncertain':
                ids = sorted({reference,end,*[round(reference+(end-reference)*i/4) for i in (1,2,3)]})
                value = session.call('semantic_select_review',ids,prompt,validator,256)
            if value['decision'] == 'uncertain':
                raise RuntimeError(f'semantic decision {reference}..{end} uncertain')
            if value['decision'] == 'changed':
                keys.append(end)
            trace.append(dict(mode='baseline',reference=reference,candidate=end,result=value))
        cursor = end
        if progress:
            progress(cursor,frames-1,started)
    if keys[-1] != frames-1:
        keys.append(frames-1)
    baseline.validate_keys(keys,frames)
    return keys,trace


def make_session(root, plan, model=None):
    """Use the existing sealed response cache with crop geometry in its identity."""
    from .qwen35_feedback import Session
    from .qwen35_backend import LocalQwen35

    class RegionalModel:
        def __init__(self, config):
            self.base = model or LocalQwen35(config)
            self.runtime = self.base.runtime

        def generate(self, images, labels, prompt, tokens):
            extra, extra_labels = [], []
            try:
                for region in session.regions:
                    im = images[session.ids.index(region['frame'])]
                    extra.append(im.crop(region['box']).resize((192,192)))
                    extra_labels.append(f"SOURCE frame {region['frame']} DETAIL CROP {region['box']}; same time")
                return self.base.generate(images+extra, labels+extra_labels,prompt,tokens)
            finally:
                for im in extra:
                    im.close()

    class EventSession(Session):
        regions = (); ids = ()

        def call_event(self, ids, prompt, validator, regions):
            self.regions, self.ids = regions, ids
            try:
                request = prompt+'\nDETAIL CROPS: '+json.dumps(regions)
                repaired = request + ('\nSCHEMA REVIEW: SELECTABLE WINDOW IDS lists permitted IDs, not required keys. '
                    'Never copy the complete list to fill the keys array. The accepted reference is already kept '
                    'and MUST NOT be returned. Every selected key must have exactly one event with the SAME frame ID; '
                    'keys must equal [event.frame for event in events], sorted and unique. '
                    'An empty selection is valid. Recheck the same images and original semantic criteria; '
                    'do not invent differences just to satisfy formatting.')
                from . import auto_extraction as ae
                # On resume reuse a successful schema review, avoiding the already failed request.
                for path in (self.root/'calls/event_window_schema').glob('*.json'):
                    record=ae.unseal(path)
                    if (record['identity']['plan_signature']==self.plan['signature']
                            and record['identity']['ids']==ids
                            and record['identity']['prompt'].startswith(repaired+'\nVALID SOURCE')):
                        return self.call('event_window_schema',ids,repaired,validator,640)
                try:
                    return self.call('event_window',ids,request,validator,640)
                except RuntimeError as exc:
                    if not str(exc).startswith('three invalid responses:'):
                        raise
                    return self.call('event_window_schema',ids,repaired,validator,640)
            finally:
                self.regions, self.ids = (), ()

    session = EventSession(Path(root),plan,RegionalModel)
    return session
