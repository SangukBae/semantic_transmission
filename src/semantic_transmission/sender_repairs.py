"""Opt-in sender repairs; no change to the frozen LGVSC/Qwen defaults.

Event decisions must come from source-image inference, never a description regex.
The budget is a ceiling; retained semantic evidence has priority over sparsity.
"""
import math

EVENT_PROMPT = '''Inspect the labeled source images in chronological order. Assess
the proposed event using the images, not the proposal's wording. Return ONLY JSON:
{"classification":"none|camera_only|object_change|scene_change|uncertain",
"evidence":"one concise sentence describing visible before/after evidence"}.
Use camera_only only for continuous viewpoint/pan/zoom changes with no visible
object action, entry/exit, occlusion, appearance change or newly revealed salient
object. Use object_change for those changes, even brief ones, and scene_change for
a shot cut. Use none only if no meaningful change is visibly supported. If the
images cannot resolve whether a semantic change occurs, use uncertain. Do not
infer absence of motion from an unchanged background. Do not invent tiny details.
Proposed event: '''


def validate_event(value):
    if (not isinstance(value, dict) or set(value) != {"classification", "evidence"}
            or value["classification"] not in {"none", "camera_only", "object_change", "scene_change", "uncertain"}
            or not isinstance(value["evidence"], str) or not value["evidence"].strip()):
        raise ValueError("invalid source event review")
    return value


def evidence_pairs(pairs, reviews):
    if len(pairs) != len(reviews):
        raise ValueError("every proposal needs a source review")
    return [dict(p) for p, r in zip(pairs, reviews)
            if validate_event(r)["classification"] in {"object_change", "scene_change", "uncertain"}]


def capped_keys(frames, cap, pairs, prior_keys, novelty, max_gap=24):
    """Proven lexicographic optimum: event coverage, minimum count, prior/novelty.

Keep the existing maximum gap as a conservative first experiment. An uncertain
event retains its original weight. Budget-infeasible pairs remain in the audit.
"""
    import numpy as np
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import lil_matrix, vstack
    from .qwen35_event_pairs import paired_keys
    if (type(frames) is not int or type(cap) is not int or type(max_gap) is not int
            or max_gap < 1 or not 2 <= cap <= frames
            or cap < math.ceil((frames-1)/max_gap)+1):
        raise ValueError("infeasible key budget ceiling")
    if len(novelty) != frames or not np.isfinite(novelty).all():
        raise ValueError("invalid source novelty")
    n = frames + len(pairs)
    rows, lo, hi = [], [], []
    def add(row, low, high):
        rows.append(row); lo.append(low); hi.append(high)
    add({i: 1 for i in range(frames)}, 2, cap)
    add({0: 1}, 1, 1); add({frames-1: 1}, 1, 1)
    for start in range(1, frames-max_gap):
        add({i: 1 for i in range(start, start+max_gap)}, 1, np.inf)
    for j, pair in enumerate(pairs):
        if (not 0 <= pair['before'] < pair['after'] < frames
                or not math.isfinite(pair['weight']) or pair['weight'] <= 0):
            raise ValueError("invalid semantic evidence pair")
        for i in (pair['before'], pair['after']):
            add({frames+j: 1, i: -1}, -np.inf, 0)
    matrix = lil_matrix((len(rows), n))
    for j, row in enumerate(rows):
        for i, value in row.items(): matrix[j, i] = value
    matrix = matrix.tocsc()
    weights = np.array([p['weight'] for p in pairs])
    options = dict(time_limit=60, mip_rel_gap=0., presolve=True)
    first = milp(np.r_[np.zeros(frames), -weights], integrality=np.ones(n), bounds=Bounds(0, 1),
                 constraints=LinearConstraint(matrix, lo, hi), options=options)
    if not first.success: raise RuntimeError(f"event optimum unproven: {first.message}")
    reward = lil_matrix((1, n)); reward[0, frames:] = weights
    optimum = round(-first.fun, 6)
    second = milp(np.r_[np.ones(frames), np.zeros(len(pairs))], integrality=np.ones(n), bounds=Bounds(0, 1),
                  constraints=LinearConstraint(vstack([matrix, reward.tocsc()], format='csc'),
                      np.r_[lo, optimum-1e-7], np.r_[hi, np.inf]), options=options)
    if not second.success: raise RuntimeError(f"minimum count unproven: {second.message}")
    count = int(round(second.fun))
    keys, audit = paired_keys(frames, count, pairs, prior_keys, novelty, max_gap)
    if abs(audit['weighted_pair_coverage']-optimum) > 1e-5:
        raise ValueError("event coverage changed during tie breaking")
    audit.update(budget_cap=cap, minimum_key_count=count, max_gap=max_gap,
                 policy='evidence_coverage_then_minimum_count_v1',
                 omitted_pairs=[i for i in range(len(pairs)) if i not in audit['covered_pairs']])
    return keys, audit


def distinct_samples(frames):
    if type(frames) is not int or frames < 1:
        raise ValueError("clip must contain at least one frame")
    count = min(4, frames)
    return [round(i*(frames-1)/(count-1)) for i in range(count)] if count > 1 else [0]


def calibrated_flow(displacements, indices, fps):
    """Duration-weighted L1 displacement at the legacy 10-frame/24-FPS scale.

This is a scale-preserving normalization, not a learned quality calibration.
Single-image clips have no measured velocity; record that explicitly upstream.
"""
    if (not math.isfinite(fps) or fps <= 0 or not indices or indices != sorted(set(indices))
            or len(displacements) != len(indices)-1
            or any(not math.isfinite(x) or x < 0 for x in displacements)):
        raise ValueError("invalid motion samples")
    if len(indices) == 1:
        return None
    seconds = (indices[-1]-indices[0])/fps
    return sum(displacements)/seconds*(10/24)


def flow_worker(cfg, repo, run):
    import sys
    import time
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torchvision.transforms.functional import pil_to_tensor
    from . import auto_extraction as ae
    from .artifacts import write_json
    from .semantic_clips import sample_clip_frames
    from .workers import semantic_clips
    from .validation_progress import emit
    keys = ae.read(run/'keyframes.json')['indices']
    rows = ae.read(run/'captions.json')
    clips = semantic_clips(run, keys, cfg['fps'], cfg)
    if len(rows) != len(keys)-1 or len(clips) != len(rows):
        raise ValueError("caption/clip count mismatch")
    sys.path.insert(0, str(repo/'.local/vendor/Open-Sora'))
    from tools.scoring.optical_flow.unimatch import UniMatch
    torch.manual_seed(cfg['seed'])
    model = UniMatch(feature_channels=128, num_scales=2, upsample_factor=4, num_head=1,
        ffn_dim_expansion=4, num_transformer_layers=6, reg_refine=True, task='flow')
    model.load_state_dict(torch.load(repo/'.local/checkpoints/unimatch.pth', map_location='cpu')['model'])
    model = model.cuda().eval()
    audit, started = [], time.monotonic()
    for i, (a, b) in enumerate(zip(keys, keys[1:])):
        indices = distinct_samples(b-a)
        images, actual = sample_clip_frames(clips[i], indices)
        if actual != indices: raise ValueError("source clip shorter than its declared interval")
        values = []
        if len(indices) > 1:
            x = torch.stack([pil_to_tensor(im) for im in images]).float().cuda()
            x = F.interpolate(x, size=(320, 576), mode='bilinear', align_corners=True)
            with torch.inference_mode():
                y = model(x[:-1], x[1:], attn_type='swin', attn_splits_list=[2,8],
                    corr_radius_list=[-1,4], prop_radius_list=[-1,1], num_reg_refine=6,
                    task='flow', pred_bidir_flow=False)['flow_preds'][-1]
                values = y.abs().mean(dim=(1,2,3)).cpu().tolist()
        score = calibrated_flow(values, indices, cfg['fps'])
        # The inherited metadata schema requires a finite scalar; distinguish an
        # unavailable observation from measured zero in the sampling receipt.
        rows[i]['flow'] = 0.0 if score is None else score
        audit.append(dict(segment=i, clip=str(clips[i]), requested_indices=indices,
            actual_indices=actual, pair_l1_displacements=values, score=score,
            motion_observed=score is not None, reference_interval_seconds=10/24,
            policy='distinct_duration_normalized_v1'))
        for im in images: im.close()
        emit(i+1, len(rows))
        print(f"FLOW {i+1}/{len(rows)} ({100*(i+1)/len(rows):.1f}%) elapsed={time.monotonic()-started:.1f}s", flush=True)
    write_json(run/'metadata_tx.json', rows)
    write_json(run/'flow_sampling.json', audit)
