"""Source-only detail-aware auxiliary selection without a temporal gap cap.

No codec, interpolation model, reconstructed frames, or quality feedback is used.
The scores are source heuristics, not a guarantee of receiver quality.
"""
import math
from pathlib import Path
import time

from .artifacts import sha256
from .sparc_selection import validate_anchors, validate_cuts

POLICY = 'sparc_source_detail_selection_v1'
DEFAULT = dict(resolution=[288,160],tile_size=16,tail_fraction=.10,
               color_threshold=6.,local_threshold=12.,edge_threshold=4.,
               prune_redundant=True)


def config(scale=1.):
    if type(scale) not in (int,float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('detail threshold scale must be finite and positive')
    return dict(DEFAULT,threshold_scale=float(scale),policy=POLICY)


class Signals:
    def __init__(self,images,settings):
        import cv2
        import numpy as np
        self.np = np
        self.config = settings
        pixels = np.asarray(images)
        if (pixels.ndim != 4 or pixels.shape[-1] != 3 or len(pixels)<2 or
                not np.isfinite(pixels).all() or pixels.min()<0 or pixels.max()>255):
            raise ValueError('expected finite RGB source images in [0,255]')
        self.rgb = pixels.astype(np.float32)
        self.edges = []
        for frame in self.rgb:
            gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
            # Signed spatial derivatives retain small moving boundaries.
            self.edges.append(np.stack([
                cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3,scale=1/8),
                cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3,scale=1/8)],axis=-1))
        self.edges = np.stack(self.edges)
        self.cache = {}
        self.evaluated_frames = 0

    def interval(self,a,b):
        """Largest normalized source error inside an endpoint interval."""
        if (a,b) in self.cache:
            return self.cache[a,b]
        np,c = self.np,self.config
        records = []
        if b-a <= 1:
            result = dict(score=0.,index=None,color=0.,local=0.,edge=0.)
        else:
            h,w = self.rgb.shape[1:3]
            tile = min(c['tile_size'],h,w)
            # Preserve incomplete edge tiles by padding with their edge values.
            ph,pw = (-h)%tile,(-w)%tile
            for i in range(a+1,b):
                fraction = (i-a)/(b-a)
                predicted = self.rgb[a]*(1-fraction)+self.rgb[b]*fraction
                error = np.abs(self.rgb[i]-predicted).mean(axis=-1)
                color = float(error.mean())
                padded = np.pad(error,((0,ph),(0,pw)),mode='edge')
                tiles = padded.reshape((h+ph)//tile,tile,(w+pw)//tile,tile).mean(axis=(1,3)).ravel()
                count = max(1,math.ceil(len(tiles)*c['tail_fraction']))
                local = float(np.partition(tiles,len(tiles)-count)[-count:].mean())
                predicted_edges = self.edges[a]*(1-fraction)+self.edges[b]*fraction
                edge = float(np.abs(self.edges[i]-predicted_edges).mean())
                score = max(color/c['color_threshold'],local/c['local_threshold'],
                            edge/c['edge_threshold'])/c['threshold_scale']
                records.append(dict(score=score,index=i,color=color,local=local,edge=edge))
            result = max(records,key=lambda r:(r['score'],-abs(r['index']-(a+b)/2),-r['index']))
            self.evaluated_frames += len(records)
        self.cache[a,b] = result
        return result


def choose(images,semantic_keys,cuts,settings):
    """Split on source error, then remove redundant nonmandatory anchors."""
    started = time.perf_counter()
    frames = len(images)
    validate_anchors(sorted(set(semantic_keys)|{i for c in cuts for i in (c-1,c)}),
                     frames,None,cuts)
    # Validate the original key map, not just the deduplicated merged map.
    validate_anchors(semantic_keys,frames,None,[])
    validate_cuts(cuts,frames)
    signal = Signals(images,settings)
    mandatory = set(semantic_keys)|{i for c in cuts for i in (c-1,c)}
    selected = set(mandatory)
    boundaries = sorted(selected)
    pending = list(zip(boundaries,boundaries[1:]))
    additions = []
    while pending:
        a,b = pending.pop()
        record = signal.interval(a,b)
        if record['score'] <= 1:
            continue
        index = record['index']
        selected.add(index)
        additions.append(dict(index=index,interval=[a,b],reason='source_detail',**{
            k:record[k] for k in ('score','color','local','edge')}))
        pending.extend(((a,index),(index,b)))
    removals = []
    if settings['prune_redundant']:
        while True:
            indices = sorted(selected)
            candidates = []
            for k,i in enumerate(indices[1:-1],1):
                if i in mandatory:
                    continue
                a,b = indices[k-1],indices[k+1]
                record = signal.interval(a,b)
                if record['score'] <= 1:
                    candidates.append((record['score'],i,a,b))
            if not candidates:
                break
            score,i,a,b = min(candidates)
            selected.remove(i)
            removals.append(dict(index=i,interval=[a,b],score=score,reason='redundant_source_detail'))
    indices = sorted(selected)
    validate_anchors(indices,frames,None,cuts)
    intervals = [dict(start=a,end=b,**signal.interval(a,b))
                 for a,b in zip(indices,indices[1:])]
    assert all(row['score']<=1 for row in intervals)
    return dict(indices=indices,semantic_keys=semantic_keys,cuts=cuts,
        auxiliary_keys=sorted(selected-set(semantic_keys)),
        transmitted_auxiliary_keys=sorted(selected-set(semantic_keys)),
        additions=additions,removals=removals,max_gap=None,
        selector_policy=POLICY,selector_config=settings,mode='source-detail',
        semantic_keys_preserved=True,selection_reads_source_only=True,
        receiver_quality_used=False,codec_calls_during_selection=0,
        neural_interpolation_calls_during_selection=0,
        quality_preservation_guaranteed=False,interval_checks=intervals,
        evaluated_source_frames=signal.evaluated_frames,
        cached_intervals=len(signal.cache),analysis_seconds=time.perf_counter()-started)


def select(source,semantic_keys,profile,settings,progress=None):
    import cv2
    import numpy as np
    from PIL import Image
    cv2.setNumThreads(2)
    started = time.perf_counter()
    frames,cuts = [],[]
    previous = None
    for i in range(source['video']['frames']):
        path = Path(source['directory'])/f'{i}.png'
        if sha256(path) != source['hashes'][path.name]:
            raise ValueError('source frame changed during detail selection')
        with Image.open(path) as image:
            current = np.asarray(image.convert('RGB'))
        if previous is not None and float(np.abs(
                current.astype(np.float32)-previous.astype(np.float32)).mean())>profile['scene_cut_mad_threshold']:
            cuts.append(i)
        frames.append(cv2.resize(current,tuple(settings['resolution']),interpolation=cv2.INTER_AREA))
        previous = current
        if progress:
            progress(i+1,source['video']['frames'])
    result = choose(np.stack(frames),semantic_keys,cuts,settings)
    result['selection_seconds'] = time.perf_counter()-started
    return result
