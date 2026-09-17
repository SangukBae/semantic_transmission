"""RGB-only observers for the frozen ERE/STA evaluation extension.

Caches contain locally generated observations keyed by pixels and frozen code.
No case type, intervention, annotation, or renderer state enters an observer.
"""
from collections import OrderedDict
import gzip
import hashlib
from pathlib import Path
import pickle

import numpy as np

from .event_metric import EventMetric, score_events
from .motion_metric import describe_flow, score_motion
from .object_metric import score_tracks
from .tracking_baselines import identity_baselines


def pixels(x):
    return hashlib.sha256(x.tobytes()).hexdigest()


class VisualObserver:
    def __init__(self, protocol, root, signature):
        self.root = Path(root) / 'visual_cache' / signature
        self.root.mkdir(parents=True, exist_ok=True)
        self.metric = EventMetric(model_root=protocol['models']['root'],
                                  cache_root=Path(root) / 'object_cache',
                                  parameters=protocol['event_parameters'])
        self.recent = OrderedDict()

    def observe(self, frames):
        key = pixels(frames)
        if key in self.recent:
            self.recent.move_to_end(key)
            return self.recent[key]
        path = self.root / (key + '.pkl.gz')
        if path.exists():
            # Only this process family writes these local, hash-keyed caches.
            with gzip.open(path, 'rb') as f:
                value = pickle.load(f)
            if value['pixel_sha256'] != key:
                raise ValueError('observation cache provenance mismatch')
        else:
            value = self.metric.extract(frames)
            value['flow'] = self.metric.flow(frames)
            value['motion'] = describe_flow(value['flow'])
            value['pixel_sha256'] = key
            temporary = path.with_suffix('.tmp')
            with gzip.open(temporary, 'wb', compresslevel=1) as f:
                pickle.dump(value, f, protocol=5)
            temporary.replace(path)
        self.recent[key] = value
        while len(self.recent) > 4:
            self.recent.popitem(last=False)
        return value

    def compare(self, a, b):
        if a.shape != b.shape:
            raise ValueError('shared RGB dimensions and timeline required')
        x, y = self.observe(a), self.observe(b)
        result = score_events(x['events'], y['events'])
        result.update(score_tracks(x['tracks'], y['tracks']))
        result.update(identity_baselines(x['tracks'], y['tracks']))
        motion = score_motion(x['motion'], y['motion'])
        result.update({k: v for k, v in motion.items() if k.startswith('mte_') or k.startswith('camera_')})
        curve = np.linalg.norm(x['flow'] - y['flow'], axis=-1).mean((1, 2))
        result.update(tof_raft_small=float(curve.mean()), tof_raft_curve=curve.tolist())
        return result


class PixelObserver:
    def __init__(self):
        from .automatic_metrics import AutomaticMetrics
        from .temporal_baselines import TemporalBaselines
        self.metric = AutomaticMetrics()
        self.temporal = TemporalBaselines('cuda', self.metric.lpips)

    def compare(self, a, b):
        self.temporal.cache.clear()
        result = self.metric.evaluate(a, b)
        result.update(self.temporal.evaluate(a, b))
        fa, la = self.temporal.transitions(a)
        fb, lb = self.temporal.transitions(b)
        result.update(tof_farneback_curve=np.linalg.norm(fa - fb, axis=-1).mean((1, 2)).tolist(),
                      tlp_alex_curve=np.abs(la - lb).tolist())
        return result
