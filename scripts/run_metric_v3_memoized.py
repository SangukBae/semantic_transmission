#!/usr/bin/env python3
"""Operational memoization of the unchanged mask-centroid function.

This wrapper changes no metric, parameter, or match rule. An equality probe on
already scored development pairs is required before a resumed worker uses it.
"""
import argparse
import gzip
import json
from pathlib import Path
import pickle
from types import SimpleNamespace
import time

from semantic_transmission import object_metric, sta_video_validation
from semantic_transmission.artifacts import sha256
from semantic_transmission.metric_v3_features import VisualObserver, pixels
from semantic_transmission.metric_v3_formal import array, read, run, save, verify

original_center = object_metric.center
cache = {}


def memoized_center(mask):
    key = id(mask)
    if key not in cache:
        if len(cache) >= 10000:
            cache.clear()
        # Strong reference prevents Python identity reuse while this entry exists.
        cache[key] = (mask, original_center(mask))
    return cache[key][1]


def enable():
    object_metric.center = memoized_center
    old_compare = VisualObserver.compare
    def compare(self, a, b):
        try:
            return old_compare(self, a, b)
        finally:
            cache.clear()
    VisualObserver.compare = compare
    old_attribution = sta_video_validation.score_video_attribution
    def attribution(*args, **kwargs):
        try:
            return old_attribution(*args, **kwargs)
        finally:
            cache.clear()
    sta_video_validation.score_video_attribution = attribution


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--benchmark', action='store_true'); ap.add_argument('--split', choices=('development', 'heldout'))
    args = ap.parse_args(); root = args.output; p, _ = verify(root)
    if args.benchmark:
        if any(read(f)['split'] == 'heldout' for f in (root / 'scores').rglob('*.json')):
            raise ValueError('operational probe must precede heldout scoring')
        results = []
        for stem, variant in (('davis__varanus-cage', 'brightness20'), ('davis__koala', 'identity')):
            base = Path(p['base_run']); a = array(base / 'inputs' / (stem + '.npz'))
            b = array(base / 'cases' / (stem + '__' + variant + '.npz'))
            observations = {}
            for frames in (a, b):
                key = pixels(frames)
                with gzip.open(root / 'visual_cache' / p['signature'] / (key + '.pkl.gz'), 'rb') as f:
                    observations[key] = pickle.load(f)
            observer = VisualObserver.__new__(VisualObserver)
            observer.observe = lambda frames: observations[pixels(frames)]
            started = time.time(); original = observer.compare(a, b); before = time.time() - started
            object_metric.center = memoized_center
            started = time.time(); optimized = observer.compare(a, b); after = time.time() - started
            object_metric.center = original_center; cache.clear()
            equal = json.dumps(original, sort_keys=True) == json.dumps(optimized, sort_keys=True)
            if not equal:
                raise ValueError('memoization changes scores')
            results.append({'case': stem + '__' + variant, 'all_outputs_exactly_equal': equal,
                            'uncached_s': before, 'memoized_s': after})
        save(root / 'memoization_probe.json', {'created_unix': time.time(), 'script_sha256': sha256(Path(__file__)),
            'heldout_scores_observed': 0, 'results': results, 'scope': 'CPU scoring of existing development RGB observations only'})
        print(json.dumps(results), flush=True)
    else:
        if args.split is None:
            ap.error('--split required')
        if read(root / 'memoization_probe.json')['script_sha256'] != sha256(Path(__file__)):
            raise ValueError('memoization wrapper changed since exact-equality check')
        enable()
        save(root / ('memoization_runtime_' + args.split + '.json'), {'started_unix': time.time(),
            'wrapper_sha256': sha256(Path(__file__)), 'metric_code_changed': False, 'parameters_changed': False,
            'cache_key': 'numpy mask object identity held by strong reference; cache cleared after each comparison'})
        run(SimpleNamespace(output=root, part='visual', split=args.split))


if __name__ == '__main__':
    main()
