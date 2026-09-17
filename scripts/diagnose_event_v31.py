#!/usr/bin/env python3
"""Choose turn parameters on synthetic development sources before re-audit."""
import argparse
import itertools
import json
from pathlib import Path
import shutil
import time

import numpy as np

from semantic_transmission.artifacts import sha256
from semantic_transmission.automatic_validation import write
from semantic_transmission.event_metric import EventMetric, PARAMETERS, compact_tracks, event_pairs, presence_events
from semantic_transmission.metric_v3_cases import annotation_events


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-run', type=Path, default=Path('outputs/ere_sta_20260914_v1'))
    p.add_argument('--output', type=Path, default=Path('outputs/ere_sta_v31_development'))
    args = p.parse_args(); root = args.output
    root.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((args.source_run / 'protocol.json').read_text())
    if list((args.source_run / 'scores').rglob('*.json')):
        raise ValueError('source run contains candidate scores; reassess independence')
    grid = [dict(smooth_frames=s, minimum_speed_px_per_frame=v, turn_angle_degrees=a, turn_context_frames=c)
            for s, v, a, c in itertools.product((3, 5, 7), (.25, .5, 1.), (90., 120., 150.), (1, 2))]
    declaration = {'created_unix': time.time(), 'heldout_sources_used': 0, 'candidate_scores_observed': 0,
        'grid': grid, 'selection': 'maximize minimum of event recall/precision/turn recall, then event F1, then proximity to baseline parameters',
        'source_protocol_sha256': sha256(args.source_run / 'protocol.json'),
        'event_code_sha256': sha256(Path('src/semantic_transmission/event_metric.py')),
        'script_sha256': sha256(Path(__file__))}
    write(root / 'declaration.json', declaration)
    cache = root / 'object_cache'; cache.mkdir()
    for old in (args.source_run / 'object_cache').glob('*.npz'):
        shutil.copy2(old, cache / old.name)
    import torch
    torch.set_num_threads(4); torch.manual_seed(20260914)
    extractor = EventMetric(model_root=protocol['models']['root'], cache_root=cache)
    observations = []
    manifest = {r['source_id']: r for r in (json.loads(s) for s in (args.source_run / 'cases.jsonl').read_text().splitlines())}
    import hashlib
    for item in protocol['inventory']:
        if item['split'] != 'development' or item['domain'] != 'rendered':
            continue
        stem = item['source_id'].replace('/', '__')
        with np.load(args.source_run / 'inputs' / (stem + '.npz')) as f:
            frames = f['frames']
        if hashlib.sha256(frames.tobytes()).hexdigest() != manifest[item['source_id']]['source_pixel_sha256']:
            raise ValueError('source pixels changed')
        observation = extractor.observe(frames)
        # Truth is read only AFTER the RGB observations and is not passed back.
        with np.load(args.source_run / 'inputs' / (stem + '_truth.npz')) as f:
            truth = annotation_events(f['centers'], f['visible'])
        record = {'source_id': item['source_id'], 'split': item['split'], 'domain': item['domain'],
                  'tracks': compact_tracks(observation['tracks']), 'reference_events': truth,
                  'frames': len(frames), 'shape': list(frames.shape[1:3])}
        observations.append(record)
        write(root / 'observations' / (stem + '.json'), record)
        print('observed', item['source_id'], len(record['tracks']), flush=True)
    results = []
    for params in grid:
        ntrue = npred = hit = nturn = hitturn = 0
        by_source = []
        for row in observations:
            events = presence_events(row['tracks'], row['frames'], row['shape'], parameters=params)
            pairs = event_pairs(row['reference_events'], events)
            turns = sum(e['type'] == 'turn' for e in row['reference_events'])
            matched_turns = sum(row['reference_events'][i]['type'] == 'turn' for i, _ in pairs)
            ntrue += len(row['reference_events']); npred += len(events); hit += len(pairs)
            nturn += turns; hitturn += matched_turns
            by_source.append({'source_id': row['source_id'], 'true': len(row['reference_events']),
                              'predicted': len(events), 'matched': len(pairs), 'turns': turns, 'matched_turns': matched_turns})
        recall, precision, turn_recall = hit / ntrue, hit / npred if npred else 0., hitturn / nturn
        baseline_distance = sum(params[k] != PARAMETERS[k] for k in params)
        results.append({'parameters': params, 'true_events': ntrue, 'predicted_events': npred, 'matched_events': hit,
            'recall': recall, 'precision': precision, 'turn_recall': turn_recall,
            'selection_key': [min(recall, precision, turn_recall), 2 * hit / (ntrue + npred), -baseline_distance],
            'by_source': by_source})
    chosen = max(results, key=lambda x: x['selection_key'])
    write(root / 'diagnostics.json', {'status': 'COMPLETED', 'completed_unix': time.time(),
        'declaration_sha256': sha256(root / 'declaration.json'), 'development_sources': len(observations),
        'heldout_sources_used': 0, 'formal_candidate_scores_observed': 0,
        'candidate_configurations': len(results), 'selected': chosen, 'results': results,
        'scope': 'development tuning; these rates are not independent validation',
        'event_code_sha256': declaration['event_code_sha256']})
    print(json.dumps({k: v for k, v in chosen.items() if k != 'by_source'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
