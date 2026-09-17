#!/usr/bin/env python3
"""Reproduce annotation-only target-alignment diagnostics without rescoring RGB."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from semantic_transmission.artifacts import sha256
from semantic_transmission.metric_v3_cases import annotation_events
from semantic_transmission.metric_v3_formal import read, save, verify
from semantic_transmission.metric_v3_statistics import truth_extra


def changes(a, b):
    extra = int(truth_extra(a, b)); matched = len(b) - extra
    return len(a) - matched, extra, matched


def category(missing, extra, unchanged):
    return 'extra_only' if not missing and extra else 'mixed' if missing and extra else 'missing_only' if missing else unchanged


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--verify', action='store_true'); args = ap.parse_args()
    root = args.output; protocol, _ = verify(root); base = Path(protocol['base_run'])
    truth = {r['case_id']: r for r in read(base / 'truth_audit.json')['results']}
    ere = []
    for row in [json.loads(x) for x in (root / 'cases.jsonl').read_text().splitlines()]:
        if row['target'] != 'motion':
            continue
        item = truth[row['case_id']]; a, b = item['reference_events'], item['reconstruction_events']
        missing, extra, matched = changes(a, b)
        ere.append({k: row[k] for k in ('case_id', 'domain', 'split', 'kind', 'severity')} |
            {'true_reference_events': len(a), 'true_reconstruction_events': len(b), 'true_matched_events': matched,
             'true_missing_events': missing, 'true_extra_events': extra,
             'category': category(missing, extra, 'no_event_change')})
    counts = {'/'.join((domain, split, kind)): dict(Counter(r['category'] for r in ere
        if r['domain'] == domain and r['split'] == split and r['kind'] == kind))
        for domain in ('rendered', 'public_video') for split in ('development', 'heldout')
        for kind in sorted({r['kind'] for r in ere if r['domain'] == domain})}
    products = {'truth_metric_scope.json': {'cases': ere, 'by_group': counts}}
    sta, cache = [], {}
    for row in [json.loads(x) for x in (root / 'sta_cases.jsonl').read_text().splitlines()]:
        if row['kind'] == 'unsupported_addition':
            continue
        stem = row['source_stem']
        if stem not in cache:
            with np.load(base / 'inputs' / (stem + '_truth.npz')) as f:
                cache[stem] = f['centers'], f['visible']
        centers, visible = cache[stem]
        indices = read(root / 'sta_inputs' / row['case_id'] / 'transport.json')['reconstruction_indices']
        a = annotation_events(centers, visible); b = annotation_events(centers[indices], visible[indices])
        missing, extra, _ = changes(a, b)
        sta.append({k: row[k] for k in ('case_id', 'source_id', 'split', 'kind', 'label', 'severity')} |
            {'true_missing_events': missing, 'true_extra_events': extra,
             'event_change_category': category(missing, extra, 'none')})
    counts = {split + '/' + kind: dict(Counter(r['event_change_category'] for r in sta
        if r['split'] == split and r['kind'] == kind))
        for split in ('development', 'heldout') for kind in sorted({r['kind'] for r in sta})}
    products['sta_truth_target_audit.json'] = {'cases': sta, 'by_group': counts}
    for name, values in products.items():
        if args.verify:
            existing = read(root / name)
            assert all(existing[key] == value for key, value in values.items()), name
        else:
            if (root / name).exists():
                raise FileExistsError(root / name)
            save(root / name, {'recorded_unix': time.time(), 'script_sha256': sha256(Path(__file__)),
                'scope': 'Annotation-only explanatory audit; no RGB score or primary label is changed', **values})
    if args.verify:
        save(root / 'target_audit_reproduction.json', {'status': 'PASSED', 'recorded_unix': time.time(),
            'script_sha256': sha256(Path(__file__)), 'verified_files': list(products),
            'ere_cases': len(ere), 'sta_cases': len(sta)})
    print('Target diagnostics', 'verified' if args.verify else 'written', flush=True)


if __name__ == '__main__':
    main()
