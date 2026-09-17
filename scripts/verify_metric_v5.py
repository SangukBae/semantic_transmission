#!/usr/bin/env python3
"""Final integrity verification for the FSO/EOI/UEP run.

Checks frozen code, calibration immutability, development/heldout separation,
score provenance, stored RGB hashes, and recomputes a sample of candidate scores
from the run's own observation cache.
"""
import argparse
import gzip
import json
from pathlib import Path
import pickle
import time

import numpy as np

from semantic_transmission.artifacts import sha256
from semantic_transmission.forbidden_state_metric import evaluate, unsupported_presence
from semantic_transmission.metric_v3_formal import read, save
from semantic_transmission.metric_v5_pipeline import CANDIDATES, SIGNATURE, rows, verify

CHECKS = []


def check(name, condition, detail=None):
    CHECKS.append({'check': name, 'passed': bool(condition), 'detail': detail})
    print(('PASS ' if condition else 'FAIL ') + name + ('' if detail is None else ' ' + str(detail)), flush=True)
    return bool(condition)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=40)
    args = parser.parse_args()
    root = args.output
    protocol = verify(root, True)
    check('frozen code and previous protocols unchanged', True, len(protocol['code_sha256']))
    calibration = read(root / 'calibration.json')
    check('calibration digest matches its lock file',
          sha256(root / 'calibration.json') == (root / 'calibration.sha256').read_text().strip())
    check('calibration was written with zero heldout scores observed',
          calibration['heldout_scores_observed'] == 0)
    check('statistics were declared before scores',
          read(root / 'statistics_declaration.json')['heldout_scores_observed'] == 0)
    check('extractor gate passed before the thresholds were frozen',
          read(root / 'extractor_gate.json')['passed']
          and calibration['extractor_gate_sha256'] == sha256(root / 'extractor_gate.json'))
    manifest = rows(root / 'cases.jsonl')
    prep = read(root / 'preparation.json')
    check('case manifest unchanged', sha256(root / 'cases.jsonl') == prep['manifest_sha256'])
    development = {r['source_id'] for r in manifest if r['split'] == 'development'}
    heldout = {r['source_id'] for r in manifest if r['split'] == 'heldout'}
    check('development and heldout sources are disjoint', not development & heldout,
          f'{len(development)} development, {len(heldout)} heldout')
    development_rgb = {r['source_pixel_sha256'] for r in manifest if r['split'] == 'development'}
    heldout_rgb = {r['source_pixel_sha256'] for r in manifest if r['split'] == 'heldout'}
    check('development and heldout source RGB are disjoint', not development_rgb & heldout_rgb)
    check('no source RGB overlaps a previous experiment', prep['source_RGB_overlap_with_previous'] is False)
    stale = [name for name, digest in prep['RGB_and_truth_file_sha256'].items() if sha256(root / name) != digest]
    check('stored RGB and truth files unchanged', not stale, f'{len(prep["RGB_and_truth_file_sha256"])} files')
    digest = sha256(root / 'calibration.json')
    heldout_rows = [r for r in manifest if r['split'] == 'heldout']
    missing, mismatched = [], []
    for row in heldout_rows:
        for part in ('visual', 'pixel'):
            path = root / 'scores' / part / (row['case_id'] + '.json')
            if not path.exists():
                missing.append(str(path))
                continue
            record = read(path)
            if (record['calibration_sha256'] != digest
                    or record['protocol_sha256'] != sha256(root / 'protocol.json')
                    or record['source_pixel_sha256'] != row['source_pixel_sha256']
                    or record['reconstruction_pixel_sha256'] != row['reconstruction_pixel_sha256']):
                mismatched.append(row['case_id'])
    check('every heldout case has visual and pixel scores', not missing, f'{len(heldout_rows)} cases')
    check('score provenance matches the frozen protocol and calibration', not mismatched)
    check('no development case was scored as heldout',
          not any((root / 'scores' / part / (r['case_id'] + '.json')).exists()
                  for part in ('visual', 'pixel') for r in manifest if r['split'] == 'development'))
    truth = read(root / 'truth_audit.json')
    check('every error case passed the occupancy truth gate',
          all(r['status'] == 'ACCEPTED' and (r['oracle'] is None or r['oracle'] > (r['source_oracle'] or 0))
              for r in truth['results']), f'{truth["accepted"]} accepted')
    cache = root / 'visual_cache' / SIGNATURE
    rng = np.random.default_rng(20260915)
    sample = [heldout_rows[i] for i in rng.choice(len(heldout_rows), min(args.samples, len(heldout_rows)), replace=False)]
    differences = {}
    recomputed = 0
    for row in sample:
        paths = [cache / (row[k] + '.pkl.gz') for k in ('source_pixel_sha256', 'reconstruction_pixel_sha256')]
        if not all(p.exists() for p in paths):
            continue
        loaded = []
        for path in paths:
            with gzip.open(path, 'rb') as f:
                loaded.append(pickle.load(f))
        value = evaluate(*loaded)
        if row['corpus'] == 'sta':
            rx = cache / (row['rx_pixel_sha256'] + '.pkl.gz')
            if rx.exists():
                with gzip.open(rx, 'rb') as f:
                    value.update(unsupported_presence(loaded[0], loaded[1], pickle.load(f),
                                                      read(root / 'cases' / row['case_id'] / 'transport.json')['rx_status']))
        stored = read(root / 'scores' / 'visual' / (row['case_id'] + '.json'))['scores']
        recomputed += 1
        for name in CANDIDATES:
            a, b = stored.get(name), value.get(name)
            if (a is None) != (b is None) or (a is not None and abs(a - b) > 1e-12):
                differences[row['case_id'] + '/' + name] = [a, b]
    check('recomputed candidate scores reproduce the stored values', not differences,
          f'{recomputed} sampled cases, {len(CANDIDATES)} candidates each')
    summary = read(root / 'summary.json')
    check('report used the frozen calibration',
          summary['calibration_sha256'] == digest and summary['protocol_sha256'] == sha256(root / 'protocol.json'))
    check('no human review and no novelty claim is recorded',
          summary['new_human_review'] is False and summary['novelty_demonstrated'] is False)
    passed = all(c['passed'] for c in CHECKS)
    save(root / 'final_integrity.json', {'status': 'PASSED' if passed else 'FAILED',
         'completed_unix': time.time(), 'checks': CHECKS,
         'recomputation_differences': differences,
         'scope': 'file, provenance and determinism checks; not evidence that the candidates are valid metrics'})
    print(('ALL CHECKS PASSED' if passed else 'INTEGRITY FAILED'), len(CHECKS), 'checks', flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
