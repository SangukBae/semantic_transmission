#!/usr/bin/env python3
"""Predeclared simple pixel baselines with the same TX/RX access as STA.

These supplemental baselines isolate extra-input value from a new support rule.
They do not alter the frozen main experiment, ERE/STA, or its adoption criteria.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from semantic_transmission.artifacts import sha256
from semantic_transmission.metric_v3_formal import array, read, save, verify
from semantic_transmission.metric_v3_statistics import fit_classifier, predict_classifier, classification_report


def stage_distances(a, tx, rx, b):
    result = {}
    for stage, x, y in (('source_tx', a, tx), ('tx_rx', tx, rx), ('rx_reconstruction', rx, b)):
        delta = (x.astype(np.float32) - y.astype(np.float32)) / 255.
        result[stage + '_mse'] = float(np.square(delta).mean())
        result[stage + '_mae'] = float(np.abs(delta).mean())
    return result


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--phase', choices=('declare', 'run'), required=True)
    args = ap.parse_args(); root = args.output; p, _ = verify(root)
    declaration_path = root / 'same_input_baseline_declaration.json'
    if args.phase == 'declare':
        if declaration_path.exists():
            raise FileExistsError(declaration_path)
        if any(read(f)['split'] == 'heldout' for f in (root / 'scores').rglob('*.json')):
            raise ValueError('supplement must be declared before any heldout score')
        save(declaration_path, {'created_unix': time.time(), 'heldout_scores_observed': 0,
            'script_sha256': sha256(Path(__file__)), 'protocol_sha256': sha256(root / 'protocol.json'),
            'inputs': 'source RGB, decoded TX/RX RGB, reconstruction RGB; same four views as STA',
            'features': 'MSE and MAE in [0,1] RGB on source/TX, TX/RX, RX/reconstruction',
            'classifiers': 'three-view MSE, three-view MAE, all six; fixed development standardized nearest centroids',
            'scope': 'supplemental comparison, no main metric/formula/threshold changes'})
        return
    d = read(declaration_path)
    if d['script_sha256'] != sha256(Path(__file__)) or d['protocol_sha256'] != sha256(root / 'protocol.json'):
        raise ValueError('supplement definition changed')
    rows = [json.loads(line) for line in (root / 'sta_cases.jsonl').read_text().splitlines()]
    for split in ('development', 'heldout'):
        if split == 'heldout' and not (root / 'same_input_baseline_calibration.json').exists():
            raise ValueError('development fit must precede heldout')
        selected = [r for r in rows if r['split'] == split and r['eligible']]
        scored = []
        for i, row in enumerate(selected):
            folder = root / 'sta_inputs' / row['case_id']
            a = array(Path(p['base_run']) / 'inputs' / (row['source_stem'] + '.npz'), row['source_pixel_sha256'])
            tx, rx, b = [array(folder / (key + '.npz'), row['pixel_sha256'][key]) for key in ('tx_frames', 'rx_frames', 'reconstruction')]
            scored.append({**row, 'scores': stage_distances(a, tx, rx, b)})
            if (i + 1) % 64 == 0:
                print(split, i + 1, '/', len(selected), flush=True)
        save(root / ('same_input_baseline_' + split + '.json'), {'rows': scored})
        if split == 'development':
            names = list(scored[0]['scores'])
            classifiers = {n: fit_classifier(scored, [k for k in names if k.endswith(n)]) for n in ('mse', 'mae')}
            classifiers['mse_and_mae'] = fit_classifier(scored, names)
            save(root / 'same_input_baseline_calibration.json', {'created_unix': time.time(), 'classifiers': classifiers,
                 'declaration_sha256': sha256(declaration_path), 'heldout_scores_observed_by_this_supplement': 0})
        else:
            classifiers = read(root / 'same_input_baseline_calibration.json')['classifiers']
            report = {n: classification_report(scored, [predict_classifier(r, f) for r in scored]) for n, f in classifiers.items()}
            save(root / 'same_input_baseline_results.json', {'completed_unix': time.time(), 'results': report,
                'scope': 'same four observed RGB views as STA, simple stage pixel distances, source-separated train/test',
                'calibration_sha256': sha256(root / 'same_input_baseline_calibration.json')})
            print(json.dumps({n: r['macro_accuracy'] for n, r in report.items()}), flush=True)


if __name__ == '__main__':
    main()
