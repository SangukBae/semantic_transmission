#!/usr/bin/env python3
"""Predeclared same-classifier comparison using the four frozen STA features."""
import argparse
import json
from pathlib import Path
import time

from semantic_transmission.artifacts import sha256
from semantic_transmission.metric_v3_formal import read, save, verify
from semantic_transmission.metric_v3_statistics import (
    CLASSES, classification_report, fit_classifier, merged, predict_classifier,
)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--phase', choices=('declare', 'fit', 'evaluate'), required=True)
    args = ap.parse_args(); root = args.output; verify(root)
    declaration = root / 'sta_classifier_head_declaration.json'
    if args.phase == 'declare':
        if declaration.exists():
            raise FileExistsError(declaration)
        if any(read(f)['split'] == 'heldout' for f in (root / 'scores').rglob('*.json')):
            raise ValueError('Declare before any heldout score')
        save(declaration, {'created_unix': time.time(), 'heldout_scores_observed': 0,
            'script_sha256': sha256(Path(__file__)), 'protocol_sha256': sha256(root / 'protocol.json'),
            'features': CLASSES, 'classifier': 'Same development standardized nearest class centroid as all registered baselines',
            'scope': 'Supplementary classifier-controlled comparison; primary STA largest-component rule unchanged; no normal-class classifier'})
        return
    if read(declaration)['script_sha256'] != sha256(Path(__file__)):
        raise ValueError('Supplement changed')
    if sha256(root / 'calibration.json') != (root / 'calibration.sha256').read_text().strip():
        raise ValueError('Main calibration changed')
    calibration = root / 'sta_classifier_head_calibration.json'
    if args.phase == 'fit':
        if calibration.exists():
            raise FileExistsError(calibration)
        rows = [{**r, 'scores': r['scores']['sta']} for r in merged(root, 'sta', 'development')]
        save(calibration, {'created_unix': time.time(), 'declaration_sha256': sha256(declaration),
            'heldout_scores_observed_by_this_supplement': 0, 'classifier': fit_classifier(rows, list(CLASSES))})
        print('STA classifier frozen from development', len(rows), flush=True)
    else:
        fitted = read(calibration)['classifier']
        rows = [{**r, 'scores': r['scores']['sta']} for r in merged(root, 'sta', 'heldout')]
        predictions = [predict_classifier(r, fitted) for r in rows]
        report = classification_report(rows, predictions)
        save(root / 'sta_classifier_head_results.json', {'completed_unix': time.time(),
            'calibration_sha256': sha256(calibration), 'results': report,
            'predictions': [{k: r[k] for k in ('case_id', 'source_id', 'label')} | {'prediction': p}
                            for r, p in zip(rows, predictions)],
            'scope': 'Same classifier as baselines on STA features; primary adoption test remains unchanged'})
        print(json.dumps({'macro_accuracy': report['macro_accuracy'], 'class_recall': report['class_recall']}), flush=True)


if __name__ == '__main__':
    main()
