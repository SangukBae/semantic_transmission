#!/usr/bin/env python3
"""Predeclared STA nuisance test on truth-audited semantic-preserving outputs.

TX/RX are the original decoded PNG video; final reconstruction is a previously
audited normal variant. Uses cached RGB observations and frozen STA thresholds.
"""
import argparse
from collections import OrderedDict
import json
from pathlib import Path
import runpy
import time

from semantic_transmission.artifacts import sha256
from semantic_transmission.metric_v3_features import VisualObserver, pixels
from semantic_transmission.metric_v3_formal import array, read, save, verify
from semantic_transmission.metric_v3_statistics import classify_sta, bootstrap_mean, area, CLASSES


class CachedObserver(VisualObserver):
    def __init__(self, root, signature):
        self.root = root / 'visual_cache' / signature
        self.recent = OrderedDict()

    def observe(self, frames):
        path = self.root / (pixels(frames) + '.pkl.gz')
        if not path.exists():
            raise FileNotFoundError('Formal RGB extraction must finish first: ' + str(path))
        return super().observe(frames)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--phase', choices=('declare', 'development', 'heldout'), required=True)
    args = ap.parse_args(); root = args.output; protocol, _ = verify(root)
    declaration_path = root / 'sta_semantic_controls_declaration.json'
    if args.phase == 'declare':
        if declaration_path.exists():
            raise FileExistsError(declaration_path)
        if any(read(f)['split'] == 'heldout' for f in (root / 'scores').rglob('*.json')):
            raise ValueError('Declare before any heldout score')
        save(declaration_path, {'created_unix': time.time(), 'heldout_scores_observed': 0,
            'script_sha256': sha256(Path(__file__)), 'protocol_sha256': sha256(root / 'protocol.json'),
            'inputs': 'reference/TX/RX identical decoded original PNG frames, reconstruction is an existing eligible rendered normal variant',
            'normal_variants': 'all 10 registered controls per rendered source, including identity; 160 development, 320 heldout',
            'decision': 'unchanged main calibration and classify_sta; no supplementary threshold fit',
            'outputs': 'false alarm rate by normal family/variant, source bootstrap CI, each component fault-vs-semantic-normal AUC',
            'scope': 'predeclared supplemental robustness test, not a replacement or relaxed primary adoption criterion'})
        return
    declaration = read(declaration_path)
    if declaration['script_sha256'] != sha256(Path(__file__)):
        raise ValueError('Supplement script changed')
    calibration = read(root / 'calibration.json')
    if sha256(root / 'calibration.json') != (root / 'calibration.sha256').read_text().strip():
        raise ValueError('Main thresholds changed')
    if args.phase == 'heldout' and not (root / 'sta_semantic_controls_development.json').exists():
        raise ValueError('Development supplement must finish before heldout')
    wrapper = Path('scripts/run_metric_v3_memoized.py')
    if read(root / 'memoization_probe.json')['script_sha256'] != sha256(wrapper):
        raise ValueError('Operational wrapper changed')
    runpy.run_path(str(wrapper), run_name='memoization_helpers')['enable']()
    from semantic_transmission.sta_video_validation import score_video_attribution
    observer = CachedObserver(root, protocol['signature'])
    selected = [json.loads(x) for x in (root / 'cases.jsonl').read_text().splitlines()]
    selected = [r for r in selected if r['split'] == args.phase and r['domain'] == 'rendered' and r['target'] == 'control']
    scored = []
    for index, row in enumerate(selected):
        base = Path(protocol['base_run'])
        a = array(base / 'inputs' / (row['source_stem'] + '.npz'), row['source_pixel_sha256'])
        b = array(base / 'cases' / (row['case_id'] + '.npz'), row['reconstruction_pixel_sha256'])
        values = score_video_attribution(observer, a, b, a, a, ['ok'] * len(a), ['ok'] * len(a))
        scored.append({**row, 'scores': {n: values[n] for n in CLASSES},
                       'prediction': classify_sta(values, calibration['sta_thresholds'])})
        if (index + 1) % 20 == 0:
            print(args.phase, index + 1, '/', len(selected), flush=True)
    groups = {'all': scored}
    groups.update({family: [r for r in scored if r['family'] == family] for family in ('pixel', 'rerender')})
    groups.update({v: [r for r in scored if r['variant'] == v] for v in sorted({r['variant'] for r in scored})})
    reports = {}
    for name, rows in groups.items():
        sources = sorted({r['source_id'] for r in rows})
        rates = [sum(r['prediction'] != 'normal' for r in rows if r['source_id'] == sid) /
                 sum(r['source_id'] == sid for r in rows) for sid in sources]
        reports[name] = {'cases': len(rows), 'false_alarms': sum(r['prediction'] != 'normal' for r in rows),
            'false_alarm_rate': sum(r['prediction'] != 'normal' for r in rows) / len(rows),
            'source_bootstrap_95ci': bootstrap_mean(rates)}
    if args.phase == 'heldout':
        faults = read(root / 'sta_predictions.json')
        component_auc = {n: area([{'scores': r['components']} for r in faults if r['label'] == n], scored, n) for n in CLASSES}
    else:
        component_auc = None
    save(root / ('sta_semantic_controls_' + args.phase + '.json'), {'completed_unix': time.time(),
        'declaration_sha256': sha256(declaration_path), 'calibration_sha256': sha256(root / 'calibration.json'),
        'rows': scored, 'normal_false_alarm': reports, 'component_fault_vs_semantic_normal_auc': component_auc,
        'scope': 'predeclared semantic-preserving final RGB controls; primary experiment unchanged'})
    print(json.dumps(reports['all']), flush=True)


if __name__ == '__main__':
    main()
