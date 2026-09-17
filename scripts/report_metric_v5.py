#!/usr/bin/env python3
"""Frozen statistics for the FSO/EOI/UEP experiment.

Declared with `--declare` before any heldout score exists; the declared script
hash is checked again at report time.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path
import time

import numpy as np

from semantic_transmission.artifacts import sha256
from semantic_transmission.automatic_validation import auc
from semantic_transmission.metric_v3_formal import read, save
from semantic_transmission.metric_v5_pipeline import (BASELINES, CANDIDATES, ABLATIONS, GATE,
                                                      METRICS, PRIMARY, rows, verify)

ITERATIONS, SEED = 1000, 20260915


def finite(x):
    return x is not None and bool(np.isfinite(x))


def interval(values):
    if not values:
        return None
    rng = np.random.default_rng(SEED)
    x = np.asarray(values, float)
    return np.quantile(x[rng.integers(len(x), size=(ITERATIONS, len(x)))].mean(1), [.025, .975]).tolist()


def rate(group, metric, threshold):
    valid = [r for r in group if finite(r['scores'].get(metric))]
    detected = [r for r in valid if threshold is not None and r['scores'][metric] > threshold]
    sources = sorted({r['source_id'] for r in group})
    by_source = [sum(r['source_id'] == s for r in detected) / sum(r['source_id'] == s for r in group)
                 for s in sources]
    return {'cases': len(group), 'measured': len(valid),
            'coverage': len(valid) / len(group) if group else None, 'detected': len(detected),
            'rate': len(detected) / len(group) if group else None, 'source_95ci': interval(by_source)}


def area(group, control, metric):
    return auc([r['scores'][metric] for r in group if finite(r['scores'].get(metric))],
               [r['scores'][metric] for r in control if finite(r['scores'].get(metric))])


def detection(group, control, metric, threshold):
    a = area(group, control, metric)
    p, n = rate(group, metric, threshold), rate(control, metric, threshold)
    gate = bool(a is not None and a >= GATE['auc'] and p['rate'] is not None and p['rate'] >= GATE['tpr']
                and n['rate'] is not None and n['rate'] <= GATE['fpr']
                and p['coverage'] is not None and n['coverage'] is not None
                and min(p['coverage'], n['coverage']) >= GATE['coverage'])
    return {'auc': a, 'positive': p, 'normal': n, 'threshold': threshold, 'detection_gate': gate}


def paired_auc(group, controls, candidate, baseline):
    group = [r for r in group if all(finite(r['scores'].get(n)) for n in (candidate, baseline))]
    controls = [r for r in controls if all(finite(r['scores'].get(n)) for n in (candidate, baseline))]
    differences, sources = [], []
    for source in sorted({r['source_id'] for r in group}):
        p = [r for r in group if r['source_id'] == source]
        n = [r for r in controls if r['source_id'] == source]
        a, b = area(p, n, candidate), area(p, n, baseline)
        if a is not None and b is not None:
            differences.append(a - b)
            sources.append(source)
    ci = interval(differences)
    return {'paired_sources': len(sources),
            'mean_source_AUC_difference': float(np.mean(differences)) if differences else None,
            'source_95ci': ci, 'positive_lower_bound': bool(ci is not None and ci[0] > 0)}


def oracle_agreement(group):
    """Independent renderer occupancy against the measured target-event record."""
    measured, all_rows = [], []
    for row in group:
        records = [r for r in row['scores'].get('target_records', []) if r['status'] == 'observed']
        value = max((r['score'] for r in records), default=None)
        item = {'case_id': row['case_id'], 'source_id': row['source_id'], 'kind': row['kind'],
                'severity': row['severity'], 'oracle': row['oracle'].get('occupancy'),
                'observed': value, 'records': len(row['scores'].get('target_records', [])),
                'observed_records': len(records)}
        all_rows.append(item)
        if finite(value):
            measured.append(item)
    errors = [abs(r['observed'] - r['oracle']) for r in measured]
    by_source = [float(np.mean([abs(r['observed'] - r['oracle']) for r in measured if r['source_id'] == s]))
                 for s in sorted({r['source_id'] for r in measured})]
    return {'cases': len(group), 'measured': len(measured),
            'target_event_coverage': len(measured) / len(group) if group else None,
            'mean_absolute_error': float(np.mean(errors)) if errors else None,
            'MAE_source_95ci': interval(by_source), 'rows': all_rows,
            'scope': 'renderer occupancy oracle vs the candidate record for the declared target event; '
                     'missed source events stay in the coverage denominator'}


def extractor_audit(group):
    """Heldout source-event recall against renderer annotation, identity cases only."""
    totals = Counter()
    by_type = {kind: Counter() for kind in ('enter', 'exit', 'start', 'stop', 'turn')}
    sources = 0
    for row in group:
        if row['variant'] != 'identity':
            continue
        sources += 1
        audited = row['scores']['source_event_audit']
        totals['truth'] += len(audited)
        totals['matched'] += sum(e['observed'] is not None for e in audited)
        for e in audited:
            by_type[e['type']]['truth'] += 1
            by_type[e['type']]['matched'] += e['observed'] is not None
    return {'sources': sources, 'truth_events': totals['truth'], 'matched_events': totals['matched'],
            'recall': totals['matched'] / totals['truth'] if totals['truth'] else None,
            'by_type': {k: {**dict(v), 'recall': v['matched'] / v['truth'] if v['truth'] else None}
                        for k, v in by_type.items()},
            'scope': 'renderer annotation audit of the heldout sources; not a candidate score'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--declare', action='store_true')
    args = parser.parse_args()
    root = args.output
    verify(root)
    declaration = root / 'statistics_declaration.json'
    if args.declare:
        if declaration.exists() or (root / 'scores').exists():
            raise ValueError('Declare before any score')
        save(declaration, {'created_unix': time.time(), 'script_sha256': sha256(Path(__file__)),
                           'heldout_scores_observed': 0,
                           'scope': 'frozen statistics for the FSO/EOI/UEP heldout analysis'})
        print('statistics declared')
        return
    verify(root, True)
    if read(declaration)['script_sha256'] != sha256(Path(__file__)):
        raise ValueError('Statistics changed after declaration')
    calibration = read(root / 'calibration.json')
    thresholds = calibration['thresholds']
    merged = []
    for row in rows(root / 'cases.jsonl'):
        if row['split'] != 'heldout':
            continue
        scores = {}
        for part in ('visual', 'pixel'):
            record = read(root / 'scores' / part / (row['case_id'] + '.json'))
            if record['calibration_sha256'] != sha256(root / 'calibration.json'):
                raise ValueError('score calibration mismatch')
            if record['source_pixel_sha256'] != row['source_pixel_sha256']:
                raise ValueError('source mismatch')
            scores.update(record['scores'])
        merged.append({**row, 'scores': scores})
    events = [r for r in merged if r['corpus'] == 'event']
    controls = [r for r in events if r['target'] == 'control']
    sta = [r for r in merged if r['corpus'] == 'sta']
    sta_controls = [r for r in sta if r['label'] == 'normal']
    results = {}
    for kind, candidate in PRIMARY.items():
        if kind == 'unsupported_addition':
            group = [r for r in sta if r['kind'] == kind]
            negatives = sta_controls
        else:
            group = [r for r in events if r['kind'] == kind]
            negatives = controls
        if not group:
            continue
        stats = {n: detection(group, negatives, n, thresholds.get(n)) for n in METRICS}
        entry = {'candidate': candidate, 'positives': len(group), 'metrics': stats,
                 'by_severity': {str(s): detection([r for r in group if r['severity'] == s], negatives,
                                                   candidate, thresholds.get(candidate))
                                 for s in sorted({r['severity'] for r in group})},
                 'by_family': {f: detection([r for r in group if r['family'] == f],
                                            [r for r in negatives if r['family'] == f],
                                            candidate, thresholds.get(candidate))
                               for f in sorted({r['family'] for r in group})},
                 'paired_comparisons': {n: paired_auc(group, negatives, candidate, n)
                                        for n in METRICS if n != candidate},
                 'truth_event_changed': dict(Counter(str(r['truth_event_changed']) for r in group))}
        if kind not in ('order_swap', 'unsupported_addition'):
            entry['oracle_agreement'] = oracle_agreement(group)
            agreement = (entry['oracle_agreement']['mean_absolute_error'] is not None
                         and entry['oracle_agreement']['mean_absolute_error'] <= GATE['oracle_mean_absolute_error']
                         and entry['oracle_agreement']['target_event_coverage'] >= GATE['target_event_coverage'])
        else:
            agreement = True
        entry['adoption_gate'] = bool(stats[candidate]['detection_gate'] and agreement)
        entry['superiority_all_comparators'] = bool(
            all(v['positive_lower_bound'] for n, v in entry['paired_comparisons'].items() if n in BASELINES))
        results[kind] = entry
    # UEP must not fire on the other three transport faults.
    other_faults = {kind: rate([r for r in sta if r['kind'] == kind], 'uep', thresholds.get('uep'))
                    for kind in sorted({r['kind'] for r in sta if r['label'] not in ('normal', 'ghr')})}
    normal_by_variant = {v: {n: rate([r for r in controls if r['variant'] == v], n, thresholds.get(n))
                             for n in CANDIDATES + ABLATIONS}
                         for v in sorted({r['variant'] for r in controls})}
    summary = {'status': 'COMPLETED', 'completed_unix': time.time(),
               'protocol_sha256': sha256(root / 'protocol.json'),
               'calibration_sha256': sha256(root / 'calibration.json'),
               'thresholds': thresholds, 'gates': GATE,
               'heldout_sources': len({r['source_id'] for r in merged}),
               'case_counts': dict(Counter(r['corpus'] for r in merged)),
               'error_detection': results,
               'uep_other_transport_faults': other_faults,
               'normal_false_alarms_by_variant': normal_by_variant,
               'heldout_extractor_audit': extractor_audit(events),
               'candidates_passing': [k for k, v in results.items() if v['adoption_gate']],
               'candidates_superior_to_all_comparators': [k for k, v in results.items()
                                                          if v['adoption_gate'] and v['superiority_all_comparators']],
               'new_human_review': False, 'novelty_demonstrated': False,
               'scope': read(root / 'protocol.json')['evaluation_scope']}
    save(root / 'summary.json', summary)
    with (root / 'scores.csv').open('w', newline='') as f:
        columns = ['case_id', 'source_id', 'family', 'split', 'corpus', 'target', 'kind', 'severity',
                   'variant', 'truth_event_changed', *METRICS]
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for r in merged:
            writer.writerow({**{k: r.get(k) for k in columns}, **{k: r['scores'].get(k) for k in METRICS}})
    print('PASSED:', summary['candidates_passing'])
    print('SUPERIOR:', summary['candidates_superior_to_all_comparators'])
    for kind, entry in results.items():
        primary = entry['metrics'][entry['candidate']]
        print(f"  {kind:22s} {entry['candidate']:14s} auc={primary['auc']} tpr={primary['positive']['rate']} "
              f"fpr={primary['normal']['rate']} cov={primary['positive']['coverage']} gate={entry['adoption_gate']}")


if __name__ == '__main__':
    main()
