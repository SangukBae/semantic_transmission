#!/usr/bin/env python3
"""Descriptive breakdown of completed v4 scores; never changes registered gates."""
import argparse
from collections import Counter
import csv
from pathlib import Path
import time

from semantic_transmission.metric_v3_formal import read, save
from semantic_transmission.metric_v4_pipeline import rows, verify, METRICS


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    root = ap.parse_args().output
    verify(root, True)
    summary = read(root / 'summary.json')
    calibration = read(root / 'calibration.json')
    manifest = rows(root / 'cases.jsonl')
    events = []
    for row in manifest:
        if row['corpus'] != 'event':
            continue
        scores = {}
        for part in ('visual', 'pixel'):
            scores.update(read(root / 'scores' / part / (row['case_id'] + '.json'))['scores'])
        events.append({**row, 'scores': scores})
    audit = [r['scores']['external_audit'] for r in events if r['variant'] == 'identity']
    unique_source = {name: sum(r[name] for r in audit) for name in
                     ('truth_events', 'matched_source_events', 'predicted_source_events')}
    unique_source.update(sources=len(audit),
        recall=fraction(unique_source['matched_source_events'], unique_source['truth_events']),
        precision=fraction(unique_source['matched_source_events'], unique_source['predicted_source_events']))

    def normal_breakdown(group):
        result = {'cases': len(group)}
        for name in ('ghost_max', 'delay_max'):
            measured = [r['scores'][name] for r in group if r['scores'][name] is not None]
            alarms = sum(x > calibration['thresholds'][name] for x in measured)
            result[name] = {'measured': len(measured), 'alarms': alarms,
                            'alarm_rate': fraction(alarms, len(group)),
                            'saturated_at_one': sum(x == 1 for x in measured)}
        return result

    controls = [r for r in events if r['target'] == 'control']
    normal = {style: normal_breakdown([r for r in controls if r['variant'] == style])
              for style in sorted({r['variant'] for r in controls})}
    predictions = {(r['case_id'], r['classifier']): r for r in summary['STA_predictions']}
    sta = [r for r in manifest if r['corpus'] == 'sta']

    def classify(group, classifier, raw=False):
        valid = [r for r in group if r['eligible']]
        correct = accepted = false_alarm = rejected = 0
        confusion = Counter()
        rejected_raw_correct = 0
        for row in valid:
            p = predictions[row['case_id'], classifier]
            label = (p['raw_prediction'] or 'abstain') if raw else p['prediction']
            correct += label == row['label']
            accepted += label != 'abstain'
            rejected += label == 'abstain'
            rejected_raw_correct += label == 'abstain' and p['raw_prediction'] == row['label']
            false_alarm += row['label'] == 'normal' and label not in ('normal', 'abstain')
            confusion[row['label'] + ' -> ' + label] += 1
        normal_count = sum(r['label'] == 'normal' for r in valid)
        return {'attempts': len(group), 'eligible': len(valid), 'excluded': len(group)-len(valid),
                'correct': correct, 'accepted': accepted, 'abstained': rejected,
                'accuracy': fraction(correct, len(valid)), 'coverage': fraction(accepted, len(valid)),
                'normal_false_alarms': false_alarm, 'normal_cases': normal_count,
                'normal_false_alarm_rate': fraction(false_alarm, normal_count),
                'abstentions_with_correct_raw_prediction': rejected_raw_correct,
                'confusion': dict(confusion)}

    classification = {}
    for name in ('sta', 'stage'):
        classification[name] = {
            'overall': classify(sta, name), 'overall_raw': classify(sta, name, True),
            'by_family': {s: classify([r for r in sta if r['family'] == s], name)
                          for s in sorted({r['family'] for r in sta})},
            'by_kind': {s: classify([r for r in sta if r['kind'] == s], name)
                        for s in sorted({r['kind'] for r in sta})},
            'by_normal_style': {s: classify([r for r in sta if r['label'] == 'normal' and r['variant'] == s], name)
                                for s in sorted({r['variant'] for r in sta if r['label'] == 'normal'})}}

    target_rows = {r['case_id']: r for key in ('ghost', 'delay')
                   for r in summary['event'][key]['numerical_agreement']['rows']}
    csv_rows = []
    for row in events:
        record = {key: row.get(key) for key in ('case_id', 'source_id', 'family', 'target', 'kind', 'variant', 'severity')}
        record.update({key: row['scores'].get(key) for key in METRICS})
        query = target_rows.get(row['case_id'])
        for key in ('ghost_auc', 'delay_s', 'delay_penalty', 'delay_status'):
            record['truth_' + key] = row['oracle'].get(key)
            record['target_' + key] = (query.get('observed') or {}).get(key) if query else None
        csv_rows.append(record)
    with (root / 'event_cases.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        writer.writeheader(); writer.writerows(csv_rows)
    with (root / 'STA_cases.csv').open('w', newline='') as f:
        fields = ('case_id', 'source_id', 'family', 'kind', 'severity', 'label', 'eligible',
                  'sta_prediction', 'sta_raw', 'sta_reason', 'stage_prediction', 'stage_raw', 'stage_reason')
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for row in sta:
            record = {key: row[key] for key in fields[:7]}
            for classifier in ('sta', 'stage'):
                p = predictions.get((row['case_id'], classifier), {})
                record.update({classifier + '_prediction': p.get('prediction'),
                               classifier + '_raw': p.get('raw_prediction'), classifier + '_reason': p.get('reason')})
            writer.writerow(record)

    ghost = [r for r in target_rows.values() if r['kind'] == 'ghost_return']
    reappearance = {'cases': len(ghost),
                    'measured': sum(r['measured'] for r in ghost),
                    'correct_one_return': sum(r['measured'] and r['observed']['ghost_returns'] == 1 for r in ghost)}
    save(root / 'descriptive_diagnostics.json', {
        'created_unix': time.time(), 'status': 'COMPLETED',
        'scope': 'Post-score descriptive breakdown only; no changes to scores, thresholds, or registered decisions',
        'unique_source_extractor_audit': unique_source, 'normal_styles': normal,
        'ghost_reappearance_count': reappearance, 'STA': classification})
    print({'source_extractor': unique_source, 'ghost_returns': reappearance}, flush=True)


if __name__ == '__main__':
    main()
