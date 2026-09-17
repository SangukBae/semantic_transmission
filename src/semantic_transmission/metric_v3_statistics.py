"""Development calibration and source-paired heldout ERE/STA statistics."""
from collections import Counter
import csv
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from .artifacts import sha256
from .automatic_validation import auc
from .metric_v3_formal import BASELINES, SIGNS, read, save, verify
from .sta_video_validation import CLASSES

CURVES = ('tlp_alex_curve', 'tof_farneback_curve', 'tof_raft_curve')
SEGMENTS = dict(zip(CURVES, ('tlp_alex_segments', 'tof_farneback_segments', 'tof_raft_segments')))
PRIMARY = {'reverse': 'ere', 'freeze': 'ere', 'swap': 'ere', 'lag': 'ere',
           'motion_reverse': 'ere', 'motion_freeze': 'ere',
           'object_omission': 'oor', 'object_addition': 'hor', 'object_shape': 'odr'}


def finite(value):
    return value is not None and bool(np.isfinite(value))


def quantile(values):
    values = [v for v in values if finite(v)]
    return float(np.quantile(values, .95, method='higher')) if values else None


def segment_score(values, threshold):
    if threshold is None or not values:
        return None
    active = np.asarray(values) > threshold
    starts = active & np.r_[True, ~active[:-1]]
    return float(starts.sum() / len(active))


def merged(root, corpus, split):
    manifest = 'cases.jsonl' if corpus == 'ere' else 'sta_cases.jsonl'
    expected = [json.loads(x) for x in (root / manifest).read_text().splitlines()]
    expected = [r for r in expected if r['split'] == split and r.get('eligible', True)]
    result = []
    for row in expected:
        values = {}
        for part in ('visual', 'pixel'):
            path = root / 'scores' / part / corpus / (row['case_id'] + '.json')
            data = read(path)
            if data['protocol_sha256'] != sha256(root / 'protocol.json'):
                raise ValueError('score provenance mismatch')
            if data['source_pixel_sha256'] != row['source_pixel_sha256']:
                raise ValueError('source mismatch')
            values.update(data['scores'])
        result.append({**row, 'scores': values})
    return result


def apply_segments(rows, thresholds):
    for row in rows:
        for curve, name in SEGMENTS.items():
            row['scores'][name] = segment_score(row['scores'].get(curve, []), thresholds[row['domain']][curve])


def fit_classifier(rows, names):
    rows = [r for r in rows if r['label'] in CLASSES]
    raw = np.asarray([[r['scores'].get(n) if finite(r['scores'].get(n)) else np.nan for n in names] for r in rows], float)
    count = np.isfinite(raw).sum(0)
    mean = np.divide(np.nansum(raw, axis=0), count, out=np.zeros(len(names)), where=count > 0)
    filled = np.where(np.isfinite(raw), raw, mean)
    scale = filled.std(0); scale[scale < 1e-12] = 1.
    x = (filled - mean) / scale
    # Equal source weighting within each class also balances the two CLR variants.
    centers = []
    for label in CLASSES:
        source_centers = []
        for sid in sorted({r['source_id'] for r in rows if r['label'] == label}):
            indices = [i for i, r in enumerate(rows) if r['label'] == label and r['source_id'] == sid]
            source_centers.append(x[indices].mean(0))
        if not source_centers:
            raise ValueError('no development examples for attribution class ' + label)
        centers.append(np.mean(source_centers, axis=0).tolist())
    return {'features': names, 'imputation_mean': mean.tolist(), 'scale': scale.tolist(), 'centroids': centers,
            'training_sources': sorted({r['source_id'] for r in rows}), 'classes': CLASSES}


def predict_classifier(row, fitted):
    raw = np.array([row['scores'].get(n) if finite(row['scores'].get(n)) else np.nan for n in fitted['features']])
    raw = np.where(np.isfinite(raw), raw, fitted['imputation_mean'])
    x = (raw - fitted['imputation_mean']) / fitted['scale']
    return CLASSES[int(np.square(np.asarray(fitted['centroids']) - x).sum(1).argmin())]


def calibrate(root):
    p, _ = verify(root)
    if (root / 'calibration.json').exists():
        raise FileExistsError('calibration is immutable once written')
    for path in (root / 'scores').rglob('*.json'):
        if read(path)['split'] == 'heldout':
            raise ValueError('cannot calibrate after heldout scores')
    rows, sta = merged(root, 'ere', 'development'), merged(root, 'sta', 'development')
    domains = sorted({r['domain'] for r in rows})
    curves = {d: {c: quantile([v for r in rows if r['domain'] == d and r['target'] == 'control'
                               for v in r['scores'].get(c, [])]) for c in CURVES} for d in domains}
    apply_segments(rows, curves); apply_segments(sta, curves)
    thresholds = {d: {n: quantile([SIGNS[n] * r['scores'][n] for r in rows if r['domain'] == d and
                       r['target'] == 'control' and finite(r['scores'].get(n))]) for n in SIGNS} for d in domains}
    sta_thresholds = {n: quantile([r['scores']['sta'].get(n) for r in sta if r['label'] == 'normal']) for n in CLASSES}
    classifiers = {n: fit_classifier(sta, [n]) for n in BASELINES}
    classifiers['all_baselines'] = fit_classifier(sta, list(BASELINES))
    value = {'created_unix': time.time(), 'protocol_sha256': sha256(root / 'protocol.json'),
             'split': 'development', 'heldout_scores_observed': 0, 'thresholds': thresholds,
             'curve_thresholds': curves, 'sta_thresholds': sta_thresholds, 'sta_classifiers': classifiers,
             'development_case_count': len(rows), 'sta_development_case_count': len(sta),
             'development_source_ids': sorted({r['source_id'] for r in rows})}
    save(root / 'calibration.json', value)
    (root / 'calibration.sha256').write_text(sha256(root / 'calibration.json') + '\n')
    print('development thresholds frozen', len(rows), len(sta), flush=True)


def bootstrap_mean(values):
    if not values:
        return None
    values = np.asarray(values, float)
    rng = np.random.default_rng(20260914)
    draws = rng.integers(0, len(values), (500, len(values)))
    return np.quantile(values[draws].mean(1), [.025, .975]).tolist()


def rate(rows, metric, threshold):
    measurable = [r for r in rows if finite(r['scores'].get(metric))]
    hits = sum(SIGNS.get(metric, 1) * r['scores'][metric] > threshold for r in measurable) if threshold is not None else 0
    source_rates = []
    for sid in sorted({r['source_id'] for r in rows}):
        selected = [r for r in rows if r['source_id'] == sid]
        source_rates.append(sum(finite(r['scores'].get(metric)) and threshold is not None and
            SIGNS.get(metric, 1) * r['scores'][metric] > threshold for r in selected) / len(selected))
    return {'cases': len(rows), 'measured': len(measurable), 'coverage': len(measurable) / len(rows) if rows else None,
            'detected': hits, 'rate_abstentions_as_no_detection': hits / len(rows) if rows else None,
            'rate_measurable_only': hits / len(measurable) if measurable else None,
            'source_bootstrap_95ci': bootstrap_mean(source_rates)}


def area(positive, negative, metric):
    pos = [SIGNS.get(metric, 1) * r['scores'][metric] for r in positive if finite(r['scores'].get(metric))]
    neg = [SIGNS.get(metric, 1) * r['scores'][metric] for r in negative if finite(r['scores'].get(metric))]
    return auc(pos, neg) if pos and neg else None


def paired(positive, negative, candidate, baseline):
    differences, sources = [], []
    for sid in sorted({r['source_id'] for r in positive + negative}):
        pos = [r for r in positive if r['source_id'] == sid and finite(r['scores'].get(candidate)) and finite(r['scores'].get(baseline))]
        neg = [r for r in negative if r['source_id'] == sid and finite(r['scores'].get(candidate)) and finite(r['scores'].get(baseline))]
        if pos and neg:
            differences.append(area(pos, neg, candidate) - area(pos, neg, baseline)); sources.append(sid)
    ci = bootstrap_mean(differences)
    return {'mean_within_source_auc_difference': float(np.mean(differences)) if differences else None,
            'source_bootstrap_95ci': ci, 'paired_sources': len(sources), 'source_ids': sources,
            'positive_lower_bound': ci is not None and ci[0] > 0,
            'scope': 'jointly measurable cases, source-matched AUC; coverage separately reported'}


def error_summary(pos, controls, candidate, thresholds):
    reports = {}
    for metric in dict.fromkeys((candidate, *BASELINES)):
        th = thresholds[metric]
        positive, negative = rate(pos, metric, th), rate(controls, metric, th)
        a = area(pos, controls, metric)
        passed = (a is not None and a >= .9 and positive['rate_abstentions_as_no_detection'] >= .8 and
                  negative['rate_abstentions_as_no_detection'] <= .1 and
                  positive['coverage'] >= .95 and negative['coverage'] >= .95 and th is not None)
        reports[metric] = {'threshold': th, 'auc': a, 'positive': positive, 'control': negative, 'passed': passed,
            'by_severity': {str(s): rate([r for r in pos if r['severity'] == s], metric, th) for s in sorted({r['severity'] for r in pos})}}
    comparisons = {m: paired(pos, controls, candidate, m) for m in BASELINES if m != candidate}
    return {'candidate': candidate, 'metrics': reports, 'paired_comparisons': comparisons,
            'candidate_adoption_gate': reports[candidate]['passed'],
            'superiority_all_comparators': bool(comparisons) and all(v['positive_lower_bound'] for v in comparisons.values())}


def truth_extra(a, b):
    costs = np.full((len(a), len(b)), 1e6)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            diff = abs(x['direction'] - y['direction'])
            if x['object_id'] == y['object_id'] and x['type'] == y['type'] and min(diff, 8 - diff) <= 1 and abs(x['time_s'] - y['time_s']) <= .25 + 1e-9:
                costs[i, j] = abs(x['time_s'] - y['time_s'])
    matches = sum(costs[i, j] < 1e6 for i, j in zip(*linear_sum_assignment(costs)))
    return len(b) - matches


def classify_sta(values, thresholds):
    candidates = [(values[n], -i, n) for i, n in enumerate(CLASSES) if finite(values.get(n)) and
                  thresholds[n] is not None and values[n] > thresholds[n]]
    return max(candidates)[2] if candidates else 'normal'


def classification_report(rows, predictions):
    faults = [(r, p) for r, p in zip(rows, predictions) if r['label'] in CLASSES]
    counts = {label: Counter(p for r, p in faults if r['label'] == label) for label in CLASSES}
    recalls = {label: counts[label][label] / sum(counts[label].values()) if counts[label] else None for label in CLASSES}
    source_accuracies = []
    for sid in sorted({r['source_id'] for r, _ in faults}):
        vals = []
        for label in CLASSES:
            selected = [(r, p) for r, p in faults if r['source_id'] == sid and r['label'] == label]
            if selected:
                vals.append(sum(p == label for _, p in selected) / len(selected))
        source_accuracies.append(float(np.mean(vals)))
    return {'macro_accuracy': float(np.mean(list(recalls.values()))), 'class_recall': recalls,
            'confusion_counts': {k: dict(v) for k, v in counts.items()}, 'fault_cases': len(faults),
            'source_bootstrap_95ci': bootstrap_mean(source_accuracies),
            'by_kind_severity': {kind + '/' + str(s): sum(p == r['label'] for r, p in faults if r['kind'] == kind and r['severity'] == s) /
                sum(r['kind'] == kind and r['severity'] == s for r, _ in faults)
                for kind, s in sorted({(r['kind'], r['severity']) for r, _ in faults})}}


def summarize(root):
    p, _ = verify(root)
    cal = read(root / 'calibration.json')
    if sha256(root / 'calibration.json') != (root / 'calibration.sha256').read_text().strip():
        raise ValueError('frozen calibration changed')
    rows, sta = merged(root, 'ere', 'heldout'), merged(root, 'sta', 'heldout')
    apply_segments(rows, cal['curve_thresholds']); apply_segments(sta, cal['curve_thresholds'])
    if set(cal['development_source_ids']) & {r['source_id'] for r in rows}:
        raise ValueError('development/test source overlap')
    results = {}
    for domain in sorted({r['domain'] for r in rows}):
        controls = [r for r in rows if r['domain'] == domain and r['target'] == 'control']
        for kind, candidate in PRIMARY.items():
            pos = [r for r in rows if r['domain'] == domain and r['kind'] == kind]
            if pos:
                result = error_summary(pos, controls, candidate, cal['thresholds'][domain])
                result['scope'] = p['domains'][domain]
                result['controls_by_family'] = {family: {n: rate([r for r in controls if r['family'] == family], n, cal['thresholds'][domain][n])
                                                       for n in dict.fromkeys((candidate, *BASELINES))}
                    for family in sorted({r['family'] for r in controls})}
                result['auc_by_control_family'] = {family: {n: area(pos, [r for r in controls if r['family'] == family], n)
                    for n in dict.fromkeys((candidate, *BASELINES))} for family in sorted({r['family'] for r in controls})}
                results[domain + '/' + kind] = result
    truth = read(Path(p['base_run']) / 'truth_audit.json')
    from .event_metric import event_pairs
    truth_lookup = {r['case_id']: r for r in truth['results']}
    extractor = {}
    for domain in sorted({r['domain'] for r in rows}):
        examples = []
        for row in rows:
            if row['domain'] != domain or row['variant'] != 'identity':
                continue
            actual = truth_lookup[row['case_id']]['reference_events']
            observed = row['scores']['reference_events']
            matches = event_pairs(actual, observed)
            examples.append({'source_id': row['source_id'], 'true': len(actual), 'predicted': len(observed),
                'matched': len(matches), 'true_turns': sum(e['type'] == 'turn' for e in actual),
                'matched_turns': sum(actual[i]['type'] == 'turn' for i, _ in matches)})
        total = {k: sum(r[k] for r in examples) for k in ('true', 'predicted', 'matched', 'true_turns', 'matched_turns')}
        extractor[domain] = {**total, 'sources': len(examples), 'by_source': examples,
            'recall': total['matched'] / total['true'] if total['true'] else None,
            'precision': total['matched'] / total['predicted'] if total['predicted'] else None,
            'turn_recall': total['matched_turns'] / total['true_turns'] if total['true_turns'] else None,
            'public_precision_scope': 'selected-object annotation correspondence, not exhaustive event precision'}
    extra_ids = {r['case_id'] for r in truth['results'] if r['accepted'] and
                 truth_extra(r['reference_events'], r['reconstruction_events']) > 0}
    ser_results = {}
    for domain in sorted({r['domain'] for r in rows}):
        pos = [r for r in rows if r['domain'] == domain and r['target'] == 'motion' and r['case_id'] in extra_ids]
        neg = [r for r in rows if r['domain'] == domain and r['target'] == 'control']
        if pos:
            ser_results[domain] = error_summary(pos, neg, 'ser', cal['thresholds'][domain])
            ser_results[domain]['scope'] = 'supplementary truth-extra-event subset; existing motion interventions'
    predictions = [classify_sta(r['scores']['sta'], cal['sta_thresholds']) for r in sta]
    sta_summary = classification_report(sta, predictions)
    normals = [(r, pred) for r, pred in zip(sta, predictions) if r['label'] == 'normal']
    sta_summary['normal_false_alarm_rate'] = sum(pred != 'normal' for _, pred in normals) / len(normals)
    components = {}
    for n in CLASSES:
        mapped = [{**r, 'scores': r['scores']['sta']} for r in sta]
        pos = [r for r in mapped if r['label'] == n]; neg = [r for r in mapped if r['label'] == 'normal']
        components[n] = {'target_vs_normal_auc': area(pos, neg, n), 'target_vs_other_fault_auc': area(pos, [r for r in mapped if r['label'] in CLASSES and r['label'] != n], n),
                         'threshold': cal['sta_thresholds'][n], 'target': rate(pos, n, cal['sta_thresholds'][n]), 'normal': rate(neg, n, cal['sta_thresholds'][n])}
    sta_summary['components'] = components
    sta_summary['adoption_gate'] = (sta_summary['macro_accuracy'] >= .8 and sta_summary['normal_false_alarm_rate'] <= .1 and
        all(v['target_vs_normal_auc'] is not None and v['target_vs_normal_auc'] >= .9 for v in components.values()))
    sta_summary['baselines'] = {n: classification_report(sta, [predict_classifier(r, f) for r in sta]) for n, f in cal['sta_classifiers'].items()}
    sta_summary['chance_macro_accuracy'] = .25
    sta_summary['scope'] = p['sta_simulator']
    save(root / 'sta_predictions.json', [{'case_id': r['case_id'], 'source_id': r['source_id'], 'label': r['label'],
        'prediction': pred, 'components': {k: r['scores']['sta'][k] for k in CLASSES}} for r, pred in zip(sta, predictions)])
    summary = {'status': 'COMPLETED', 'protocol_sha256': sha256(root / 'protocol.json'), 'calibration_sha256': sha256(root / 'calibration.json'),
        'heldout_ere_cases': len(rows), 'heldout_sta_cases': len(sta), 'heldout_sources': dict(Counter(r['domain'] for r in {r['source_id']: r for r in rows}.values())),
        'error_detection': results, 'ser_supplementary': ser_results, 'sta': sta_summary,
        'heldout_source_extractor_audit': extractor,
        'ere_adoption_gate': all(r['candidate_adoption_gate'] for k, r in results.items() if k.startswith('rendered/') and r['candidate'] == 'ere'),
        'ere_superiority_all_comparators': all(r['superiority_all_comparators'] for k, r in results.items() if k.startswith('rendered/') and r['candidate'] == 'ere'),
        'novelty_demonstrated': False, 'new_human_review': False}
    save(root / 'summary.json', summary)
    with (root / 'scores.csv').open('w', newline='') as f:
        columns = ['case_id', 'source_id', 'split', 'domain', 'kind', 'severity', *SIGNS]
        writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, r['scores'].get(k)) for k in columns})
    print('heldout analysis complete', len(rows), len(sta), flush=True)
