#!/usr/bin/env python3
"""Descriptive analysis of completed, frozen ERE/STA scores.

No thresholds or classifiers are fitted here. Examples and plots are post-hoc
descriptions; they are not a second confirmatory experiment.
"""
import argparse
from pathlib import Path

import numpy as np

from semantic_transmission.metric_v3_formal import BASELINES, read, save, verify
from semantic_transmission.metric_v3_statistics import (
    CLASSES, area, bootstrap_mean, finite, merged, predict_classifier, rate,
)


def distribution(rows, metric):
    values = [r['scores'][metric] for r in rows if finite(r['scores'].get(metric))]
    return {'cases': len(rows), 'measured': len(values), 'null': len(rows) - len(values),
            'zero': sum(v == 0 for v in values),
            'minimum': min(values) if values else None,
            'median': float(np.median(values)) if values else None,
            'q95_higher': float(np.quantile(values, .95, method='higher')) if values else None,
            'maximum': max(values) if values else None}


def source_accuracy(rows, predictions):
    result = {}
    for sid in sorted({r['source_id'] for r in rows if r['label'] in CLASSES}):
        recalls = []
        for label in CLASSES:
            selected = [(r, p) for r, p in zip(rows, predictions)
                        if r['source_id'] == sid and r['label'] == label]
            recalls.append(sum(r['label'] == p for r, p in selected) / len(selected))
        result[sid] = float(np.mean(recalls))
    return result


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True)
    root = ap.parse_args().output; verify(root)
    summary, calibration = read(root / 'summary.json'), read(root / 'calibration.json')
    development, heldout = merged(root, 'ere', 'development'), merged(root, 'ere', 'heldout')
    controls = {}
    for domain in ('rendered', 'public_video'):
        controls[domain] = {split: {family: distribution(
            [r for r in rows if r['domain'] == domain and r['target'] == 'control'
             and (family == 'all' or r['family'] == family)], 'ere')
            for family in ['all', *sorted({r['family'] for r in rows
                if r['domain'] == domain and r['target'] == 'control'})]}
            for split, rows in (('development', development), ('heldout', heldout))}
    examples = []
    for domain in ('rendered', 'public_video'):
        threshold = calibration['thresholds'][domain]['ere']
        rows = [r for r in heldout if r['domain'] == domain]
        groups = {'normal_false_alarm': [r for r in rows if r['target'] == 'control'
            and finite(r['scores'].get('ere')) and threshold is not None and r['scores']['ere'] > threshold]}
        for kind in ('reverse', 'freeze', 'swap', 'lag', 'motion_reverse', 'motion_freeze'):
            groups[kind + '_short_miss'] = [r for r in rows if r['kind'] == kind and r['severity'] == .125
                and (not finite(r['scores'].get('ere')) or threshold is None or r['scores']['ere'] <= threshold)]
        for reason, candidates in groups.items():
            candidates.sort(key=lambda r: r['case_id'])
            if candidates:
                r = candidates[0]
                examples.append({k: r[k] for k in ('case_id', 'source_id', 'domain', 'kind', 'severity')} |
                    {'reason': reason, 'selection': 'lexicographically first matching case, after scoring',
                     'ere': r['scores']['ere'], 'threshold': threshold,
                     'reference_event_count': r['scores']['reference_event_count'],
                     'reconstruction_event_count': r['scores']['reconstruction_event_count'],
                     'matched_event_count': len(r['scores']['event_pairs'])})
    sta = read(root / 'sta_predictions.json')
    sta_by_source = source_accuracy(sta, [r['prediction'] for r in sta])
    head_report = read(root / 'sta_classifier_head_results.json')
    head = head_report['predictions']
    head_by_source = source_accuracy(head, [r['prediction'] for r in head])
    supplement = read(root / 'same_input_baseline_heldout.json')['rows']
    fitted = read(root / 'same_input_baseline_calibration.json')['classifiers']
    paired, head_paired = {}, {}
    for name, classifier in fitted.items():
        per_source = source_accuracy(supplement, [predict_classifier(r, classifier) for r in supplement])
        if set(per_source) != set(sta_by_source):
            raise ValueError('same-input comparison source mismatch')
        differences = [sta_by_source[sid] - per_source[sid] for sid in sorted(per_source)]
        paired[name] = {'mean_source_macro_accuracy_difference_STA_minus_baseline': float(np.mean(differences)),
            'source_bootstrap_95ci': bootstrap_mean(differences), 'paired_sources': len(per_source),
            'sta_accuracy_by_source': sta_by_source, 'baseline_accuracy_by_source': per_source,
            'scope': 'descriptive paired analysis of predeclared same-input baselines; no retuning'}
        differences = [head_by_source[sid] - per_source[sid] for sid in sorted(per_source)]
        head_paired[name] = {'mean_source_macro_accuracy_difference_STA_features_minus_baseline': float(np.mean(differences)),
            'source_bootstrap_95ci': bootstrap_mean(differences), 'paired_sources': len(per_source),
            'scope': 'predeclared STA feature classifier and same-input baselines; same classifier family, post-hoc paired CI, no retuning'}
    row_lookup = {r['case_id']: r for r in merged(root, 'sta', 'heldout')}
    support_patterns = {}
    for row in sta:
        raw = row_lookup[row['case_id']]['scores']['sta']
        key = row['label'] + '->' + row['prediction']
        group = support_patterns.setdefault(key, {'cases': 0, 'reference_unit_counts': [],
            'reconstructed_unit_counts': [], 'supported_in_RX_but_not_TX': 0, 'example_case_ids': []})
        group['cases'] += 1
        group['reference_unit_counts'].append(len(raw['units_reference']))
        group['reconstructed_unit_counts'].append(len(raw['units_reconstruction']))
        group['supported_in_RX_but_not_TX'] += raw['rx_supported_without_tx_support']
        if len(group['example_case_ids']) < 3:
            group['example_case_ids'].append(row['case_id'])
    for group in support_patterns.values():
        for name in ('reference_unit_counts', 'reconstructed_unit_counts'):
            values = group[name]
            group[name] = {'min': min(values), 'mean': float(np.mean(values)), 'max': max(values)}
    truth_scope = {r['case_id']: r for r in read(root / 'truth_metric_scope.json')['cases']}
    alignment = {}
    for domain in ('rendered', 'public_video'):
        normal = [r for r in heldout if r['domain'] == domain and r['target'] == 'control']
        for category in ('extra_only', 'has_missing'):
            selected = [r for r in heldout if r['domain'] == domain and r['target'] == 'motion'
                and (truth_scope[r['case_id']]['true_missing_events'] > 0 if category == 'has_missing'
                     else truth_scope[r['case_id']]['category'] == 'extra_only')]
            alignment[domain + '/' + category] = {'cases': len(selected),
                'scope': 'post-hoc annotation-defined subgroup, frozen thresholds, not a replacement adoption test',
                'metrics': {name: {'auc': area(selected, normal, name),
                    'positive': rate(selected, name, calibration['thresholds'][domain][name])}
                    for name in ('ere', 'ser', 'mte_tail', 'lpips_alex', 'tof_raft_small')},
                'by_kind': {kind: {name: rate([r for r in selected if r['kind'] == kind], name,
                    calibration['thresholds'][domain][name]) for name in ('ere', 'ser')}
                    for kind in sorted({r['kind'] for r in selected})}}
    sta_truth = {r['case_id']: r for r in read(root / 'sta_truth_target_audit.json')['cases']}
    gfr_groups = {}
    for category in ('has_missing', 'extra_only'):
        selected = [r for r in sta if r['label'] == 'gfr' and
                    (sta_truth[r['case_id']]['true_missing_events'] > 0 if category == 'has_missing'
                     else sta_truth[r['case_id']]['event_change_category'] == 'extra_only')]
        gfr_groups[category] = {'cases': len(selected),
            'predicted_gfr': sum(r['prediction'] == 'gfr' for r in selected),
            'predicted_gfr_rate': sum(r['prediction'] == 'gfr' for r in selected) / len(selected) if selected else None,
            'predictions': {label: sum(r['prediction'] == label for r in selected) for label in (*CLASSES, 'normal')},
            'scope': 'post-hoc target-alignment diagnostic; original primary labels/gate retained; extra-only events are outside an omission-only GFR target'}
    save(root / 'analysis.json', {'scope': 'post-hoc explanation; frozen primary statistics unchanged',
        'control_ere_distributions': controls, 'deterministic_examples': examples,
        'same_input_paired_STA_comparisons': paired, 'sta_support_patterns': support_patterns,
        'same_classifier_head_source_paired_comparisons': head_paired,
        'metric_target_alignment': alignment, 'gfr_target_alignment': gfr_groups})

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ['ere', *BASELINES]
    kinds = ['reverse', 'freeze', 'swap', 'lag', 'motion_reverse', 'motion_freeze']
    fig, axes = plt.subplots(1, 2, figsize=(14, 10), layout='constrained')
    for ax, field, title in zip(axes, ('auc', 'tpr'), ('AUC', 'Detection rate at frozen threshold')):
        matrix = []
        for name in names:
            values = []
            for kind in kinds:
                result = summary['error_detection']['rendered/' + kind]['metrics'][name]
                values.append(result['auc'] if field == 'auc' else result['positive']['rate_abstentions_as_no_detection'])
            matrix.append(values)
        values = np.asarray(matrix, dtype=float)
        im = ax.imshow(values, vmin=0, vmax=1, cmap='viridis', aspect='auto')
        for y in range(len(names)):
            for x in range(len(kinds)):
                value = values[y, x]
                ax.text(x, y, 'null' if not np.isfinite(value) else f'{value:.2f}', ha='center', va='center',
                        color='black' if value > .55 else 'white', fontsize=8)
        ax.set(xticks=range(len(kinds)), xticklabels=kinds, yticks=range(len(names)), yticklabels=names, title=title)
        ax.tick_params(axis='x', labelrotation=35)
    fig.suptitle('Heldout synthetic sources: ERE and all registered comparators')
    fig.colorbar(im, ax=axes, shrink=.6)
    fig.savefig(root / 'ere_baseline_matrix.png', dpi=170); fig.savefig(root / 'ere_baseline_matrix.svg'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    labels = [*CLASSES, 'normal']
    confusion = np.asarray([[sum(r['label'] == actual and r['prediction'] == pred for r in sta)
        for pred in labels] for actual in labels])
    rates = confusion / confusion.sum(1, keepdims=True)
    axes[0].imshow(rates, vmin=0, vmax=1, cmap='Blues')
    for y in range(5):
        for x in range(5):
            axes[0].text(x, y, f'{confusion[y,x]}\n{rates[y,x]:.0%}', ha='center', va='center',
                         color='white' if rates[y,x] > .55 else 'black', fontsize=9)
    axes[0].set(xticks=range(5), xticklabels=labels, yticks=range(5), yticklabels=labels,
                xlabel='STA prediction', ylabel='Known intervention', title='STA confusion (normal: unchanged PNG)')
    supplemental = read(root / 'same_input_baseline_results.json')['results']
    bars = [('A/B only\nall baselines', summary['sta']['baselines']['all_baselines']),
            ('STA\nA, TX, RX, B', summary['sta']),
            ('STA + classifier\nA, TX, RX, B', head_report['results']),
            ('Stage MSE\nA, TX, RX, B', supplemental['mse']),
            ('Stage MAE\nA, TX, RX, B', supplemental['mae']),
            ('MSE+MAE\nA, TX, RX, B', supplemental['mse_and_mae'])]
    values = [r['macro_accuracy'] for _, r in bars]
    intervals = [r['source_bootstrap_95ci'] for _, r in bars]
    yerr = np.array([[max(0, v - ci[0]), max(0, ci[1] - v)] for v, ci in zip(values, intervals)]).T
    axes[1].bar(range(len(bars)), values, yerr=yerr, capsize=4,
                color=['gray', 'tab:blue', 'tab:green', 'tab:orange', 'tab:orange', 'tab:orange'])
    for i, value in enumerate(values):
        axes[1].text(i, min(1.10, max(value, intervals[i][1]) + .025), f'{value:.1%}', ha='center', fontsize=9)
    axes[1].axhline(.25, color='gray', linestyle=':'); axes[1].axhline(.8, color='gray', linestyle='--')
    axes[1].set(xticks=range(len(bars)), xticklabels=[n for n, _ in bars], ylim=(0, 1.15),
                ylabel='Four-class macro accuracy', title='Attribution comparison; source bootstrap 95% CI')
    axes[1].tick_params(axis='x', labelsize=8)
    fig.supxlabel('Right: classification among known fault cases; STA + classifier has no normal/rejection class.', fontsize=9)
    fig.savefig(root / 'sta_input_matched_comparison.png', dpi=170)
    fig.savefig(root / 'sta_input_matched_comparison.svg'); plt.close(fig)
    normal = read(root / 'sta_semantic_controls_heldout.json')['normal_false_alarm']
    names = ['all', 'identity', 'brightness20', 'gamma075', 'jpeg75', 'restyle_texture',
             'restyle_palette', 'restyle_lighting', 'restyle_camera', 'restyle_instance', 'restyle_gait']
    values = [normal[n]['false_alarm_rate'] for n in names]
    ci = [normal[n]['source_bootstrap_95ci'] for n in names]
    xerr = np.asarray([[max(0, v - c[0]), max(0, c[1] - v)] for v, c in zip(values, ci)]).T
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    ax.barh(range(len(names)), values, xerr=xerr, capsize=3, color=['tab:blue', *(['tab:orange'] * 10)])
    for i, n in enumerate(names):
        ax.text(min(1.10, max(values[i], ci[i][1]) + .02), i,
                f"{normal[n]['false_alarms']}/{normal[n]['cases']} ({values[i]:.1%})", va='center', fontsize=8)
    ax.axvline(.1, color='gray', linestyle='--')
    ax.set(yticks=range(len(names)), yticklabels=names, xlim=(0, 1.28),
           xlabel='False alarm rate; frozen primary STA rule',
           title='STA: semantic-preserving normal outputs (source bootstrap 95% CI)')
    ax.invert_yaxis()
    fig.savefig(root / 'sta_normal_false_alarms.png', dpi=170)
    fig.savefig(root / 'sta_normal_false_alarms.svg'); plt.close(fig)
    print('Descriptive analysis and figures complete', flush=True)


if __name__ == '__main__':
    main()
