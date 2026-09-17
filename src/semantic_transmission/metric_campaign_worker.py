"""Isolated operations for metric_campaign (NumPy/CUDA environments are separate)."""
import argparse
from collections import defaultdict
from pathlib import Path
import time

import cv2
import numpy as np

from .artifacts import sha256
from .metric_campaign import (REPO, PHASES, check_files, digest, file_inventory, read,
                              receipt, resolve, runtime_inventory, save, tasks, verify_receipt)
from .metric_campaign_cases import PRIMARY, cross_corpus, natural_corpus, rx_corpus
from .metric_campaign_real import attach_truth, load_array, real_inputs, reconstruct
from .metric_campaign_scoring import ALL_METRICS, analyse, component_scores, finite, interval, visual_scores


def calibrate(root, config):
    from .metric_v5_pipeline import CachedObservations, DURATION
    base = resolve(config['base_run'])
    normals = read(base / 'development_normals.json')['rows']
    index = {}
    input_paths = [DURATION / 'cases.jsonl']
    input_paths += sorted((DURATION / 'development_event').glob('*.json'))
    for path in input_paths[1:]:
        row = read(path)
        index[row['case_id']] = row
    import json
    for line in input_paths[0].read_text().splitlines():
        row = json.loads(line)
        if row.get('corpus') == 'event' and row.get('target') == 'control':
            index[row['case_id']] = row
    if len(normals) != 600 or len({r['case_id'] for r in normals}) != 600:
        raise ValueError('Expected the frozen 600 distinct development normals')
    cached = CachedObservations()
    # Freeze cache bytes before using them, including across interrupted calibration.
    for normal in normals:
        row = index[normal['case_id']]
        for key in ('source_pixel_sha256', 'reconstruction_pixel_sha256'):
            candidates = [p / (row[key] + '.pkl.gz') for p in cached.roots]
            path = next((p for p in candidates if p.is_file()), None)
            if path is None:
                raise FileNotFoundError('Missing calibration observation: ' + row[key])
            input_paths.append(path)
    freeze = root / '01_cross/calibration_inputs.json'
    if freeze.exists():
        check_files(read(freeze))
    else:
        save(freeze, file_inventory(input_paths))
    results, artifacts = [], [freeze]
    for i, normal in enumerate(normals):
        row = index[normal['case_id']]
        destination = root / '01_cross/development_scores' / (row['case_id'] + '.json')
        signature = digest([normal, row, sha256(freeze)])
        if destination.exists():
            record = read(destination)
            if record.get('input_signature') != signature or record.get('scores_sha256') != digest(record['scores']):
                raise ValueError('Development score changed')
        else:
            a, b = cached(row['source_pixel_sha256']), cached(row['reconstruction_pixel_sha256'])
            # Baseline and original candidate scores already exist for these
            # exact pixels. Recompute only the newly added component ablations.
            from .forbidden_state_metric import evaluate
            values = {**normal['scores'], **evaluate(a, b)}
            scores = {metric: normal['scores'].get(metric) for metric in ALL_METRICS}
            scores.update(component_scores(a, b, values, a, ['ok'] * a['frame_count']))
            record = {'case_id': row['case_id'], 'scores': scores, 'input_signature': signature,
                      'scores_sha256': digest(scores)}
            save(destination, record)
        results.append(record['scores'])
        artifacts.append(destination)
        if i % 50 == 0:
            print('calibration', i + 1, '/', len(normals), flush=True)
    old = read(base / 'calibration.json')['thresholds']
    thresholds, counts = {}, {}
    for metric in ALL_METRICS:
        values = [r[metric] for r in results if finite(r.get(metric))]
        counts[metric] = len(values)
        thresholds[metric] = old[metric] if metric in old else float(np.quantile(values, .95, method='higher')) if values else None
    destination = root / '01_cross/calibration.json'
    save(destination, {'thresholds': thresholds, 'normal_cases': len(normals), 'measured_per_metric': counts,
                       'base_calibration_sha256': sha256(base / 'calibration.json'),
                       'scope': 'original thresholds unchanged; added ablations use only the same old 600 normals',
                       'heldout_scores_observed': 0})
    return artifacts + [destination]


def score_rows(root, phase, operation, config, campaign):
    from .metric_v3_features import PixelObserver, VisualObserver
    real = operation.startswith('real_')
    pixel = operation.endswith('pixel')
    manifest = root / phase / ('real_pairs.json' if real else 'cases.json')
    rows = read(manifest)['rows']
    if not rows or len({r['case_id'] for r in rows}) != len(rows):
        raise ValueError('Empty or duplicate scoring manifest')
    calibration = root / '01_cross/calibration.json'
    signature = digest([campaign['files'], sha256(calibration)])
    observer = None
    artifacts = []
    for i, row in enumerate(rows):
        path = root / phase / 'scores' / operation / (row['case_id'] + '.json')
        input_signature = digest([row, signature, operation])
        if path.exists():
            record = read(path)
            if record.get('input_signature') != input_signature or record.get('result_sha256') != digest(record['result']):
                raise ValueError('Score input/provenance changed: ' + row['case_id'])
        else:
            if real:
                a, b, rx, status, alignment = real_inputs(row)
            else:
                a, b = load_array(row['source']), load_array(row['reconstruction'])
                rx = load_array(row['rx']) if row.get('rx') else None
                status, alignment = row.get('rx_status'), None
            if observer is None:
                if pixel:
                    observer = PixelObserver()
                else:
                    protocol = {'models': {'root': str(REPO / '.local/metric_v2_models')},
                                'event_parameters': read(REPO / 'outputs/ere_sta_formal_20260914_v1/protocol.json')['event_parameters']}
                    observer = VisualObserver(protocol, root, signature)
            if pixel:
                scores = {k: v for k, v in observer.compare(a, b).items() if k in ALL_METRICS}
            else:
                a_observed, b_observed = observer.observe(a), observer.observe(b)
                scores = visual_scores(a_observed, b_observed, observer.observe(rx) if rx is not None else None, status)
            # JSON null means unavailable. NaN/Inf also become explicit unavailable values.
            scores = {k: float(v) if finite(v) else None for k, v in scores.items()}
            result = {'scores': scores, 'alignment': alignment}
            record = {'case_id': row['case_id'], 'input_signature': input_signature,
                      'result': result, 'result_sha256': digest(result)}
            save(path, record)
        artifacts.append(path)
        if i % 10 == 0 or i + 1 == len(rows):
            print(operation, i + 1, '/', len(rows), row['case_id'], flush=True)
    return artifacts


def scored(root, phase, real=False):
    rows = read(root / phase / ('real_pairs.json' if real else 'cases.json'))['rows']
    operations = ['real_visual', 'real_pixel'] if real else ['visual', 'pixel']
    if phase == '02_components':
        operations = ['visual']
    result = []
    for row in rows:
        scores, alignment = {}, None
        for operation in operations:
            value = read(root / phase / 'scores' / operation / (row['case_id'] + '.json'))
            if value['case_id'] != row['case_id'] or value['result_sha256'] != digest(value['result']):
                raise ValueError('Score identity/hash mismatch')
            scores.update(value['result']['scores'])
            alignment = value['result']['alignment']
        result.append({**row, 'scores': scores, 'alignment': alignment})
    return result


def rx_counterfactual(rows, config):
    groups = defaultdict(list)
    for row in rows:
        groups[row['counterfactual_group']].append(row)
    result = {}
    for metric in ('uep', 'uep_source_only', 'uep_rx_only'):
        successes, maes = defaultdict(list), defaultdict(list)
        for values in groups.values():
            if len({digest([r['source'], r['reconstruction']]) for r in values}) != 1:
                raise ValueError('RX counterfactual changed source/reconstruction')
            absent = next(r for r in values if r['kind'] == 'absent')
            for row in values:
                if finite(row['scores'].get(metric)):
                    maes[row['source_id']].append(abs(row['scores'][metric] - row['truth']['uep']))
                delta = row['truth']['uep'] - absent['truth']['uep']
                if delta == 0:
                    continue
                measured = finite(row['scores'].get(metric)) and finite(absent['scores'].get(metric))
                successes[row['source_id']].append(bool(measured and
                    np.sign(row['scores'][metric] - absent['scores'][metric]) == np.sign(delta)))
        rates = [float(np.mean(v)) for v in successes.values()]
        result[metric] = {'source_mean_direction_agreement': float(np.mean(rates)) if rates else None,
                          'source_95ci': interval(rates, config), 'sources': len(rates),
                          'measured_mean_absolute_error': float(np.mean([np.mean(v) for v in maes.values()])) if maes else None}
    return result


def report(root, phase, config):
    thresholds = read(root / '01_cross/calibration.json')['thresholds']
    rows = scored(root, phase)
    result = {'execution_status': 'COMPLETED', 'phase': phase, 'cases': len(rows),
              'scope': read(root / phase / 'cases.json')['scope'],
              'metrics': analyse(rows, thresholds, config), 'novelty_established': False}
    if phase == '02_components':
        result['fso_eoi_components_on_crossed_cases'] = read(root / '01_cross/report.json')['metrics']
        result['rx_counterfactual'] = rx_counterfactual(rows, config)
        result['scope'] += '; FSO/EOI ablations reuse stage-1 observations; pixel baselines apply there'
    if phase == '03_natural':
        real_rows = attach_truth(scored(root, phase, real=True),
                                 resolve(config['reconstruction_truth']) if config.get('reconstruction_truth') else None)
        result['generated'] = generated_report(real_rows, thresholds, config)
        template = root / phase / 'independent_truth.template.json'
        save(template, {'schema': 'metric-campaign-independent-truth-v1',
                        'rows': [{k: row[k] for k in ('case_id', 'source_sha256', 'reconstruction_sha256')} |
                                 {'alignment_sha256': digest(row['alignment']), 'provenance': '', 'evidence_files': {},
                                  'uep_evidence_scope': 'received_keyframes_only',
                                  'truth': {m: None for m in PRIMARY}} for row in real_rows]})
    destination = root / phase / 'report.json'
    save(destination, result)
    return [destination] + ([template] if phase == '03_natural' else [])


def generated_report(rows, thresholds, config):
    metrics = analyse(rows, thresholds, config)
    labels = any(any(finite(v) for v in r['truth'].values()) for r in rows)
    return {'status': 'INDEPENDENT_TRUTH_EVALUATED' if labels else 'REQUIRES_INDEPENDENT_RECONSTRUCTION_TRUTH',
            'rows': rows, 'metrics': metrics,
            'scope': 'physical-timestamp diagnostics; sparse RX support; sampling-step comparison, not model superiority',
            'ranking_by_metric': model_change_agreement(rows, config),
            'model_ranking_validated': False}


def model_change_agreement(rows, config):
    """Compare measured change against independent truth; never rank from scores alone."""
    groups = defaultdict(list)
    for row in rows:
        groups[row['source_id']].append(row)
    result = {}
    for metric in PRIMARY:
        agreements, records = [], []
        for sid, values in groups.items():
            base = next((r for r in values if r.get('steps') == config['reconstruction_steps'][0]), None)
            if base is None:
                continue
            directions = []
            for row in values:
                if row is base:
                    continue
                if not all(finite(r['truth'].get(metric)) for r in (base, row)):
                    continue
                truth_delta = row['truth'][metric] - base['truth'][metric]
                if truth_delta == 0:
                    records.append({'source_id': sid, 'case_id': row['case_id'], 'status': 'TRUTH_TIE'})
                    continue
                measured = all(finite(r['scores'].get(metric)) for r in (base, row))
                delta = row['scores'][metric] - base['scores'][metric] if measured else None
                correct = bool(measured and np.sign(delta) == np.sign(truth_delta))
                directions.append(correct)
                records.append({'source_id': sid, 'case_id': row['case_id'], 'truth_delta': truth_delta,
                                'score_delta': delta, 'agrees': correct})
            if directions:
                agreements.append(float(np.mean(directions)))
        mean = float(np.mean(agreements)) if agreements else None
        sufficient = len(agreements) >= config['minimum_positive_sources']
        result[metric] = {'sources_with_independent_truth_change': len(agreements),
                          'direction_agreement': mean, 'source_95ci': interval(agreements, config), 'pairs': records,
                          'status': ('PASSED' if mean >= .8 else 'NOT_PASSED') if sufficient else 'INSUFFICIENT_TRUTH'}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['prepare', 'calibrate', 'visual', 'pixel', 'report', 'evaluate_truth',
                                             'reconstruct', 'real_visual', 'real_pixel'])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--phase', required=True, choices=PHASES)
    parser.add_argument('--truth', type=Path)
    args = parser.parse_args(argv)
    root, phase, operation = args.output.resolve(), args.phase, args.operation
    campaign = read(root / 'campaign.json')
    config = campaign['config']
    check_files(campaign['files'])
    if runtime_inventory(config) != campaign['runtime']:
        raise ValueError('Model files or Python environment changed')
    ordered = tasks(config)
    index = len(ordered) if operation == 'evaluate_truth' else ordered.index((phase, operation))
    # Require every predecessor, including artifacts, so direct worker calls
    # cannot accidentally score heldout data before frozen calibration.
    for p, op in ordered[:index]:
        verify_receipt(receipt(root, p, op))
    if operation == 'evaluate_truth':
        if args.truth is None:
            raise ValueError('--truth required')
        rows = attach_truth(scored(root, '03_natural', real=True), args.truth)
        thresholds = read(root / '01_cross/calibration.json')['thresholds']
        destination = root / 'independent_assessments' / (sha256(args.truth) + '.json')
        save(destination, {'truth_manifest': str(args.truth.resolve()), 'truth_sha256': sha256(args.truth),
                           'generated': generated_report(rows, thresholds, config)})
        print('Independent assessment:', destination, flush=True)
        return
    if receipt(root, phase, operation).exists():
        verify_receipt(receipt(root, phase, operation))
        return
    cv2.setNumThreads(1)
    np.random.seed(config['natural_seed'])
    if operation in ('visual', 'pixel', 'real_visual', 'real_pixel'):
        import torch
        torch.set_num_threads(4)
        torch.manual_seed(config['natural_seed'])
    start = time.time()
    if operation == 'prepare':
        if phase == '01_cross':
            paths = cross_corpus(root, config)
        elif phase == '02_components':
            paths = rx_corpus(root, config)
        else:
            paths = natural_corpus(root, config, campaign['natural_inventory'])
    elif operation == 'calibrate':
        paths = calibrate(root, config)
    elif operation == 'reconstruct':
        paths = reconstruct(root, config)
    elif operation == 'report':
        paths = report(root, phase, config)
    else:
        paths = score_rows(root, phase, operation, config, campaign)
    if not paths:
        raise RuntimeError('Operation produced no artifacts')
    save(receipt(root, phase, operation), {'execution_status': 'COMPLETED', 'phase': phase, 'operation': operation,
                                         'elapsed_s': time.time() - start, 'artifacts': file_inventory(paths)})


if __name__ == '__main__':
    main()
