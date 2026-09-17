"""Reproducible FSO/EOI/UEP validation on fresh scene families.

Phase order: prepare -> development -> visual/pixel -> report.
`development` observes only the new development sources and the previous runs'
normal cases, measures the pre-score extractor gate, and freezes thresholds.
Heldout RGB is never observed before `calibration.json` exists.
"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import pickle
import shutil
import time

import numpy as np

from .artifacts import sha256
from .event_duration_metric import evaluate_observations, sfr_observed
from .forbidden_state_metric import PARAMETERS, evaluate, truth_occupancy, unsupported_presence
from .metric_v3_cases import annotation_events, truth_event_change
from .metric_v3_features import PixelObserver, VisualObserver, pixels
from .metric_v3_formal import array, read, save
from .metric_v4_audit import audit_events
from .metric_v5_cases import FAMILIES, INTERVENTIONS, SWAP, TARGETS, draw, scene, trajectory, truth_order_inversions
from .sta_video_validation import LABEL, encode_video, simulate

PREVIOUS = ('outputs/automatic_validation_20260912_v1', 'outputs/mte_otf_20260914_v1',
            'outputs/ere_sta_20260914_v2', 'outputs/ere_sta_formal_20260914_v1',
            'outputs/event_duration_sta_20260914_v1')
FORMAL = Path('outputs/ere_sta_formal_20260914_v1')
DURATION = Path('outputs/event_duration_sta_20260914_v1')
SIGNATURE = 'b33a03c10649587a0ea5f6457b9db65e31a50e68e33f7035855e892285866e61'
CODES = ('forbidden_state_metric.py', 'metric_v5_cases.py', 'metric_v5_pipeline.py',
         'event_duration_metric.py', 'event_metric.py', 'object_metric.py', 'motion_metric.py',
         'metric_v3_features.py', 'metric_v3_cases.py', 'sta_video_validation.py',
         'attribution_metric.py', 'automatic_metrics.py', 'temporal_baselines.py', 'tracking_baselines.py')
CANDIDATES = ('fso_presence', 'fso_premature', 'fso_motion', 'fso_heading', 'fso_max', 'eoi', 'uep')
ABLATIONS = ('fso_presence_raw', 'fso_premature_raw', 'fso_motion_raw', 'fso_heading_raw', 'fso_max_raw')
BASELINES = ('ghost_max', 'delay_max', 'sfr_inferred', 'ere', 'ser', 'tlp_alex', 'mte_tail',
             'idf1_mask_error', 'lpips_alex', 'tof_farneback')
METRICS = CANDIDATES + ABLATIONS + BASELINES
PRIMARY = {'ghost_hold': 'fso_presence', 'ghost_return': 'fso_presence',
           'premature_enter': 'fso_premature', 'stop_overrun': 'fso_motion',
           'turn_overrun': 'fso_heading', 'order_swap': 'eoi', 'unsupported_addition': 'uep'}
GATE = {'auc': .9, 'tpr': .8, 'fpr': .1, 'coverage': .95, 'target_event_coverage': .8,
        'oracle_mean_absolute_error': .25, 'extractor_recall': .8, 'extractor_precision': .8,
        'anchor_type_recall': .8}
DEVELOPMENT_SEED, HELDOUT_SEED, PER_FAMILY = 260915300, 260915600, 8


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def verify(root, calibrated=False):
    p = read(root / 'protocol.json')
    for name, digest in p['code_sha256'].items():
        if sha256(Path(__file__).with_name(name)) != digest:
            raise ValueError('Frozen code changed: ' + name)
    for name, digest in p['previous_protocol_sha256'].items():
        if sha256(Path(name) / 'protocol.json') != digest:
            raise ValueError('Previous experiment changed: ' + name)
    if calibrated:
        if sha256(root / 'calibration.json') != (root / 'calibration.sha256').read_text().strip():
            raise ValueError('Calibration changed')
    return p


class CachedObservations:
    """Read-only access to the previous runs' hash-keyed observation caches."""

    def __init__(self):
        self.roots = [run / 'visual_cache' / SIGNATURE for run in (FORMAL, DURATION)]

    def __call__(self, digest):
        for root in self.roots:
            path = root / (digest + '.pkl.gz')
            if path.exists():
                with gzip.open(path, 'rb') as f:
                    value = pickle.load(f)
                if value['pixel_sha256'] != digest:
                    raise ValueError('observation cache provenance mismatch')
                return value
        raise FileNotFoundError('no cached observation for ' + digest)


def prepare(root):
    if root.exists():
        raise FileExistsError('Use a fresh output directory: ' + str(root))
    root.mkdir(parents=True)
    protocol = {
        'schema': 'forbidden_state_occupancy_v1', 'created_unix': time.time(),
        'candidates': CANDIDATES, 'ablations': ABLATIONS, 'comparators': BASELINES,
        'primary_target': PRIMARY, 'parameters': PARAMETERS, 'gates': GATE,
        'code_sha256': {name: sha256(Path(__file__).with_name(name)) for name in CODES},
        'previous_protocol_sha256': {name: sha256(Path(name) / 'protocol.json') for name in PREVIOUS},
        'observation_signature': SIGNATURE,
        'families': FAMILIES, 'sources_per_family': PER_FAMILY,
        'development_seed_base': DEVELOPMENT_SEED, 'heldout_seed_base': HELDOUT_SEED,
        'threshold': 'pooled development normal 95th percentile higher; detect strictly greater',
        'development_normals': 'previous rendered normal cases (480 v3 + 120 v4, all now development) '
                               'plus the new development families 120 controls',
        'truth_criterion': 'renderer occupancy oracle for the targeted instance must strictly increase; '
                           'event-sequence change within the 0.25s matching tolerance is reported separately',
        'bootstrap': {'unit': 'source', 'iterations': 1000, 'seed': 20260915},
        'evaluation_scope': 'controlled new procedural scene families; not natural-video validity, '
                            'not actual LGVSC reconstruction, not human agreement',
        'new_human_review': False, 'novelty_demonstrated': False, 'heldout_scores_observed': 0}
    save(root / 'protocol.json', protocol)
    (root / 'frozen_source').mkdir()
    for name in CODES:
        shutil.copy2(Path(__file__).with_name(name), root / 'frozen_source' / name)
    shutil.copy2('docs/FSO_UEP_PROTOCOL.md', root / 'PROTOCOL.frozen.md')
    manifest, sources, audit = [], [], []
    for split, base in (('development', DEVELOPMENT_SEED), ('heldout', HELDOUT_SEED)):
        for fi, family in enumerate(FAMILIES):
            for k in range(PER_FAMILY):
                seed = base + fi * 100 + k
                sid = family + '/' + str(seed)
                stem = sid.replace('/', '__')
                state = scene(seed, family)
                frames, truth = draw(state)
                folder = root / 'sources' / stem
                folder.mkdir(parents=True)
                np.savez_compressed(folder / 'source.npz', frames=frames)
                np.savez_compressed(folder / 'truth.npz', **truth)
                events = annotation_events(truth['centers'], truth['visible'])
                source = {'source_id': sid, 'source_stem': stem, 'family': family, 'split': split,
                          'source_pixel_sha256': pixels(frames), 'truth_events': events,
                          'anchor_times': state['times']}
                sources.append(source)

                def add(case_id, recon, recon_truth, metadata, extra=None):
                    dest = root / 'cases' / case_id
                    dest.mkdir(parents=True)
                    np.savez_compressed(dest / 'reconstruction.npz', frames=recon)
                    if recon_truth is not None:
                        np.savez_compressed(dest / 'truth.npz', **recon_truth)
                    record = {**{k: v for k, v in source.items() if k != 'truth_events'},
                              'case_id': case_id, 'reconstruction_pixel_sha256': pixels(recon), **metadata}
                    if extra is not None:
                        for name, value in (('tx', extra['tx_frames']), ('rx', extra['rx_frames'])):
                            np.savez_compressed(dest / (name + '.npz'), frames=value)
                            record[name + '_pixel_sha256'] = pixels(value)
                        save(dest / 'transport.json', {key: extra[key] for key in
                             ('tx_status', 'rx_status', 'tx_indices', 'rx_indices', 'reconstruction_indices')})
                        record['transport_sha256'] = sha256(dest / 'transport.json')
                        with (dest / 'packets.pkl').open('wb') as f:
                            pickle.dump({key: extra[key] for key in ('tx_packets', 'rx_packets')}, f, protocol=5)
                        record['packets_sha256'] = sha256(dest / 'packets.pkl')
                    manifest.append(record)

                controls = []
                for style in ('identity', 'brightness20', 'texture', 'camera', 'palette'):
                    recon, recon_truth = draw(state, style=style)
                    if not (np.array_equal(truth['centers'], recon_truth['centers'])
                            and np.array_equal(truth['visible'], recon_truth['visible'])):
                        raise ValueError('control changed the simulated state')
                    controls.append((style, recon, recon_truth))
                    add(stem + '__' + style, recon, recon_truth,
                        {'corpus': 'event', 'target': 'control', 'kind': 'control', 'instance': None,
                         'severity': 0., 'target_object': None, 'target_event': None, 'target_frame': None,
                         'variant': style, 'oracle': {}, 'truth_event_changed': False, 'eligible': True})
                if split == 'heldout':
                    for kind, instance, delays in INTERVENTIONS:
                        event, obj = TARGETS[instance]
                        frame = state['times'][event]
                        for d in delays:
                            centers, visible = trajectory(state, override={'kind': kind, 'delay': d})
                            recon, recon_truth = draw(state, centers, visible)
                            oracle = truth_occupancy(instance, recon_truth['centers'], recon_truth['visible'], obj, frame)
                            baseline = truth_occupancy(instance, truth['centers'], truth['visible'], obj, frame)
                            if oracle['occupancy'] is None or oracle['occupancy'] <= baseline['occupancy']:
                                raise ValueError('intervention did not increase the truth occupancy')
                            changed = truth_event_change(events, annotation_events(recon_truth['centers'], recon_truth['visible']))
                            case_id = stem + '__' + kind + '_' + str(d)
                            add(case_id, recon, recon_truth,
                                {'corpus': 'event', 'target': instance, 'kind': kind, 'instance': instance,
                                 'severity': d / 8., 'target_object': obj, 'target_event': event,
                                 'target_frame': frame, 'variant': kind + '_' + str(d), 'eligible': True,
                                 'oracle': {'occupancy': oracle['occupancy'], 'source_occupancy': baseline['occupancy'],
                                            'observed_s': oracle['observed_s']},
                                 'truth_event_changed': bool(changed)})
                            audit.append({'case_id': case_id, 'instance': instance, 'status': 'ACCEPTED',
                                          'oracle': oracle['occupancy'], 'source_oracle': baseline['occupancy'],
                                          'truth_event_changed': bool(changed)})
                    first, second = SWAP[family]
                    times = dict(state['times'])
                    times[first], times[second] = times[second], times[first]
                    centers, visible = trajectory(state, times)
                    recon, recon_truth = draw(state, centers, visible)
                    order = truth_order_inversions(truth, recon_truth)
                    if not order['truth_inversions']:
                        raise ValueError('order intervention did not invert a truth pair')
                    case_id = stem + '__order_swap'
                    add(case_id, recon, recon_truth,
                        {'corpus': 'event', 'target': 'order', 'kind': 'order_swap', 'instance': 'order',
                         'severity': abs(state['times'][first] - state['times'][second]) / 8.,
                         'target_object': None, 'target_event': None, 'target_frame': None,
                         'variant': 'order_swap', 'eligible': True, 'oracle': order,
                         'truth_event_changed': True, 'swapped_events': [first, second]})
                    audit.append({'case_id': case_id, 'instance': 'order', 'status': 'ACCEPTED',
                                  'oracle': order['truth_eoi'], 'source_oracle': 0., 'truth_event_changed': True})
                    encoded = encode_video(frames)
                    for style, recon, recon_truth in controls:
                        add(stem + '__sta_control_' + style, recon, recon_truth,
                            {'corpus': 'sta', 'target': 'control', 'kind': 'control', 'instance': None,
                             'label': 'normal', 'severity': 0., 'eligible': True, 'variant': style,
                             'truth_event_changed': False},
                            {'tx_frames': frames, 'rx_frames': frames, 'tx_status': ['ok'] * len(frames),
                             'rx_status': ['ok'] * len(frames), 'tx_indices': list(range(len(frames))),
                             'rx_indices': list(range(len(frames))), 'reconstruction_indices': list(range(len(frames))),
                             'tx_packets': encoded, 'rx_packets': encoded})
                    for kind in ('tx_omission', 'packet_loss', 'bit_corruption', 'generation_freeze', 'unsupported_addition'):
                        for severity in (.125, .25):
                            sim = simulate(frames, kind, severity, encoded)
                            indices = sim['reconstruction_indices']
                            recon_truth = {key: value[indices] for key, value in truth.items()}
                            changed = (kind == 'unsupported_addition' or
                                       truth_event_change(events, annotation_events(recon_truth['centers'], recon_truth['visible'])))
                            add(stem + '__sta_' + kind + '_' + str(severity), sim['reconstruction'], recon_truth,
                                {'corpus': 'sta', 'target': kind, 'kind': kind, 'instance': None,
                                 'label': LABEL[kind], 'severity': severity, 'eligible': bool(changed),
                                 'variant': kind, 'truth_event_changed': bool(changed),
                                 'addition_frames': int(sim['addition_masks'].any((1, 2)).sum())}, sim)
                print('prepared', split, family, k + 1, '/', PER_FAMILY, len(manifest), flush=True)
    previous = set()
    for run in PREVIOUS:
        listing = Path(run) / 'cases.jsonl'
        if listing.exists():
            previous |= {r.get('source_pixel_sha256') for r in rows(listing)}
    fresh = {s['source_pixel_sha256'] for s in sources}
    if previous & fresh:
        raise ValueError('source RGB overlaps a previous experiment')
    (root / 'cases.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in manifest))
    save(root / 'sources.json', sources)
    save(root / 'truth_audit.json', {'status': 'PASSED', 'accepted': len(audit), 'results': audit,
        'truth_event_changed_counts': dict(Counter(str(r['truth_event_changed']) for r in audit)),
        'scope': 'renderer occupancy/order oracle; the 0.25s event tolerance flag is informational'})
    save(root / 'preparation.json', {
        'sources': dict(Counter(s['split'] for s in sources)),
        'cases': dict(Counter(r['split'] + '/' + r['corpus'] for r in manifest)),
        'sta_eligibility': dict(Counter(str(r.get('eligible')) for r in manifest if r['corpus'] == 'sta')),
        'source_RGB_overlap_with_previous': False, 'manifest_sha256': sha256(root / 'cases.jsonl'),
        'RGB_and_truth_file_sha256': {str(f.relative_to(root)): sha256(f) for f in sorted(root.glob('**/*.npz'))}})


def compare_observations(x, y):
    """Transcription of the frozen `VisualObserver.compare` body for two loaded
observations, so previous-run caches and new RGB take one identical code path.
`development` asserts equality against the frozen method on a real pair."""
    from .event_metric import score_events
    from .motion_metric import score_motion
    from .object_metric import score_tracks
    from .tracking_baselines import identity_baselines
    result = score_events(x['events'], y['events'])
    result.update(score_tracks(x['tracks'], y['tracks']))
    result.update(identity_baselines(x['tracks'], y['tracks']))
    motion = score_motion(x['motion'], y['motion'])
    result.update({k: v for k, v in motion.items() if k.startswith('mte_') or k.startswith('camera_')})
    curve = np.linalg.norm(x['flow'] - y['flow'], axis=-1).mean((1, 2))
    result.update(tof_raft_small=float(curve.mean()), tof_raft_curve=curve.tolist())
    return result


def score_observations(ao, bo, rx=None, rx_status=None):
    """Candidates plus the previous runs' visual comparators from observations only."""
    result = compare_observations(ao, bo)
    duration = evaluate_observations(ao, bo)
    result.update({k: v for k, v in duration.items() if k in ('ghost_max', 'delay_max')})
    result['sfr_inferred'] = sfr_observed(ao, bo, duration['entity_groups'])
    result.update(evaluate(ao, bo))
    if rx is not None:
        result.update(unsupported_presence(ao, bo, rx, rx_status))
    return result


def _serialisable(result):
    drop = ('fso_records', 'eoi_rows', 'eoi_matches', 'entity_groups', 'uep_rows',
            'reference_events', 'reconstruction_events', 'event_pairs', 'track_pairs',
            'omission_intervals', 'addition_intervals', 'distortion_intervals',
            'tof_raft_curve', 'tlp_alex_curve', 'tof_farneback_curve')
    return {k: v for k, v in result.items() if k not in drop}


def _previous_normals(cached):
    """The 600 previous rendered normal cases, all declared development data.

Their comparator scores are read from the frozen runs on disk; the new
candidates are recomputed from the stored RGB observations of the same pixels.
"""
    normals = []
    for path in sorted((DURATION / 'development_event').glob('*.json')):
        row = read(path)
        ao, bo = cached(row['source_pixel_sha256']), cached(row['reconstruction_pixel_sha256'])
        value = evaluate(ao, bo)
        value.update(unsupported_presence(ao, bo, ao, ['ok'] * ao['frame_count']))
        normals.append({'origin': 'previous_v3_rendered_normal', 'case_id': row['case_id'],
                        'source_id': row['source_id'], 'variant': row['variant'],
                        'scores': {**row['scores'], **_serialisable(value)}})
    for row in rows(DURATION / 'cases.jsonl'):
        if row['corpus'] != 'event' or row['target'] != 'control':
            continue
        scores = {}
        for part in ('visual', 'pixel'):
            scores.update(read(DURATION / 'scores' / part / (row['case_id'] + '.json'))['scores'])
        ao, bo = cached(row['source_pixel_sha256']), cached(row['reconstruction_pixel_sha256'])
        value = evaluate(ao, bo)
        value.update(unsupported_presence(ao, bo, ao, ['ok'] * ao['frame_count']))
        normals.append({'origin': 'previous_v4_rendered_normal', 'case_id': row['case_id'],
                        'source_id': row['source_id'], 'variant': row['variant'],
                        'scores': {**scores, **_serialisable(value)}})
    return normals


def _extractor_gate(observer, root):
    """Source-event recall/precision on the new development families only."""
    gate_rows = []
    for row in rows(root / 'cases.jsonl'):
        if row['split'] != 'development' or row['variant'] != 'identity':
            continue
        source_folder = root / 'sources' / row['source_stem']
        a = array(source_folder / 'source.npz', row['source_pixel_sha256'])
        ao = observer.observe(a)
        with np.load(source_folder / 'truth.npz') as f:
            truth = {k: f[k] for k in f.files}
        audited = audit_events(ao, {'events': [{**e, 'source_time_s': e['time_s'],
                                                'source_track_id': e['track_id']} for e in ao['events']],
                                    'entity_groups': {}}, truth)
        gate_rows.append({'source_id': row['source_id'], 'family': row['family'],
                          'truth_events': audited['truth_events'],
                          'predicted_events': audited['predicted_source_events'],
                          'matched_events': audited['matched_source_events'],
                          'by_type': {kind: {'truth': sum(e['type'] == kind for e in audited['source_events']),
                                             'matched': sum(e['type'] == kind and e['observed'] is not None
                                                            for e in audited['source_events'])}
                                      for kind in ('enter', 'exit', 'start', 'stop', 'turn')}})
        print('gate', len(gate_rows), row['source_id'], audited['matched_source_events'], '/',
              audited['truth_events'], flush=True)
    totals = {k: sum(r[k] for r in gate_rows) for k in ('truth_events', 'predicted_events', 'matched_events')}
    by_type = {kind: {'truth': sum(r['by_type'][kind]['truth'] for r in gate_rows),
                      'matched': sum(r['by_type'][kind]['matched'] for r in gate_rows)}
               for kind in ('enter', 'exit', 'start', 'stop', 'turn')}
    gate = {'rows': gate_rows, **totals, 'by_type': by_type,
            'recall': totals['matched_events'] / totals['truth_events'] if totals['truth_events'] else None,
            'precision': totals['matched_events'] / totals['predicted_events'] if totals['predicted_events'] else None,
            'anchor_type_recall': {kind: (by_type[kind]['matched'] / by_type[kind]['truth']
                                          if by_type[kind]['truth'] else None)
                                   for kind in ('enter', 'exit', 'stop', 'turn')},
            'scope': 'development families only; measured before any heldout observation'}
    gate['passed'] = bool(gate['recall'] is not None and gate['recall'] >= GATE['extractor_recall']
                          and gate['precision'] is not None and gate['precision'] >= GATE['extractor_precision']
                          and all(v is not None and v >= GATE['anchor_type_recall']
                                  for v in gate['anchor_type_recall'].values()))
    return gate


def development(root):
    verify(root)
    if (root / 'calibration.json').exists():
        raise ValueError('Development is locked')
    if any((root / 'scores' / part).exists() for part in ('visual', 'pixel')):
        raise ValueError('Heldout scores already exist')
    import torch
    import cv2
    torch.set_num_threads(4)
    torch.manual_seed(20260915)
    cv2.setNumThreads(1)
    observer = VisualObserver({'models': {'root': '.local/metric_v2_models'},
                               'event_parameters': read(FORMAL / 'protocol.json')['event_parameters']},
                              root, SIGNATURE)
    cached = CachedObservations()
    normals = _previous_normals(cached)
    print('previous normal cases', len(normals), flush=True)
    gate = _extractor_gate(observer, root)
    save(root / 'extractor_gate.json', gate)
    print('extractor gate', gate['recall'], gate['precision'], gate['anchor_type_recall'], gate['passed'], flush=True)
    # Declared diagnostic: the same candidates on the new development controls.
    # These never enter the frozen thresholds, matching the v4 precedent of
    # calibrating on the previous rendered normal set alone.
    diagnostic, checked = [], False
    for row in rows(root / 'cases.jsonl'):
        if row['split'] != 'development':
            continue
        a = array(root / 'sources' / row['source_stem'] / 'source.npz', row['source_pixel_sha256'])
        b = array(root / 'cases' / row['case_id'] / 'reconstruction.npz', row['reconstruction_pixel_sha256'])
        ao, bo = observer.observe(a), observer.observe(b)
        if not checked:
            frozen = observer.compare(a, b)
            mirrored = compare_observations(ao, bo)
            if {k: v for k, v in frozen.items() if isinstance(v, float)} != \
                    {k: v for k, v in mirrored.items() if isinstance(v, float)}:
                raise ValueError('compare_observations diverged from the frozen observer')
            checked = True
        value = score_observations(ao, bo, ao, ['ok'] * ao['frame_count'])
        diagnostic.append({'origin': 'new_development_control', 'case_id': row['case_id'],
                           'source_id': row['source_id'], 'family': row['family'],
                           'variant': row['variant'], 'scores': _serialisable(value)})
        print('development control', len(diagnostic), row['case_id'], flush=True)
    save(root / 'development_normals.json', {'rows': normals, 'count': len(normals),
         'by_origin': dict(Counter(r['origin'] for r in normals)),
         'frozen_observer_transcription_checked': checked})
    save(root / 'development_new_family_diagnostic.json', {'rows': diagnostic, 'count': len(diagnostic),
         'scope': 'declared diagnostic on the new development families; not used for any frozen threshold'})
    if not gate['passed']:
        raise SystemExit('pre-score extractor gate failed; thresholds are not frozen')
    names = CANDIDATES + ABLATIONS + BASELINES
    thresholds, measured = {}, {}
    for name in names:
        values = [r['scores'].get(name) for r in normals]
        values = [v for v in values if v is not None and np.isfinite(v)]
        measured[name] = len(values)
        thresholds[name] = float(np.quantile(values, .95, method='higher')) if values else None
    alternative = {}
    for name in names:
        values = [r['scores'].get(name) for r in diagnostic]
        values = [v for v in values if v is not None and np.isfinite(v)]
        alternative[name] = float(np.quantile(values, .95, method='higher')) if values else None
    save(root / 'calibration.json', {'created_unix': time.time(), 'protocol_sha256': sha256(root / 'protocol.json'),
         'thresholds': thresholds, 'normal_cases': len(normals), 'measured_per_metric': measured,
         'extractor_gate_sha256': sha256(root / 'extractor_gate.json'),
         'new_family_alternative_thresholds_unused': alternative,
         'heldout_scores_observed': 0,
         'scope': 'candidates and comparators share the pooled previous rendered normal set '
                  '(480 v3 + 120 v4 cases, 72 sources); the new development controls are diagnostic only'})
    (root / 'calibration.sha256').write_text(sha256(root / 'calibration.json') + '\n')
    print('FROZEN', {k: v for k, v in thresholds.items() if k in CANDIDATES}, flush=True)


def run(root, part):
    verify(root, True)
    prep = read(root / 'preparation.json')
    if sha256(root / 'cases.jsonl') != prep['manifest_sha256']:
        raise ValueError('Changed manifest')
    import torch
    import cv2
    torch.set_num_threads(4)
    torch.manual_seed(20260915)
    cv2.setNumThreads(1)
    if part == 'visual':
        observer = VisualObserver({'models': {'root': '.local/metric_v2_models'},
                                   'event_parameters': read(FORMAL / 'protocol.json')['event_parameters']},
                                  root, SIGNATURE)
    else:
        observer = PixelObserver()
    save(root / ('runtime_' + part + '.json'), {'started_unix': time.time(), 'torch': torch.__version__,
         'numpy': np.__version__, 'GPU': torch.cuda.get_device_name(), 'part': part,
         'protocol_sha256': sha256(root / 'protocol.json')})
    manifest = [r for r in rows(root / 'cases.jsonl') if r['split'] == 'heldout']
    digest = sha256(root / 'calibration.json')
    for index, row in enumerate(manifest):
        dest = root / 'scores' / part / (row['case_id'] + '.json')
        if dest.exists():
            if read(dest)['calibration_sha256'] != digest:
                raise ValueError('Resume calibration differs')
            continue
        started = time.time()
        folder = root / 'cases' / row['case_id']
        source_folder = root / 'sources' / row['source_stem']
        a = array(source_folder / 'source.npz', row['source_pixel_sha256'])
        b = array(folder / 'reconstruction.npz', row['reconstruction_pixel_sha256'])
        if part == 'pixel':
            result = observer.compare(a, b)
        else:
            rx_observation = rx_status = None
            if row['corpus'] == 'sta':
                if sha256(folder / 'transport.json') != row['transport_sha256']:
                    raise ValueError('Transport changed')
                rx_observation = observer.observe(array(folder / 'rx.npz', row['rx_pixel_sha256']))
                rx_status = read(folder / 'transport.json')['rx_status']
            ao, bo = observer.observe(a), observer.observe(b)
            result = score_observations(ao, bo, rx_observation, rx_status)
            with np.load(source_folder / 'truth.npz') as f:
                truth = {k: f[k] for k in f.files}
            records = result['fso_records']
            result = _serialisable(result)
            if row['target'] in TARGETS:
                matches = [r for r in records if r['instance'] == row['target']]
                result['target_records'] = matches
            result['source_event_audit'] = audit_events(
                ao, {'events': [{**e, 'source_time_s': e['time_s'], 'source_track_id': e['track_id']}
                                for e in ao['events']], 'entity_groups': {}}, truth)['source_events']
        save(dest, {**row, 'scores': result, 'elapsed_s': time.time() - started,
                    'protocol_sha256': sha256(root / 'protocol.json'), 'calibration_sha256': digest})
        print(part, index + 1, '/', len(manifest), row['case_id'], round(time.time() - started, 2), flush=True)
    save(root / ('completed_' + part + '.json'), {'status': 'COMPLETED', 'completed_unix': time.time(),
         'cases': len(manifest)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('prepare', 'development', 'visual', 'pixel', 'verify'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare(args.output)
    elif args.phase == 'development':
        development(args.output)
    elif args.phase in ('visual', 'pixel'):
        run(args.output, args.phase)
    else:
        verify(args.output, (args.output / 'calibration.json').exists())
        print('frozen inputs verified')


if __name__ == '__main__':
    main()
