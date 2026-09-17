"""Frozen post-gate ERE/STA experiment orchestration, with separate test stage."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import time

import numpy as np

from .artifacts import sha256
from .metric_v3_validation import require_gates, verify as verify_base
from .metric_v3_cases import annotation_events, truth_event_change
from .metric_v3_features import pixels

CODE = ('metric_v3_formal.py', 'metric_v3_features.py', 'sta_video_validation.py', 'metric_v3_statistics.py')
SCRIPTS = ('run_metric_v3_formal.sh', 'report_metric_v3_formal.py')
BASELINES = ('psnr_db', 'ssim', 'lpips_alex', 'clip_cosine', 'rte', 'lssd', 'tlp_alex',
             'tof_farneback', 'tof_raft_small', 'mte_tail', 'otf_error', 'oor', 'hor', 'odr',
             'idf1_mask_error', 'tlp_alex_segments', 'tof_farneback_segments', 'tof_raft_segments')
SIGNS = {key: (-1 if key in ('psnr_db', 'ssim', 'clip_cosine') else 1) for key in ('ere', 'ser', *BASELINES)}


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def array(path, digest=None):
    with np.load(path) as f:
        frames = f['frames']
    if digest and pixels(frames) != digest:
        raise ValueError('RGB changed: ' + str(path))
    return frames


def verify(root):
    p = read(root / 'protocol.json')
    base = Path(p['base_run']); b = verify_base(base); require_gates(base)
    if sha256(base / 'protocol.json') != p['base_protocol_sha256']:
        raise ValueError('base protocol changed')
    for name, expected in p['code_sha256'].items():
        if sha256(Path(__file__).with_name(name)) != expected:
            raise ValueError('frozen extension changed: ' + name)
    for name, expected in p['script_sha256'].items():
        if sha256(Path('scripts') / name) != expected:
            raise ValueError('frozen extension script changed: ' + name)
    ready = read(root / 'preparation.json')
    for name, digest in ready['artifact_sha256'].items():
        if sha256(root / name) != digest:
            raise ValueError('frozen experiment artifact changed: ' + name)
    return p, b


def prepare(args):
    require_gates(args.base)
    base_protocol = verify_base(args.base)
    models = base_protocol['models']; mr = Path(models['root'])
    for file, key in ((mr / 'sam2.1_hiera_tiny.pt', 'sam2_weights_sha256'),
                      (mr / 'dinov2_vits14_pretrain.pth', 'dino_weights_sha256'),
                      (Path.home() / '.cache/torch/hub/checkpoints/raft_small_C_T_V2-01064c6d.pth', 'raft_weights_sha256'),
                      (Path.home() / '.cache/clip/ViT-B-32.pt', 'clip_weights_sha256')):
        if sha256(file) != models[key]:
            raise ValueError('model bytes changed: ' + str(file))
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.output
    code = {n: sha256(Path(__file__).with_name(n)) for n in CODE}
    scripts = {n: sha256(Path('scripts') / n) for n in SCRIPTS}
    p = {'schema': 'ere-sta-postgate-v1', 'created_unix': time.time(), 'base_run': str(args.base.resolve()),
        'base_protocol_sha256': sha256(args.base / 'protocol.json'), 'code_sha256': code, 'script_sha256': scripts,
        'signature': hashlib.sha256(json.dumps([code, scripts], sort_keys=True).encode()).hexdigest(),
        'event_parameters': base_protocol['event_parameters'], 'metrics': SIGNS, 'comparators': BASELINES,
        'threshold': 'separate domain/development control 95th percentile higher; detect strictly greater',
        'temporal_segments': 'transition threshold from pooled development-control curves, contiguous run count / transitions',
        'gates': {'auc': .9, 'tpr': .8, 'fpr': .1, 'coverage': .95, 'sta_accuracy': .8},
        'bootstrap': {'replicates': 500, 'seed': 20260914, 'unit': 'source', 'paired': True},
        'domains': {'rendered': 'primary', 'public_video': 'diagnostic_only_predeclared'},
        'baseline_sta_classifier': 'per-score and all-score standardized nearest class centroid; development-only fit; four equal-prior classes; CLR variants weighted within source/class',
        'sta_decision': 'largest component strictly above its own development normal threshold; ties EDR,CLR,GFR,GHR; no active component=normal; undefined components cannot win',
        'sta_simulator': '32 PNG keyframes, no captions, checksum verification, zero-order-hold raster decoder; no actual generative model',
        'sta_units': 'RGB-predicted events plus whole object tracks, equal unit weights; no truth input',
        'sta_variants': ['normal', 'tx_omission', 'packet_loss', 'bit_corruption', 'generation_freeze', 'unsupported_addition'],
        'sta_severities': [.125, .25, .5],
        'sta_truth_policy': 'separate renderer annotation events under decoded index map; unchanged semantic interventions excluded and counted',
        'ser_policy': 'supplementary sensitivity on eligible motion variants that introduce unmatched truth events; not a new separately generated corpus',
        'formal_candidate_scores_observed_at_freeze': 0, 'human_review': False}
    save(root / 'protocol.json', p)
    for n in CODE:
        dst = root / 'frozen_source' / n; dst.parent.mkdir(exist_ok=True); shutil.copy2(Path(__file__).with_name(n), dst)
    for n in SCRIPTS:
        shutil.copy2(Path('scripts') / n, root / 'frozen_source' / n)
    shutil.copy2('docs/ERE_STA_FORMAL_EXECUTION.md', root / 'EXECUTION.frozen.md')
    truth = read(args.base / 'truth_audit.json')
    accepted = {x['case_id'] for x in truth['results'] if x['accepted']}
    manifest = [json.loads(line) for line in (args.base / 'cases.jsonl').read_text().splitlines()]
    rows = [x for x in manifest if x['case_id'] in accepted]
    (root / 'cases.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows))
    # Preserve the accepted/excluded lists; observers never receive their truth.
    save(root / 'truth_eligibility.json', {'base_truth_sha256': sha256(args.base / 'truth_audit.json'),
        'total': truth['total'], 'accepted': truth['accepted'], 'rejected': truth['rejected']})
    from .sta_video_validation import simulate, encode_video, KINDS, LABEL
    sta_rows = []
    for item in base_protocol['inventory']:
        if item['domain'] != 'rendered':
            continue
        stem = item['source_id'].replace('/', '__')
        frames = array(args.base / 'inputs' / (stem + '.npz'))
        with np.load(args.base / 'inputs' / (stem + '_truth.npz')) as f:
            centers, visible = f['centers'], f['visible']
        original_events = annotation_events(centers, visible)
        encoded = encode_video(frames)
        for kind in KINDS:
            for severity in ([0.] if kind == 'normal' else p['sta_severities']):
                case_id = stem + '__sta_' + kind + '_' + str(severity)
                sim = simulate(frames, kind, severity, encoded=encoded)
                indices = sim['reconstruction_indices']
                revents = annotation_events(centers[indices], visible[indices])
                changed = truth_event_change(original_events, revents)
                eligible = kind in ('normal', 'unsupported_addition') or changed
                folder = root / 'sta_inputs' / case_id; folder.mkdir(parents=True)
                for key in ('tx_frames', 'rx_frames', 'reconstruction'):
                    np.savez_compressed(folder / (key + '.npz'), frames=sim[key])
                with (folder / 'packets.pkl').open('wb') as f:
                    pickle.dump({k: sim[k] for k in ('tx_packets', 'rx_packets')}, f, protocol=5)
                status = {k: sim[k] for k in ('tx_status', 'rx_status', 'tx_indices', 'rx_indices', 'reconstruction_indices', 'interval', 'caption_policy')}
                save(folder / 'transport.json', status)
                sta_rows.append({'case_id': case_id, 'source_id': item['source_id'], 'source_stem': stem,
                    'domain': 'rendered', 'split': item['split'], 'kind': kind, 'label': LABEL[kind], 'severity': severity,
                    'eligible': eligible, 'truth_event_changed': changed, 'source_pixel_sha256': pixels(frames),
                    'pixel_sha256': {k: pixels(sim[k]) for k in ('tx_frames', 'rx_frames', 'reconstruction')},
                    'transport_sha256': sha256(folder / 'transport.json'), 'packets_sha256': sha256(folder / 'packets.pkl'),
                    'truth_scope': 'known intervention plus independent event change/inserted object; not candidate score'})
        print('prepared STA', item['source_id'], len(sta_rows), flush=True)
    (root / 'sta_cases.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in sta_rows))
    save(root / 'preparation.json', {'sources': len(base_protocol['inventory']), 'ere_cases': len(rows),
        'sta_attempts': len(sta_rows), 'sta_eligible': sum(x['eligible'] for x in sta_rows),
        'sta_by_kind_eligibility': dict(Counter(x['kind'] + '/' + str(x['eligible']) for x in sta_rows)),
        'artifact_sha256': {n: sha256(root / n) for n in ('protocol.json', 'cases.jsonl', 'sta_cases.jsonl', 'truth_eligibility.json', 'EXECUTION.frozen.md')}})


def run(args):
    root = args.output; p, base = verify(root)
    if args.split == 'heldout':
        calibration = read(root / 'calibration.json')
        if sha256(root / 'calibration.json') != (root / 'calibration.sha256').read_text().strip():
            raise ValueError('frozen calibration changed')
        if calibration['protocol_sha256'] != sha256(root / 'protocol.json'):
            raise ValueError('calibration provenance mismatch')
    import torch
    import cv2
    torch.set_num_threads(4); torch.manual_seed(20260914); cv2.setNumThreads(1)
    from .metric_v3_features import VisualObserver, PixelObserver
    from .sta_video_validation import score_video_attribution
    observer = VisualObserver(base, root, p['signature']) if args.part == 'visual' else PixelObserver()
    base_root = Path(p['base_run'])
    save(root / ('runtime_' + args.part + '_' + args.split + '.json'), {'started_unix': time.time(), 'torch': torch.__version__,
        'numpy': np.__version__, 'gpu': torch.cuda.get_device_name(), 'part': args.part, 'split': args.split})
    # Pair cache is local to the frozen execution, not inherited from earlier metrics.
    cache_root = root / 'pair_cache' / args.part; cache_root.mkdir(parents=True, exist_ok=True)
    def pair(a, b):
        key = pixels(a) + '_' + pixels(b)
        path = cache_root / (key + '.json')
        if path.exists():
            return read(path)
        value = observer.compare(a, b); save(path, value); return value
    for corpus, filename in (('ere', 'cases.jsonl'), ('sta', 'sta_cases.jsonl')):
        rows = [json.loads(x) for x in (root / filename).read_text().splitlines()]
        rows = [x for x in rows if x['split'] == args.split and x.get('eligible', True)]
        for index, row in enumerate(rows):
            output = root / 'scores' / args.part / corpus / (row['case_id'] + '.json')
            if output.exists():
                prior = read(output)
                if prior['protocol_sha256'] != sha256(root / 'protocol.json'):
                    raise ValueError('resumed score provenance mismatch')
                continue
            start = time.time()
            a = array(base_root / 'inputs' / (row['source_stem'] + '.npz'), row['source_pixel_sha256'])
            if corpus == 'ere':
                b = array(base_root / 'cases' / (row['case_id'] + '.npz'), row['reconstruction_pixel_sha256'])
                scores = pair(a, b)
            else:
                folder = root / 'sta_inputs' / row['case_id']
                b = array(folder / 'reconstruction.npz', row['pixel_sha256']['reconstruction'])
                scores = pair(a, b)
                if args.part == 'visual':
                    status = read(folder / 'transport.json')
                    if sha256(folder / 'transport.json') != row['transport_sha256']:
                        raise ValueError('transport metadata changed')
                    tx, rx = [array(folder / (key + '.npz'), row['pixel_sha256'][key]) for key in ('tx_frames', 'rx_frames')]
                    scores = {**scores, 'sta': score_video_attribution(observer, a, b, tx, rx, status['tx_status'], status['rx_status'])}
            save(output, {**row, 'scores': scores, 'elapsed_s': time.time() - start,
                          'protocol_sha256': sha256(root / 'protocol.json')})
            print(args.part, args.split, corpus, index + 1, '/', len(rows), row['case_id'], round(time.time() - start, 2), flush=True)
    save(root / ('completed_' + args.part + '_' + args.split + '.json'), {'completed_unix': time.time(), 'status': 'COMPLETED'})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('prepare', 'run', 'calibrate', 'summarize', 'verify'))
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--base', type=Path, default=Path('outputs/ere_sta_20260914_v2'))
    ap.add_argument('--part', choices=('visual', 'pixel'))
    ap.add_argument('--split', choices=('development', 'heldout'))
    args = ap.parse_args()
    if args.stage == 'prepare':
        prepare(args)
    elif args.stage == 'run':
        if args.part is None or args.split is None:
            ap.error('run requires --part and --split')
        run(args)
    elif args.stage == 'verify':
        verify(args.output); print('frozen execution verified')
    else:
        from .metric_v3_statistics import calibrate, summarize
        (calibrate if args.stage == 'calibrate' else summarize)(args.output)


if __name__ == '__main__':
    main()
