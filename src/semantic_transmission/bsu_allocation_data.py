"""Source-only feature cache and grouped, measured candidate labels for BSU.

Labels live outside the model input dictionary. All crops, seeds, budgets and
rates from an original video must share source_id. Hash equality additionally
detects identical source files disguised by different IDs.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch

from .artifacts import sha256, write_json
from .bsu_allocation_transport import bundle, receipt, verified_path, validate_candidate, wire_cost

SCHEMA = 'bsu_joint_allocation_dataset_v1'
FEATURE_SCHEMA = 'sender_rgb4x4_delta4x4_farneback_laplacian_v1'
METRICS = ('psnr_db', 'lpips_vgg', 'flow_error_mean')
MODEL_INPUTS = ('features', 'times', 'frame_mask', 'key_positions', 'etas', 'key_mask',
                'candidate_mask', 'budget', 'snr', 'baseline_index')


def extract_features(frame_paths, fps, destination):
    """71 fixed-scale features per SOURCE frame; no reconstruction is read.

    A cached frozen learned backbone can replace this via the same NPZ interface
    and a different feature_schema. This helper does not pretend to be DINOv3.
    """
    import cv2
    from PIL import Image
    if not math.isfinite(fps) or fps <= 0 or len(frame_paths) < 2:
        raise ValueError('feature extraction requires >=2 frames and positive FPS')
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    cv2.setNumThreads(1)
    values, hashes = [], []
    previous = None
    for path in frame_paths:
        hashes.append(sha256(path))
        rgb = np.asarray(Image.open(path).convert('RGB'))
        rgb = cv2.resize(rgb, (144, 80), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        pooled = cv2.resize(rgb, (4, 4), interpolation=cv2.INTER_AREA).reshape(-1) / 255.
        if previous is None:
            delta, motion = np.zeros(16), np.zeros(6)
        else:
            diff = np.abs(gray.astype(np.float32)-previous.astype(np.float32)) / 255.
            delta = cv2.resize(diff, (4, 4), interpolation=cv2.INTER_AREA).reshape(-1)
            flow = cv2.calcOpticalFlowFarneback(previous, gray, None, .5, 3, 15, 3, 5, 1.2, 0)
            magnitude = np.linalg.norm(flow, axis=-1)
            motion = np.array([*flow.mean((0, 1)), *flow.std((0, 1)),
                               magnitude.mean(), np.percentile(magnitude, 90)]) / 32.
        texture = np.abs(cv2.Laplacian(gray.astype(np.float32)/255., cv2.CV_32F)).mean()
        values.append(np.r_[pooled, delta, motion, texture])
        previous = gray
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        np.savez_compressed(stream, features=np.asarray(values, dtype=np.float32),
            times=np.arange(len(values), dtype=np.float32)/fps,
            frame_indices=np.arange(len(values), dtype=np.int64))
    return dict(**receipt(destination), source_frame_sha256=hashes, source_only=True)


def _integer(value, name):
    if type(value) is not int or not 0 < value < 2**63:
        raise ValueError(f'{name} must be a positive int64')


def load_dataset(path, *, require_labels=True):
    path = Path(path).resolve(strict=True)
    document = json.loads(path.read_text())
    if document.get('schema') != SCHEMA or not document.get('feature_schema') or not document.get('pipeline_id'):
        raise ValueError('unknown dataset/feature/pipeline schema')
    _integer(document['feature_dim'], 'feature_dim')
    if not document.get('records'):
        raise ValueError('empty allocation dataset')
    ids, groups, source_hashes, records = set(), {}, {}, []
    for row in document['records']:
        if row['id'] in ids:
            raise ValueError('duplicate record ID')
        ids.add(row['id'])
        split = row['split']
        if split not in ('train', 'validation', 'test', 'development', 'inference') or not row['source_id']:
            raise ValueError('invalid source group/split')
        if (not isinstance(row['source_sha256'], str) or len(row['source_sha256']) != 64
                or any(c not in '0123456789abcdef' for c in row['source_sha256'])):
            raise ValueError('invalid source SHA256')
        for mapping, key in ((groups, row['source_id']), (source_hashes, row['source_sha256'])):
            if key in mapping and mapping[key] != split:
                raise ValueError('source leakage across splits')
            mapping[key] = split
        _integer(row['frames'], 'frames')
        _integer(row['budget'], 'budget')
        if not math.isfinite(row['snr_db']):
            raise ValueError('nonfinite SNR')
        feature = row['feature']
        if feature.get('source_only') is not True:
            raise ValueError('features must be sender/source-only')
        with np.load(verified_path(feature), allow_pickle=False) as cache:
            if set(cache.files) != {'features', 'times', 'frame_indices'}:
                raise ValueError('unexpected fields in feature cache')
            features = cache['features'].astype(np.float32)
            times = cache['times'].astype(np.float32)
            indices = cache['frame_indices']
        if (features.ndim != 2 or features.shape[1] != document['feature_dim'] or len(features) < 2
                or times.shape != (len(features),) or indices.shape != times.shape
                or indices.dtype.kind not in 'iu' or indices[0] != 0 or indices[-1] != row['frames']-1
                or np.any(np.diff(indices.astype(np.int64)) <= 0) or times[0] != 0
                or np.any(np.diff(times) <= 0) or not np.isfinite(features).all()
                or not np.isfinite(times).all()):
            raise ValueError('invalid source feature/timestamp shapes or values')
        names, costs, positions, labels = set(), [], [], []
        for candidate in row['candidates']:
            if candidate['id'] in names:
                raise ValueError('duplicate candidate ID')
            names.add(candidate['id'])
            validate_candidate(candidate, row['frames'], row.get('minimum_key_spacing', 1))
            lookup = {int(index): i for i, index in enumerate(indices)}
            if not set(candidate['keys']) <= set(lookup):
                raise ValueError('candidate key missing from feature timeline')
            positions.append([lookup[i] for i in candidate['keys']])
            actual = wire_cost(candidate['bundle'], candidate, row['frames'])['total_uses']
            _integer(actual, 'actual cost')
            costs.append(actual)
            if require_labels:
                if 'label_evidence' in candidate:
                    verified_path(candidate['label_evidence'])
                metric = candidate['metrics']
                label = [float(metric[m]) for m in METRICS]
                if not all(math.isfinite(v) for v in label) or any(v < 0 for v in label[1:]):
                    raise ValueError('missing/invalid measured candidate metrics')
                labels.append(label)
        if not names or row['baseline_id'] not in names:
            raise ValueError('missing candidates or baseline')
        if not any(c <= row['budget'] for c in costs):
            raise ValueError('INFEASIBLE_REGISTERED_BUDGET: dataset row has no feasible candidate')
        baseline = next(i for i, c in enumerate(row['candidates']) if c['id'] == row['baseline_id'])
        if costs[baseline] > row['budget']:
            raise ValueError('reference baseline must fit the registered budget')
        records.append(dict(row=row, features=features, times=times, positions=positions,
                            costs=costs, labels=labels, baseline=baseline))
    return document, records


def collate(records, objective, device='cpu', *, labels=True):
    """Explicit allow-list: evaluator labels and source IDs never reach forward()."""
    b = len(records)
    t = max(len(r['features']) for r in records)
    c = max(len(r['costs']) for r in records)
    k = max(len(p) for r in records for p in r['positions'])
    f = records[0]['features'].shape[1]
    def zeros(*shape, dtype=torch.float32):
        return torch.zeros(*shape, dtype=dtype, device=device)
    inputs = dict(features=zeros(b, t, f), times=zeros(b, t), frame_mask=zeros(b, t, dtype=torch.bool),
        key_positions=zeros(b, c, k, dtype=torch.long), etas=torch.ones(b, c, k, device=device),
        key_mask=zeros(b, c, k, dtype=torch.bool), candidate_mask=zeros(b, c, dtype=torch.bool),
        budget=torch.tensor([r['row']['budget'] for r in records], dtype=torch.long, device=device),
        snr=torch.tensor([r['row']['snr_db'] for r in records], dtype=torch.float32, device=device),
        baseline_index=torch.tensor([r['baseline'] for r in records], dtype=torch.long, device=device))
    costs = torch.ones(b, c, dtype=torch.long, device=device)
    targets, utility = zeros(b, c, len(METRICS)), zeros(b, c)
    scales = torch.tensor(objective['scales'], device=device)
    signs = torch.tensor([1., -1., -1.], device=device)
    weights = torch.tensor(objective['weights'], device=device)
    if (scales.shape != (len(METRICS),) or weights.shape != scales.shape
            or not torch.isfinite(scales).all() or not torch.isfinite(weights).all()
            or (scales <= 0).any() or (weights < 0).any() or weights.sum() <= 0):
        raise ValueError('invalid predeclared metric scales/weights')
    for i, r in enumerate(records):
        nt, nc = len(r['features']), len(r['costs'])
        inputs['features'][i, :nt] = torch.as_tensor(r['features'], device=device)
        inputs['times'][i, :nt] = torch.as_tensor(r['times'], device=device)
        inputs['frame_mask'][i, :nt] = True
        inputs['candidate_mask'][i, :nc] = True
        costs[i, :nc] = torch.tensor(r['costs'], dtype=torch.long, device=device)
        for j, (positions, candidate) in enumerate(zip(r['positions'], r['row']['candidates'])):
            nk = len(positions)
            inputs['key_positions'][i, j, :nk] = torch.tensor(positions, device=device)
            inputs['etas'][i, j, :nk] = torch.tensor(candidate['etas'], device=device)
            inputs['key_mask'][i, j, :nk] = True
        if labels:
            measured = torch.tensor(r['labels'], dtype=torch.float32, device=device)
            target = (measured-measured[r['baseline']]) / scales * signs
            targets[i, :nc] = target
            utility[i, :nc] = (target * weights).sum(-1)
    return dict(inputs=inputs, costs=costs, targets=targets, utility=utility)


def import_pilot(root, destination):
    """Read-only reuse of the completed two-video pilot, DEVELOPMENT ONLY."""
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    result = json.loads((root/'RESULT.json').read_text())
    protocol = json.loads((root/'protocol.json').read_text())
    if result['status'] != 'COMPLETE_BSU_KEY_BUDGET_PILOT' or result['signature'] != protocol['signature']:
        raise ValueError('completed matching pilot required')
    destination.mkdir(parents=True)
    records = []
    for case in protocol['cases']:
        digest = sha256(case['source'])
        selection = json.loads((root/'cases'/case['id']/'source/selection.json').read_text())
        paths = [root/'cases'/case['id']/'source'/f'{i:05d}.png' for i in range(case['video']['frames'])]
        if digest != selection['source_sha256'] or any(sha256(p) != selection['source_frame_hashes'][str(i)]
                                                      for i, p in enumerate(paths)):
            raise ValueError('pilot source/frame provenance changed')
        feature = extract_features(paths, case['video']['fps'], destination/(case['id']+'.npz'))
        for seed in protocol['seeds']:
            candidates = []
            matched = [r for r in result['records'] if r['id'] == case['id'] and r['seed'] == seed]
            for r in matched:
                transport = Path(r['transport'])
                tx = transport/'run/transmitter'
                aux = transport/'trajectory_sender/trajectory.bin'
                candidate = dict(id=r['arm'], keys=r['keys'], etas=[r['eta']]*len(r['keys']),
                    bundle=bundle(tx/'visual.c64', tx/'metadata.bin', [aux] if aux.exists() else []))
                cost = wire_cost(candidate['bundle'], candidate, r['frames'])
                evaluation = Path(r['output'])/'evaluation/refined/quality.json'
                measured = json.loads(evaluation.read_text())
                if (measured != r['refined'] or measured['source_sha256'] != digest
                        or sha256(Path(r['output'])/'refined/reconstruction.mp4') != measured['video_sha256']
                        or cost['total_uses'] != r['total_uses']):
                    raise ValueError('pilot evaluation/actual cost mismatch')
                candidate['metrics'] = {m: measured['quality' if m != 'flow_error_mean' else 'temporal'][m]
                                        for m in METRICS}
                candidate['label_evidence'] = receipt(evaluation)
                candidates.append(candidate)
            if len(candidates) != 3:
                raise ValueError('pilot must have all three measured arms')
            records.append(dict(id=f'{case["id"]}_seed_{seed}', source_id=case['id'], source_sha256=digest,
                split='development', frames=case['video']['frames'], snr_db=protocol['policy']['snr_db'],
                budget=matched[0]['budget'], baseline_id='baseline', feature=feature, candidates=candidates,
                minimum_key_spacing=protocol['policy']['minimum_key_spacing']))
    document = dict(schema=SCHEMA, feature_schema=FEATURE_SCHEMA, feature_dim=71,
        pipeline_id=protocol['signature'], records=records, pilot_result=receipt(root/'RESULT.json'),
        generalization_proven=False, purpose='development integration/overfit only; never an independent test')
    path = destination/'dataset.json'
    write_json(path, document)
    load_dataset(path)
    return path
