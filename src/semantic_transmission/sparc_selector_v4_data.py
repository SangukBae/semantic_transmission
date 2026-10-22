"""Immutable adaptive-reference targets and spatial labels from delivered MP4s."""
from collections import OrderedDict
import hashlib
import io
import math
from pathlib import Path
import subprocess

import numpy as np
import torch

from . import sparc_selector_budget as budget
from . import sparc_selector_v3_data as old
from . import sparc_selector_train as supervision
from .sparc_selector_data import read, save, sha, digest
from .sparc_selector_risk import quality_feasible
from .sparc_selector_search_v4 import Session


def split_rows(rows, config):
    groups = {split: {r['parent'] for r in rows if r['split'] == split} for split in ('train','val','test')}
    if any(groups[a] & groups[b] for a, b in [('train','val'), ('train','test'), ('val','test')]):
        raise ValueError('source parent leakage between splits')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('duplicate source id')
    count = config['calibration']['clips'] + config['search']['validation_clips']
    heldout = old.sample_parents(rows, 'val', count, config['seed'])
    if len(heldout) < count:
        raise ValueError('not enough disjoint validation parents for calibration and model selection')
    n = config['calibration']['clips']
    return dict(calibration_ids=[r['id'] for r in heldout[:n]], validation_ids=[r['id'] for r in heldout[n:]],
                test_ids=[r['id'] for r in old.sample_parents(rows, 'test', config['search']['test_clips'], config['seed'])])


def source_array_hash(frames):
    buffer = io.BytesIO()
    np.save(buffer, frames)
    return hashlib.sha256(buffer.getbuffer()).hexdigest()


def spatial_errors(frames, movie, grid=(3, 5)):
    """One log-MSE label per source time and spatial patch, measured after CRF18."""
    n, h, w, _ = frames.shape
    if min(h // grid[0], w // grid[1]) < 1:
        raise ValueError('spatial label grid exceeds source geometry')
    command = ['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(movie),
               '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-threads', '2', '-']
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    labels = []
    ys = np.linspace(0, h, grid[0]+1).astype(int)
    xs = np.linspace(0, w, grid[1]+1).astype(int)
    try:
        for frame in frames:
            supervision.interrupted()
            raw = process.stdout.read(h*w*3)
            if len(raw) != h*w*3:
                raise ValueError('spatial label reconstruction timeline is short')
            restored = np.frombuffer(raw, np.uint8).reshape(h, w, 3)
            error = ((frame.astype(np.float32) - restored) / 255) ** 2
            labels.append([float(error[a:b, c:d].mean())
                           for a, b in zip(ys, ys[1:]) for c, d in zip(xs, xs[1:])])
        if process.stdout.read(1) or process.wait() != 0:
            raise ValueError('spatial label reconstruction timeline or decoder failed')
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
        process.wait()
    values = np.asarray(labels, np.float32)
    if values.shape != (n, math.prod(grid)) or not np.isfinite(values).all():
        raise ValueError('invalid spatial labels')
    return np.log(np.maximum(values, 1e-8))


def locate_observation(roots, row, observation, protocol, ready, array_hash):
    key = digest(observation['identity'])[:24]
    for root in roots:
        path = root/'clips'/row['id']/'observations'/key/'result.json'
        if path.exists():
            checked = old.verify_observation(path, protocol, ready, array_hash)
            if checked != old.compact_observation(observation):
                raise ValueError('candidate differs from its measured artifacts')
            return path.parent/'delivered.mp4'
    raise ValueError('candidate has no verified reconstruction artifacts: '+row['id']+'/'+key)


def prepare(root, parent, source_root, rows, p, progress):
    selected = [r for r in rows if r['split'] == 'train' or r['id'] in p['calibration_ids']]
    teacher_ids = {r['id'] for r in old.sample_parents(rows, 'train', p['search']['teacher_clips'], p['seed'])}
    try:
        for index, row in enumerate(selected):
            supervision.interrupted()
            progress('prepare_v4_labels', index, len(selected), clip=row['id'])
            folder = root/'clips'/row['id']
            folder.mkdir(parents=True, exist_ok=True)
            receipt = folder/'risk_inputs.json'
            if receipt.exists():
                verify_cache(folder, p)
                continue
            frames, ready = old.load_source(source_root/'clips'/row['id'])
            if ready['input'] != row or ready['protocol'] != p['supervision_protocol']['signature']:
                raise ValueError('prepared source identity changed')
            save(folder/'prepared.json', ready)
            src = parent/'clips'/row['id']
            old_ready = read(src/'prepared.json')
            if any(old_ready['source'][k] != ready['source'][k] for k in ('video', 'hashes')):
                raise ValueError('parent features refer to different source pixels')
            inputs = read(src/'inputs.json')
            if (sha(src/'features.npz') != inputs['features_sha256']
                    or sha(src/'base_observations.json') != inputs['observations_sha256']):
                raise ValueError('parent feature or observation cache changed')
            with np.load(src/'features.npz') as z:
                old.atomic_npz(folder/'features.npz', **{k:z[k] for k in z.files})
                proxies = z['proxies'].copy()
            observations = read(src/'base_observations.json')
            teacher = src/'teacher.json'
            if teacher.exists():
                if row['split'] != 'train':
                    raise ValueError('teacher candidates in non-training split')
                observations += read(teacher)['observations']
            observations = list({tuple(o['identity']['indices']):o for o in observations}.values())
            array_hash = source_array_hash(frames)
            roots = [root, parent, source_root]
            movies = {tuple(o['identity']['indices']):locate_observation(
                roots, row, o, p['supervision_protocol'], ready, array_hash) for o in observations}
            save(folder/'base_observations.json', observations)
            measure = old.Measurer(root, source_root, row, p)
            try:
                baseline = measure(list(budget.seeds(proxies, ready['keys'], ready['cuts'])[0]))
                if baseline is None:
                    raise ValueError('fixed adaptive baseline failed channel verification')
                if row['id'] in teacher_ids:
                    session = Session(row['id'], proxies, ready['keys'], ready['cuts'], measure.pixels,
                        measure, p['search'], p['quality_goal'], stop=supervision.interrupted)
                    session.initialize()
                    # Retain original measured evidence and explore across all registered budgets.
                    session.measured.update({tuple(o['identity']['indices']):o for o in observations})
                    session.visited.update(session.measured)
                    for target in p['training_budgets']:
                        session.step(target, p['search']['per_round'])
                observations = list(measure.cached.values())
                observations.sort(key=lambda o: tuple(o['identity']['indices']))
                local = []
                for o in observations:
                    key = tuple(o['identity']['indices'])
                    movie = movies.get(key)
                    if movie is None:
                        movie = locate_observation(roots, row, o, p['supervision_protocol'], ready, array_hash)
                    local.append(spatial_errors(frames, movie))
                save(folder/'observations.json', observations)
                save(folder/'baseline.json', baseline)
                old.atomic_npz(folder/'risk_labels.npz', local_log_mse=np.stack(local))
                files = ('prepared.json', 'features.npz', 'observations.json', 'baseline.json', 'risk_labels.npz')
                save(receipt, dict(schema='sparc_risk_labels_v4', protocol=p['signature'],
                    source_array_sha256=array_hash, baseline_identity=baseline['identity'],
                    files={name:sha(folder/name) for name in files},
                    measurements=len(observations), spatial_grid=[3,5], target_source='delivered H264 CRF18'))
            finally:
                measure.close()
        progress('prepare_v4_labels', len(selected), len(selected))
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def verify_cache(folder, p):
    receipt = read(folder/'risk_inputs.json')
    required = {'prepared.json', 'features.npz', 'observations.json', 'baseline.json', 'risk_labels.npz'}
    if (receipt['schema'] != 'sparc_risk_labels_v4' or receipt['protocol'] != p['signature']
            or set(receipt['files']) != required
            or any(sha(folder/name) != expected for name, expected in receipt['files'].items())):
        raise ValueError('immutable v4 label cache changed')
    return receipt


class Corpus:
    def __init__(self, root, rows, p):
        self.root, self.p = root, p
        self.rows = dict(train=[r for r in rows if r['split'] == 'train'],
                         calibration=[r for r in rows if r['id'] in p['calibration_ids']])
        split_rows(rows, p)
        calibrators = {r['parent'] for r in self.rows['calibration']}
        validators = {r['parent'] for r in rows if r['id'] in p['validation_ids']}
        if calibrators & validators:
            raise ValueError('calibration/model-selection parent leakage')
        self.records, self.prepared, self.cache = {}, {}, OrderedDict()
        hashes, targets, deltas = {}, [], []
        local_sum = local_square = local_count = 0
        for row in self.rows['train'] + self.rows['calibration']:
            folder = root/'clips'/row['id']
            receipt = verify_cache(folder, p)
            observations, baseline = read(folder/'observations.json'), read(folder/'baseline.json')
            ready = read(folder/'prepared.json')
            with np.load(folder/'features.npz') as z:
                expected = list(budget.seeds(z['proxies'], ready['keys'], ready['cuts'])[0])
            if baseline['identity']['indices'] != expected or baseline not in observations:
                raise ValueError('fixed adaptive baseline absent or changed')
            if receipt['baseline_identity'] != baseline['identity']:
                raise ValueError('baseline receipt mismatch')
            v = ready['source']['video']
            pixels = v['frames']*v['height']*v['width']
            absolute = np.log(np.maximum([[*[o['quality'][k] for k in budget.METRICS], budget.cbr(o, pixels)]
                                         for o in observations], 1e-8)).astype(np.float32)
            delta = np.asarray([[o['quality']['psnr_db']-baseline['quality']['psnr_db'],
                                 o['quality']['lpips_vgg']-baseline['quality']['lpips_vgg']]
                                for o in observations], np.float32)
            unsafe = np.asarray([not quality_feasible(o, baseline, p['quality_goal']) for o in observations], np.float32)
            if not np.isfinite(absolute).all() or not np.isfinite(delta).all():
                raise ValueError('nonfinite training targets')
            self.records[row['id']] = dict(observations=observations, baseline=baseline, pixels=pixels,
                absolute=torch.from_numpy(absolute), delta=torch.from_numpy(delta), unsafe=torch.from_numpy(unsafe))
            self.prepared[row['id']] = ready
            hashes[row['id']] = sha(folder/'risk_inputs.json')
            if row['split'] == 'train':
                targets.append(absolute)
                deltas.append(delta)
                with np.load(folder/'risk_labels.npz') as z:
                    a = z['local_log_mse'].astype(np.float64)
                    if a.shape != (len(observations), v['frames'], 15) or not np.isfinite(a).all():
                        raise ValueError('invalid local error target shape')
                    local_sum += a.sum(); local_square += (a*a).sum(); local_count += a.size
        if not targets or not self.rows['calibration']:
            raise ValueError('training and independent calibration data required')
        a, d = torch.from_numpy(np.concatenate(targets)), torch.from_numpy(np.concatenate(deltas))
        mean = local_sum/local_count
        self.normalization = dict(mean=a.mean(0), std=a.std(0, unbiased=False).clamp_min(.05),
            delta_mean=d.mean(0), delta_std=d.std(0, unbiased=False).clamp_min(torch.tensor([.05,.001])),
            local_mean=torch.tensor(mean, dtype=torch.float32),
            local_std=torch.tensor(max(.05, math.sqrt(max(0,local_square/local_count-mean*mean))), dtype=torch.float32))
        self.data_hash = digest(hashes)

    def tensors(self, row):
        key = row['id']
        if key not in self.cache:
            folder = self.root/'clips'/key
            with np.load(folder/'features.npz') as z:
                features = [torch.from_numpy(z[k].copy()) for k in ('patches','motion')]
            with np.load(folder/'risk_labels.npz') as z:
                local = torch.from_numpy(z['local_log_mse'].copy())
            expected = (len(self.records[key]['observations']), len(features[0]), 15)
            if local.shape != expected or not torch.isfinite(local).all():
                raise ValueError('invalid spatial calibration/training targets')
            self.cache[key] = (features, local)
        self.cache.move_to_end(key)
        while len(self.cache) > 4:
            self.cache.popitem(last=False)
        return self.cache[key]
