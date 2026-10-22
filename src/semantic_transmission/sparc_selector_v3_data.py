"""Read-only v2 evidence reuse, frozen pretrained features and v3 teacher search."""
from collections import OrderedDict
import json
import hashlib
import io
import math
import os
from pathlib import Path
import subprocess

import numpy as np
import torch
import torch.nn.functional as F

from . import sparc_selector_train as supervision
from .sparc_selector_data import read, save, sha, digest, source_features
from .sparc_selector_workers import REPO
from .sparc_selector_budget import METRICS, search, seeds, cbr


def load_source(folder):
    from PIL import Image
    ready = read(folder/'prepared.json')
    source = ready['source']
    images = []
    for i in range(source['video']['frames']):
        path = Path(source['directory'])/f'{i}.png'
        if sha(path) != source['hashes'][f'{i}.png']:
            raise ValueError('source frame checksum mismatch')
        images.append(np.asarray(Image.open(path).convert('RGB')))
    return np.stack(images), ready


def backbone_weights():
    from torchvision.models import ResNet50_Weights
    weights = ResNet50_Weights.IMAGENET1K_V2
    path = Path(torch.hub.get_dir())/'checkpoints'/Path(weights.url).name
    if not path.exists():
        torch.hub.load_state_dict_from_url(weights.url, check_hash=True, map_location='cpu')
    if not sha(path).startswith('11ad3fa6'):
        raise ValueError('unexpected official ResNet50 V2 weights')
    return path


class Extractor:
    def __init__(self, device):
        from torchvision.models import resnet50
        path = backbone_weights()
        net = resnet50(weights=None)
        net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        self.net = torch.nn.Sequential(*list(net.children())[:7]).to(device).eval()
        self.net.requires_grad_(False)
        self.device = device
        self.identity = dict(model='torchvision_resnet50_imagenet1k_v2_layer3',
            weights_sha256=sha(path), input_hw=[192, 336], grid=[3, 5], motion='frozen_source_features_v1')

    def extract(self, frames):
        mean = torch.tensor([.485, .456, .406], device=self.device)[None, :, None, None]
        std = torch.tensor([.229, .224, .225], device=self.device)[None, :, None, None]
        patches = []
        with torch.inference_mode():
            for start in range(0, len(frames), 8):
                supervision.interrupted()
                x = torch.from_numpy(frames[start:start+8].copy()).permute(0, 3, 1, 2).to(self.device).float()/255
                x = F.interpolate(x, (192, 336), mode='bilinear', align_corners=False)
                z = self.net((x-mean)/std)
                z = F.adaptive_avg_pool2d(z, (3, 5)).flatten(2).transpose(1, 2)
                patches.append(z.cpu().half().numpy())
        small, motion, cuts = source_features(frames)
        # Do not interpret a shot cut as physical motion.
        motion[cuts, 8:] = 0
        return dict(patches=np.concatenate(patches), motion=motion,
                    proxies=(small*255).round().astype(np.uint8))


def atomic_npz(path, **values):
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **values)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def compact_observation(obs):
    return {k: obs[k] for k in ('identity', 'channel', 'quality')}


def verify_observation(path, protocol, prepared, source_sha256=None):
    obs = read(path)
    i = obs['identity']
    if (i['protocol'] != protocol['signature'] or digest(i)[:24] != path.parent.name
            or i['keys'] != prepared['keys'] or i['cuts'] != prepared['cuts']):
        raise ValueError('observation provenance mismatch')
    if source_sha256 is not None and i['source_sha256'] != source_sha256:
        raise ValueError('observation source array differs from prepared RGB')
    if (obs['channel']['bit_errors'] or obs['receiver']['source_pixels_at_receiver']
            or obs['sender']['complex_channel_uses'] != obs['channel']['complex_channel_uses']
            or obs['artifacts']['sent.bin'] != obs['artifacts']['received.bin']):
        raise ValueError('invalid real transport observation')
    for name, expected in obs['artifacts'].items():
        if sha(path.parent/name) != expected:
            raise ValueError('observation artifact changed')
    if not all(math.isfinite(obs['quality'][k]) for k in (*METRICS, 'psnr_db')):
        raise ValueError('nonfinite reconstruction metrics')
    return compact_observation(obs)


def prepare(root, parent, rows, p, progress):
    """No test labels/features are read until the final fixed-checkpoint evaluation."""
    extractor = None
    selected = [r for r in rows if r['split'] in ('train', 'val')]
    try:
        for index, row in enumerate(selected):
            supervision.interrupted()
            progress('prepare_v3_evidence', index, len(selected), clip=row['id'])
            old = parent/'clips'/row['id']
            dest = root/'clips'/row['id']
            dest.mkdir(parents=True, exist_ok=True)
            identity = dict(parent_labels=sha(old/'labels.json'), parent_prepared=sha(old/'prepared.json'),
                            feature=p['features'], supervision=p['supervision_protocol']['signature'])
            receipt = dest/'inputs.json'
            if receipt.exists():
                value = read(receipt)
                if (value['identity'] != identity or sha(dest/'features.npz') != value['features_sha256']
                        or sha(dest/'base_observations.json') != value['observations_sha256']):
                    raise ValueError('cached v3 evidence changed')
                continue
            frames, ready = load_source(old)
            buffer = io.BytesIO(); np.save(buffer, frames)
            array_hash = hashlib.sha256(buffer.getbuffer()).hexdigest()
            del buffer
            labels = read(old/'labels.json')
            if labels['sha256'] != sha(old/'labels.npz') or labels['input'] != row:
                raise ValueError('parent label receipt changed')
            refs = sorted({r[k] for r in labels['records'] for k in ('before_observation','after_observation')})
            observations = [verify_observation(old/'observations'/name/'result.json',
                            p['supervision_protocol'], ready, array_hash) for name in refs]
            if extractor is None:
                extractor = Extractor(p['train']['device'])
            if extractor.identity != p['features']:
                raise ValueError('pretrained feature identity mismatch')
            atomic_npz(dest/'features.npz', **extractor.extract(frames))
            save(dest/'prepared.json', ready)
            save(dest/'base_observations.json', observations)
            save(receipt, dict(identity=identity, features_sha256=sha(dest/'features.npz'),
                              observations_sha256=sha(dest/'base_observations.json')))
        progress('prepare_v3_evidence', len(selected), len(selected))
    finally:
        del extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def sample_parents(rows, split, count, seed):
    """Equal-domain, source-disjoint search clips; prefer long views per parent."""
    by_parent = {}
    for row in rows:
        if row['split'] != split:
            continue
        parent = row['parent']
        if parent not in by_parent or row.get('duration_seconds', 0) > by_parent[parent].get('duration_seconds', 0):
            by_parent[parent] = row
    domains = {}
    for row in by_parent.values():
        domains.setdefault(row.get('dataset', 'default'), []).append(row)
    for group in domains.values():
        group.sort(key=lambda r: digest([seed, r['id']]))
    selected = []
    while len(selected) < count and any(domains.values()):
        for name in sorted(domains):
            if domains[name] and len(selected) < count:
                selected.append(domains[name].pop(0))
    return selected


class Measurer:
    """Own new artifacts; reuse parent measurements without modifying them."""
    def __init__(self, root, parent, row, p, progress=None):
        self.root, self.row, self.p = root, row, p
        self.folder = root/'clips'/row['id']
        self.folder.mkdir(parents=True, exist_ok=True)
        self.frames, self.ready = load_source(parent/'clips'/row['id'])
        self.array = self.folder/'source.npy'
        np.save(self.array, self.frames)
        self.evaluator = supervision.Evaluator(root, p['supervision_protocol'])
        self.cached = {}
        self.progress = progress
        base = self.folder/'base_observations.json'
        if base.exists():
            self.cached = {tuple(o['identity']['indices']): o for o in read(base)}
        self.calls = 0

    def __call__(self, indices):
        supervision.interrupted()
        key = tuple(indices)
        if key in self.cached:
            return self.cached[key]
        self.calls += 1
        if self.progress:
            self.progress(self.calls)
        identity = dict(protocol=self.p['supervision_protocol']['signature'], source_sha256=sha(self.array),
                        indices=indices, keys=self.ready['keys'], cuts=self.ready['cuts'])
        out = self.folder/'observations'/digest(identity)[:24]
        rejected = out/'rejected.json'
        if rejected.exists():
            record = read(rejected)
            if record['identity'] != identity or any(sha(out/k) != v for k,v in record['artifacts'].items()):
                raise ValueError('rejected measurement provenance changed')
            return None
        try:
            value = self.evaluator.evaluate(self.folder, self.array, indices,
                                           self.ready['keys'], self.ready['cuts'])
        except RuntimeError as error:
            if 'channel integrity failure; no oracle replacement permitted' not in str(error):
                raise
            client = self.evaluator.clients.pop('channel', None)
            if client is not None:
                client.close()
            # Retain the corrupted payload and verify reproducibility at the same SNR/seed.
            script = '''import json,sys
from pathlib import Path
from semantic_transmission.metadata_channel import transmit
raw=Path(sys.argv[1]).read_bytes()
received,report=transmit(raw,10.,42,bits_per_symbol=4)
Path(sys.argv[2]).write_bytes(received)
print(json.dumps(report))
'''
            env = dict(os.environ, PYTHONPATH=str(REPO/'src'), PYTHONNOUSERSITE='1', CUDA_VISIBLE_DEVICES='-1',
                       OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
            with (self.root/'channel_rejections.log').open('ab') as log:
                report = json.loads(subprocess.check_output([supervision.runtimes()['channel'], '-c', script,
                    str(out/'sent.bin'), str(out/'failed_received.bin')], env=env, stderr=log, text=True, timeout=180))
            a = np.frombuffer((out/'sent.bin').read_bytes(), np.uint8)
            b = np.frombuffer((out/'failed_received.bin').read_bytes(), np.uint8)
            if a.shape != b.shape or report['bit_errors'] <= 0 or int(np.unpackbits(a^b).sum()) != report['bit_errors']:
                raise ValueError('channel failure did not reproduce') from error
            save(rejected, dict(identity=identity, status='REJECTED_CHANNEL', report=report,
                artifacts={k:sha(out/k) for k in ('sent.bin','failed_received.bin')}, use_for_training=False))
            return None
        value = compact_observation(value)
        self.cached[key] = value
        return value

    @property
    def pixels(self):
        return math.prod(self.frames.shape[:3])

    def close(self):
        self.evaluator.close()
        self.array.unlink(missing_ok=True)


def teacher(root, parent, rows, p, progress):
    chosen = sample_parents(rows, 'train', p['search']['teacher_clips'], p['seed'])
    save(root/'teacher_manifest.json', chosen)
    for i, row in enumerate(chosen):
        supervision.interrupted()
        folder = root/'clips'/row['id']
        result_path = folder/'teacher.json'
        if result_path.exists():
            cached = read(result_path)
            if cached['protocol'] != p['signature']:
                raise ValueError('teacher protocol changed')
            verify_teacher(folder, cached, p)
            continue
        progress('teacher_search', i, len(chosen), clip=row['id'])
        measure = Measurer(root, parent, row, p,
            lambda calls: progress('teacher_search', i, len(chosen), clip=row['id'], candidate_measurements=calls))
        try:
            with np.load(folder/'features.npz') as data:
                proxies = data['proxies'].copy()
            budget = math.floor(p['target_cbr']*3*measure.pixels)
            result, observations, baseline = search(proxies, measure.ready['keys'], measure.ready['cuts'],
                budget, measure, known=list(measure.cached.values()),
                rounds=p['search']['rounds'], beam=p['search']['beam'],
                per_round=p['search']['per_round'], stop=supervision.interrupted)
            save(result_path, dict(protocol=p['signature'], result=result, baseline=baseline,
                                  observations=observations, target_cbr=p['target_cbr']))
        finally:
            measure.close()
    progress('teacher_search', len(chosen), len(chosen))


def verify_teacher(folder, teacher, p):
    base = {tuple(o['identity']['indices']): o for o in read(folder/'base_observations.json')}
    ready = read(folder/'prepared.json')
    for observation in teacher['observations']:
        key = tuple(observation['identity']['indices'])
        if key in base:
            original = base[key]
        else:
            original = verify_observation(folder/'observations'/digest(observation['identity'])[:24]/'result.json',
                                          p['supervision_protocol'], ready)
        if original != observation:
            raise ValueError('teacher differs from measured evidence')


class Corpus:
    def __init__(self, root, rows, p):
        self.root, self.p = root, p
        self.rows = {s: [r for r in rows if r['split'] == s] for s in ('train','val','test')}
        self.parents = {s: {r['parent'] for r in group} for s, group in self.rows.items()}
        if any(self.parents[a]&self.parents[b] for a, b in [('train','val'),('train','test'),('val','test')]):
            raise ValueError('parent leakage between splits')
        self.records, self.prepared, self.cache = {}, {}, OrderedDict()
        hashes, targets = {}, []
        for row in self.rows['train']+self.rows['val']:
            folder = root/'clips'/row['id']
            ready = read(folder/'prepared.json')
            observations = read(folder/'base_observations.json')
            inputs = read(folder/'inputs.json')
            if sha(folder/'features.npz') != inputs['features_sha256']:
                raise ValueError('feature cache changed')
            if sha(folder/'base_observations.json') != inputs['observations_sha256']:
                raise ValueError('measurement cache changed')
            paths = ['inputs.json', 'prepared.json', 'base_observations.json']
            if (folder/'teacher.json').exists():
                t = read(folder/'teacher.json')
                if t['protocol'] != p['signature'] or row['split'] != 'train':
                    raise ValueError('teacher split/protocol mismatch')
                verify_teacher(folder, t, p)
                observations += t['observations']
                paths += ['teacher.json']
            observations = list({tuple(o['identity']['indices']): o for o in observations}.values())
            if not observations:
                continue
            v = ready['source']['video']
            pixels = v['frames']*v['height']*v['width']
            y = np.log(np.maximum([[*[o['quality'][k] for k in METRICS], cbr(o, pixels)]
                                    for o in observations], 1e-8)).astype(np.float32)
            if not np.isfinite(y).all():
                raise ValueError('nonfinite absolute training target')
            self.records[row['id']] = (observations, torch.from_numpy(y), pixels)
            self.prepared[row['id']] = ready
            hashes[row['id']] = {name: sha(folder/name) for name in paths}
            if row['split'] == 'train':
                targets.append(y)
        for split in ('train','val'):
            self.rows[split] = [r for r in self.rows[split] if r['id'] in self.records]
        if not targets or not self.rows['val']:
            raise ValueError('nonempty disjoint train/val data required')
        y = torch.from_numpy(np.concatenate(targets))
        self.normalization = dict(mean=y.mean(0), std=y.std(0).clamp_min(.05))
        self.data_hash = digest(hashes)

    def features(self, row):
        key = row['id']
        if key not in self.cache:
            with np.load(self.root/'clips'/key/'features.npz') as z:
                self.cache[key] = (torch.from_numpy(z['patches'].copy()), torch.from_numpy(z['motion'].copy()))
        self.cache.move_to_end(key)
        while len(self.cache) > 4:
            self.cache.popitem(last=False)
        return self.cache[key]
