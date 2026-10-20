"""Pinned public clips, source-only screening, and disjoint allocation splits."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess

from .artifacts import sha256, write_json
from .qwen35_pairs_video import freeze
from .webvid_ablation import Stages
from .webvid5 import fingerprint

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO/'configs/bsu_allocation_labels.json'


def read(path):
    return json.loads(Path(path).read_text())


def catalog(root, policy):
    path = root/'catalog.json'
    if path.exists():
        value = read(path)
        if value['policy'] != policy:
            raise ValueError('corpus acquisition policy changed')
        return value
    import requests
    url = (f"https://huggingface.co/api/datasets/{policy['dataset']}/tree/"
           f"{policy['dataset_revision']}/{policy['dataset_directory']}?limit=1000")
    files = []
    while url:
        response = requests.get(url, timeout=60); response.raise_for_status()
        files.extend(v for v in response.json() if v['path'].endswith('.mp4'))
        url = response.links.get('next', {}).get('url')
    files.sort(key=lambda v: v['path'])
    if len(files) != 1000 or any(not v.get('lfs', {}).get('oid') for v in files):
        raise ValueError('expected pinned 1000-video Inter4K catalog with content hashes')
    random.Random(policy['seed']).shuffle(files)
    value = dict(policy=policy, files=files,
        official_source='https://github.com/alexandrosstergiou/Inter4K',
        download_mirror='https://huggingface.co/datasets/'+policy['dataset'],
        original_filming_identity_available=False)
    freeze(path, value)
    return value


def download(entry, policy, cache):
    cache.mkdir(parents=True, exist_ok=True)
    target = cache/Path(entry['path']).name
    expected = entry['lfs']['oid']
    if target.exists():
        if sha256(target) != expected:
            raise ValueError(f'changed downloaded source: {target}')
        return target
    temporary = target.with_suffix('.mp4.part')
    url = (f"https://huggingface.co/datasets/{policy['dataset']}/resolve/"
           f"{policy['dataset_revision']}/{entry['path']}?download=true")
    subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
        '--retry', '3', '--retry-delay', '2', '--connect-timeout', '30',
        '--max-time', '600', '--continue-at', '-', '--output', str(temporary), url], check=True)
    if temporary.stat().st_size != entry['size'] or sha256(temporary) != expected:
        raise ValueError('download does not match pinned LFS size/SHA256')
    temporary.rename(target)
    return target


def thumbnails(source):
    import cv2
    import numpy as np
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-nostdin', '-threads', '2',
        '-i', str(source), '-vf', 'fps=1,scale=32:32', '-frames:v', '5',
        '-threads', '2', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
    rgb = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 32, 32, 3)
    if len(rgb) != 5:
        raise ValueError('source is shorter than the registered five-second clip')
    hashes = []
    for frame in rgb:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
        block = cv2.dct(gray)[:8, :8].flatten()[1:]
        hashes.append(''.join('1' if v > np.median(block) else '0' for v in block))
    return dict(phash=hashes, rgb8=(rgb[:, ::4, ::4]/255.).tolist())


def similar(a, b):
    """Conservative automatic screen; not proof of distinct filming sessions."""
    import numpy as np
    matches = 0
    for i, x in enumerate(a['phash']):
        for j, y in enumerate(b['phash']):
            if sum(c != d for c, d in zip(x, y)) <= 6:
                error = np.abs(np.asarray(a['rgb8'][i])-np.asarray(b['rgb8'][j])).mean()
                if error < .08:
                    matches += 1
                    break
    return matches >= 2


def split_rows(rows, train_count, seed):
    if len({v['source_sha256'] for v in rows}) != len(rows):
        raise ValueError('duplicate original source SHA256')
    order = sorted(rows, key=lambda v: hashlib.sha256(f"{seed}:{v['id']}".encode()).hexdigest())
    train = {v['id'] for v in order[:train_count]}
    return [dict(v, split='train' if v['id'] in train else 'validation') for v in rows]


def prepare_clip(source, dest, policy, identity):
    import numpy as np
    from .bsu_allocation_data import extract_features
    from .video_io import probe
    dest.mkdir(parents=True)
    rng = random.Random(f"{policy['seed']}:{identity}")
    start = round(rng.uniform(0, 5-policy['frames']/policy['fps']-.1), 6)
    movie = dest/'source.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-threads', '2', '-ss', str(start),
        '-i', str(source), '-vf', f"fps={policy['fps']},scale={policy['width']}:{policy['height']}:flags=lanczos",
        '-frames:v', str(policy['frames']), '-an', '-c:v', 'libx264', '-preset', 'fast',
        '-crf', '0', '-pix_fmt', 'yuv420p', '-threads', '2', str(movie)], check=True)
    info = probe(movie)
    if any(info[k] != policy[k] for k in ('width', 'height', 'frames', 'fps')):
        raise ValueError('normalized source geometry mismatch')
    frames = dest/'frames'; frames.mkdir()
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-threads', '2', '-i', str(movie),
        '-threads', '2', '-start_number', '0', str(frames/'%05d.png')], check=True)
    paths = [frames/f'{i:05d}.png' for i in range(policy['frames'])]
    feature = extract_features(paths, policy['fps'], dest/'features.npz')
    with np.load(dest/'features.npz') as features:
        motion = float(features['features'][1:, 68].mean())
    write_json(dest/'source.json', dict(id=identity, source_id='inter4k:'+identity,
        source_sha256=sha256(source), original=str(source), source=str(movie),
        normalized_sha256=sha256(movie), start_seconds=start,
        video={k:policy[k] for k in ('width', 'height', 'frames', 'fps')},
        feature=feature, mean_motion_feature=motion))


def prepare(root, limit=None):
    root = root.resolve(); root.mkdir(parents=True, exist_ok=True)
    policy = read(CONFIG); data = catalog(root, policy)
    code = {str(p):sha256(p) for p in (Path(__file__), CONFIG,
        Path(__file__).with_name('bsu_allocation_data.py'))}
    identity = fingerprint(dict(catalog=data, code=code))
    selected, rejected = [], []
    target = policy['sources'] if limit is None else limit
    for entry in data['files']:
        video_id = Path(entry['path']).stem
        original = download(entry, policy, root/'downloads')
        thumb = root/'screening'/f'{video_id}.json'
        if not thumb.exists():
            write_json(thumb, dict(original_sha256=sha256(original), **thumbnails(original)))
        descriptor = read(thumb)
        if descriptor['original_sha256'] != sha256(original):
            raise ValueError('screening source changed')
        duplicate = next((v for v in selected if v['source_sha256'] == sha256(original)
                         or similar(v['descriptor'], descriptor)), None)
        if duplicate:
            rejected.append(dict(id=video_id, duplicate_of=duplicate['id'], reason='source-only near-duplicate screen'))
            continue
        folder = root/'sources'/video_id
        stages = Stages(folder.parent, identity+':'+video_id)
        stages.step('source_'+video_id, [video_id], [video_id],
            lambda _,o=original,d=folder,i=video_id:prepare_clip(o,d,policy,i))
        row = read(folder/'source.json'); row['descriptor'] = descriptor
        selected.append(row)
        write_json(root/'acquisition_status.json', dict(status='RUNNING', selected=len(selected),
            target=policy['sources'], rejected=rejected, last_id=video_id))
        print(f'CORPUS {len(selected)}/{policy["sources"]}: Inter4K {video_id}', flush=True)
        if len(selected) == target:
            break
    if len(selected) != target:
        raise ValueError('not enough distinct screened sources')
    if target != policy['sources']:
        return selected
    rows = split_rows(selected, policy['train_sources'], policy['seed'])
    frozen = dict(schema='bsu_allocation_corpus_v1', acquisition_identity=identity,
        policy=policy, records=rows, rejected=rejected, screening='full-clip five-thumbnail pHash+RGB',
        limits=['dataset file groups; no original filming-session IDs in mirror',
                'automatic near-duplicate screening is not semantic independence proof',
                'one 49-frame clip/source; one SNR and generation seed; validation is not an untouched test set'])
    freeze(root/'corpus.json', frozen)
    write_json(root/'acquisition_status.json', dict(status='COMPLETE', selected=target, target=target,
        train_sources=policy['train_sources'], validation_sources=target-policy['train_sources']))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    prepare(args.output, args.limit)
