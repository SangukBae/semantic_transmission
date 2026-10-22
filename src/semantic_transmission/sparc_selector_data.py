"""Public sources and state-dependent, source-only auxiliary-selection features.

No codec, channel, reference reconstruction or quality label enters features.
"""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import tarfile
import time


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read(path):
    return json.loads(Path(path).read_text())


def split(parent):
    n = int(digest(['sparc-selector-v1', parent])[:8], 16) % 100
    return 'train' if n < 70 else 'val' if n < 85 else 'test'


def status(root, stage, done=None, total=None, **extra):
    old = read(root/'status.json') if (root/'status.json').exists() else {}
    now = time.time()
    began = old.get('stage_started_unix', now) if old.get('stage') == stage else now
    save(root/'status.json', dict(status='RUNNING', stage=stage, done=done, total=total,
         started_unix=old.get('started_unix', now), stage_started_unix=began,
         updated_unix=now, pid=os.getpid(), **extra))


class RemoteParts:
    """Bounded memory, retryable byte stream across the official split archive."""
    def __init__(self, revision, entries, progress):
        self.revision, self.entries, self.progress = revision, entries, progress
        self.part = self.offset = self.total = 0
        self.response = None

    def read(self, size):
        import requests
        while self.part < len(self.entries):
            entry = self.entries[self.part]
            if self.offset == entry['size']:
                if self.response:
                    self.response.close()
                self.part += 1
                self.offset = 0
                self.response = None
                continue
            for attempt in range(6):
                try:
                    if self.response is None:
                        url = ('https://huggingface.co/datasets/astergiou/LAVIB/resolve/'
                               + self.revision + '/' + entry['path'])
                        headers = {'Range': f'bytes={self.offset}-'} if self.offset else {}
                        self.response = requests.get(url, headers=headers, stream=True, timeout=(30, 90))
                        self.response.raise_for_status()
                        if self.offset and (self.response.status_code != 206 or not
                                self.response.headers.get('Content-Range', '').startswith(f'bytes {self.offset}-')):
                            raise ValueError('server did not honor resume offset')
                    data = self.response.raw.read(min(size, entry['size']-self.offset))
                    if not data:
                        raise IOError('premature archive EOF')
                    self.offset += len(data)
                    self.total += len(data)
                    self.progress(self.total, entry['path'], self.offset)
                    return data
                except Exception:
                    if self.response:
                        self.response.close()
                    self.response = None
                    if attempt == 5:
                        raise
                    time.sleep(min(2**attempt, 15))
        return b''

    def close(self):
        if self.response:
            self.response.close()


def acquire_lavib(root, count):
    """Extract a bounded parent-balanced subset, retaining exact member provenance.

    Archive traversal is sequential; this is explicitly NOT a uniform full-LAVIB
    sample. Per-parent cap and reported coverage prevent silent duplicate inflation.
    On interruption completed member files are retained; archive scanning restarts.
    """
    import requests
    data = root/'data/lavib'
    data.mkdir(parents=True, exist_ok=True)
    manifest = data/'manifest.json'
    if manifest.exists():
        result = read(manifest)
        if len(result) != count or any(sha(r['path']) != r['sha256'] for r in result):
            raise ValueError('LAVIB manifest changed')
        return result
    provenance = data/'upstream.json'
    if not provenance.exists():
        info = requests.get('https://huggingface.co/api/datasets/astergiou/LAVIB', timeout=60)
        info.raise_for_status()
        revision = info.json()['sha']
        tree = requests.get(f'https://huggingface.co/api/datasets/astergiou/LAVIB/tree/{revision}?recursive=true&limit=100', timeout=60)
        tree.raise_for_status()
        entries = sorted([e for e in tree.json() if re.fullmatch('lavib[0-9]{2}', e['path'])],
                         key=lambda e: e['path'])
        if len(entries) != 11:
            raise ValueError('unexpected official archive layout')
        save(provenance, dict(revision=revision, entries=entries, source='https://github.com/alexandrosstergiou/LAVIB',
             license='CC-BY-NC-SA-4.0', sampling='archive-order, at most two temporal crops per parent, one spatial crop',
             limitation='bounded archive prefix; not a uniform sample of the full benchmark'))
        for name in ('train', 'val', 'test', 'yt_ids'):
            response = requests.get(f'https://huggingface.co/datasets/astergiou/LAVIB/resolve/{revision}/annotations/{name}.csv', timeout=60)
            response.raise_for_status()
            (data/(name+'.csv')).write_bytes(response.content)
    upstream = read(provenance)
    official = {}
    for name in ('train', 'val', 'test'):
        for row in csv.DictReader((data/(name+'.csv')).open()):
            key = tuple(row[k] for k in ('name', 'shot', 'tmp_crop', 'vrt_crop', 'hrz_crop'))
            official[key] = name
    records = [read(p) for p in sorted(data.glob('members/*.json'))]
    parents = {}
    for r in records:
        if sha(r['path']) != r['sha256']:
            raise ValueError('saved LAVIB member corrupted')
        parents.setdefault(r['parent'], set()).add(tuple(r['temporal_id']))
    seen = {r['member'] for r in records}
    last = [0.]
    def progress(n, part, offset):
        if time.time()-last[0] >= 5:
            status(root, 'download_lavib', len(records), count, archive_bytes_read=n,
                   archive_part=part, part_offset=offset,
                   resume='retains complete videos; replays archive prefix')
            last[0] = time.time()
    stream = RemoteParts(upstream['revision'], upstream['entries'], progress)
    try:
        archive = tarfile.open(fileobj=stream, mode='r|gz')
        for item in archive:
            if len(records) >= count:
                break
            if not item.isfile() or not item.name.endswith('/vid.mp4') or item.name in seen:
                continue
            name = Path(item.name).parent.name
            match = re.fullmatch(r'(\d+)_(?:shot|hot)(\d+)_(\d+)_(\d+)_(\d+)', name)
            if not match:
                raise ValueError('unrecognized LAVIB parent ID: '+name)
            key = match.groups()
            parent = 'lavib:'+key[0]
            temporal = (key[1], key[2])
            used = parents.get(parent, set())
            if key not in official or len(used) >= 2 or temporal in used:
                continue
            if shutil.disk_usage(root).free < 25*1024**3:
                raise RuntimeError('less than 25 GiB free; acquisition paused without deleting existing data')
            dest = data/'videos'/(name+'.mp4')
            dest.parent.mkdir(exist_ok=True)
            tmp = dest.with_suffix('.partial')
            with tmp.open('wb') as out, archive.extractfile(item) as src:
                shutil.copyfileobj(src, out)
            if tmp.stat().st_size != item.size:
                raise ValueError('truncated tar member')
            os.replace(tmp, dest)
            record = dict(id='lavib_'+name, parent=parent, split=split(parent),
                          dataset='LAVIB', path=str(dest), sha256=sha(dest), member=item.name,
                          official_split=official[key], temporal_id=list(temporal),
                          start_seconds=0., duration_seconds=1.,
                          source_revision=upstream['revision'])
            save(data/'members'/(name+'.json'), record)
            records.append(record)
            seen.add(item.name)
            parents.setdefault(parent, set()).add(temporal)
    finally:
        stream.close()
    if len(records) != count:
        raise ValueError(f'insufficient LAVIB members: {len(records)}/{count}')
    save(manifest, records)
    return records


def acquire_existing(root, base):
    """Long public source windows; old harbor/Kinetics evaluation clips excluded."""
    manifest = root/'data/existing.json'
    if manifest.exists():
        rows = read(manifest)
        if any(sha(r['path']) != r['sha256'] for r in rows):
            raise ValueError('existing source changed')
        return rows
    rows = []
    sources = sorted((base/'data/webvid55/raw').glob('*.mp4'))
    if not sources:
        raise ValueError('WebVid original files unavailable')
    for i, path in enumerate(sources):
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_format',
                                                   '-of', 'json', str(path)]))
        duration = float(probe['format']['duration'])
        h = sha(path)
        # Hash-identical files are one parent regardless of filename.
        parent = 'webvid:'+h
        if any(r['parent'] == parent for r in rows):
            continue
        for j, length in enumerate((2., 4., 8., 12.)):
            length = min(length, duration)
            start = max(0., (duration-length)*j/3)
            rows.append(dict(id=f'webvid_{h[:16]}_{j}', parent=parent, split=split(parent),
                dataset='WebVid-existing', path=str(path), sha256=h,
                start_seconds=start, duration_seconds=length))
        status(root, 'inventory_public_sources', i+1, len(sources))
    save(manifest, rows)
    return rows


def load_frames(row):
    import numpy as np
    if sha(row['path']) != row['sha256']:
        raise ValueError('source identity changed')
    cmd = ['ffmpeg', '-v', 'error', '-threads', '2', '-ss', str(row['start_seconds']),
           '-i', row['path'], '-t', str(row['duration_seconds']), '-an',
           '-vf', 'fps=24,scale=576:320:flags=lanczos', '-threads', '2',
           '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    raw = subprocess.check_output(cmd)
    frames = np.frombuffer(raw, np.uint8).reshape(-1, 320, 576, 3).copy()
    if len(frames) < 3:
        raise ValueError('too few frames')
    return frames


def source_features(frames):
    import cv2
    import numpy as np
    cv2.setNumThreads(2)
    small = np.stack([cv2.resize(f, (128, 72), interpolation=cv2.INTER_AREA) for f in frames])
    x = small.astype(np.float32)/255
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in small])
    features, cuts = [], []
    for i, (f, g) in enumerate(zip(x, gray)):
        edge = cv2.Laplacian(g.astype(np.float32)/255, cv2.CV_32F)
        delta = np.abs(f-x[max(0, i-1)]).mean(-1)
        flow = cv2.calcOpticalFlowFarneback(gray[max(0, i-1)], g, None, .5, 2, 15, 2, 5, 1.1, 0)
        backward = cv2.calcOpticalFlowFarneback(g, gray[max(0, i-1)], None, .5, 2, 15, 2, 5, 1.1, 0)
        yy, xx = np.mgrid[:72, :128].astype(np.float32)
        warped = cv2.remap(backward, xx+flow[..., 0], yy+flow[..., 1], cv2.INTER_LINEAR)
        inconsistency = np.linalg.norm(flow+warped, axis=-1)
        mag = np.linalg.norm(flow, axis=-1)/128
        fvec = [*f.mean((0, 1)), *f.std((0, 1)), float(np.abs(edge).mean()),
                float(edge.std()), *np.quantile(delta, [.5, .9, .99]), float(delta.mean()),
                *np.quantile(mag, [.5, .9, .99]), *flow.mean((0, 1))/128,
                float((inconsistency > 1).mean()), float(inconsistency.mean()/128)]
        features.append(fvec)
        # Exactly the source MAD cut criterion used by the frozen SPARC baseline.
        if i and np.abs(frames[i].astype(np.float32)-frames[i-1]).mean() > 40:
            cuts.append(i)
    return x, np.asarray(features, np.float32), cuts


def state_features(x, features, indices, candidate):
    import numpy as np
    selected = sorted(indices)
    n = len(x)
    a = max((j for j in selected if j < candidate), default=0)
    b = min((j for j in selected if j > candidate), default=n-1)
    t = (candidate-a)/max(1, b-a)
    proxy = np.abs(x[candidate]-((1-t)*x[a]+t*x[b])).mean(-1)
    gaps = np.diff(selected)
    context = features[a:b+1]
    return np.concatenate([features[candidate], features[a], features[b],
        context.mean(0), context.max(0), features.mean(0),
        [candidate/max(1,n-1), (candidate-a)/24, (b-candidate)/24,
         len(selected)/n, np.max(gaps)/24, np.mean(gaps)/24,
         ((len(selected)-1) % 8)/8, (8-(len(selected)-1)%8)%8/8,
         sum(j < candidate for j in selected)%8/8, float(candidate in selected),
         proxy.mean(), *np.quantile(proxy, [.5, .9, .99]), n/24]]).astype(np.float32)


def action_features(x, features, before, after, cuts, keys):
    import numpy as np
    removed, added = sorted(set(before)-set(after)), sorted(set(after)-set(before))
    focus = (added or removed)[0]
    other = (removed or added)[0]
    return np.concatenate([state_features(x, features, before, focus),
        state_features(x, features, after, other),
        [len(added), len(removed), len(cuts)/len(x), len(keys)/len(x),
         min(abs(focus-k) for k in keys)/24]]).astype(np.float32)


def candidates(x, features, keys, cuts, seed, states=6, actions=4):
    """Vary density; include additions, removals and relocations without gap cap."""
    import numpy as np
    from .sparc_selection import adaptive
    rng = random.Random(seed)
    n = len(x)
    required = set(keys) | {i for c in cuts for i in (c-1, c)}
    pools = [sorted(required)]
    for threshold in (3., 6., 12., 24.):
        pools.append(adaptive(x*255, keys, cuts, max_gap=None, threshold=threshold)['indices'])
    optional = sorted(set(range(n))-required)
    pools.append(sorted(required | set(rng.sample(optional, int(.6*len(optional))))))
    pairs = []
    seen = set()
    for state in pools[:states]:
        removable = sorted(set(state)-required)
        unused = sorted(set(range(n))-set(state))
        options = []
        if unused:
            # Mix source proxy proposals and unbiased random proposals.
            scores = [(float(state_features(x, features, state, i)[-5]), i) for i in unused]
            top = [i for _, i in sorted(scores, reverse=True)[:max(1, len(unused)//4)]]
            for i in [rng.choice(top), rng.choice(unused)]:
                options.append(sorted(set(state)|{i}))
        if removable:
            options.append(sorted(set(state)-{rng.choice(removable)}))
        if removable and unused:
            options.append(sorted((set(state)-{rng.choice(removable)})|{rng.choice(unused)}))
        rng.shuffle(options)
        for after in options[:actions]:
            key = (tuple(state), tuple(after))
            if key not in seen and state != after:
                assert required <= set(after)
                seen.add(key)
                pairs.append((state, after))
    return pairs
