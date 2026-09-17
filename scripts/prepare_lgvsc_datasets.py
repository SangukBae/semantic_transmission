#!/usr/bin/env python3
"""Acquire LGVSC data on legend; keep exact paper inputs separate from candidates.

Requires requests and Pillow. Commands are resumable. No deletion is performed.
OpenImages is a seeded replacement training subset, not the unpublished paper split.
"""
import argparse
import concurrent.futures as cf
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import shutil
import subprocess
import tarfile
import threading
import time
from urllib.parse import unquote, urlparse

import requests
from PIL import Image

ROOT = Path('/legend/semantic_transmission/datasets')
REPO = Path(__file__).resolve().parents[1]
LOCAL = threading.local()


def session():
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
    return LOCAL.session


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.part')
    for attempt in range(3):
        try:
            with session().get(url, timeout=(15, 90), stream=True) as r:
                r.raise_for_status()
                with tmp.open('wb') as f:
                    for block in r.iter_content(1024 * 1024):
                        f.write(block)
            tmp.replace(path)
            return
        except (requests.RequestException, OSError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def manifest(root, kind):
    path = root / 'manifests' / f'eval_manifest_{kind}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        shutil.copy2(REPO / 'docs' / path.name, path)
    return list(csv.DictReader(path.open()))


def accept(root, subset, row, path, source):
    actual = sha(path)
    exact = actual == row['sha256']
    dest = root / subset / ('raw' if exact else 'candidates') / row['filename']
    dest.parent.mkdir(parents=True, exist_ok=True)
    if path != dest:
        path.replace(dest)
    result = {**row, 'actual_sha256': actual, 'status': 'exact' if exact else 'hash_mismatch',
              'path': str(dest), 'source': source}
    save(root / subset / 'records' / (row['filename'] + '.json'), result)
    print(subset, result['status'], row['filename'], flush=True)
    return result


def webvid(args):
    root = args.root
    metadata = root / 'webvid55/metadata/webvid_val.csv'
    # Public metadata mirror; videos are fetched from the original publisher URLs.
    url = 'https://huggingface.co/datasets/TempoFunk/webvid-10M/resolve/461f7da9a310b67c7fa05fcbd6e312ddbffc8ba5/data/val/partitions/0000.csv'
    if not metadata.exists():
        download(url, metadata)
    rows = list(csv.DictReader(metadata.open()))
    by_name = {unquote(urlparse(r['contentUrl']).path.rsplit('/', 1)[-1]): r for r in rows}
    save(root / 'webvid55/metadata/provenance.json', {'url': url, 'sha256': sha(metadata)})

    def one(row):
        name = row['filename']
        source = by_name.get(name, {}).get('contentUrl')
        raw = root / 'webvid55/raw' / name
        if raw.exists() and sha(raw) == row['sha256']:
            return accept(root, 'webvid55', row, raw, source or 'bundled_sample')
        try:
            if not source:
                raise ValueError('filename absent from source metadata')
            path = root / 'webvid55/downloads' / name
            download(source, path)
            return accept(root, 'webvid55', row, path, source)
        except Exception as e:
            result = {**row, 'status': 'unavailable', 'source': source, 'error': str(e)}
            save(root / 'webvid55/records' / (name + '.json'), result)
            print('webvid55 unavailable', name, str(e), flush=True)
            return result

    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(one, manifest(root, 'webvid')))
    save(root / 'reports/webvid_acquisition.json', results)


def kinetics(args):
    root = args.root
    wanted = {r['filename']: r for r in manifest(root, 'kinetics')}
    metadata = root / 'kinetics400_14/metadata'
    for name, url in {
        'val.csv': 'https://s3.amazonaws.com/kinetics/400/annotations/val.csv',
        'archives.txt': 'https://s3.amazonaws.com/kinetics/400/val/k400_val_path.txt',
    }.items():
        if not (metadata / name).exists():
            download(url, metadata / name)
    labels = list(csv.DictReader((metadata / 'val.csv').open()))
    label_names = sorted({r['label'] for r in labels})
    by_filename = {f"{r['youtube_id']}_{int(r['time_start']):06d}_{int(r['time_end']):06d}.mp4": r for r in labels}
    for name, row in wanted.items():
        info = by_filename[name]
        assert label_names[int(row['kinetics_label_id'])] == info['label'], (name, info)
    save(metadata / 'selected_annotations.json', {name: by_filename[name] for name in wanted})
    if all((root / 'kinetics400_14/raw' / name).exists()
           and sha(root / 'kinetics400_14/raw' / name) == row['sha256']
           for name, row in wanted.items()):
        print('All 14 Kinetics sources and labels already verified.', flush=True)
        audit(args)
        return

    def archive(url):
        part = url.rsplit('/', 1)[-1]
        record = root / 'kinetics400_14/archive_records' / (part + '.json')
        if record.exists():
            old = json.loads(record.read_text())
            if old.get('complete') and all(Path(r['path']).exists() and sha(Path(r['path'])) == r['actual_sha256'] for r in old.get('found', [])):
                return old
        found = []
        started = time.monotonic()
        try:
            with session().get(url, timeout=(20, 120), stream=True) as response:
                response.raise_for_status()
                with tarfile.open(fileobj=response.raw, mode='r|gz', bufsize=1024 * 1024) as archive_file:
                    for member in archive_file:
                        name = Path(member.name).name
                        if not member.isfile() or name not in wanted:
                            continue
                        tmp = root / 'kinetics400_14/downloads' / (part + '_' + name)
                        tmp.parent.mkdir(parents=True, exist_ok=True)
                        with archive_file.extractfile(member) as src, tmp.open('wb') as dst:
                            shutil.copyfileobj(src, dst)
                        found.append(accept(root, 'kinetics400_14', wanted[name], tmp, url))
            result = {'source': url, 'complete': True, 'found': found, 'elapsed_sec': time.monotonic() - started}
        except Exception as e:
            result = {'source': url, 'complete': False, 'found': found, 'error': str(e)}
        save(record, result)
        print('kinetics archive', part, result['complete'], 'found', len(found), flush=True)
        return result

    urls = (metadata / 'archives.txt').read_text().splitlines()
    if args.archive_number:
        selected_names = {f'part_{number}.tar.gz' for number in args.archive_number}
        urls = [url for url in urls if url.rsplit('/', 1)[-1] in selected_names]
        if len(urls) != len(selected_names):
            raise ValueError('Unknown Kinetics archive number')
    with cf.ThreadPoolExecutor(max_workers=min(args.workers, 8)) as pool:
        results = list(pool.map(archive, urls))
    save(root / 'reports/kinetics_acquisition.json', results)


def openimages(args):
    root = args.root / 'openimages_ntscc'
    metadata = root / 'metadata/train-images-boxable-with-rotation.csv'
    url = 'https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv'
    if not metadata.exists():
        print('Downloading OpenImages metadata', flush=True)
        download(url, metadata)
    selected = root / 'metadata/train_100k_selection.csv'
    if selected.exists():
        rows = list(csv.DictReader(selected.open()))
        assert len(rows) == args.count, 'Existing selection has a different count'
    else:
        rng = random.Random(20260911)
        rows = []
        with metadata.open() as f:
            for index, row in enumerate(csv.DictReader(f)):
                if index < args.count:
                    rows.append(row)
                else:
                    j = rng.randrange(index + 1)
                    if j < args.count:
                        rows[j] = row
        rows.sort(key=lambda r: r['ImageID'])
        with selected.open('w') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    save(root / 'metadata/provenance.json', {
        'source': url, 'source_sha256': sha(metadata), 'selection_sha256': sha(selected),
        'count': len(rows), 'seed': 20260911, 'algorithm': 'reservoir sampling over source CSV order; sorted by ImageID',
        'paper_exact_training_subset': False,
        'reason': 'LGVSC does not publish the exact OpenImages training image IDs.',
        'image_source': 'https://open-images-dataset.s3.amazonaws.com/train/{ImageID}.jpg',
    })

    def one(row):
        image_id = row['ImageID']
        path = root / 'train' / (image_id + '.jpg')
        source = f'https://open-images-dataset.s3.amazonaws.com/train/{image_id}.jpg'
        try:
            if not path.exists():
                download(source, path)
            with Image.open(path) as img:
                width, height = img.size
                img.verify()
            return {'image_id': image_id, 'status': 'ok', 'sha256': sha(path), 'bytes': path.stat().st_size,
                    'width': width, 'height': height, 'source': source, 'error': ''}
        except Exception as e:
            return {'image_id': image_id, 'status': 'failed', 'sha256': '', 'bytes': 0,
                    'width': 0, 'height': 0, 'source': source, 'error': str(e)}

    done = failed = total_bytes = 0
    started = time.monotonic()
    report = args.root / 'reports/openimages_acquisition.json'
    with (root / 'metadata/download_results.csv').open('w') as f, cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        writer = csv.DictWriter(f, fieldnames=['image_id', 'status', 'sha256', 'bytes', 'width', 'height', 'source', 'error'])
        writer.writeheader()
        for result in pool.map(one, rows):
            writer.writerow(result)
            done += 1
            failed += result['status'] != 'ok'
            total_bytes += result['bytes']
            if done % 250 == 0 or done == len(rows):
                f.flush()
                status = {'target': len(rows), 'processed': done, 'downloaded_verified': done - failed,
                          'failed': failed, 'bytes': total_bytes, 'elapsed_sec': time.monotonic() - started,
                          'complete': done == len(rows) and failed == 0, 'paper_exact_training_subset': False}
                save(report, status)
                print('openimages', json.dumps(status), flush=True)


def audit(args):
    report = {}
    for kind, subset in [('webvid', 'webvid55'), ('kinetics', 'kinetics400_14')]:
        rows = []
        for row in manifest(args.root, kind):
            path = args.root / subset / 'raw' / row['filename']
            actual = sha(path) if path.exists() else None
            rows.append({**row, 'actual_sha256': actual, 'status': 'exact' if actual == row['sha256'] else ('missing' if actual is None else 'hash_mismatch')})
        report[subset] = {'target': len(rows), 'exact': sum(r['status'] == 'exact' for r in rows), 'rows': rows}
    save(args.root / 'reports/evaluation_audit.json', report)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'rows'} for k, v in report.items()}, indent=2))


def preprocess(args):
    """Apply the release's MoviePy then FFmpeg operations to verified raw videos."""
    from moviepy.editor import VideoFileClip

    def one(item):
        subset, row = item
        source = args.root / subset / 'raw' / row['filename']
        if not source.exists():
            return {'filename': row['filename'], 'subset': subset, 'status': 'source_missing'}
        if sha(source) != row['sha256']:
            raise ValueError(f'Source hash mismatch: {source}')
        dest = args.root / subset / 'processed' / row['filename']
        record = args.root / subset / 'preprocess_records' / (row['filename'] + '.json')
        if record.exists() and dest.exists():
            previous = json.loads(record.read_text())
            if previous.get('source_sha256') == row['sha256'] and previous.get('sha256') == sha(dest):
                return previous
        temp = args.root / subset / 'preprocessing_temp' / row['filename']
        temp.parent.mkdir(parents=True, exist_ok=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with VideoFileClip(str(source)) as original:
            clip = original.subclip(0, 16) if original.duration > 16 else original
            if clip.fps > 24:
                clip = clip.set_fps(24)
            clip.write_videofile(str(temp), codec='libx264', fps=24, logger=None)
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-y', '-i', str(temp),
                        '-vf', 'crop=iw:ih:((iw-576)/2):((ih-320)/2),scale=576:320',
                        '-c:v', 'libx264', '-crf', '18', '-preset', 'medium', str(dest)], check=True)
        subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-nostdin', '-i', str(dest),
                        '-map', '0:v:0', '-f', 'null', '-'], check=True)
        info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                          '-count_frames', '-show_entries', 'stream=width,height,avg_frame_rate,nb_read_frames,duration',
                          '-of', 'json', str(dest)], text=True))['streams'][0]
        assert (info['width'], info['height'], info['avg_frame_rate']) == (576, 320, '24/1'), info
        assert 0 < float(info['duration']) <= 16 + 1/24, info
        result = {'filename': source.name, 'subset': subset, 'status': 'verified', 'path': str(dest),
                  'source_sha256': row['sha256'], 'sha256': sha(dest), **info}
        save(record, result)
        temp.unlink()
        print('preprocessed', subset, source.name, info['nb_read_frames'], flush=True)
        return result

    items = [(subset, row) for kind, subset in [('webvid', 'webvid55'), ('kinetics', 'kinetics400_14')]
             for row in manifest(args.root, kind)]
    with cf.ThreadPoolExecutor(max_workers=min(args.workers, 2)) as pool:
        results = list(pool.map(one, items))
    for subset in ['webvid55', 'kinetics400_14']:
        with (args.root / subset / 'processed/videos.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=['path'])
            writer.writeheader()
            writer.writerows({'path': r['path']} for r in results if r['subset'] == subset and r['status'] == 'verified')
    save(args.root / 'reports/preprocessing.json', results)


def verify_images(args):
    root = args.root / 'openimages_ntscc'
    selected = list(csv.DictReader((root / 'metadata/train_100k_selection.csv').open()))
    expected_ids = {r['ImageID'] for r in selected}
    if len(expected_ids) != len(selected):
        raise ValueError('Duplicate OpenImages IDs in selection')
    results = {r['image_id']: r for r in csv.DictReader((root / 'metadata/download_results.csv').open())}

    def one(row):
        image_id = row['ImageID']
        try:
            result = results[image_id]
            path = root / 'train' / (image_id + '.jpg')
            assert result['status'] == 'ok' and sha(path) == result['sha256'], 'missing or changed download'
            with Image.open(path) as img:
                img.load()
                assert img.size == (int(result['width']), int(result['height'])), 'dimensions changed'
            return None
        except Exception as e:
            return {'image_id': image_id, 'error': str(e)}

    failures = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, result in enumerate(pool.map(one, selected), 1):
            if result:
                failures.append(result)
            if index % 5000 == 0:
                print('verified image decode and checksum', index, 'failures', len(failures), flush=True)
    unexpected = sorted({p.stem for p in (root / 'train').glob('*.jpg')} - expected_ids)
    report = {'selected': len(selected), 'unique_ids': len(expected_ids),
              'verified': len(selected) - len(failures), 'failures': failures, 'unexpected_ids': unexpected,
              'complete': len(selected) == args.count and not failures and not unexpected,
              'paper_exact_training_subset': False}
    save(args.root / 'reports/openimages_integrity.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['webvid', 'kinetics', 'openimages', 'audit', 'preprocess', 'verify-images'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--count', type=int, default=100000)
    parser.add_argument('--archive-number', type=int, action='append', help='optional Kinetics archive subset')
    args = parser.parse_args()
    if args.workers < 1 or args.count < 1:
        parser.error('workers and count must be positive')
    globals()[args.command.replace('-', '_')](args)
