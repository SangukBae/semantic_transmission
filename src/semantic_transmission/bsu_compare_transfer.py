"""Read-only remote inventory and resumable, manifest-limited artifact transfer.

This file uses only the standard library so its inventory function can execute
over SSH without installing code or touching the remote experiment directory.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import tempfile
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def safe_relative(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or '..' in path.parts or '\\' in name
            or any(ord(c) < 32 for c in name) or str(path) != name):
        raise ValueError(f'unsafe artifact path: {name!r}')
    return path


def inventory(repo, benchmark, ids, snrs):
    """Select completed conditions only; use their original artifact receipts."""
    repo = Path(repo).resolve(strict=True)
    root = repo / safe_relative(benchmark)
    files = {}

    def add(path, expected=None):
        path = Path(path)
        name = path.relative_to(repo).as_posix()
        safe_relative(name)
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(repo):
            raise ValueError(f'nonlocal or symlink artifact: {name}')
        value = dict(sha256=expected or digest(path), size=path.stat().st_size)
        if not re.fullmatch('[a-f0-9]{64}', value['sha256']):
            raise ValueError('invalid artifact digest')
        if name in files and files[name] != value:
            raise ValueError(f'conflicting original receipts: {name}')
        files[name] = value
        return name

    def read(path, expected=None):
        add(path, expected)
        if expected and digest(path) != expected:
            raise ValueError(f'metadata receipt mismatch: {path}')
        return json.loads(Path(path).read_text())

    plan = read(root/'protocol.json')
    if plan['mode'] != 'combined':
        raise ValueError('reference benchmark must use combined refinement')
    sources, cases = [], []
    if len(ids) != len(set(ids)) or len(snrs) != len(set(snrs)) or not ids or not snrs:
        raise ValueError('unique nonempty IDs and SNRs required')
    for video_id in ids:
        matches = [v for v in plan['videos'] if v['id'] == video_id]
        if len(matches) != 1 or not re.fullmatch('[A-Za-z0-9_-]+', video_id):
            raise ValueError(f'unknown/unsafe video: {video_id}')
        v = matches[0]
        src = read(v['prepared'], v['prepared_sha256'])
        n = src['frames']
        if ((src['width'], src['height'], src['fps']) != (576, 320, 24)
                or n != v['frames'] or src['source_sha256'] != v['source_sha256']):
            raise ValueError('source identity or full timeline mismatch')
        source_frames = [add(Path(src['source_frames'])/f'{i}.png', src['source_frame_hashes'][str(i)])
                         for i in range(n)]
        source_movie = add(src['normalized_video'], src['normalized_sha256'])
        sources.append(dict(id=video_id, frames=n, fps=24, width=576, height=320,
                            source_sha256=src['source_sha256'], frames_paths=source_frames,
                            movie=source_movie))
        prior_keys = prior_captions = None
        for snr in snrs:
            job = root/f'snr_{snr:g}'/video_id
            result = read(job/'RESULT.json')
            if (not result['status'].startswith('COMPLETE_RECONSTRUCTION_METRICS')
                    or result['signature'] != plan['signature'] or result['mode'] != 'combined'
                    or result['id'] != video_id or result['snr_db'] != snr
                    or result['frames'] != n or result['source_sha256'] != src['source_sha256']):
                raise ValueError(f'incomplete or mismatched condition: {video_id} SNR {snr}')
            base = Path(result['base'])
            if base != job/'base':
                raise ValueError('base path does not belong to this condition')
            run = base/'reconstruction/event_pairs/run'
            receipts = {}
            for stage in ('reconstruct', 'receive', 'channel', 'evaluate', 'output-audit'):
                receipt = read(base/f'reconstruction/event_pairs/stages/{stage}.json')
                if receipt['status'] != 'PASSED':
                    raise ValueError('base stage did not pass')
                for name, value in receipt['artifacts'].items():
                    safe_relative(name)
                    path = str(base/'reconstruction/event_pairs'/name)
                    if path in receipts and receipts[path] != value:
                        raise ValueError('conflicting base receipts')
                    receipts[path] = value

            def base_add(path):
                return add(path, receipts[str(path)])

            selection = read(base/'selection.json')
            read(base/'protocol.json')
            captions = read(base/'captions/event_pairs.json')['captions']
            indices = selection['keys']
            if indices != sorted(set(indices)) or indices[0] != 0 or indices[-1] != n-1:
                raise ValueError('keys do not cover the complete source')
            texts = [c['caption'] for c in captions]
            if prior_keys is not None and (indices != prior_keys or texts != prior_captions):
                raise ValueError('SNR conditions changed selection/captions')
            prior_keys, prior_captions = indices, texts
            movie = Path(result['base_video'])
            frames = movie.with_name(movie.stem+'_frames')
            raw = [base_add(frames/f'{i:05d}.png') for i in range(n)]
            base_movie = base_add(movie)
            base_packet = base_add(run/'received/metadata.bin')
            key_paths = [base_add(run/f'receiver/frames/sample/key_frames_received/{i}.png') for i in indices]
            channel = read(run/'channel_accounting.json', receipts[str(run/'channel_accounting.json')])
            if not channel['metadata_exact_match'] or channel['bit_errors'] != 0:
                raise ValueError('base metadata was not received exactly')
            quality = read(run/'quality.json', receipts[str(run/'quality.json')])
            if quality['video_sha256'] != files[base_movie]['sha256']:
                raise ValueError('base video/quality identity mismatch')
            artifacts = result['artifacts']

            def refined_add(name):
                safe_relative(name)
                return add(job/name, artifacts[name])

            current = [refined_add(f'refined/frames/{i:05d}.png') for i in range(n)]
            current_movie = refined_add('refined/reconstruction.mp4')
            refined_add('refined/quality.json')
            if files[current_movie]['sha256'] != result['refined_quality']['video_sha256']:
                raise ValueError('current video/quality identity mismatch')
            trajectory = None
            if result['trajectory_audit']['packet_present']:
                trajectory = refined_add('received/trajectory.bin')
                ch = read(job/'received/channel.json', artifacts['received/channel.json'])
                if not ch['exact_match'] or ch['bit_errors'] != 0:
                    raise ValueError('trajectory not received exactly')
            if result['total_channel_uses'] != result['base_channel_uses']+result['trajectory_channel_uses']:
                raise ValueError('inconsistent channel accounting')
            for field in ('base_quality', 'refined_quality'):
                for boundary in ('lossless_frames', 'delivered_mp4'):
                    if result[field][boundary]['frames'] != n:
                        raise ValueError('partial quality result')
            cases.append(dict(id=video_id, snr=snr, frames=n, keys=indices, raw_frames=raw,
                              base_movie=base_movie, base_packet=base_packet, key_paths=key_paths,
                              trajectory=trajectory, current_frames=current, current_movie=current_movie,
                              result_path=(job/'RESULT.json').relative_to(repo).as_posix(),
                              base_channel_uses=result['base_channel_uses'],
                              trajectory_channel_uses=result['trajectory_channel_uses']))
    return dict(version=1, remote_repo=str(repo), benchmark=benchmark, ids=ids, snrs=snrs,
                sources=sources, cases=cases, files=files, benchmark_signature=plan['signature'],
                remote_writes=False, includes_models=False, includes_visual_bitstreams=False)


def validate_manifest(manifest):
    if fingerprint({k:v for k,v in manifest.items() if k != 'signature'}) != manifest['signature']:
        raise ValueError('transfer manifest changed')
    for name, item in manifest['files'].items():
        safe_relative(name)
        if (not re.fullmatch('[a-f0-9]{64}', item['sha256'])
                or type(item['size']) is not int or item['size'] < 0):
            raise ValueError('invalid transfer file identity')
    return manifest


def local_file(root, name):
    safe_relative(name)
    path = root/name
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('unsafe local mirror path')
    return path


def ssh_args(host):
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*', host):
        raise ValueError('invalid SSH target; use user@host')
    cache = Path.home()/'.cache/bsu-compare-ssh'
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    if cache.is_symlink() or cache.stat().st_uid != os.getuid():
        raise ValueError('unsafe SSH control directory')
    cache.chmod(0o700)
    return ['ssh', '-S', str(cache/'%C'), '-o', 'ControlMaster=auto',
            '-o', 'ControlPersist=600', '-o', 'ConnectTimeout=15']


def authenticate(host):
    # Foreground inherited TTY: passwords are never arguments, files or logs.
    subprocess.run([*ssh_args(host), host, 'true'], check=True)


def remote_inventory(host, repo, benchmark, ids, snrs):
    source = Path(__file__).read_text()
    script = source+'\nprint(json.dumps(inventory(**'+repr(dict(repo=repo, benchmark=benchmark,
                                                               ids=ids, snrs=snrs))+')))\n'
    result = subprocess.run([*ssh_args(host), '-o', 'BatchMode=yes', host, 'python3 -'],
                            input=script, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError('remote inventory failed: '+result.stderr[-4000:])
    manifest = json.loads(result.stdout)
    manifest['host'] = host
    manifest['signature'] = fingerprint(manifest)
    return validate_manifest(manifest)


def inspect_files(manifest, mirror, progress=None):
    missing = []
    for i, (name, identity) in enumerate(manifest['files'].items(), 1):
        path = local_file(mirror, name)
        if (not path.is_file() or path.stat().st_size != identity['size']
                or digest(path) != identity['sha256']):
            missing.append(name)
        if progress and (i == 1 or i % 100 == 0 or i == len(manifest['files'])):
            progress(i, len(manifest['files']))
    return missing


def fetch(manifest, mirror, progress=None, offline=False):
    validate_manifest(manifest)
    mirror.mkdir(parents=True, exist_ok=True)
    missing = inspect_files(manifest, mirror, progress)
    if not missing:
        return dict(reused=True, files=len(manifest['files']), transferred_files=0)
    if offline:
        raise ValueError('offline mirror has missing or corrupt files: '+str(missing[:4]))
    needed = sum(manifest['files'][n]['size'] for n in missing)
    if shutil.disk_usage(mirror).free < needed+256*1024**2:
        raise RuntimeError(f'insufficient disk space for {needed} transfer bytes')
    with tempfile.NamedTemporaryFile(dir=mirror.parent, prefix='transfer-', suffix='.list') as listing:
        listing.write(b'\0'.join(n.encode() for n in missing)+b'\0'); listing.flush()
        transport = shlex.join([*ssh_args(manifest['host']), '-o', 'BatchMode=yes'])
        command = ['rsync', '-rlt', '--protect-args', '--ignore-times', '--partial',
                   '--partial-dir=.rsync-partial', '--from0', '--files-from='+listing.name,
                   '--info=progress2', '--outbuf=L', '-e', transport,
                   manifest['host']+':'+manifest['remote_repo'].rstrip('/')+'/', str(mirror)+'/']
        print('COMPARE_PROGRESS '+json.dumps(dict(stage='파일 전송 준비 (진행률 미확정)')), flush=True)
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as process:
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    match = re.match(r'\s*([\d,]+)\s+(\d+)%\s+(\S+)', line)
                    if match:
                        copied = min(needed, int(match[1].replace(',', '')))
                        print('COMPARE_PROGRESS '+json.dumps(dict(stage=
                            f'파일 전송: {copied}/{needed} bytes ({100*copied/max(1,needed):.1f}%) | {match[3]}')),
                              flush=True)
                if process.wait():
                    raise subprocess.CalledProcessError(process.returncode, command)
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
                raise
    bad = inspect_files(manifest, mirror, progress)
    if bad:
        raise ValueError('transferred artifacts differ from original receipts: '+str(bad[:4]))
    return dict(reused=False, files=len(manifest['files']), transferred_files=len(missing),
                transferred_bytes=needed)
