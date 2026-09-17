"""Actual LGVSC runs and independently labelled (or explicitly unlabelled) pairs."""
import bisect
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

from .artifacts import git_state, sha256
from .metric_campaign import REPO, check_files, digest, file_inventory, read, resolve, save, worker_env
from .pair_inputs import load_pair


def load_array(descriptor):
    path = Path(descriptor['path'])
    if sha256(path) != descriptor['file_sha256']:
        raise ValueError('RGB artifact changed: ' + str(path))
    with np.load(path) as archive:
        frames = archive['frames']
    from .metric_v3_features import pixels
    if pixels(frames) != descriptor['pixel_sha256']:
        raise ValueError('RGB pixel hash changed')
    return frames


def export_source(descriptor, destination):
    """Lossless H.264 input, 8 Hz annotations repeated at 24 Hz, fixed 576x320."""
    frames = load_array(descriptor)
    temporary = destination.with_suffix('.partial.mp4')
    command = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', f'{frames.shape[2]}x{frames.shape[1]}', '-r', '8', '-i', 'pipe:0',
               '-vf', 'scale=576:320:flags=lanczos,fps=24', '-an', '-c:v', 'libx264',
               '-preset', 'fast', '-crf', '0', '-pix_fmt', 'yuv420p', str(temporary)]
    subprocess.run(command, input=frames.tobytes(), check=True)
    temporary.replace(destination)


def validated_attempt(attempt, profile, source):
    from .resume import completed_runs
    return completed_runs([attempt], profile, [source]).get(source['id'])


def run_attempt(attempt, profile, source, config, runner=subprocess.run):
    """Use the existing nine workers while preserving the configured driver path.

    The generic research runner removes LD_LIBRARY_PATH before its NVML probe;
    this workstation requires its locally matched driver libraries. Keep that
    fix local to this campaign and keep the existing completed-run contract.
    """
    from .input_contract import video_config
    local = read(REPO / '.local/settings.json')
    attempt.mkdir(parents=True, exist_ok=False)
    run = attempt / source['id']
    (run / 'logs').mkdir(parents=True)
    save(attempt / 'profile.json', profile)
    save(run / 'run_config.json', dict(video_config(profile, source), input=source['path']))
    record = {'id': source['id'], 'status': 'RUNNING', 'started_unix': time.time(), 'stages': []}
    batch = {'status': 'RUNNING', 'inputs': [source], 'runs': [record], 'code': git_state(REPO),
             'expected_videos': 1, 'completed_videos': 0, 'failed_videos': 0,
             'environment_declaration': str(attempt.parents[4] / 'campaign.json')}
    save(attempt / 'batch_manifest.json', batch)
    stages = [(s, 'semantic_transmission.workers') for s in ('prepare', 'select', 'caption', 'flow')]
    stages += [(s, 'semantic_transmission.codec_transport') for s in ('send', 'channel', 'receive', 'reconstruct')]
    stages += [('evaluate', 'semantic_transmission.research_quality')]
    try:
        for stage, module in stages:
            env = worker_env(config)
            env['PYTHONHASHSEED'] = str(profile['seed'])
            if stage == 'channel':
                env['CUDA_VISIBLE_DEVICES'] = '-1'
            python = local['channel_python'] if stage == 'channel' else local['python']
            command = [python, '-m', module, stage, str(run)]
            entry = {'stage': stage, 'status': 'RUNNING', 'command': command, 'started_unix': time.time()}
            record['stages'].append(entry)
            save(run / 'run_manifest.json', record)
            save(attempt / 'batch_manifest.json', batch)
            print(source['id'], stage, flush=True)
            with (run / 'logs' / (stage + '.log')).open('w') as stream:
                result = runner(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
            entry.update(returncode=result.returncode, seconds=time.time() - entry['started_unix'],
                         status='PASSED' if result.returncode == 0 else 'FAILED')
            if result.returncode:
                raise RuntimeError(f"LGVSC {stage} failed; see {run / 'logs' / (stage + '.log')}")
        record['status'] = batch['status'] = 'PASSED'
        batch['completed_videos'] = 1
    except BaseException as error:
        record['status'] = batch['status'] = 'FAILED'
        record['error'] = str(error)
        batch['failed_videos'] = 1
        raise
    finally:
        record['finished_unix'] = time.time()
        save(run / 'run_manifest.json', record)
        save(attempt / 'batch_manifest.json', batch)


def reconstruct(root, config):
    """Resume whole completed videos; preserve interrupted attempts for inspection."""
    from .video_io import probe
    sources = read(root / '03_natural/sources.json')[:config['reconstruction_sources']]
    destination = root / '03_natural/lgvsc'
    destination.mkdir(parents=True, exist_ok=True)
    pairs, artifacts = [], []
    for item in sources:
        sid = item['source_id'].replace('/', '__')
        input_dir = destination / sid / 'input'
        input_dir.mkdir(parents=True, exist_ok=True)
        video = input_dir / (sid + '.mp4')
        export_receipt = input_dir / 'export.json'
        if export_receipt.exists():
            record = read(export_receipt)
            if record['input'] != item['array']:
                raise ValueError('Export input changed')
            check_files(record['files'])
        else:
            export_source(item['array'], video)
            save(export_receipt, {'input': item['array'], 'files': file_inventory([video])})
        source = {'id': sid, 'path': str(video), 'sha256': sha256(video), **probe(video)}
        artifacts += [video, export_receipt]
        for steps in config['reconstruction_steps']:
            folder = destination / sid / f'steps_{steps}'
            folder.mkdir(exist_ok=True)
            profile = read(resolve(config['reconstruction_profile']))
            profile['steps'] = steps
            profile['models'] = read(REPO / '.local/model_paths.json')
            profile_path = folder / 'profile.json'
            if profile_path.exists() and read(profile_path) != profile:
                raise ValueError('Reconstruction profile changed')
            save(profile_path, profile)
            completed = folder / 'completed.json'
            if completed.exists():
                record = read(completed)
                check_files(record['files'])
                result = validated_attempt(Path(record['attempt']), profile, source)
                if not result:
                    raise ValueError('Completed reconstruction is no longer valid')
            else:
                result = None
                attempts = sorted(folder.glob('attempt_*'))
                for attempt in attempts:
                    result = validated_attempt(attempt, profile, source)
                    if result:
                        break
                if not result:
                    attempt = folder / f'attempt_{len(attempts) + 1:03d}'
                    print('LGVSC', sid, 'steps', steps, 'attempt', attempt.name, flush=True)
                    run_attempt(attempt, profile, source, config)
                    result = validated_attempt(attempt, profile, source)
                    if not result:
                        raise RuntimeError('LGVSC exited without a verified completed video')
                run = result['path']
                # Protect receiver evidence and stage manifests as well as the output video.
                files = [p for p in run.rglob('*') if p.is_file()]
                files += [attempt / 'batch_manifest.json', attempt / 'profile.json']
                save(completed, {'attempt': str(attempt), 'files': file_inventory(files)})
            run = result['path']
            reconstruction = next((run / 'receiver/reconstruction').glob('*.mp4'))
            row = {'case_id': sid + f'__steps_{steps}', 'source_id': item['source_id'],
                   'model': f'LGVSC_steps_{steps}', 'steps': steps,
                   'source': str(video), 'source_sha256': source['sha256'],
                   'reconstruction': str(reconstruction), 'reconstruction_sha256': sha256(reconstruction),
                   'receiver': str(run / 'receiver'), 'profile_sha256': sha256(profile_path),
                   'truth': {}, 'style': 'generated', 'family': 'natural',
                   'truth_provenance': None}
            pairs.append(row)
            artifacts += [profile_path, completed, *map(Path, read(completed)['files'])]
    path = root / '03_natural/real_pairs.json'
    save(path, {'rows': pairs, 'count': len(pairs),
                'scope': 'sampling-step comparison, not a new-model improvement claim; generated labels unavailable by default'})
    return artifacts + [path]


def real_inputs(row):
    a, b, alignment = load_pair(row, width=320, height=192, sample_fps=8.)
    descriptor = read(Path(row['receiver']) / 'decoder_inputs.json')
    times = [i / descriptor['video']['fps'] for i in descriptor['indices']]
    if not times or times != sorted(times) or len(times) != len(set(times)):
        raise ValueError('Invalid received keyframe indices')
    frames = []
    for index in descriptor['indices']:
        path = Path(row['receiver']) / 'frames/sample/key_frames_received' / (str(index) + '.png')
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError('Missing received keyframe: ' + str(path))
        frames.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (320, 192), interpolation=cv2.INTER_AREA))
    rx = np.stack([frames[max(0, bisect.bisect_right(times, t / 8.) - 1)] for t in range(len(a))])
    status = ['not_transmitted'] * len(a)
    for time_s in times:
        index = int(np.ceil(time_s * 8 - 1e-9))
        if index < len(a):
            status[index] = 'ok'
    alignment['rx_scope'] = ('diagnostic sparse keyframe hold; only received slots provide support; '
                             'caption and optical-flow semantic support are not interpreted')
    return a, b, rx, status, alignment


def attach_truth(rows, path):
    """Require exact pair and alignment hashes; never infer truth from candidate scores."""
    if path is None:
        return rows
    declaration = read(path)
    if declaration.get('schema') != 'metric-campaign-independent-truth-v1':
        raise ValueError('Unsupported independent truth schema')
    labels = declaration['rows']
    by_id = {r['case_id']: r for r in labels}
    if len(by_id) != len(labels) or set(by_id) - {r['case_id'] for r in rows}:
        raise ValueError('Duplicate or unknown truth case ID')
    output = []
    from .metric_campaign_cases import PRIMARY
    for row in rows:
        label = by_id.get(row['case_id'])
        if label is None:
            output.append(row)
            continue
        for key in ('source_sha256', 'reconstruction_sha256'):
            if row[key] != label.get(key):
                raise ValueError('Independent truth pair changed: ' + row['case_id'])
        if label.get('alignment_sha256') != digest(row['alignment']):
            raise ValueError('Independent truth timeline changed')
        if not label.get('provenance') or not label.get('evidence_files'):
            raise ValueError('Independent annotation evidence and provenance required')
        check_files(label['evidence_files'])
        for metric, value in label['truth'].items():
            if metric not in PRIMARY or (value is not None and
                    (isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1)):
                raise ValueError('Invalid truth value')
        # UEP labels must describe the same sparse-keyframe support contract.
        if label['truth'].get('uep') is not None and label.get('uep_evidence_scope') != 'received_keyframes_only':
            raise ValueError('UEP labels must specify received_keyframes_only support')
        output.append({**row, 'truth': label['truth'], 'truth_provenance': label['provenance']})
    return output
