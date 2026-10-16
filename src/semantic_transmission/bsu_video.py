"""Full raw-video input to the adopted BSU receiver, with verified resume.

Selection and captions are inferred here. Only sender stages of the historical
semantic pipeline are used; generation runs once, after the current cost router.
"""
import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from . import bsu, qwen35_pairs_video as semantic
from .artifacts import sha256, write_json
from .etri_60s_check import lock
from .webvid5 import fingerprint
from .webvid_ablation import Stages

REPO = bsu.REPO
DEFAULT_ROOT = REPO / 'outputs/bsu_etri01_latest_20261016_v1'
SOURCE = REPO / 'data/etri_video_eval/raw/01_person_walk.mp4'
TEMPLATE = REPO / '.local/deployment/bsu_template.json'
SENDER_STAGES = ('input-audit', 'semantic-clips', 'caption', 'flow', 'send')
MODULE = 'semantic_transmission.bsu_video'


def read(path):
    return json.loads(Path(path).read_text())


def event(stage, state='실행 중'):
    print('COMPARE_PROGRESS ' + json.dumps(dict(overall='전체 영상 0/1 (0%)',
        current='ETRI 01_person_walk', stage=stage, state=state), ensure_ascii=False), flush=True)


def preparation_module():
    spec = importlib.util.spec_from_file_location('bsu_source_preparation',
        REPO / 'scripts/prepare_fc_dataset_sources.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preflight(args):
    source, root, template = args.input.resolve(), args.root.resolve(), args.template.resolve()
    semantic.pc.forbid_overlap(root, [source, template, REPO / 'data', REPO / '.local'])
    for name in ('ffmpeg', 'ffprobe'):
        if not shutil.which(name):
            raise FileNotFoundError(name)
    profile = bsu.profile()
    t = read(template)
    runtimes = dict(t['interpreters'], codec=str(REPO / '.local/dcvc_rt_env/bin/python'),
                    controller=str(REPO / '.local/bsu_codec_eval_env/bin/python'))
    for runtime in runtimes.values():
        if not Path(runtime).is_file() or not os.access(runtime, os.X_OK):
            raise FileNotFoundError(runtime)
    if t['receiver']['snr_db'] != 10 or t['receiver']['visual_channel'] != 'source_frame_seeded_numpy_awgn_v1':
        raise ValueError('template must use the adopted 10dB frame-seeded AWGN channel')
    for relative, digest in t['code'].items():
        if sha256(REPO / relative) != digest:
            raise ValueError('template implementation changed: ' + relative)
    semantic.pc.verify_adapter(t['caption']['caption_adapter'])
    for path, entry in t['base_model_files'].items():
        stat = Path(path).stat()
        if (stat.st_size, stat.st_mtime_ns) != (entry['size'], entry['mtime_ns']):
            raise ValueError('Qwen model changed: ' + path)
    for name in ('stdit', 'vae', 't5', 'vae2d'):
        if not Path(t['receiver']['models'][name]).is_dir():
            raise FileNotFoundError(t['receiver']['models'][name])
    from .bsu_dcvc_keys import WEIGHTS
    from .dense_temporal_receiver import WEIGHT_SHA, NETWORK_SHA, WARP_SHA
    assets = {REPO / f'.local/checkpoints/dcvc_rt/dcvc_rt_{k}.tar': d for k, d in WEIGHTS.items()}
    assets.update({REPO / '.local/checkpoints/rife/v4.25/flownet.pkl': WEIGHT_SHA,
        REPO / '.local/checkpoints/rife/v4.25/IFNet_HDv3.py': NETWORK_SHA,
        REPO / '.local/vendor/Practical-RIFE/model/warplayer.py': WARP_SHA})
    for path, digest in assets.items():
        if sha256(path) != digest:
            raise ValueError('pinned predictive model changed: ' + str(path))
    for name in ('ntscc_hyperprior_quality_4_psnr.pth', 'unimatch.pth'):
        if not (REPO / '.local/checkpoints' / name).is_file():
            raise FileNotFoundError(name)
    info = preparation_module().probe(source)
    duration = float(info['streams'][0].get('duration', info['format']['duration']))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('source duration must be positive and finite')
    modules = (*bsu.CODE, 'bsu_video', 'bsu_kinetics_base', 'bsu_webvid_base',
               'bsu_sequence_refinement', 'sparse_trajectory', 'qwen35_pairs_video')
    code = {f'src/semantic_transmission/{n}.py': sha256(REPO / f'src/semantic_transmission/{n}.py')
            for n in modules}
    for path in ('scripts/prepare_fc_dataset_sources.py', 'scripts/run_bsu_video.sh'):
        code[path] = sha256(REPO / path)
    value = dict(policy='raw_video_adopted_bsu_v1', input=str(source), input_sha256=sha256(source),
        root=str(root), template=str(template), template_sha256=sha256(template),
        source_duration_seconds=duration, profile=profile, runtimes=runtimes, code=code,
        normalization=preparation_module().RECIPE, trajectory_extra_rate_ratio=.05,
        trajectory_adapter=None, selection_format_repair='selection_word_count_feedback_v1')
    value['signature'] = fingerprint(value)
    if (root / 'request.json').exists() and read(root / 'request.json') != value:
        raise ValueError('frozen input/model/code changed; choose a new --root')
    return value


def prepare(args, request):
    root = args.root.resolve()
    semantic.freeze(root / 'request.json', request)
    event('원본 전체 길이 정규화·프레임 준비')
    item = dict(id='video', dataset='input', source_id=args.input.stem,
        source_path=request['input'], source_sha256=request['input_sha256'],
        source_duration_seconds=request['source_duration_seconds'], benchmark_memberships=[])
    source = preparation_module().prepare_one(root / 'inputs', item)
    semantic.validate_source_window(source)
    template = copy.deepcopy(read(args.template))
    template['code'].update(request['code'])
    semantic.freeze(root / 'config/template.json', template)
    p = semantic.plan(root / 'inputs/input/video/source_prepared.json', root / 'semantic',
        template_path=root / 'config/template.json')
    semantic.freeze(root / 'semantic/plan.json', p)
    write_json(root / 'status.json', dict(status='PREPARED_INFERENCE_NOT_STARTED', frames=source['frames'],
        seconds=source['frames']/source['fps'], request_signature=request['signature']))
    return p


def launch(command, log, env=None):
    """Stream worker progress; terminate its entire process group on interruption."""
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    environment = dict(os.environ, PYTHONPATH=str(REPO / 'src'), PYTHONUNBUFFERED='1',
                       PYTHONNOUSERSITE='1', **(env or {}))
    with log.open('x') as stream:
        process = subprocess.Popen(list(map(str, command)), cwd=REPO, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            start_new_session=True)
        try:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end='', flush=True)
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
        finally:
            process.stdout.close()
            write_json(log.with_suffix('.resource.json'), dict(command=list(map(str, command)),
                seconds=time.monotonic()-started, returncode=process.returncode))


def sender_jobs():
    jobs = list(semantic.pc.reconstruction_jobs())
    selected = jobs[:len(SENDER_STAGES)]
    if tuple(stage for stage, _, _ in selected) != SENDER_STAGES:
        raise ValueError('semantic sender stage order changed')
    return selected


def semantic_sender(root, p, request):
    work = root / 'semantic'
    # Shared GPU lock ends before the adopted router acquires its own lock.
    with lock(REPO / '.local/etri_60s_check.lock'):
        event('Qwen 키 선택·캡션 추출 (유효한 개별 추론 캐시 재사용)')
        if not (work / 'captions/event_pairs.json').exists():
            launch([request['runtimes']['qwen'], '-m', 'semantic_transmission.bsu_kinetics_base',
                '--root', work, '--worker', 'extract'], work / f'logs/extract_{time.time_ns()}.log')
        protocol = semantic.pc.load_protocol(work)
        semantic.pc.validate_bundle(work, semantic.ARM, protocol)
        dest = work / 'reconstruction/event_pairs'
        cfg, keys = protocol['receiver'], protocol['keys'][semantic.ARM]
        stages = Stages(dest, protocol['signature'])
        stages.step('initialize', ['run'], semantic.pc.initialization_products(p['source']['frames']),
            lambda _: semantic.pc.initialize(dest, p['source'], dict(indices=keys, selector=cfg['selector']), cfg))
        for stage, module, products in sender_jobs():
            event('송신단: ' + stage)
            if module == semantic.pc.MODULE:
                module = 'semantic_transmission.bsu_kinetics_base'
                argv = ['--worker', stage, '--root', work]
            elif module == semantic.pc.fc.ETRI:
                argv = ['--worker', stage, '--run-dir', dest / 'run']
            else:
                argv = [stage, dest / 'run']
            required = ['run/' + product for product in products]
            stages.step(stage, required, required,
                lambda log, cmd=[request['runtimes']['python'], '-m', module, *argv]:
                    launch(cmd, log, {'ETRI_RUN_SIGNATURE': protocol['signature']}))
    return dest / 'run'


def trajectory_sender(run, source, output):
    """Same trajectory policy, without requiring a prior decoded video/audit."""
    import cv2
    import numpy as np
    from PIL import Image
    from . import bsu_sequence_refinement as trajectory
    from .packets import accounting
    from .wire import unpack
    packet = (run / 'transmitter/metadata.bin').read_bytes()
    header, _ = unpack(packet)
    keys = [key['index'] for key in header['keyframes']]
    trajectory.validate_keys(keys, header['video']['frames'])
    base_uses = (run / 'transmitter/visual.c64').stat().st_size // 8 + accounting(
        len(packet), bits_per_symbol=4)['complex_channel_uses']
    budget = math.floor(base_uses * .05)
    cv2.setNumThreads(2)
    cv2.setRNGSeed(2025)
    candidates, started = [], time.monotonic()
    intervals = list(zip(keys, keys[1:])) if budget >= accounting(1, bits_per_symbol=4)['complex_channel_uses'] else []
    for i, (a, b) in enumerate(intervals, 1):
        if b-a+1 > 65535:
            raise ValueError('trajectory interval exceeds wire format')
        frames = []
        for index in range(a, b+1):
            with Image.open(source / f'{index}.png') as im:
                frames.append(cv2.resize(np.asarray(im.convert('RGB')), (288, 160), interpolation=cv2.INTER_AREA))
        candidates.append(trajectory.sender_candidates(np.stack(frames), a, b))
        trajectory.progress('TRAJECTORY sender windows', i, len(keys)-1, started)
    if intervals:
        encoded, audit = trajectory.choose_packet(candidates, header['video'],
            sha256(run / 'transmitter/metadata.bin'), budget, 4)
    else:
        encoded, audit = None, dict(reason='mode does not request trajectories or budget below one LDPC block', sent_windows=0)
    output.mkdir(parents=True, exist_ok=True)
    if encoded is not None:
        trajectory.decode_packet(encoded, packet)
        (output / 'trajectory.bin').write_bytes(encoded)
    write_json(output / 'sender.json', dict(packet_present=encoded is not None, audit=audit,
        base_channel_uses=base_uses, maximum_extra_channel_uses=budget,
        extra_channel_uses=accounting(len(encoded), bits_per_symbol=4)['complex_channel_uses'] if encoded else 0,
        mode='combined', bits_per_symbol=4, budget_reference='measured same-video BSU base transmission'))


def verify_movie(movie, source):
    data = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
        '-select_streams', 'v:0', '-show_streams', '-of', 'json', str(movie)], text=True))['streams'][0]
    from fractions import Fraction
    actual = (int(data['nb_read_frames']), int(data['width']), int(data['height']),
              float(Fraction(data['avg_frame_rate'])))
    expected = tuple(source[k] for k in ('frames', 'width', 'height', 'fps'))
    if actual != expected or abs(float(data['duration']) - source['frames']/source['fps']) > 1/source['fps']:
        raise ValueError(f'delivered full-timeline video differs: {actual} vs {expected}')
    subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-nostdin', '-i', str(movie),
                    '-map', '0:v:0', '-f', 'null', '-'], check=True, stdout=subprocess.DEVNULL)
    return dict(frames=actual[0], width=actual[1], height=actual[2], fps=actual[3],
                seconds=float(data['duration']), decoded_to_eof=True)


def run(args, request, p):
    root = args.root.resolve()
    write_json(root / 'status.json', dict(status='RUNNING', request_signature=request['signature']))
    run_dir = semantic_sender(root, p, request)
    stages = Stages(root, request['signature'])
    event('송신단: 유료 움직임 정보 준비')
    stages.step('trajectory', ['sender'], ['sender'],
        lambda _: trajectory_sender(run_dir, Path(p['source']['source_frames']), root / 'sender'))
    event('최신 BSU: 예측 부호화·경로 선택·채널 전송·수신 복원')
    command = [request['runtimes']['controller'], '-m', 'semantic_transmission.bsu', 'run',
        '--semantic-run', run_dir, '--source-frames', p['source']['source_frames'], '--output', root / 'bsu']
    if (root / 'sender/trajectory.bin').exists():
        command += ['--trajectory', root / 'sender/trajectory.bin']
    launch(command, root / f'logs/adopted_bsu_{time.time_ns()}.log')
    result = read(root / 'bsu/RESULT.json')
    movie = Path(result['movie'])
    if (result['status'] != 'COMPLETE_ADAPTIVE_DENSE_BSU' or sha256(movie) != result['movie_sha256']
            or result['source_pixels_at_receiver'] or not result['receiver_ran_afresh']):
        raise ValueError('adopted receiver completion/provenance mismatch')
    event('완료 영상: 전체 프레임·길이·디코딩 검증')
    audit = verify_movie(movie, p['source'])
    comparison = root / 'source_vs_bsu.mp4'
    def compare(_):
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-y', '-i', p['source']['normalized_video'],
            '-i', str(movie), '-filter_complex', '[0:v][1:v]hstack=inputs=2[v]', '-map', '[v]',
            '-an', '-c:v', 'libx264', '-threads', '2', '-crf', '18', '-preset', 'fast',
            '-pix_fmt', 'yuv420p', str(comparison)], check=True)
    stages.step('comparison', ['source_vs_bsu.mp4'], ['source_vs_bsu.mp4'], compare)
    final = dict(status='COMPLETE_RAW_VIDEO_BSU', request_signature=request['signature'],
        input=request['input'], input_sha256=request['input_sha256'], profile=request['profile'],
        mode=result['mode'], movie=str(movie), movie_sha256=sha256(movie),
        comparison=str(comparison), comparison_sha256=sha256(comparison), full_timeline=audit,
        receiver_result=str(root / 'bsu/RESULT.json'), receiver_result_sha256=sha256(root / 'bsu/RESULT.json'),
        selection_sha256=sha256(root / 'semantic/selection.json'),
        receiver_ran_afresh=True, source_pixels_at_receiver=False, visual_quality_review='NOT_PERFORMED')
    write_json(root / 'RESULT.json', final)
    write_json(root / 'status.json', final)
    print('COMPARE_PROGRESS ' + json.dumps(dict(overall='전체 영상 1/1 (100%)',
        stage='전체 길이 복원·파일 검증 완료', state='완료'), ensure_ascii=False), flush=True)
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


def status(root):
    path = root / 'RESULT.json'
    if path.exists():
        value = read(path)
        for filename, digest in ((value['movie'], value['movie_sha256']),
                (value['comparison'], value['comparison_sha256']),
                (value['receiver_result'], value['receiver_result_sha256'])):
            if sha256(filename) != digest:
                raise ValueError('completed artifact changed: ' + filename)
        value['completed_artifacts_verified'] = True
        return value
    path = root / 'status.json'
    return read(path) if path.exists() else dict(status='NOT_STARTED', root=str(root))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', default='run', choices=('check', 'prepare', 'run', 'status'))
    parser.add_argument('--input', type=Path, default=SOURCE)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--template', type=Path, default=TEMPLATE)
    args = parser.parse_args(argv)
    if args.action == 'status':
        print(json.dumps(status(args.root), ensure_ascii=False, indent=2)); return
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('interrupted')))
    event('입력·모델·고정 설정 확인')
    request = preflight(args)
    if args.action == 'check':
        print(json.dumps(dict(status='READY', input=request['input'], profile=request['profile'],
            seconds=request['source_duration_seconds'], model_inference_started=False), ensure_ascii=False, indent=2))
        return
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with lock(root / 'execution.lock'):
        try:
            p = prepare(args, request)
            if args.action == 'run':
                run(args, request, p)
            else:
                event('원본·설정 준비 완료 (모델 추론 미실행)', '준비 완료')
        except BaseException as exc:
            write_json(root / 'status.json', dict(status='INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'FAILED',
                error=str(exc), resume='rerun the same command; completed artifacts and valid inference calls are verified'))
            raise


def progress_main():
    # An SSH PTY hangup must reach terminal_progress's cleanup handler, even
    # though its ordinary CLI only installs SIGINT/SIGTERM handlers.
    from . import terminal_gpu_progress
    signal.signal(signal.SIGHUP, lambda *_: os.kill(os.getpid(), signal.SIGTERM))
    terminal_gpu_progress.main()


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode)
