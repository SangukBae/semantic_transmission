"""BSU-Predict: the pre-unification DCVC-RT/RIFE path, without routing.

Sender, channel, decoder and interpolation run in separate processes. Only the
received packet and pinned models reach the receiver. Completed stages are
hash-verified on resume; a channel/decoder failure stops this predictive path.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from .artifacts import sha256, write_json
from .packets import accounting
from .webvid5 import fingerprint
from .wire import pack, unpack

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / 'configs/bsu_predict.json'
MODULE = 'semantic_transmission.bsu_predict'
POLICY = 'bsu_predict_v1'
CODE = ('bsu_predict', 'predictive_codec', 'dense_temporal_receiver',
        'metadata_channel', 'packets', 'wire', 'artifacts', 'webvid5',
        'webvid_ablation', 'qwen35_pairs_video', 'etri_60s_check')


def read(path):
    return json.loads(Path(path).read_text())


def profile():
    value = read(CONFIG)
    expected = dict(name='BSU-Predict', policy=POLICY, route='predictive_only',
                    qp=32, temporal_stride=2, snr_db=10, channel_seed=42,
                    adaptive_rate_search=False, generative_fallback=False,
                    semantic_input_required=False)
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError('BSU-Predict fixed profile changed; register a new policy')
    return value


def validate_video(video):
    if (set(video) != {'frames', 'width', 'height', 'fps'} or
            any(type(video[k]) is not int or video[k] < 2 for k in ('frames', 'width', 'height')) or
            video['width'] % 2 or video['height'] % 2 or
            type(video['fps']) not in (int, float) or not math.isfinite(video['fps']) or video['fps'] <= 0):
        raise ValueError('invalid video timeline or dimensions')


def inspect_frames(source, fps):
    from PIL import Image
    files = list(source.glob('*.png'))
    expected = [source / f'{i}.png' for i in range(len(files))]
    if set(files) != set(expected) or len(files) < 2:
        raise ValueError('source frames must be contiguous 0.png, 1.png, ... (at least two)')
    size = None
    hashes = {}
    for path in expected:
        with Image.open(path) as frame:
            if size is not None and frame.size != size:
                raise ValueError('source frame dimensions differ')
            size = frame.size
            frame.verify()
        hashes[path.name] = sha256(path)
    video = dict(frames=len(files), width=size[0], height=size[1], fps=fps)
    validate_video(video)
    return dict(directory=str(source), video=video, hashes=hashes)


def temporal_identity():
    from . import dense_temporal_receiver as temporal
    return dict(policy=temporal.POLICY, stride=2, weights=temporal.WEIGHT_SHA,
                network=temporal.NETWORK_SHA, warp=temporal.WARP_SHA)


def make_packet(video, coded):
    packet = pack(dict(schema=POLICY, video=video, temporal=temporal_identity()), coded)
    unpack_packet(packet)
    return packet


def unpack_packet(packet):
    from .predictive_codec import WEIGHTS
    from .dense_temporal_receiver import validate_indices
    header, coded = unpack(packet)
    if (set(header) != {'schema', 'video', 'temporal'} or header['schema'] != POLICY or
            header['temporal'] != temporal_identity()):
        raise ValueError('unsupported BSU-Predict packet or temporal model')
    video = header['video']; validate_video(video)
    codec, _ = unpack(coded)
    if (codec.get('schema') != 'bsu_dcvc_rt_keys_v1' or codec.get('weights') != WEIGHTS or
            codec.get('upstream') != 'cbdae87a5445114cdc7f48816da63ea80bdeac40' or
            codec.get('color') != 'full_range_ycbcr444_bt709' or
            (codec.get('width'), codec.get('height')) != (video['width'], video['height'])):
        raise ValueError('inconsistent predictive codec identity or dimensions')
    validate_indices(codec['indices'], video['frames'], header['temporal']['stride'])
    return header, coded


def runtime_paths(args):
    settings = read(REPO / '.local/settings.json') if (REPO / '.local/settings.json').exists() else {}
    return dict(codec=str(args.codec_python or REPO / '.local/dcvc_rt_env/bin/python'),
                receiver=str(args.receiver_python or REPO / '.local/bsu_codec_eval_env/bin/python'),
                channel=str(args.channel_python or settings.get('channel_python', '')))


def preflight(args):
    from .predictive_codec import WEIGHTS
    from . import dense_temporal_receiver as temporal
    value = profile(); root = args.output.resolve()
    source = (args.video or args.source_frames).resolve(strict=True)
    if root == source or root in source.parents or source in root.parents:
        raise ValueError('output must be separate from source')
    if not math.isfinite(args.fps) or args.fps <= 0:
        raise ValueError('fps must be positive and finite')
    if args.video:
        if not source.is_file() or min(args.width, args.height) < 2 or args.width % 2 or args.height % 2:
            raise ValueError('video requires a file and positive even normalization dimensions')
        inputs = dict(video=str(source), sha256=sha256(source), fps=args.fps,
                      width=args.width, height=args.height)
    else:
        inputs = inspect_frames(source, args.fps)
    runtimes = runtime_paths(args)
    for name, path in runtimes.items():
        if not path or not Path(path).is_file() or not os.access(path, os.X_OK):
            raise ValueError(f'{name} runtime missing: {path}; supply --{name}-python')
        runtimes[name] = str(Path(path).absolute())  # Preserve venv executable paths.
    binaries = {}
    for name in ('ffmpeg', 'ffprobe'):
        path = shutil.which(name)
        if not path:
            raise ValueError(f'{name} is required')
        binaries[name] = dict(path=path, sha256=sha256(path))
    assets = {REPO / f'.local/checkpoints/dcvc_rt/dcvc_rt_{k}.tar': v for k, v in WEIGHTS.items()}
    assets.update({REPO / '.local/checkpoints/rife/v4.25/flownet.pkl': temporal.WEIGHT_SHA,
                   REPO / '.local/checkpoints/rife/v4.25/IFNet_HDv3.py': temporal.NETWORK_SHA,
                   REPO / '.local/vendor/Practical-RIFE/model/warplayer.py': temporal.WARP_SHA})
    for path, digest in assets.items():
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f'pinned model/implementation missing or changed: {path}')
    vendor = REPO / '.local/vendor/DCVC-RT'
    if not (vendor / 'src/models/video_model.py').is_file():
        raise ValueError(f'DCVC-RT source missing: {vendor}')
    record = dict(profile=value, output=str(root), source=inputs, runtimes=runtimes,
                  assets={str(p): d for p, d in assets.items()}, binaries=binaries,
                  vendor_code={str(p.relative_to(vendor)): sha256(p) for p in sorted(vendor.rglob('*.py'))},
                  config_sha256=sha256(CONFIG),
                  code={n: sha256(REPO / f'src/semantic_transmission/{n}.py') for n in CODE})
    record['signature'] = fingerprint(record)
    prior = root / 'protocol.json'
    if prior.exists() and read(prior) != record:
        raise ValueError('frozen input/code changed; use a new output')
    return record


def prepare(record, root):
    source = record['source']
    if 'video' in source and isinstance(source['video'], str):
        if sha256(source['video']) != source['sha256']:
            raise ValueError('source video changed')
        dest = root / 'prepared'; dest.mkdir()
        subprocess.run([record['binaries']['ffmpeg']['path'], '-v', 'error', '-nostdin',
                        '-i', source['video'], '-map', '0:v:0', '-an',
                        '-vf', f"fps={source['fps']},scale={source['width']}:{source['height']}",
                        '-start_number', '0', str(dest / '%d.png')], check=True)
        source = inspect_frames(dest, source['fps'])
    elif inspect_frames(Path(source['directory']), source['video']['fps']) != source:
        raise ValueError('source frames changed')
    write_json(root / 'source.json', source)


def worker(job, stage):
    root = Path(job['output'])
    if stage == 'sender':
        import numpy as np
        from PIL import Image
        from .predictive_codec import Codec
        from .dense_temporal_receiver import select_indices
        source = job['source']; images = []
        for i in range(source['video']['frames']):
            path = Path(source['directory']) / f'{i}.png'
            if sha256(path) != source['hashes'][path.name]:
                raise ValueError('source frame changed')
            with Image.open(path) as frame:
                images.append(np.asarray(frame.convert('RGB')))
        images = np.stack(images)
        indices = select_indices(images, stride=2)
        codec = Codec(REPO / '.local/vendor/DCVC-RT', REPO / '.local/checkpoints/dcvc_rt')
        coded = codec.encode(images[indices], indices, qp=32)
        data = make_packet(source['video'], coded)
        dest = root / 'sender'; dest.mkdir()
        (dest / 'packet.bin').write_bytes(data)
        write_json(dest / 'sender.json', dict(model='BSU-Predict', route='predictive_only',
            qp=32, indices=indices, intra_frames=1, predictive_frames=len(indices)-1,
            channel_uses=accounting(len(data))['complex_channel_uses'],
            packet_sha256=sha256(dest / 'packet.bin')))
    elif stage == 'channel':
        from .metadata_channel import transmit
        packet = Path(job['packet']).read_bytes(); unpack_packet(packet)
        received, report = transmit(packet, 10, 42, bits_per_symbol=4)
        if received != packet or report['bit_errors']:
            raise ValueError('predictive packet channel integrity failure')
        unpack_packet(received)
        if report['complex_channel_uses'] != accounting(len(packet))['complex_channel_uses']:
            raise ValueError('channel accounting mismatch')
        dest = root / 'received'; dest.mkdir()
        (dest / 'packet.bin').write_bytes(received)
        write_json(dest / 'channel.json', report)
    elif stage == 'decode':
        from PIL import Image
        from .predictive_codec import Codec
        packet = Path(job['packet']); header, coded = unpack_packet(packet.read_bytes())
        codec = Codec(REPO / '.local/vendor/DCVC-RT', REPO / '.local/checkpoints/dcvc_rt')
        indices, frames = codec.decode(coded)
        expected, _ = unpack(coded)
        v = header['video']
        if indices != expected['indices'] or frames.shape != (len(indices), v['height'], v['width'], 3):
            raise ValueError('decoded anchor timeline/shape mismatch')
        dest = root / 'decoded'; dest.mkdir()
        for i, frame in zip(indices, frames):
            Image.fromarray(frame).save(dest / f'{i:05d}.png')
        write_json(root / 'decoded.json', dict(**header, indices=indices,
                   packet_sha256=sha256(packet), encoder_state_reused=False))
    elif stage == 'restore':
        import numpy as np
        from PIL import Image
        from .dense_temporal_receiver import interpolate
        decoded = Path(job['decoded']); info = read(decoded.parent / 'decoded.json')
        anchors = []
        for i in info['indices']:
            with Image.open(decoded / f'{i:05d}.png') as frame:
                anchors.append(np.asarray(frame.convert('RGB')))
        v = info['video']
        frames = interpolate(np.stack(anchors), info['indices'], v['frames'], 2,
                             REPO / '.local/vendor/Practical-RIFE', REPO / '.local/checkpoints/rife/v4.25')
        if frames.shape != (v['frames'], v['height'], v['width'], 3):
            raise ValueError('restored video shape mismatch')
        if not np.array_equal(frames[info['indices']], np.stack(anchors)):
            raise ValueError('RIFE changed a received anchor')
        dest = root / 'reconstruction'; paths = dest / 'frames'; paths.mkdir(parents=True)
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(paths / f'{i:05d}.png')
        movie = dest / 'reconstruction.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-framerate', str(v['fps']),
                        '-i', str(paths / '%05d.png'), '-frames:v', str(v['frames']),
                        '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', str(movie)], check=True)
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
            '-select_streams', 'v:0', '-show_entries', 'stream=width,height,nb_read_frames,avg_frame_rate',
            '-of', 'json', str(movie)]))['streams'][0]
        from fractions import Fraction
        if (int(probe['nb_read_frames']), probe['width'], probe['height']) != (v['frames'], v['width'], v['height']) or not math.isclose(float(Fraction(probe['avg_frame_rate'])), v['fps'], rel_tol=1e-6):
            raise ValueError('delivered MP4 timeline changed')
        write_json(dest / 'receiver.json', dict(source_pixels_at_receiver=False,
            packet_sha256=info['packet_sha256'], video=v, probe=probe,
            received_anchors_preserved_before_mp4=True, generative_fallback=False))
    else:
        raise ValueError(f'unknown predictive stage: {stage}')


def launch(stage, job, log, runtime):
    from .webvid_ablation import command
    env = dict(os.environ, PYTHONPATH=str(REPO / 'src'), PYTHONNOUSERSITE='1',
               PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    env.pop('LD_LIBRARY_PATH', None)
    if stage == 'channel':
        env['CUDA_VISIBLE_DEVICES'] = '-1'
    command(REPO, [runtime, '-m', MODULE, '--worker', stage, '--job', str(job)], log, env)


def execute(record):
    from .etri_60s_check import lock
    from .qwen35_pairs_video import freeze
    from .webvid_ablation import Stages
    root = Path(record['output']); root.mkdir(parents=True, exist_ok=True)
    (REPO / '.local').mkdir(exist_ok=True)
    with lock(REPO / '.local/etri_60s_check.lock'), lock(root / 'execution.lock'):
        freeze(root / 'protocol.json', record)
        stages = Stages(root, record['signature'])
        def progress(stage, done, state='RUNNING'):
            write_json(root / 'status.json', dict(status=state, stage=stage, completed_stages=done,
                       total_stages=5, model='BSU-Predict'))
            print('COMPARE_PROGRESS ' + json.dumps(dict(stage=stage, done=done, total=5,
                current=root.name, state=state,
                overall=f'{done}/5 ({100*done/5:.1f}%) stages complete')), flush=True)
        current, done = 'prepare', 0
        try:
            progress(current, done)
            owned = ['source.json', 'prepared'] if isinstance(record['source']['video'], str) else ['source.json']
            stages.step('prepare', owned, owned, lambda _: prepare(record, root))
            source = read(root / 'source.json')
            jobs = dict(sender=dict(output=str(root), source=source),
                        channel=dict(output=str(root), packet=str(root / 'sender/packet.bin')),
                        decode=dict(output=str(root), packet=str(root / 'received/packet.bin')),
                        restore=dict(output=str(root), decoded=str(root / 'decoded')))
            specs = [('sender', ['sender'], 'codec'), ('channel', ['received'], 'channel'),
                     ('decode', ['decoded', 'decoded.json'], 'codec'),
                     ('restore', ['reconstruction'], 'receiver')]
            for done, (current, products, runtime) in enumerate(specs, 1):
                progress(current, done)
                job = root / f'{current}_job.json'; freeze(job, jobs[current])
                stages.step(current, products, products,
                            lambda log, s=current, j=job, r=runtime: launch(s, j, log, record['runtimes'][r]))
            movie = root / 'reconstruction/reconstruction.mp4'
            write_json(root / 'RESULT.json', dict(status='COMPLETE_BSU_PREDICT', model='BSU-Predict',
                route='predictive_only', movie=str(movie), movie_sha256=sha256(movie),
                channel_uses=read(root / 'received/channel.json')['complex_channel_uses'],
                source_pixels_at_receiver=False, generative_fallback=False,
                protocol_sha256=sha256(root / 'protocol.json'), quality_evaluation_performed=False))
            progress('complete', 5, 'COMPLETE_BSU_PREDICT')
        except BaseException as error:
            # A previous completion record must not survive a failed integrity
            # check as the apparent result of the latest invocation.
            result = root / 'RESULT.json'
            if result.exists():
                archive = root / 'failed_attempts' / f'result_{time.time_ns()}.json'
                archive.parent.mkdir(parents=True, exist_ok=True)
                result.rename(archive)
            progress(current, done, 'INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED')
            raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', nargs='?', default='run', choices=('profile', 'check', 'run', 'status'))
    inputs = p.add_mutually_exclusive_group()
    inputs.add_argument('--video', type=Path, help='normalize the complete video at --fps/--width/--height')
    inputs.add_argument('--source-frames', type=Path, help='normalized sender frames: 0.png, 1.png, ...')
    p.add_argument('--fps', type=float, default=24, help='source PNG FPS / video normalization FPS (default 24)')
    p.add_argument('--width', type=int, default=576, help='video normalization width')
    p.add_argument('--height', type=int, default=320, help='video normalization height')
    p.add_argument('--output', type=Path)
    for runtime in ('codec', 'receiver', 'channel'):
        p.add_argument(f'--{runtime}-python', type=Path)
    p.add_argument('--worker', choices=('sender', 'channel', 'decode', 'restore'), help=argparse.SUPPRESS)
    p.add_argument('--job', type=Path, help=argparse.SUPPRESS)
    return p


def main(argv=None):
    p = parser(); args = p.parse_args(argv)
    if args.worker:
        if not args.job:
            p.error('--job is required for a worker')
        return worker(read(args.job), args.worker)
    if args.action == 'profile':
        print(json.dumps(profile(), indent=2)); return
    if not args.output:
        p.error('--output is required')
    if args.action == 'status':
        path = args.output / 'status.json'
        print(json.dumps(read(path) if path.exists() else dict(status='NOT_STARTED'), indent=2)); return
    if not (args.video or args.source_frames):
        p.error('--video or --source-frames is required')
    record = preflight(args)
    if args.action == 'check':
        print(json.dumps(dict(status='READY_FILES_VERIFIED', profile=record['profile'],
            runtime_imports_and_gpu_inference_verified=False, model_inference_started=False), indent=2)); return
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    execute(record)


if __name__ == '__main__':
    main()
