"""NTSCC+ anchor JSCC and received-only RIFE temporal reconstruction.

Reuse the BSU-Gen image codec unchanged. This is NOT a temporally predictive
latent codec: each anchor is independently encoded. Only metadata uses LDPC;
complex visual symbols go directly through AWGN. No new training is required.
"""
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from . import bsu_predict as base
from . import bsu_metadata_compression as compression
from . import dense_temporal_receiver as temporal
from . import ntscc_plus_codec as codec
from .artifacts import sha256, write_json
from .packets import accounting
from .webvid5 import fingerprint
from .wire import pack, unpack

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / 'configs/bsu_predict_jscc.json'
MODULE = 'semantic_transmission.bsu_predict_jscc'
POLICY = 'bsu_predict_ntscc_plus_rife_v2'
COMPLETE = 'COMPLETE_BSU_PREDICT_JSCC'
CODE = ('bsu_predict_jscc', 'bsu_predict', 'ntscc_plus_codec',
        'bsu_metadata_compression', 'dense_temporal_receiver', 'metadata_channel',
        'packets', 'wire', 'artifacts', 'webvid5', 'webvid_ablation',
        'qwen35_pairs_video', 'qwen35_pairs_comparison', 'etri_60s_check',
        'bsu_predict_compressed', 'bsu_predict_compression', 'validation_progress')


def profile():
    expected = dict(name='BSU-Predict-JSCC', policy=POLICY,
        model='NTSCC+ + Practical-RIFE4.25', route='jscc_anchors_rife',
        temporal_stride=2, snr_db=10, channel_seed=42, encoder_seed=2025,
        metadata_compression='auto', visual_ldpc=False, metadata_ldpc=True,
        temporal_latent_prediction=False, generative_fallback=False,
        semantic_input_required=False)
    if base.read(CONFIG) != expected:
        raise ValueError('JSCC fixed profile changed; register a new policy')
    from .bsu_predict_compressed import RESTORE_NUMERICS
    return dict(expected, visual_codec=codec.SPEC,
                checkpoint_sha256=codec.CHECKPOINT_SHA256,
                restore_numerics=RESTORE_NUMERICS)


def parse_packet(packet):
    """Validate every received descriptor before model/GPU allocation."""
    header, rates = unpack(packet)
    fields = {'schema', 'video', 'visual_codec', 'checkpoint_sha256',
              'complex_dtype', 'temporal', 'keyframes'}
    if (set(header) != fields or header['schema'] != POLICY or
            header['visual_codec'] != codec.SPEC or
            header['checkpoint_sha256'] != codec.CHECKPOINT_SHA256 or
            header['complex_dtype'] != 'little_endian_complex64' or
            header['temporal'] != base.temporal_identity()):
        raise ValueError('unsupported JSCC packet or model identity')
    video = header['video']; base.validate_video(video)
    keys = header['keyframes']
    if not isinstance(keys, list) or not keys:
        raise ValueError('missing JSCC anchors')
    fields = {'index', 'complex_offset', 'complex_count', 'average_power',
              'rate_offset', 'rate_bytes', 'rate_count'}
    left, right, top, bottom = codec.padding(video['height'], video['width'])
    expected_rates = 2 * ((video['height'] + top + bottom) // 32) * ((video['width'] + left + right) // 16)
    offset = rate_offset = 0
    for key in keys:
        if (not isinstance(key, dict) or set(key) != fields or
                any(type(key[k]) is not int or key[k] < 0 for k in fields - {'average_power'}) or
                key['complex_offset'] != offset or key['rate_offset'] != rate_offset or
                key['rate_count'] != expected_rates or key['rate_bytes'] != expected_rates or
                type(key['average_power']) not in (int, float) or
                not math.isfinite(key['average_power']) or key['average_power'] <= 0):
            raise ValueError('invalid JSCC anchor descriptor')
        values = codec.unpack_rates(rates[rate_offset:rate_offset + expected_rates], expected_rates)
        # The released checkpoint masks rate-1 entries to zero.
        real_count = sum(0 if codec.RATES[r] == 1 else codec.RATES[r] for r in values)
        if key['complex_count'] != (real_count + 1) // 2 or not real_count:
            raise ValueError('visual symbol count disagrees with rate allocation')
        offset += key['complex_count']; rate_offset += expected_rates
    temporal.validate_indices([k['index'] for k in keys], video['frames'], 2)
    if rate_offset != len(rates):
        raise ValueError('unconsumed JSCC rate payload')
    return header, rates


def read_symbols(path, header):
    import numpy as np
    count = sum(k['complex_count'] for k in header['keyframes'])
    if path.stat().st_size != count * 8:
        raise ValueError('visual payload byte length mismatch')
    symbols = np.fromfile(path, dtype='<c8')
    if not np.isfinite(symbols).all():
        raise ValueError('nonfinite visual payload')
    return symbols


def preflight(args):
    value = profile(); root = args.output.resolve()
    source = (args.video or args.source_frames).resolve(strict=True)
    if root == source or root in source.parents or source in root.parents:
        raise ValueError('output must be separate from source')
    if not math.isfinite(args.fps) or args.fps <= 0:
        raise ValueError('fps must be positive and finite')
    if args.video:
        if not source.is_file() or min(args.width, args.height) < 2 or args.width % 2 or args.height % 2:
            raise ValueError('video requires a file and positive even dimensions')
        inputs = dict(video=str(source), sha256=sha256(source), fps=args.fps,
                      width=args.width, height=args.height)
    else:
        inputs = base.inspect_frames(source, args.fps)
    runtimes = base.runtime_paths(args)
    runtimes['codec'] = str(args.codec_python or REPO / '.local/ntscc_plus_env/bin/python')
    for name, path in runtimes.items():
        if not path or not Path(path).is_file() or not os.access(path, os.X_OK):
            raise ValueError(f'{name} runtime missing: {path}; supply --{name}-python')
        runtimes[name] = str(Path(path).absolute())
    versions = json.loads(subprocess.check_output([runtimes['codec'], '-c',
        'import json,importlib.metadata as m; print(json.dumps('
        f'{{k:m.version(k) for k in {list(codec.VERSIONS)!r}}}))'], text=True,
        env=codec.runtime_environment(runtimes['codec'])))
    if versions != codec.VERSIONS:
        raise ValueError('NTSCC+ runtime differs from validated versions')
    binaries = {}
    for name in ('ffmpeg', 'ffprobe'):
        path = shutil.which(name)
        if not path:
            raise ValueError(f'{name} is required')
        binaries[name] = dict(path=path, sha256=sha256(path))
    assets = {REPO / codec.CHECKPOINT: codec.CHECKPOINT_SHA256,
        REPO / '.local/checkpoints/rife/v4.25/flownet.pkl': temporal.WEIGHT_SHA,
        REPO / '.local/checkpoints/rife/v4.25/IFNet_HDv3.py': temporal.NETWORK_SHA,
        REPO / '.local/vendor/Practical-RIFE/model/warplayer.py': temporal.WARP_SHA}
    for path, digest in assets.items():
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f'pinned model/implementation missing or changed: {path}')
    vendor = REPO / '.local/vendor/NTSCC_plus'
    if not (vendor / 'net/ntscc.py').is_file():
        raise ValueError('NTSCC+ source missing')
    record = dict(profile=value, output=str(root), source=inputs, runtimes=runtimes,
        runtime_versions=versions, assets={str(p): d for p, d in assets.items()}, binaries=binaries,
        vendor_code={str(p.relative_to(vendor)): sha256(p) for p in sorted(vendor.rglob('*.py'))},
        config_sha256=sha256(CONFIG),
        code={n: sha256(REPO / f'src/semantic_transmission/{n}.py') for n in CODE})
    record['signature'] = fingerprint(record)
    if (root / 'protocol.json').exists() and base.read(root / 'protocol.json') != record:
        raise ValueError('frozen input/code changed; use a new output')
    return record


def select_source_indices(source):
    import numpy as np
    from PIL import Image
    video = source['video']
    # Streaming cut detection agrees with DCVC's select_indices without keeping
    # all full-resolution source frames in memory.
    selected = set(range(0, video['frames'], 2)) | {video['frames'] - 1}
    previous = None
    for i in range(video['frames']):
        path = Path(source['directory']) / f'{i}.png'
        if sha256(path) != source['hashes'][path.name]:
            raise ValueError('source frame changed')
        with Image.open(path) as image:
            current = np.asarray(image.convert('RGB'), dtype=np.float32)
        if previous is not None and np.abs(current - previous).mean() > 40:
            selected.update((i - 1, i))
        previous = current
    return sorted(selected)


def sender(job):
    import numpy as np
    import torch
    from PIL import Image
    from .validation_progress import emit
    root = Path(job['output']); source = job['source']; video = source['video']
    indices = select_source_indices(source)
    torch.manual_seed(2025)
    model, _ = codec.load_codec(REPO)
    header = dict(schema=POLICY, video=video, visual_codec=codec.SPEC,
        checkpoint_sha256=codec.CHECKPOINT_SHA256, complex_dtype='little_endian_complex64',
        temporal=base.temporal_identity(), keyframes=[])
    dest = root / 'sender'; dest.mkdir()
    payload = bytearray(); offset = 0
    with (dest / 'visual.c64').open('xb') as stream, torch.inference_mode():
        for number, index in enumerate(indices, 1):
            path = Path(source['directory']) / f'{index}.png'
            if sha256(path) != source['hashes'][path.name]:
                raise ValueError('source anchor changed')
            with Image.open(path) as image:
                tensor = torch.from_numpy(np.asarray(image.convert('RGB')).copy()).permute(2, 0, 1)[None].cuda().float() / 255
            values, rates, power = codec.encode_symbols(model, tensor)
            raw = codec.pack_rates(rates.cpu().tolist())
            header['keyframes'].append(dict(index=index, complex_offset=offset,
                complex_count=values.numel(), average_power=power, rate_offset=len(payload),
                rate_bytes=len(raw), rate_count=rates.numel()))
            payload.extend(raw); offset += values.numel()
            stream.write(values.cpu().numpy().astype('<c8').tobytes())
            emit(number, len(indices))
    packet = pack(header, payload); parse_packet(packet)
    wire = compression.encode(packet, 'auto')
    if compression.decode(wire) != packet:
        raise ValueError('metadata compression round-trip failed')
    (dest / 'metadata_original.bin').write_bytes(packet)
    (dest / 'metadata_wire.bin').write_bytes(wire)
    write_json(dest / 'sender.json', dict(model='BSU-Predict-JSCC', indices=indices,
        independently_coded_anchors=len(indices), temporal_latent_prediction=False,
        visual_complex_channel_uses=offset, visual_serialization_bytes=offset * 8,
        metadata_original_bytes=len(packet), metadata_wire_bytes=len(wire),
        metadata_method=compression.method_name(wire),
        digital_complex_channel_uses=accounting(len(wire))['complex_channel_uses'],
        channel_uses=offset + accounting(len(wire))['complex_channel_uses'],
        visual_ldpc=False, original_metadata_not_transmitted_separately=True))


def channel(job):
    from .metadata_channel import transmit
    from .qwen35_pairs_comparison import frame_noise
    root = Path(job['output']); source = Path(job['sender'])
    wire = (source / 'metadata_wire.bin').read_bytes()
    packet = compression.decode(wire); header, _ = parse_packet(packet)
    sent = read_symbols(source / 'visual.c64', header)
    received, report = transmit(wire, 10, 42, bits_per_symbol=4)
    if received != wire or report['bit_errors']:
        raise ValueError('JSCC metadata channel integrity failure')
    if any(report[k] != v for k, v in accounting(len(wire)).items()):
        raise ValueError('metadata channel accounting mismatch')
    restored = compression.decode(received); parse_packet(restored)
    dest = root / 'received'; dest.mkdir()
    (dest / 'metadata_wire.bin').write_bytes(received)
    (dest / 'metadata.bin').write_bytes(restored)
    with (dest / 'visual.c64').open('xb') as stream:
        for key in header['keyframes']:
            start, count = key['complex_offset'], key['complex_count']
            values = sent[start:start + count] + frame_noise(key['index'], count, 42, 10)
            values.astype('<c8').tofile(stream)
    digital = report['complex_channel_uses']
    write_json(dest / 'channel.json', dict(status='PASSED', metadata=report,
        visual_complex_channel_uses=len(sent), digital_complex_channel_uses=digital,
        complex_channel_uses=len(sent) + digital, visual_ldpc=False,
        visual_channel='source_frame_seeded_numpy_awgn_v1', snr_db=10, channel_seed=42,
        visual_tx_mean_power=float((abs(sent) ** 2).mean()),
        metadata_exact_match=restored == packet, physical_link_overhead_included=False,
        complex64_file_bytes_are_not_radio_bits=True))


def decode(job):
    import torch
    from PIL import Image
    from .validation_progress import emit
    root = Path(job['output']); source = Path(job['received'])
    packet = source / 'metadata.bin'; header, payload = parse_packet(packet.read_bytes())
    values = read_symbols(source / 'visual.c64', header)
    model, _ = codec.load_codec(REPO)
    dest = root / 'decoded'; dest.mkdir(); video = header['video']
    with torch.inference_mode():
        for number, key in enumerate(header['keyframes'], 1):
            start, count = key['complex_offset'], key['complex_count']
            rate_start = key['rate_offset']
            rates = codec.unpack_rates(payload[rate_start:rate_start + key['rate_bytes']], key['rate_count'])
            frame = codec.decode_symbols(model, torch.from_numpy(values[start:start + count].copy()).cuda(),
                torch.tensor(rates, device='cuda'), key['average_power'], video['height'], video['width'])
            pixels = (frame[0].permute(1, 2, 0) * 255).round().byte().cpu().numpy()
            Image.fromarray(pixels).save(dest / f"{key['index']:05d}.png")
            emit(number, len(header['keyframes']))
    write_json(root / 'decoded.json', dict(schema=POLICY, video=video,
        indices=[k['index'] for k in header['keyframes']], packet_sha256=sha256(packet),
        source_pixels_at_receiver=False, encoder_state_reused=False))


def worker(job, stage):
    if stage == 'restore':
        from .bsu_predict_compressed import configure_restore_numerics
        configure_restore_numerics()
        return base.worker(job, stage)
    return {'sender': sender, 'channel': channel, 'decode': decode}[stage](job)


def launch(stage, job, log, runtime):
    from .webvid_ablation import command
    env = dict(os.environ, PYTHONPATH=str(REPO / 'src'), PYTHONNOUSERSITE='1',
        PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    env.pop('LD_LIBRARY_PATH', None)
    if stage in ('sender', 'decode'):
        env = codec.runtime_environment(runtime, env)
    elif stage == 'channel':
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
        stages = Stages(root, record['signature']); started = time.time()
        def progress(stage, done, state='RUNNING'):
            write_json(root / 'status.json', dict(status=state, stage=stage, completed_stages=done,
                total_stages=5, model='BSU-Predict-JSCC', elapsed_seconds=time.time() - started))
            print('COMPARE_PROGRESS ' + json.dumps(dict(stage=stage, done=done, total=5,
                current=root.name, state=state,
                overall=f'{done}/5 ({100*done/5:.1f}%) stages complete')), flush=True)
        current, done = 'prepare', 0
        try:
            progress(current, done)
            owned = ['source.json', 'prepared'] if isinstance(record['source']['video'], str) else ['source.json']
            stages.step('prepare', owned, owned, lambda _: base.prepare(record, root))
            jobs = dict(sender=dict(output=str(root), source=base.read(root / 'source.json')),
                channel=dict(output=str(root), sender=str(root / 'sender')),
                decode=dict(output=str(root), received=str(root / 'received')),
                restore=dict(output=str(root), decoded=str(root / 'decoded')))
            specs = [('sender', ['sender'], 'codec'), ('channel', ['received'], 'channel'),
                ('decode', ['decoded', 'decoded.json'], 'codec'), ('restore', ['reconstruction'], 'receiver')]
            for done, (current, products, runtime) in enumerate(specs, 1):
                progress(current, done)
                job = root / f'{current}_job.json'; freeze(job, jobs[current])
                stages.step(current, products, products,
                    lambda log, s=current, j=job, r=runtime: launch(s, j, log, record['runtimes'][r]))
            movie = root / 'reconstruction/reconstruction.mp4'
            write_json(root / 'RESULT.json', dict(status=COMPLETE, model='BSU-Predict-JSCC',
                route='jscc_anchors_rife', movie=str(movie), movie_sha256=sha256(movie),
                channel_uses=base.read(root / 'received/channel.json')['complex_channel_uses'],
                source_pixels_at_receiver=False, generative_fallback=False,
                temporal_latent_prediction=False, new_training_performed=False,
                protocol_sha256=sha256(root / 'protocol.json'), quality_evaluation_performed=False))
            progress('complete', 5, COMPLETE)
        except BaseException as error:
            result = root / 'RESULT.json'
            if result.exists():
                archive = root / 'failed_attempts' / f'result_{time.time_ns()}.json'
                archive.parent.mkdir(parents=True, exist_ok=True); result.rename(archive)
            progress(current, done, 'INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED')
            raise


def main(argv=None):
    parser = base.parser(); parser.description = __doc__; args = parser.parse_args(argv)
    if args.worker:
        if not args.job:
            parser.error('--job is required')
        job = base.read(args.job); record = base.read(Path(job['output']) / 'protocol.json')
        if record['profile'] != profile() or record['config_sha256'] != sha256(CONFIG):
            raise ValueError('frozen JSCC profile changed')
        for name, digest in record['code'].items():
            if sha256(REPO / f'src/semantic_transmission/{name}.py') != digest:
                raise ValueError('frozen JSCC code changed: ' + name)
        for name, digest in record['vendor_code'].items():
            if sha256(REPO / '.local/vendor/NTSCC_plus' / name) != digest:
                raise ValueError('frozen NTSCC+ vendor code changed: ' + name)
        return worker(job, args.worker)
    if args.action == 'profile':
        print(json.dumps(profile(), indent=2)); return
    if not args.output:
        parser.error('--output is required')
    if args.action == 'status':
        path = args.output / 'status.json'
        print(json.dumps(base.read(path) if path.exists() else dict(status='NOT_STARTED'), indent=2)); return
    if not (args.video or args.source_frames):
        parser.error('--video or --source-frames is required')
    record = preflight(args)
    if args.action == 'check':
        print(json.dumps(dict(status='READY_FILES_VERIFIED', profile=record['profile'],
            runtime_imports_and_gpu_inference_verified=False, model_inference_started=False), indent=2)); return
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    execute(record)


if __name__ == '__main__':
    main()
