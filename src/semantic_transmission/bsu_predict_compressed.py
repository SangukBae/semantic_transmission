"""Opt-in BSU-Predict lossless metadata transport with the original GPU path."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal

from . import bsu_predict as model, bsu_predict_compression as compression
from .artifacts import sha256, write_json
from .packets import accounting

MODULE = 'semantic_transmission.bsu_predict_compressed'
RESTORE_NUMERICS = dict(seed=0, deterministic_algorithms=True, cudnn_benchmark=False,
                        cudnn_deterministic=True, matmul_tf32=False, cudnn_tf32=False,
                        cublas_workspace_config=':4096:8')


def transport(mode):
    if mode not in ('raw', 'auto'):
        raise ValueError('invalid predictive metadata compression mode')
    return dict(policy=compression.POLICY, mode=mode,
                dcvc_stream_unchanged=True, original_packet_restored_before_decode=True,
                restore_numerics=RESTORE_NUMERICS)


def configure_restore_numerics():
    # Fresh, matched controls use the same deterministic RIFE execution. The
    # first unmodified-RIFE replay exhibited repeated-run rounding differences.
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = RESTORE_NUMERICS['cublas_workspace_config']
    import torch
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@contextmanager
def execution_profile(mode):
    """Extend provenance/worker dispatch without editing a frozen base module."""
    old_profile, old_code, old_module = model.profile, model.CODE, model.MODULE
    model.profile = lambda: dict(old_profile(), metadata_transport=transport(mode))
    model.CODE = (*old_code, 'bsu_predict_compression', 'bsu_predict_compressed')
    model.MODULE = MODULE
    try:
        yield
    finally:
        model.profile, model.CODE, model.MODULE = old_profile, old_code, old_module


def worker(job, stage, mode):
    root = Path(job['output'])
    if stage == 'sender':
        model.worker(job, stage)
        dest = root / 'sender'
        original = (dest / 'packet.bin').read_bytes()
        wire = compression.encode(original, mode)
        if compression.decode(wire) != original:
            raise ValueError('predictive sender round-trip failed')
        (dest / 'packet_original.bin').write_bytes(original)
        (dest / 'packet.bin').write_bytes(wire)
        report = model.read(dest / 'sender.json')
        report.update(metadata_transport=transport(mode), method=compression.method_name(wire),
                      original_packet_bytes=len(original), wire_packet_bytes=len(wire),
                      original_channel_uses=accounting(len(original))['complex_channel_uses'],
                      channel_uses=accounting(len(wire))['complex_channel_uses'],
                      original_packet_sha256=sha256(dest / 'packet_original.bin'),
                      packet_sha256=sha256(dest / 'packet.bin'))
        write_json(dest / 'sender.json', report)
    elif stage == 'channel':
        from .metadata_channel import transmit
        packet = Path(job['packet']).read_bytes()
        original = compression.decode(packet)
        received, report = transmit(packet, 10, 42, bits_per_symbol=4)
        if received != packet or report['bit_errors']:
            raise ValueError('predictive packet channel integrity failure')
        restored = compression.decode(received)
        if restored != original:
            raise ValueError('received predictive packet restoration failed')
        if any(report[k] != value for k, value in accounting(len(packet)).items()):
            raise ValueError('compressed wire accounting mismatch')
        dest = root / 'received'; dest.mkdir()
        (dest / 'packet_wire.bin').write_bytes(received)
        (dest / 'packet.bin').write_bytes(restored)
        report.update(metadata_transport=transport(mode), method=compression.method_name(packet),
                      original_packet_restored_exactly=True, original_packet_bytes=len(restored),
                      wire_packet_bytes=len(packet), original_bytes_not_transmitted_separately=True,
                      packet_wire_sha256=sha256(dest / 'packet_wire.bin'),
                      restored_packet_sha256=sha256(dest / 'packet.bin'))
        write_json(dest / 'channel.json', report)
    else:
        # The original DCVC decoder and RIFE receive only the restored packet.
        if stage == 'restore':
            configure_restore_numerics()
        model.worker(job, stage)


def main(argv=None):
    parser = model.parser()
    parser.add_argument('--metadata-compression', choices=('raw', 'auto'), default='auto')
    args = parser.parse_args(argv)
    if args.worker:
        if not args.job:
            parser.error('--job is required for a worker')
        job = model.read(args.job)
        record = model.read(Path(job['output']) / 'protocol.json')
        for name, digest in record['code'].items():
            if sha256(model.REPO / f'src/semantic_transmission/{name}.py') != digest:
                raise ValueError('frozen predictive execution code changed: ' + name)
        spec = record['profile']['metadata_transport']
        if spec != transport(spec['mode']):
            raise ValueError('unsupported predictive transport profile')
        return worker(job, args.worker, spec['mode'])
    with execution_profile(args.metadata_compression):
        if args.action == 'profile':
            print(json.dumps(model.profile(), indent=2)); return
        if not args.output:
            parser.error('--output is required')
        if args.action == 'status':
            path = args.output / 'status.json'
            print(json.dumps(model.read(path) if path.exists() else dict(status='NOT_STARTED'), indent=2)); return
        if not (args.video or args.source_frames):
            parser.error('--video or --source-frames is required')
        record = model.preflight(args)
        if args.action == 'check':
            print(json.dumps(dict(status='READY_FILES_VERIFIED', profile=record['profile'],
                                 model_inference_started=False), indent=2)); return
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
        model.execute(record)
        result = model.read(args.output / 'RESULT.json')
        result.update(metadata_transport=record['profile']['metadata_transport'],
                      sender=model.read(args.output / 'sender/sender.json'),
                      channel=model.read(args.output / 'received/channel.json'))
        write_json(args.output / 'RESULT.json', result)


if __name__ == '__main__':
    main()
