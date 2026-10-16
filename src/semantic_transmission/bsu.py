"""Adopted BSU entry point: fixed DCVC-RT/RIFE profile, cost-routed fallback.

Consumes an existing BSU semantic transmission and its normalized sender PNGs.
Historical dataset runners retain their frozen combined_context protocols.
"""
import argparse
import json
import os
from pathlib import Path

from .artifacts import sha256, write_json
from .webvid5 import fingerprint

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / 'configs/bsu_default.json'
CODE = ('bsu', 'bsu_adaptive_dense', 'bsu_dense_router', 'bsu_dense_neural',
        'bsu_dense_temporal', 'dense_temporal_receiver', 'bsu_dcvc_keys',
        'bsu_key_reduction', 'bsu_dcvc_validation', 'bsu_dense_validation',
        'dense_video_transport', 'lossless_transport', 'codec_transport',
        'bsu_receiver', 'receiver_key_context', 'metadata_channel', 'packets', 'wire')


def profile():
    value = json.loads(CONFIG.read_text())
    expected = dict(qp=32, temporal_stride=2, min_original_keys=5,
                    protect_semantic_keys=False, snr_db=10, adaptive_rate_search=False)
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError('adopted profile changed; use the experimental runner for other settings')
    return value


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', nargs='?', default='run', choices=('profile', 'check', 'run', 'status'))
    p.add_argument('--semantic-run', type=Path, help='existing event_pairs/run containing transmitter and run_config.json')
    p.add_argument('--source-frames', type=Path, help='normalized sender RGB PNGs: 0.png, 1.png, ...')
    p.add_argument('--trajectory', type=Path, help='optional explicit original trajectory packet')
    p.add_argument('--output', type=Path, help='separate resumable output directory')
    return p


def preflight(args):
    from .wire import unpack
    from .bsu_dcvc_keys import WEIGHTS
    from .dense_temporal_receiver import WEIGHT_SHA, NETWORK_SHA, WARP_SHA
    value = profile()
    old, source, root = (p.resolve() for p in (args.semantic_run, args.source_frames, args.output))
    if any(root == p or root in p.parents or p in root.parents for p in (old, source)):
        raise ValueError('output must be separate from semantic input and source frames')
    cfg = json.loads((old / 'run_config.json').read_text())
    if cfg['snr_db'] != value['snr_db'] or cfg['visual_channel'] != 'source_frame_seeded_numpy_awgn_v1':
        raise ValueError('default BSU requires the validated 10dB frame-seeded AWGN input')
    h, _ = unpack((old / 'transmitter/metadata.bin').read_bytes())
    if h['video']['frames'] < 2:
        raise ValueError('at least two source frames required')
    from PIL import Image
    for i in range(h['video']['frames']):
        with Image.open(source / f'{i}.png') as im:
            if im.size != (h['video']['width'], h['video']['height']):
                raise ValueError('source dimensions differ from semantic metadata')
    if not (old / 'transmitter/visual.c64').is_file():
        raise FileNotFoundError(old / 'transmitter/visual.c64')
    if args.trajectory and not args.trajectory.is_file():
        raise FileNotFoundError(args.trajectory)
    local = json.loads((REPO / '.local/settings.json').read_text())
    for runtime in (REPO / '.local/dcvc_rt_env/bin/python',
                    REPO / '.local/bsu_codec_eval_env/bin/python', Path(local['channel_python'])):
        if not runtime.is_file() or not os.access(runtime, os.X_OK):
            raise ValueError(f'required runtime missing: {runtime}')
    assets = {REPO / f'.local/checkpoints/dcvc_rt/dcvc_rt_{k}.tar': v for k, v in WEIGHTS.items()}
    assets.update({REPO / '.local/checkpoints/rife/v4.25/flownet.pkl': WEIGHT_SHA,
                   REPO / '.local/checkpoints/rife/v4.25/IFNet_HDv3.py': NETWORK_SHA,
                   REPO / '.local/vendor/Practical-RIFE/model/warplayer.py': WARP_SHA})
    for path, digest in assets.items():
        if sha256(path) != digest:
            raise ValueError(f'pinned model or implementation changed: {path}')
    record = dict(profile=value, config_sha256=sha256(CONFIG),
                  code={n: sha256(REPO / f'src/semantic_transmission/{n}.py') for n in CODE})
    record['signature'] = fingerprint(record)
    prior = root / 'default_profile.json'
    if prior.exists() and json.loads(prior.read_text()) != record:
        raise ValueError('frozen default profile/code changed; use a new --output')
    return record


def main(argv=None):
    p = parser(); args = p.parse_args(argv)
    if args.action == 'profile':
        print(json.dumps(profile(), ensure_ascii=False, indent=2)); return
    if not args.output:
        p.error('--output is required')
    if args.action == 'status':
        path = args.output / 'RESULT.json'
        print(path.read_text() if path.exists() else json.dumps(dict(status='NOT_COMPLETE', output=str(args.output))))
        return
    if not args.semantic_run or not args.source_frames:
        p.error('--semantic-run and --source-frames are required')
    record = preflight(args)
    if args.action == 'check':
        print(json.dumps(dict(status='READY', profile=record['profile'], model_inference_started=False), indent=2))
        return
    from . import bsu_adaptive_dense as pipeline
    from .etri_60s_check import lock
    from .qwen35_pairs_video import freeze
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=True)
    with lock(root / 'default_execution.lock'):
        freeze(root / 'default_profile.json', record)
        for name in ('qp', 'temporal_stride', 'min_original_keys', 'protect_semantic_keys'):
            setattr(args, name, record['profile'][name])
        args.neural_run = pipeline.prepare_sender(args)
        args.output = root / 'delivery'
        pipeline.execute(args)
        write_json(root / 'RESULT.json', dict(**pipeline.read(args.output / 'RESULT.json'),
            default_profile=record['profile']['policy'],
            default_profile_sha256=sha256(root / 'default_profile.json'),
            sender_protocol_sha256=sha256(root / 'sender_protocol.json'),
            delivery_result=str(args.output / 'RESULT.json')))


if __name__ == '__main__':
    main()
