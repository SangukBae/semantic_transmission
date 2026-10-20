"""Resumable, opt-in four-factor LTX receiver. Existing defaults are unchanged.

Usage accepts received metadata, RGB keys, and an optional received trajectory.
There is deliberately no original-video or evaluation-label argument.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from . import ltx_conditioning as conditioning, ltx_receiver as ltx
from .artifacts import sha256, write_json
from .bsu_receiver import encode_frames, refine_windows
from .bsu_sequence_refinement import decode_packet
from .qwen35_pairs_video import freeze
from .webvid5 import fingerprint
from .webvid_ablation import Stages, snapshot
from .wire import unpack

MODULE = 'semantic_transmission.bsu_ltx_improvements'


def code_identity():
    names = ('bsu_ltx_improvements', 'ltx_conditioning', 'ltx_refinement', 'ltx_receiver',
             'bsu_receiver', 'bsu_sequence_refinement', 'receiver_key_context', 'trajectory_receiver',
             'sparse_trajectory', 'wire', 'webvid_ablation')
    paths = [Path(__file__).with_name(n+'.py') for n in names] + [conditioning.CONFIG, ltx.CONFIG]
    return {str(p): sha256(p) for p in paths}


def condition_identity(request):
    """Path-independent identity of actual generation inputs, including guides."""
    return dict(video=request['video'], profile=request['profile'], windows=[
        dict(start=w['start'], end=w['end'], frames=w['frames'], generated_frames=w['generated_frames'],
             prompt=w['prompt'], seed=w.get('seed', request['profile']['generation_seed']),
             images=[dict(frame_idx=i['frame_idx'], sha256=request['inputs'][i['path']],
                          strength=i.get('strength', request['profile']['image_strength'])) for i in w['images']])
        for w in request['windows']])


def verified_raw(raw, expected):
    raw = Path(raw).resolve(strict=True)
    receipt, protocol = ltx.read(raw/'generation.json'), ltx.read(raw/'protocol.json')
    old = protocol['request']
    if fingerprint({k: v for k, v in old.items() if k != 'signature'}) != old['signature']:
        raise ValueError('raw generation request signature changed')
    if (receipt['status'] != 'COMPLETE_LTX_GENERATION' or receipt['source_frames_read']
            or receipt['received_input_signature'] != protocol['request']['signature']
            or receipt['video_sha256'] != sha256(raw/'reconstruction.mp4')
            or condition_identity(protocol['request']) != condition_identity(expected)):
        raise ValueError('raw generation does not match requested received conditions')
    for path in sorted((raw/'stages').glob('*.json')):
        item = ltx.read(path)
        if item['status'] != 'PASSED' or snapshot(raw, item['required']) != item['artifacts']:
            raise ValueError('raw generation stage is incomplete or corrupted')
    if not (raw/'stages/assemble.json').is_file():
        raise ValueError('raw generation has no verified assembly receipt')
    paths = sorted((raw/'frames').glob('*.png'))
    if [p.name for p in paths] != [f'{i:05d}.png' for i in range(expected['video']['frames'])]:
        raise ValueError('raw generation coverage differs')
    return paths


def generate(request_path, output, model_dir, vendor, offload='disk'):
    value = ltx.read(request_path)
    conditioning.validate(value)
    assets = dict(ltx.runtime(model_dir, vendor), execution_offload=offload)
    return ltx.reconstruct(value, assets, output)


def refinement(value, raw, output, packet, keys_directory, trajectory):
    from . import ltx_refinement
    conditioning.validate(value)
    paths = verified_raw(raw, value)
    keys = value['receiver_improvements']['original_key_indices']
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    progress = lambda n, total: print(f'LTX refinement intervals {n}/{total} ({100*n/total:.1f}%) ETA=unknown', flush=True)
    if value['receiver_improvements']['flags']['postprocess']:
        rows = ltx_refinement.refine(paths, keys, keys_directory, output/'frames', value['video']['fps'], progress)
        mode = conditioning.policy()['policy']
    else:
        tracks = decode_packet(Path(trajectory).read_bytes(), Path(packet).read_bytes()) if trajectory else {}
        rows = refine_windows(paths, keys, Path(keys_directory), tracks, output/'frames', 'combined_context', progress)
        mode = 'combined_context'
    timeline = encode_frames(output/'frames', output/'reconstruction.mp4', len(paths), value['video']['fps'])
    write_json(output/'refinement.json', dict(mode=mode, windows=rows, timeline=timeline,
        receiver_reads_source=False, received_key_endpoints_exact=True, additional_transmitted_bytes=0,
        policy=conditioning.policy(), variant=value['receiver_improvements']['variant']))


def run(args):
    root = args.output.resolve()
    from .qwen35_pairs_comparison import forbid_overlap
    protected = [args.packet, args.keys, ltx.REPO/'src', ltx.REPO/'configs', args.model_dir, args.vendor]
    protected += [p for p in (args.trajectory, args.raw) if p is not None]
    forbid_overlap(root, protected)
    initial = ltx.request(args.packet, args.keys)
    if args.trajectory:
        decode_packet(args.trajectory.read_bytes(), args.packet.read_bytes())
        initial['inputs'][str(args.trajectory.resolve())] = sha256(args.trajectory)
    identity = dict(received_inputs=initial['inputs'], policy=conditioning.policy(), variant=args.variant,
                    seed=args.seed, implementation=code_identity(), training=False,
                    execution_offload=None if args.raw else args.offload)
    if args.raw:
        if args.variant != 'postprocess':
            raise ValueError('--raw reuse is only valid for the postprocess-only variant')
        # Postprocessing uses exactly the baseline generator inputs.
        expected = ltx.request(args.packet, args.keys)
        for w in expected['windows']:
            w['seed'] = args.seed
        verified_raw(args.raw, expected)
        identity['reused_generation'] = {str(p): sha256(p) for p in [
            args.raw/'protocol.json', args.raw/'generation.json', *sorted((args.raw/'frames').glob('*.png'))]}
    if args.action == 'check':
        ready = args.raw is not None or (args.ltx_python.is_file() and (args.model_dir/'manifest.json').is_file())
        print(dict(status='READY' if ready else 'GENERATION_RUNTIME_UNAVAILABLE',
                   received_inputs_verified=True, reuse_raw=bool(args.raw), training=False,
                   policy=conditioning.policy()['policy'], variant=args.variant))
        return
    if root.exists() and any(root.iterdir()) and not (root/'protocol.json').exists():
        raise ValueError('unrecognized nonempty output; choose a new directory')
    root.mkdir(parents=True, exist_ok=True)
    freeze(root/'protocol.json', identity)
    stages = Stages(root, fingerprint(identity))
    stages.step('prepare', ['conditions'], ['conditions'], lambda _: conditioning.prepare(
        args.packet, args.keys, root/'conditions', args.variant, args.trajectory, args.seed))
    value = ltx.read(root/'conditions/request.json')
    conditioning.validate(value)
    if args.action == 'prepare':
        write_json(root/'status.json', dict(status='PREPARED_GENERATION_NOT_RUN', variant=args.variant))
        return
    raw = args.raw.resolve() if args.raw else root/'raw'
    if not args.raw:
        assets = dict(ltx.runtime(args.model_dir, args.vendor), execution_offload=args.offload)
        freeze(root/'runtime.json', assets)
        def invoke(log):
            from .bsu_ltx_comparison import command
            env = dict(os.environ, PYTHONPATH=str(ltx.REPO/'src'), PYTHONNOUSERSITE='1',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
            command([args.ltx_python, '-m', MODULE, 'generate', '--request', root/'conditions/request.json',
                     '--output', raw, '--model-dir', args.model_dir, '--vendor', args.vendor,
                     '--offload', args.offload], log, env)
        # The inner LTX stages own intervals and retain verified partial work.
        stages.step('generate', [], ['raw'], invoke)
    stages.step('refine', ['refined'], ['refined'], lambda _: refinement(
        value, raw, root/'refined', args.packet, args.keys, args.trajectory))
    write_json(root/'RESULT.json', dict(status='COMPLETE_RECEIVED_ONLY_LTX_RECONSTRUCTION',
        variant=args.variant, training=False, source_pixels_read=False,
        generation_reused=bool(args.raw), received_inputs=identity['received_inputs'],
        additional_transmitted_bytes=0, video=str(root/'refined/reconstruction.mp4'),
        video_sha256=sha256(root/'refined/reconstruction.mp4'), evaluation='NOT_RUN',
        generalization_proven=False, default_adopted=False))
    write_json(root/'status.json', dict(status='COMPLETE_RECEIVED_ONLY_LTX_RECONSTRUCTION', variant=args.variant))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'prepare', 'run', 'generate'))
    parser.add_argument('--packet', type=Path)
    parser.add_argument('--keys', type=Path)
    parser.add_argument('--trajectory', type=Path)
    parser.add_argument('--variant', choices=conditioning.VARIANTS, default='all')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--offload', choices=('cpu', 'disk'), default='disk')
    parser.add_argument('--raw', type=Path, help='Reuse verified baseline generation for postprocess-only evaluation')
    parser.add_argument('--request', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ltx-python', type=Path, default=ltx.REPO/'.local/envs/bsu-ltx25/bin/python')
    parser.add_argument('--model-dir', type=Path, default=ltx.REPO/'.local/checkpoints/ltx25')
    parser.add_argument('--vendor', type=Path, default=ltx.REPO/'.local/vendor/LTX-2')
    args = parser.parse_args()
    if args.action == 'generate':
        if args.request is None:
            parser.error('generate requires --request')
        return generate(args.request, args.output, args.model_dir, args.vendor, args.offload)
    if args.packet is None or args.keys is None:
        parser.error('--packet and --keys are required')
    run(args)


if __name__ == '__main__':
    main()
