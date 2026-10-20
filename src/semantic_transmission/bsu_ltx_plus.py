"""Frozen-selection NTSCC+ / LTX experiment with real channel accounting.

Reuse source selection, semantic metadata and trajectory coordinates, not decoded
keys. Rebind and retransmit trajectories against the new base packet. Compare with
the completed LGVSC / Open-Sora / NTSCC-LTX cohort without overwriting it.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from .artifacts import sha256, write_json
from . import bsu_ltx_comparison as previous, ltx_receiver as ltx, ntscc_plus_codec as codec
from .webvid_ablation import Stages, environment, snapshot
from .webvid5 import fingerprint
from .qwen35_pairs_video import freeze
from .wire import pack, unpack

REPO = Path(__file__).resolve().parents[2]
MODULE = 'semantic_transmission.bsu_ltx_plus'
POLICY = 'bsu_ltx_ntscc_plus_codec_ablation_v1'
COMPLETE = 'COMPLETE_LTX_NTSCC_PLUS_COMPARISON'
read = previous.read


def audit_reference(reference):
    result = read(reference / 'RESULT.json')
    original = read(reference / 'baseline_audit.json')
    protocol = read(reference / 'protocol.json')
    if (result['status'] != 'COMPLETE_THREE_WAY_RECONSTRUCTION_COMPARISON'
            or result['videos'] != 14 or len(original['rows']) != 14
            or sha256(reference / 'baseline_audit.json') != protocol['baseline_audit_sha256']):
        raise ValueError('need the complete, unchanged 14-video reference comparison')
    files = dict(original['inputs'])
    files.update(protocol['code'])
    for name in ('RESULT.json', 'baseline_audit.json', 'protocol.json'):
        files[str(reference / name)] = sha256(reference / name)
    rows = []
    for row in original['rows']:
        dest = reference / 'videos' / row['id']; old = read(dest / 'RESULT.json')
        if (old['status'] != 'COMPLETE_LTX_RECONSTRUCTION_QUALITY'
                or old['signature'] != result['signature'] or old['frames'] != row['frames']
                or old['total_channel_uses'] != row['bsu_channel_uses']):
            raise ValueError('incomplete or inconsistent LTX baseline')
        files[str(dest / 'RESULT.json')] = sha256(dest / 'RESULT.json')
        files.update({str(dest / p): h for p, h in old['artifacts'].items()})
        base_run = Path(row['base']) / 'reconstruction/event_pairs/run'
        files[str(base_run / 'run_config.json')] = sha256(base_run / 'run_config.json')
        sender = Path(row['trajectory']).parent / 'sender/trajectory.bin'
        if sender.exists():
            files[str(sender)] = sha256(sender)
            if sender.read_bytes() != (Path(row['trajectory']) / 'trajectory.bin').read_bytes():
                raise ValueError('reference trajectory was not received exactly')
        rows.append(dict(row, ltx_quality=old['quality'], ltx_raw_quality=old['raw_quality'],
                         ltx_video=old['video']))
    # Old outputs may be reused only with the same metric implementations.
    for name in ('official_quality.py', 'research_quality.py', 'stutter_metrics.py'):
        matches = [h for p, h in protocol['code'].items() if p.endswith('/' + name)]
        if matches != [sha256(REPO / 'src/semantic_transmission' / name)]:
            raise ValueError(f'baseline evaluator changed: {name}')
    previous.verify_files(files)
    return dict(reference=str(reference), inputs=files, rows=rows, frames=sum(r['frames'] for r in rows),
                original_ntscc_ltx_result_sha256=sha256(reference / 'RESULT.json'))


def contract(args):
    from .cli import settings
    from .qwen35_pairs_comparison import forbid_overlap
    reference, root = args.reference.resolve(), args.output.resolve()
    forbid_overlap(root, [reference, REPO / 'src', REPO / '.local', REPO / 'configs'])
    if root.exists() and any(root.iterdir()) and not (root / 'reference_audit.json').exists():
        raise ValueError('unrecognized nonempty output; choose a new directory')
    if (root / 'reference_audit.json').exists():
        audit = read(root / 'reference_audit.json')
        if audit['reference'] != str(reference):
            raise ValueError('reference changed')
        previous.verify_files(audit['inputs'])
    else:
        audit = audit_reference(reference)
    local = settings(REPO)
    interpreters = dict(metrics=local['python'], channel=local['channel_python'],
        codec=str(REPO / '.local/ntscc_plus_env/bin/python'), ltx=str(args.ltx_python.absolute()))
    for p in interpreters.values():
        if not Path(p).is_file():
            raise FileNotFoundError(p)
    if sha256(REPO / codec.CHECKPOINT) != codec.CHECKPOINT_SHA256:
        raise ValueError('NTSCC+ checkpoint changed')
    code = previous.code_identity()
    code.update({str(p): sha256(p) for p in (REPO / '.local/vendor/NTSCC_plus').rglob('*.py')})
    code[str(REPO / 'scripts/compare_bsu_ltx_plus.sh')] = sha256(REPO / 'scripts/compare_bsu_ltx_plus.sh')
    value = dict(policy=POLICY, reference=str(reference), reference_identity=fingerprint(audit),
        code=code, codec=codec.installation(REPO), ltx_profile=ltx.profile(),
        runtime=ltx.runtime(args.model_dir, args.vendor), interpreters=interpreters,
        ltx_environment_sha256=sha256(args.ltx_python.parent.parent / 'bsu-environment.txt'),
        selection='reuse_exact_baseline_keys_captions_flow', snr_db=10, channel_seed=42,
        trajectory_channel_seed=43, encoder_seed=2025, generation_seed=2025,
        trajectory_policy='reuse_coordinates_rebind_packet_and_retransmit',
        trajectory_budget='frozen_baseline_allocation; not recomputed as 5 percent of new codec cost',
        equal_rate_comparison=False, independent_semantic_review='PENDING')
    return audit, value


def initialize(row, run):
    source = read(row['prepared'])
    old_run = Path(row['base']) / 'reconstruction/event_pairs/run'
    header, _ = unpack(Path(row['received_packet']).read_bytes())
    keys = [k['index'] for k in header['keyframes']]
    if header['schema'] != 'lgvsc_model_inputs_v1' or source['frames'] != header['video']['frames']:
        raise ValueError('unexpected baseline codec or timeline')
    cfg = read(old_run / 'run_config.json')
    cfg.update(visual_codec='NTSCC+', profile=POLICY)
    if any(cfg[k] != v for k, v in dict(seed=2025, channel_seed=42, snr_db=10,
                                       visual_channel='source_frame_seeded_numpy_awgn_v1').items()):
        raise ValueError('baseline channel or seed changed')
    frames = run / 'data/frames/sample'; frames.mkdir(parents=True)
    for i in keys:
        path = Path(source['source_frames']) / f'{i}.png'
        if sha256(path) != source['source_frame_hashes'][str(i)]:
            raise ValueError('source key pixels changed')
        shutil.copyfile(path, frames / path.name)
    write_json(run / 'run_config.json', cfg)
    write_json(run / 'keyframes.json', dict(indices=keys))
    write_json(run / 'metadata_tx.json', header['segments'])


def rebind_trajectory(packet, old_base, new_base):
    from .bsu_sequence_refinement import decode_packet
    old, _ = unpack(old_base); new, _ = unpack(new_base)
    if (old['video'] != new['video'] or old['segments'] != new['segments']
            or [k['index'] for k in old['keyframes']] != [k['index'] for k in new['keyframes']]):
        raise ValueError('codec experiment changed semantic conditions')
    decode_packet(packet, old_base)
    header, payload = unpack(packet)
    header['base_metadata_sha256'] = hashlib.sha256(new_base).hexdigest()
    rebound = pack(header, payload)
    decode_packet(rebound, new_base)
    return rebound


def trajectory(row, dest):
    from .metadata_channel import transmit
    from .bsu_sequence_refinement import decode_packet
    received = dest / 'trajectory_received'; received.mkdir()
    sender = dest / 'trajectory_sender'; sender.mkdir()
    original = Path(row['trajectory']) / 'trajectory.bin'
    old_report = read(Path(row['trajectory']) / 'channel.json')
    report = dict(status='PASSED', packet_present=False, complex_channel_uses=0)
    if original.exists():
        base = (dest / 'run/received/metadata.bin').read_bytes()
        packet = rebind_trajectory(original.read_bytes(), Path(row['received_packet']).read_bytes(), base)
        (sender / 'trajectory.bin').write_bytes(packet)
        decoded, report = transmit(packet, 10, 43, bits_per_symbol=4)
        if decoded != packet or report['bit_errors']:
            raise ValueError('trajectory digital channel failed')
        decode_packet(decoded, base)
        (received / 'trajectory.bin').write_bytes(decoded)
        report.update(status='PASSED', packet_present=True, exact_match=True)
    if report['complex_channel_uses'] != old_report['complex_channel_uses']:
        raise ValueError('fixed trajectory side cost changed')
    write_json(sender / 'sender.json', dict(original_trajectory=str(original),
        reused_coordinates=True, packet_rebound_and_channel_tested=original.exists(),
        budget_policy='frozen baseline coordinates; extra cost paid in full'))
    write_json(received / 'channel.json', report)


def transport_audit(row, dest):
    import numpy as np
    from PIL import Image
    from .transmission_accounting import channel_breakdown
    run = dest / 'run'; packet = (run / 'received/metadata.bin').read_bytes()
    header, _ = unpack(packet); old, _ = unpack(Path(row['received_packet']).read_bytes())
    if (header['schema'] != codec.SCHEMA or header['visual_codec'] != codec.SPEC
            or header['checkpoint_sha256'] != codec.CHECKPOINT_SHA256
            or header['segments'] != old['segments'] or header['video'] != old['video']
            or [k['index'] for k in header['keyframes']] != [k['index'] for k in old['keyframes']]
            or packet != (run / 'transmitter/metadata.bin').read_bytes()):
        raise ValueError('new codec packet or fixed semantic inputs mismatch')
    receiver = read(run / 'receiver_accounting.json')
    if receiver['source_pixels_read'] or receiver['visual_codec'] != 'NTSCC+':
        raise ValueError('receiver codec audit failed')
    channel = read(run / 'channel_accounting.json')
    ledger = channel_breakdown(packet, (run / 'transmitter/visual.c64').stat().st_size // 8,
                               header['video'], channel)
    extras = read(dest / 'trajectory_received/channel.json')['complex_channel_uses']
    keys = []
    for item in header['keyframes']:
        index = item['index']; source = np.asarray(Image.open(run / f'data/frames/sample/{index}.png'), dtype=float)
        scores = {}
        for name, path in (('NTSCC', Path(row['received_keys']) / f'{index}.png'),
                           ('NTSCC+', run / f'receiver/frames/sample/key_frames_received/{index}.png')):
            pixels = np.asarray(Image.open(path), dtype=float)
            if pixels.shape != source.shape:
                raise ValueError('decoded key geometry changed')
            scores[name] = float(10 * np.log10(255**2 / max(np.mean((source-pixels)**2), 1e-12)))
        keys.append(dict(index=index, psnr_db=scores))
    write_json(dest / 'transport_audit.json', dict(status='PASSED', ledger=ledger,
        trajectory_complex_channel_uses=extras, total_channel_uses=channel['total_complex_channel_uses'] + extras,
        old_total_channel_uses=row['bsu_channel_uses'], source_pixels_read_by_receiver=False,
        fixed_selection_and_metadata_verified=True, key_quality=keys))


def conditions(row, dest):
    modified = dict(row, received_packet=str(dest / 'run/received/metadata.bin'),
        received_keys=str(dest / 'run/receiver/frames/sample/key_frames_received'),
        trajectory=str(dest / 'trajectory_received'))
    previous.copy_conditions(modified, dest / 'conditions')


def worker(stage, dest):
    row = read(dest / 'job.json')['row']
    if stage == 'channel':
        from .qwen35_pairs_comparison import paired_channel
        paired_channel(dest / 'run')
    elif stage == 'trajectory':
        trajectory(row, dest)
    elif stage == 'audit':
        transport_audit(row, dest)
    elif stage == 'refine':
        previous.refine(dest)
    else:
        from .bsu_sequence_refinement import evaluate
        evaluate(dict(base=row['base']), dest / stage.removeprefix('evaluate_'))


def summarize(root, audit, results, signature):
    rows = audit['rows']
    if [r['id'] for r in results] != [r['id'] for r in rows]:
        raise ValueError('incomplete comparison cohort')
    for row, result in zip(rows, results):
        q = result['quality']
        if (result['status'] != 'COMPLETE_LTX_NTSCC_PLUS_VIDEO'
                or result['frames'] != row['frames'] or q['delivered_mp4']['frames'] != row['frames']
                or q['source_sha256'] != row['bsu_quality']['source_sha256']):
            raise ValueError('new quality reference or frame coverage mismatch')
    arms = {}
    specs = [('LGVSC/NTSCC', [r['lgvsc_quality'] for r in rows], [r['lgvsc_channel_uses'] for r in rows]),
        ('BSU/Open-Sora/NTSCC', [r['bsu_quality'] for r in rows], [r['bsu_channel_uses'] for r in rows]),
        ('BSU/LTX/NTSCC', [r['ltx_quality'] for r in rows], [r['bsu_channel_uses'] for r in rows]),
        ('BSU/LTX/NTSCC+', [r['quality'] for r in results], [r['total_channel_uses'] for r in results])]
    for name, qualities, costs in specs:
        arms[name] = dict(total_channel_uses=sum(costs), delivered_mp4_video_mean={
            k: sum(q['delivered_mp4'][k] for q in qualities) / len(rows) for k in previous.KEYS})
    result = dict(status=COMPLETE, signature=signature, videos=len(rows), frames=audit['frames'], arms=arms,
        equal_rate_comparison=False, fixed_source_keys_captions_flow_trajectory_coordinates=True,
        independent_semantic_review='PENDING', action_accuracy_evaluated=False,
        long_video_generalization_verified=False)
    write_json(root / 'RESULT.json', result)
    write_json(root / 'status.json', dict(status=COMPLETE, completed=len(rows), total=len(rows)))


def run(args):
    from .etri_60s_check import lock
    root = args.output.resolve()
    audit, protocol = contract(args)
    root.mkdir(parents=True, exist_ok=True)
    freeze(root / 'reference_audit.json', audit); freeze(root / 'protocol.json', protocol)
    signature = fingerprint(protocol); runtimes = protocol['interpreters']
    env = dict(environment(2025), PYTHONPATH=str(REPO / 'src'), PYTHONNOUSERSITE='1')
    env.pop('LD_LIBRARY_PATH', None); env.pop('PYTORCH_CUDA_ALLOC_CONF', None)
    results = []; started = time.monotonic()
    with lock(root / 'execution.lock'), lock(REPO / '.local/etri_60s_check.lock'):
        try:
            for row in audit['rows']:
                dest = root / 'videos' / row['id']; dest.mkdir(parents=True, exist_ok=True)
                freeze(dest / 'job.json', dict(row=row, signature=signature))
                stages = Stages(dest, signature)
                def step(name, owned, products, callback):
                    write_json(root / 'status.json', dict(status='RUNNING', video=row['id'], stage=name,
                        completed=len(results), total=len(audit['rows']), elapsed_seconds=time.monotonic()-started,
                        resume='Repeat the identical run command; completed stages and LTX intervals are hash-verified.'))
                    previous.progress(len(results), len(audit['rows']), row['id'], name)
                    stages.step(name, owned, products, callback)
                def launch(stage, runtime, log, module=MODULE, extra=None):
                    e = codec.runtime_environment(runtimes[runtime], env) if runtime == 'codec' else dict(env)
                    if runtime == 'channel':
                        e['CUDA_VISIBLE_DEVICES'] = '-1'
                    argv = extra if extra is not None else ['_worker', '--worker', stage, '--output', str(dest)]
                    previous.command([runtimes[runtime], '-m', module, *argv], log, e)
                step('initialize', ['run'], ['run/data', 'run/run_config.json', 'run/keyframes.json', 'run/metadata_tx.json'],
                     lambda _: initialize(row, dest / 'run'))
                step('send', ['run/transmitter', 'run/sender_accounting.json'],
                     ['run/transmitter', 'run/sender_accounting.json'],
                     lambda log: launch('send', 'codec', log, 'semantic_transmission.codec_transport', ['send', str(dest / 'run')]))
                step('channel', ['run/received', 'run/channel_accounting.json'],
                     ['run/received', 'run/channel_accounting.json'], lambda log: launch('channel', 'channel', log))
                step('receive', ['run/receiver', 'run/receiver_accounting.json'],
                     ['run/receiver', 'run/receiver_accounting.json'],
                     lambda log: launch('receive', 'codec', log, 'semantic_transmission.codec_transport', ['receive', str(dest / 'run')]))
                step('trajectory', ['trajectory_sender', 'trajectory_received'], ['trajectory_sender', 'trajectory_received'],
                     lambda log: launch('trajectory', 'channel', log))
                step('transport_audit', ['transport_audit.json'], ['transport_audit.json'],
                     lambda log: launch('audit', 'metrics', log))
                step('conditions', ['conditions'], ['conditions'], lambda _: conditions(row, dest))
                step('generate', [], ['raw/frames', 'raw/reconstruction.mp4', 'raw/generation.json'],
                     lambda log: launch('generate', 'ltx', log, 'semantic_transmission.ltx_receiver',
                        ['--request', str(dest / 'conditions/request.json'), '--output', str(dest / 'raw'),
                         '--model-dir', str(args.model_dir), '--vendor', str(args.vendor)]))
                step('refine', ['refined'], ['refined/frames', 'refined/reconstruction.mp4', 'refined/refinement.json'],
                     lambda log: launch('refine', 'metrics', log))
                for boundary in ('raw', 'refined'):
                    products = [f'{boundary}/quality.json', f'{boundary}/quality_lossless_frames.csv',
                                f'{boundary}/quality_delivered_mp4.csv']
                    step('evaluate_' + boundary, products, products,
                         lambda log, b=boundary: launch('evaluate_' + b, 'metrics', log))
                q = read(dest / 'refined/quality.json'); transport = read(dest / 'transport_audit.json')
                if q['source_sha256'] != row['bsu_quality']['source_sha256'] or q['delivered_mp4']['frames'] != row['frames']:
                    raise ValueError('new result reference or timeline mismatch')
                result = dict(id=row['id'], status='COMPLETE_LTX_NTSCC_PLUS_VIDEO', signature=signature,
                    frames=row['frames'], quality=q, raw_quality=read(dest / 'raw/quality.json'),
                    total_channel_uses=transport['total_channel_uses'], transport=transport,
                    video=str(dest / 'refined/reconstruction.mp4'),
                    artifacts=snapshot(dest, ['run/transmitter', 'run/received', 'run/receiver', 'conditions',
                        'trajectory_sender', 'trajectory_received', 'transport_audit.json', 'raw', 'refined']))
                write_json(dest / 'RESULT.json', result); results.append(result)
                previous.progress(len(results), len(audit['rows']), row['id'], '송수신·복원·평가 완료')
            previous.verify_files(audit['inputs']); previous.verify_files(protocol['code'])
            summarize(root, audit, results, signature)
            previous.progress(len(results), len(audit['rows']), '전체', '완료', state=COMPLETE)
        except BaseException as error:
            write_json(root / 'status.json', dict(status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED',
                error=str(error), completed=len(results), total=len(audit['rows']),
                resume='Repeat the identical run command; preserve this directory.'))
            raise


def status(root):
    if not (root / 'status.json').exists():
        return dict(status='NOT_STARTED')
    value = read(root / 'status.json')
    if value.get('video'):
        dest = root / 'videos' / value['video']; request = dest / 'conditions/request.json'
        if request.exists():
            receipts = list((dest / 'raw/stages').glob('interval_*.json'))
            value['current_video_intervals'] = dict(completed=sum(read(p).get('status') == 'PASSED' for p in receipts),
                total=len(read(request)['windows']))
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'run', 'status', '_worker'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--worker', choices=('channel', 'trajectory', 'audit', 'refine', 'evaluate_raw', 'evaluate_refined'))
    parser.add_argument('--model-dir', type=Path, default=REPO / '.local/checkpoints/ltx25')
    parser.add_argument('--vendor', type=Path, default=REPO / '.local/vendor/LTX-2')
    parser.add_argument('--ltx-python', type=Path, default=REPO / '.local/envs/bsu-ltx25/bin/python')
    args = parser.parse_args()
    if args.action == 'status':
        print(json.dumps(status(args.output), indent=2, ensure_ascii=False)); return
    if args.action == '_worker':
        worker(args.worker, args.output); return
    if args.reference is None:
        parser.error('--reference is required')
    if args.action == 'check':
        audit, _ = contract(args)
        print(json.dumps(dict(status='READY_FILES_VERIFIED', videos=len(audit['rows']), frames=audit['frames'],
                              model_inference_started=False))); return
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    run(args)


if __name__ == '__main__':
    main()
