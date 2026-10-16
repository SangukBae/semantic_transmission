"""Short, source-matched validation of the experimental sender extension.

Registers arms before generation. Evaluates delivered MP4s independently, never
uses quality results to retrain or choose a reconstruction. This is a development
smoke, not a replacement for the full adopted-BSU benchmark.
"""
import argparse
import itertools
import subprocess
from pathlib import Path

from . import unified_bsu as u
from . import unified_bsu_core as core
from .artifacts import sha256, write_json
from .qwen35_pairs_video import freeze
from .wire import pack, unpack


def verified_infeasible(run, arm):
    """Recognize a fully measured no-fit grid; never hide other worker failures."""
    paths = [run / 'stages/sender.json', run / 'sender/candidates.json', run / 'logs/sender.log']
    if not all(p.exists() for p in paths) or (run / 'RESULT.json').exists():
        return None
    stage, candidates = (core.read(p) for p in paths[:2])
    if (stage.get('status') != 'FAILED' or candidates.get('budget') != arm['budget']
            or 'ValueError: no candidate fits the complete paid packet budget' not in paths[2].read_text()):
        return None
    p = arm['profile']
    names = {f'scale{s}_stride{t}_qp{q}' for s, t, q in itertools.product(p['scales'], p['strides'], p['qps'])}
    rows = candidates['rows']
    if len(rows) != len(names) or {r['name'] for r in rows} != names:
        return None
    for row in rows:
        packet = run / 'sender' / (row['name']+'.bin')
        data = packet.read_bytes()
        if (sha256(packet) != row['packet_sha256'] or len(data) != row['packet_bytes']
                or core.channel_uses(data) != row['channel_uses'] or row['fits']
                or row['channel_uses'] <= arm['budget']):
            return None
        core.decode_packet(data)
    return dict(status='INFEASIBLE_REGISTERED_BUDGET', budget=arm['budget'],
                tested_candidates=len(rows), minimum_candidate_channel_uses=min(r['channel_uses'] for r in rows),
                actual_channel_run=False, movie=None, budget_met=False,
                candidate_receipt_sha256=sha256(paths[1]),
                reason='every measured base packet exceeds the budget; added tokens cannot make it fit')


def prepare_clip(root, clip):
    original = core.REPO / 'outputs/bsu_kinetics_context'
    source = original / 'inputs/kinetics' / clip / 'source_frames'
    packet = original / 'benchmark/snr_10' / clip / 'base/reconstruction/event_pairs/run/transmitter/metadata.bin'
    h, _ = unpack(packet.read_bytes()); keys = [x['index'] for x in h['keyframes']]
    indices = sorted({0, 16} | {i for i in keys if 0 <= i <= 16})
    segments = []
    for a, b in zip(indices, indices[1:]):
        position = max(i for i, k in enumerate(keys[:-1]) if k <= a)
        s = h['segments'][position]
        segments.append(dict(path=f'clips/sample/{a}_{b}.mp4', text=s['text'], flow=s['flow']))
    c = dict(video=dict(h['video'], frames=17), indices=indices, segments=segments, decoder={})
    output = root / 'inputs' / clip; frames = output / 'source'; frames.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for i in range(17):
        target = frames / f'{i}.png'; path = source / f'{i}.png'; hashes[f'{i}.png'] = sha256(path)
        if not target.exists():
            target.symlink_to(path.resolve())
        if sha256(target) != hashes[f'{i}.png']:
            raise ValueError('source frame changed')
    semantic = output / 'semantic/transmitter'; semantic.mkdir(parents=True, exist_ok=True)
    data = pack(dict(video=c['video'], keyframes=[dict(index=i) for i in indices], segments=segments, decoder={}))
    path = semantic / 'metadata.bin'
    if path.exists() and path.read_bytes() != data:
        raise ValueError('semantic clip changed')
    path.write_bytes(data)
    return dict(clip=clip, source=str(frames.resolve()), semantic=str(semantic.parent.resolve()),
                source_hashes=hashes, source_packet_sha256=sha256(packet), context=c)


def evaluate(root):
    import numpy as np
    from .official_quality import OfficialMetrics
    from .research_quality import read_video
    from .unified_bsu_evaluate import metrics
    from .etri_60s_check import pts_audit
    p = core.read(root / 'protocol.json'); evaluator = OfficialMetrics(); rows = []
    if any(sha256(core.REPO / f'src/semantic_transmission/{n}.py') != digest for n, digest in p['code'].items()):
        raise ValueError('frozen model code changed')
    if any(sha256(a['checkpoint']) != p['checkpoints'][a['name']] for a in p['arms']):
        raise ValueError('frozen checkpoint changed')
    for clip in p['clips']:
        source = u.load_rgb(clip['source'], 17)
        for name, digest in clip['source_hashes'].items():
            if sha256(Path(clip['source']) / name) != digest:
                raise ValueError('evaluation source changed')
        for arm in p['arms']:
            run = root / clip['clip'] / arm['name']
            infeasible = verified_infeasible(run, arm)
            if infeasible:
                rows.append(dict(clip=clip['clip'], arm=arm['name'], **infeasible))
                continue
            result = core.read(run / 'RESULT.json')
            movie = Path(result['movie'])
            if sha256(movie) != result['movie_sha256']:
                raise ValueError('delivered movie changed')
            pts_audit(movie, 17, clip['context']['video']['fps'])
            decoded = read_video(movie)
            basic = metrics(source, decoded); official = evaluator.evaluate(source, decoded)[0]
            channel = core.read(run / 'received/channel.json')
            selection = core.read(run / 'sender/selection.json')['selected']
            receiver = core.read(run / 'receiver_job.json')
            if any(k in receiver for k in ('source', 'source_frames', 'semantic_run', 'source_hashes')):
                raise ValueError('receiver has source inputs')
            rows.append(dict(clip=clip['clip'], arm=arm['name'], status='COMPLETE_RECONSTRUCTION', channel_uses=result['channel_uses'],
                budget=result['budget'], budget_met=result['budget_met'], bit_errors=channel['bit_errors'],
                selected=selection['name'], packet=core.packet_breakdown((run / 'received/packet.bin').read_bytes()),
                packet_sha256=result['packet_sha256'], movie=str(movie), movie_sha256=sha256(movie),
                basic=basic, official=official, receiver_source_access=False,
                refinement=core.read(run / 'refined/audit.json')))
    comparisons = []
    for clip in p['clips']:
        old = next(r for r in rows if r['clip'] == clip['clip'] and r['arm'] == 'previous_unified')
        for row in (r for r in rows if r['clip'] == clip['clip'] and r['arm'] != 'previous_unified'):
            if row['status'] == 'INFEASIBLE_REGISTERED_BUDGET':
                comparisons.append(dict(clip=clip['clip'], arm=row['arm'], checks=dict(budget_feasible=False),
                    registered_smoke_guard_passed=False, quality_evaluated=False,
                    channel_saving_percent=None, psnr_delta_db=None))
                continue
            q, b = row['official'], old['official']
            checks = dict(psnr=q['psnr_db'] >= b['psnr_db']-.1, ssim=q['ssim'] >= b['ssim']-.005,
                lpips=q['lpips_vgg'] <= b['lpips_vgg']*1.01, dists=q['dists'] <= b['dists']*1.01,
                clip=q['clip'] >= b['clip']-.002,
                temporal=row['basic']['temporal_rgb_mae'] <= old['basic']['temporal_rgb_mae']*1.02,
                fewer_channel_uses=row['channel_uses'] < old['channel_uses'])
            comparisons.append(dict(clip=clip['clip'], arm=row['arm'], checks=checks,
                registered_smoke_guard_passed=all(checks.values()),
                channel_saving_percent=100*(1-row['channel_uses']/old['channel_uses']),
                psnr_delta_db=q['psnr_db']-b['psnr_db']))
    result = dict(status='COMPLETE_SHORT_SENDER_CHANNEL_GENERATION_COMPARISON', rows=rows,
        comparisons=comparisons, frames_per_clip=17, source_matched=True, delivered_mp4_metrics=True,
        full_adopted_bsu_comparison=False, independent_semantic_review=False,
        action_evaluation=False, full_video_evaluation=False, generalization_proven=False)
    result.update(completed_reconstructions=sum(r['status'] == 'COMPLETE_RECONSTRUCTION' for r in rows),
        infeasible_arms=sum(r['status'] == 'INFEASIBLE_REGISTERED_BUDGET' for r in rows),
        registered_validation_code_sha256=p['validation_code_sha256'],
        evaluation_code_sha256=sha256(Path(__file__)),
        registered_budgets_and_quality_thresholds_unchanged=True)
    write_json(root / 'RESULT.json', result)
    return result


def main():
    from .bsu_video import launch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--previous-checkpoint', type=Path, required=True)
    parser.add_argument('--clips', nargs='+', default=['kinetics_003_5d588509', 'kinetics_005_11836d73'])
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args(); root = args.root.resolve(); root.mkdir(parents=True, exist_ok=True)
    if args.evaluate_only:
        from .etri_60s_check import lock
        with lock(core.REPO / '.local/etri_60s_check.lock'):
            evaluate(root)
        return
    current = core.profile(core.REPO / 'configs/bsu_unified_sender.json')
    previous = core.profile(); clips = [prepare_clip(root, name) for name in args.clips]
    arms = [dict(name='previous_unified', budget=13824, profile=previous, checkpoint=str(args.previous_checkpoint.resolve())),
            dict(name='sender_unified', budget=13824, profile=current, checkpoint=str(args.checkpoint.resolve())),
            dict(name='sender_tight', budget=9216, profile=current, checkpoint=str(args.checkpoint.resolve()))]
    protocol = dict(clips=clips, arms=arms, code={n: sha256(core.REPO / f'src/semantic_transmission/{n}.py') for n in u.CODE},
        validation_code_sha256=sha256(Path(__file__)), checkpoints={a['name']: sha256(a['checkpoint']) for a in arms},
        predefined_budgets=True, scope='two short development clips, not full benchmark')
    freeze(root / 'protocol.json', protocol)
    for clip in clips:
        for arm in arms:
            path = root / (arm['name']+'_profile.json'); freeze(path, arm['profile'])
            output = root / clip['clip'] / arm['name']
            command = ['bash', str(core.REPO / 'scripts/run_bsu_unified.sh'), '--semantic-run', clip['semantic'],
                       '--source-frames', clip['source'], '--output', output, '--budget-channel-uses', str(arm['budget']),
                       '--profile', path, '--adapter-checkpoint', arm['checkpoint']]
            if verified_infeasible(output, arm):
                continue
            try:
                launch(command, root / clip['clip'] / (arm['name']+'.log'))
            except subprocess.CalledProcessError:
                if not verified_infeasible(output, arm):
                    raise
    launch([u.paths()['python'], '-m', __spec__.name, '--root', root, '--checkpoint', args.checkpoint,
            '--previous-checkpoint', args.previous_checkpoint, '--evaluate-only'], root / 'evaluation.log')
    u.event('송신부 비교 검증 완료', 1, 1)


if __name__ == '__main__':
    main()
