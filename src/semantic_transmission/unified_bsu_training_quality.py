"""Training rounds with independent, full-duration validation between workers.

The training worker exits before codecs/generation/metrics acquire GPU memory.
Only validation sources select checkpoints. ETRI evaluation originals stay out.
No plateau, time limit, or successful run is labelled proven sufficient training.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np

from . import unified_bsu as u
from . import unified_bsu_core as core
from . import unified_bsu_training as training
from . import unified_bsu_training_policy as policy
from .adapter_training_state import load_checkpoint
from .artifacts import sha256, write_json
from .etri_60s_check import lock
from .qwen35_pairs_video import freeze
from .wire import pack


def launch(command, log):
    """Forward interruption and wait for the child's committed-update checkpoint."""
    log = Path(log); log.parent.mkdir(parents=True, exist_ok=True)
    interrupted = []
    with log.open('w') as out:
        child = subprocess.Popen([str(x) for x in command], cwd=core.REPO,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        def stop(signum, frame):
            interrupted.append(signum)
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            for line in child.stdout:
                out.write(line); out.flush(); print(line, end='', flush=True)
            code = child.wait()
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
        if interrupted:
            raise SystemExit(130)
        if code:
            raise subprocess.CalledProcessError(code, command)


def input_job(root, example):
    root = Path(root); source = root / 'source'; source.mkdir(parents=True, exist_ok=True)
    for i, (path, digest) in enumerate(zip(example['source_frames'], example['source_hashes'])):
        if sha256(path) != digest:
            raise ValueError('validation source changed')
        target = source / f'{i}.png'
        if not target.exists():
            target.symlink_to(Path(path).resolve())
        if sha256(target) != digest:
            raise ValueError('validation source link changed')
    semantic = root / 'semantic/transmitter'; semantic.mkdir(parents=True, exist_ok=True)
    c = example['context']
    data = pack(dict(video=c['video'], keyframes=[dict(index=i) for i in c['indices']],
                     segments=c['segments'], decoder={}))
    path = semantic / 'metadata.bin'
    if path.exists() and path.read_bytes() != data:
        raise ValueError('validation captions changed')
    path.write_bytes(data)
    return source, semantic.parent


def summarize_quality(rows, c):
    if not rows:
        raise ValueError('empty validation quality cohort')
    scores, guards = {}, []
    for row in rows:
        q = row['quality']; final = q['rows']['refined']; base = q['comparison']['reference']
        f, b = final['official_quality'], base['official_quality']
        score = f['lpips_vgg']/max(b['lpips_vgg'], 1e-6)+final['temporal_rgb_mae']/max(base['temporal_rgb_mae'], .1)
        if not np.isfinite(score):
            raise ValueError('nonfinite validation quality')
        scores.setdefault(row['dataset'], []).append(float(score))
        guards.append(dict(name=row['name'], budget=q['budget_met'],
            psnr=f['psnr_db'] >= b['psnr_db']-c['quality_psnr_tolerance_db'],
            lpips=f['lpips_vgg'] <= b['lpips_vgg']+c['quality_lpips_tolerance'],
            temporal=final['temporal_rgb_mae'] <= base['temporal_rgb_mae']+c['quality_temporal_tolerance'],
            worst_second=final['worst_second_temporal_rgb_mae'] <=
                base['worst_second_temporal_rgb_mae']+c['quality_temporal_tolerance']))
    return dict(score=float(np.mean([np.mean(v) for v in scores.values()])), guards=guards,
        guard_passed=all(all(v for k, v in g.items() if k != 'name') for g in guards),
        score_definition='dataset-balanced LPIPS/base + temporal RGB error/base; lower is better',
        quality_is_validation_only=True, performance_improvement_proven=False)


def evaluate_checkpoint(root, p, export):
    root = Path(root); step = export['step']; out = root / 'quality' / f'step_{step:07d}'
    out.mkdir(parents=True, exist_ok=True)
    request = dict(training_signature=p['signature'], export=export,
        evaluator_sha256=sha256(core.REPO / 'src/semantic_transmission/unified_bsu_evaluate.py'))
    freeze(out / 'protocol.json', request)
    profile_path = root / 'quality_profile.json'; freeze(profile_path, p['profile'])
    rows = []
    for i, e in enumerate(p['quality_examples']):
        if e['role'] != 'validation':
            raise ValueError('quality checkpoint selection may only use validation sources')
        u.event('긴 영상 검증 복원', i, len(p['quality_examples']), e['name'])
        source, semantic = input_job(root / 'quality_inputs' / e['name'], e)
        budget = len(e['source_frames'])*e['budget_per_frame']
        common = [p['runtime']['python'], '-m', u.MODULE, '--semantic-run', semantic,
            '--source-frames', source, '--profile', profile_path, '--budget-channel-uses', budget]
        baseline = root / 'quality_baseline' / e['name']
        launch([*common, '--output', baseline, '--base-only'], out / (e['name']+'_baseline.log'))
        ref = core.read(baseline / 'RESULT.json')
        reference = out / (e['name']+'_reference.json')
        freeze(reference, dict(source_hashes={f'{i}.png': h for i, h in enumerate(e['source_hashes'])},
            fps=e['context']['video']['fps'], movie=ref['movie'], movie_sha256=ref['movie_sha256'],
            channel_uses=ref['channel_uses']))
        candidate = out / e['name']
        launch([*common, '--output', candidate, '--adapter-checkpoint', export['path']],
               out / (e['name']+'_generate.log'))
        launch([p['runtime']['python'], '-m', 'semantic_transmission.unified_bsu_evaluate',
            '--run', candidate, '--full-metrics', '--reference', reference], out / (e['name']+'_evaluate.log'))
        q = core.read(candidate / 'quality.json')
        if q.get('metric_input') != 'decoded_final_mp4':
            raise ValueError('validation must use the same decoded MP4 domain')
        rows.append(dict(name=e['name'], dataset=e['dataset'], source_sha256=e['source_sha256'],
            seconds=len(e['source_frames'])/e['context']['video']['fps'], quality=q,
            quality_path=str(candidate / 'quality.json'), quality_sha256=sha256(candidate / 'quality.json')))
    result = dict(step=step, export=export, rows=rows, **summarize_quality(rows, p['training_config']))
    write_json(out / 'RESULT.json', result)
    return result


def run(args):
    root = args.root.resolve(); root.mkdir(parents=True, exist_ok=True)
    with lock(root / 'controller.lock'):
        if not (root / 'protocol.json').exists():
            training.plan(args)
        p = training.verify_plan(root); c = p['training_config']
        if c != core.read(args.training_config):
            raise ValueError('training policy changed; choose a new root')
        if p['profile'] != core.profile(args.profile) or p['selection_sha256'] != sha256(args.selection):
            raise ValueError('training source/profile arguments changed; choose a new root')
        if p['train']['innovation_warmup_steps'] != args.codec_warmup_steps:
            raise ValueError('training warmup changed; choose a new root')
        initial = dict(path=str(args.initialize.resolve()), sha256=sha256(args.initialize)) if args.initialize else None
        if p.get('initialize') != initial:
            raise ValueError('initial checkpoint changed; choose a new root')
        if not (root / 'prepared.json').exists():
            launch([p['runtime']['python'], '-m', training.MODULE, 'prepare', '--root', root,
                '--selection', args.selection, '--profile', args.profile, '--training-config', args.training_config,
                '--codec-warmup-steps', args.codec_warmup_steps],
                root / f'prepare_{time.time_ns()}.log')
        state_path = root / 'controller.json'
        state = core.read(state_path) if state_path.exists() else dict(signature=p['signature'], history=[])
        if state['signature'] != p['signature']:
            raise ValueError('controller state differs from frozen protocol')
        for row in state['history']:
            if sha256(row['path']) != row['sha256']:
                raise ValueError('previous checkpoint quality report changed')
        if (state.get('status', '').startswith('VALIDATION_PLATEAU')
                or state.get('status') == 'MAX_UPDATES_WITHOUT_CONFIRMED_PLATEAU'):
            print(json.dumps(state, indent=2)); return
        while True:
            saved = load_checkpoint(root, p['signature'])
            step = saved['step'] if saved else 0
            last_evaluated = state['history'][-1]['step'] if state['history'] else 0
            time_cap_reached = bool(args.seconds and saved and saved['state']['elapsed_seconds'] >= args.seconds)
            if time_cap_reached and last_evaluated >= step:
                state['status'] = 'TIME_BUDGET_WITHOUT_CONFIRMED_PLATEAU'
                write_json(state_path, state); print(json.dumps(state, indent=2)); return
            # A killed quality worker leaves a committed training checkpoint;
            # finish that evaluation before advancing the optimizer again.
            pending = root / 'quality' / f'step_{step:07d}' / 'protocol.json'
            previous_result = core.read(root / 'training_result.json') if (root / 'training_result.json').exists() else {}
            stopped_at_time_cap = previous_result.get('step') == step and previous_result.get('status') == 'TIME_BUDGET_REACHED'
            needs_quality = step > last_evaluated and (step % c['quality_every'] == 0 or step >= c['max_steps']
                or pending.exists() or time_cap_reached or stopped_at_time_cap)
            if not needs_quality:
                remaining = c['quality_every']-step % c['quality_every']
                launch([p['runtime']['python'], '-m', training.MODULE, 'train', '--root', root,
                    '--seconds', args.seconds, '--stop-after', remaining], root / f'train_{time.time_ns()}.log')
                saved = load_checkpoint(root, p['signature']); step = saved['step']
            if not step:
                raise ValueError('no committed update available for validation')
            export = core.read(root / 'export.json')
            if export['step'] != step or sha256(export['path']) != export['sha256']:
                raise ValueError('export/checkpoint mismatch')
            quality = evaluate_checkpoint(root, p, export)
            monitor, improved = policy.update_monitor(state.get('quality_monitor'), quality['score'], step, c)
            state['quality_monitor'] = monitor
            report = root / 'quality' / f'step_{step:07d}' / 'RESULT.json'
            if step > last_evaluated:
                state['history'].append(dict(step=step, path=str(report), sha256=sha256(report),
                    score=quality['score'], guard_passed=quality['guard_passed']))
            if improved:
                write_json(root / 'best_candidate_quality.json', dict(export, score=quality['score'],
                    guard_passed=quality['guard_passed'], quality=str(report), quality_sha256=sha256(report),
                    selection='validation_only_long_video_quality', sufficient_training_verified=False))
            if quality['guard_passed'] and quality['score'] < state.get('best_eligible_score', float('inf')):
                state['best_eligible_score'] = quality['score']
                write_json(root / 'best_quality.json', dict(export, score=quality['score'], guard_passed=True,
                    quality=str(report), quality_sha256=sha256(report),
                    selection='validation_quality_with_per_video_guards', sufficient_training_verified=False))
            ready = policy.plateau_ready(saved['state'], c, p['train']['innovation_warmup_steps'])
            if ready and monitor['bad'] >= c['quality_patience']:
                state['status'] = ('VALIDATION_PLATEAU' if 'best_eligible_score' in state else
                                   'VALIDATION_PLATEAU_QUALITY_TARGET_NOT_MET')
            elif step >= c['max_steps']:
                state['status'] = 'MAX_UPDATES_WITHOUT_CONFIRMED_PLATEAU'
            elif args.seconds and saved['state']['elapsed_seconds'] >= args.seconds:
                state['status'] = 'TIME_BUDGET_WITHOUT_CONFIRMED_PLATEAU'
            else:
                state['status'] = 'RUNNING'
            state.update(step=step, sufficient_training_verified=False,
                         performance_improvement_proven=False, independent_semantic_review=False)
            write_json(state_path, state)
            if state['status'] != 'RUNNING':
                print(json.dumps(state, indent=2)); return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'run', 'status'], nargs='?', default='status')
    parser.add_argument('--root', type=Path, default=core.REPO / 'outputs/bsu_unified_etri_training_v2')
    parser.add_argument('--selection', type=Path, default=training.DEFAULT_SELECTION)
    parser.add_argument('--profile', type=Path, default=core.REPO / 'configs/bsu_unified_sender.json')
    parser.add_argument('--training-config', type=Path, default=core.REPO / 'configs/bsu_unified_training_etri.json')
    parser.add_argument('--seconds', type=int, default=0, help='cumulative training-only cap; 0 uses validation plateau/update cap')
    parser.add_argument('--codec-warmup-steps', type=int, default=200)
    parser.add_argument('--initialize', type=Path)
    args = parser.parse_args(); args.smoke = False; args.budget_per_frame = 768
    if args.seconds < 0 or args.codec_warmup_steps < 0:
        parser.error('negative time/update budget')
    if args.action == 'plan':
        training.plan(args)
    elif args.action == 'run':
        run(args)
    else:
        print(json.dumps({name: core.read(args.root / name) for name in
            ('plan.json', 'prepared.json', 'latest.json', 'best_loss.json', 'best_candidate_quality.json',
             'best_quality.json', 'controller.json')
            if (args.root / name).exists()}, indent=2))


if __name__ == '__main__':
    main()
