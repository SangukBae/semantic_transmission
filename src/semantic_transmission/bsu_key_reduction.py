"""Opt-in, resumable BSU key removal trial on the frozen Kinetics-400 cohort.

Motion ranks candidates at the sender. Actual complete reconstructions determine
whether a deletion survives quality guards; a second generation seed confirms
the chosen subset. Production defaults and historical artifacts are immutable.
"""
import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

from .artifacts import sha256, write_json
from .webvid5 import fingerprint
from .webvid_ablation import Stages
from .qwen35_pairs_video import freeze
from .key_reduction_selection import CRITERIA, PARAMETERS, guard, proposals, subset_packet

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO/'outputs/bsu_key_reduction_20261015_v2'
BASE = REPO/'outputs/bsu_kinetics_context'
MODULE = 'semantic_transmission.bsu_key_reduction'
read = lambda path: json.loads(Path(path).read_text())


def event(**values):
    if 'done' in values:
        values['overall'] = f"{values['done']}/{values['total']} ({100*values['done']/values['total']:.1f}%) 영상 검증 완료"
    print('COMPARE_PROGRESS '+json.dumps(values, ensure_ascii=False), flush=True)


def link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(Path(source).resolve())


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['path', 'text', 'flow'])
        writer.writeheader(); writer.writerows(rows)


def canonical_asset(record, expected):
    """Accept a path alias only for the identical pinned asset and policy."""
    value = copy.deepcopy(record)
    if Path(value['path']).resolve(strict=True) != Path(expected['path']).resolve(strict=True):
        raise ValueError('caption adapter points to a different asset')
    value['path'] = expected['path']
    if value != expected:
        raise ValueError('caption adapter policy changed')
    for name, digest in value['files'].items():
        if sha256(Path(value['path'])/name) != digest:
            raise ValueError('caption adapter bytes changed')
    return value


def caption_config(base):
    from .qwen35_caption_adapter import default_adapter
    cfg = read(Path(base)/'plan.json')['caption']
    cfg['caption_adapter'] = canonical_asset(cfg['caption_adapter'], default_adapter())
    return cfg


def prepare(root):
    from .lgvsc_kinetics import verify_bsu
    plan, result, items = verify_bsu(BASE)
    settings = read(REPO/'.local/settings.json')
    code = dict(plan['code'])
    # Freeze all package modules: includes transitive receiver and evaluator
    # helpers, not only this controller. Never overwrite historical code.
    code.update({str(p.relative_to(REPO)): sha256(p)
                 for p in (REPO/'src/semantic_transmission').glob('*.py')})
    code['scripts/experiment_bsu_key_reduction.sh'] = sha256(REPO/'scripts/experiment_bsu_key_reduction.sh')
    cases = []
    for video, item in zip(plan['videos'], items):
        base = Path(item['base']); run = base/'reconstruction/event_pairs/run'
        caption_config(base)
        keys = read(run/'keyframes.json')['indices']
        cases.append(dict(id=video['id'], source=read(video['prepared']),
            original_keys=keys, original_run=str(run), original_base=str(base),
            original_movie=item['video'], original_channel_uses=item['total_channel_uses'],
            original_trajectory=str(Path(item['video']).parent.parent/'received/trajectory.bin'),
            original_result_sha256=sha256(BASE/'benchmark/snr_10'/video['id']/'RESULT.json')))
    value = dict(policy='bsu_motion_ranked_reconstruction_key_removal_v1',
        criteria=CRITERIA, parameters=PARAMETERS, code=code, cases=cases,
        source_result_sha256=sha256(BASE/'RESULT.json'), source_plan_sha256=sha256(BASE/'benchmark/protocol.json'),
        seed_selection=2025, seed_confirmation=2026, snr_db=10, steps=30, receiver='combined_context',
        runtimes=dict(python=settings['python'], channel=settings['channel_python'],
                      qwen=str(REPO/'.local/qwen35_env/bin/python'), action=plan['action_runtime']['python']),
        action_runtime=plan['action_runtime'],
        decision='try 20 percent of eligible interior keys, then only first ranked deletion on failure; '
                 'confirm one selected subset with seed 2026; no further search on confirmation failure',
        accounting='exact retained codec symbols and rate bytes; new captions, flows and paid trajectories; '
                   '16QAM LDPC including padding; trajectory cap 5 percent of reduced base cost',
        original_reuse='verified completed seed2025 baseline; every source and output artifact checked',
        channel_pairing='identical retained TX and RX visual symbols and decoded RGB keys; metadata actually retransmitted',
        noise_pairing='same scoped RNG seed and policy; different segment counts/shapes are NOT identical noise tensors',
        metric_boundary='delivered MP4 against normalized source; original key exclusion and boundaries fixed for both arms',
        selection_scope='14 previously seen development sources; seed confirmation is not unseen-source validation',
        action_scope='independent TimeSformer evaluation after key decisions, labels never used for selection',
        production_defaults_changed=False, independent_semantic_review='PENDING')
    value['signature'] = fingerprint(value)
    freeze(root/'protocol.json', value)
    return value


def verify_protocol(p):
    if fingerprint({k: v for k, v in p.items() if k != 'signature'}) != p['signature']:
        raise ValueError('protocol signature changed')
    for name, digest in p['code'].items():
        if sha256(REPO/name) != digest:
            raise ValueError('experiment code changed: '+name)


def execute_worker(p, stage, job_path, log):
    runtime = p['runtimes']['qwen' if stage == 'caption' else 'channel' if stage == 'channel' else 'action' if stage == 'action' else 'python']
    env = dict(os.environ, PYTHONPATH=str(REPO/'src'), PYTHONNOUSERSITE='1',
               PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    if stage == 'channel':
        env['CUDA_VISIBLE_DEVICES'] = '-1'
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log.open('x') as f:
        completed = subprocess.run([runtime, '-m', MODULE, '--worker', stage, '--job', str(job_path)],
                                   cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT)
    if completed.returncode:
        print(log.read_text(errors='replace')[-10000:], flush=True)
        raise RuntimeError(f'{stage} failed ({completed.returncode}): {log}')
    print(f'WORKER_DONE {stage} seconds={time.monotonic()-started:.1f}', flush=True)


def initialize(j):
    run = Path(j['run']); old = Path(j['case']['original_run']); run.mkdir(parents=True, exist_ok=True)
    cfg = read(old/'run_config.json'); cfg['seed'] = j['seed']
    write_json(run/'run_config.json', cfg); write_json(run/'keyframes.json', dict(indices=j['keys']))
    link(old/'data/normalized.mp4', run/'data/normalized.mp4')
    link(old/'data/frames/sample', run/'data/frames/sample')


def captions(j):
    from PIL import Image
    from . import auto_extraction as ae, qwen35_pairs_comparison as pc
    from .qwen35_backend import LocalQwen35
    run = Path(j['run']); old = Path(j['case']['original_run']); keys = j['keys']
    original_keys = j['case']['original_keys']
    prior = dict(zip(zip(original_keys, original_keys[1:]), read(old/'metadata_tx.json')))
    cfg = caption_config(j['case']['original_base'])
    source = j['case']['source']; rows = []; provenance = []; model = None
    for n, (a, b) in enumerate(zip(keys, keys[1:])):
        if (a, b) in prior:
            text = prior[a, b]['text']; proof = dict(reused_interval=[a, b])
        else:
            if model is None:
                model = LocalQwen35(cfg)
            ids = pc.sample_indices(a, b); paths = [ae.check_frame(source, i) for i in ids]
            identity = dict(protocol=j['protocol'], id=j['case']['id'], start=a, end_exclusive=b,
                            source_indices=ids, frame_hashes=[sha256(path) for path in paths])
            labels = [f'Source frame {i}, time {i/24:.6f} seconds' for i in ids]
            images = [Image.open(path).convert('RGB') for path in paths]
            cache = Path(j['cache'])/f'{a}_{b}'
            try:
                draft = pc.infer(model, cache/'draft.json', dict(identity, task='draft'), images, labels,
                    cfg['prompts']['caption'], ae.caption_validator(cfg))
                review = pc.infer(model, cache/'review.json', dict(identity, task='visual_review',
                    draft_sha256=ae.digest(draft)), images, labels, cfg['prompts']['review']+draft['caption'],
                    ae.caption_validator(cfg, True))
            finally:
                for image in images:
                    image.close()
            text = review['caption']; proof = dict(identity=identity, calls={str(x): sha256(x) for x in cache.glob('*.json')})
        rows.append(dict(path=f'clips/sample/{n:05d}.mp4', text=text, flow=0.))
        provenance.append(proof)
    write_json(run/'captions.json', rows); write_json(run/'caption_provenance.json', provenance)


def flow(j):
    from .workers import flow as measure
    run = Path(j['run']); old = Path(j['case']['original_run'])
    measure(read(run/'run_config.json'), REPO, run)
    rows = read(run/'metadata_tx.json'); original_keys = j['case']['original_keys']
    prior = dict(zip(zip(original_keys, original_keys[1:]), read(old/'metadata_tx.json')))
    differences = []
    for row, interval in zip(rows, zip(j['keys'], j['keys'][1:])):
        if interval in prior:
            differences.append(abs(row['flow']-prior[interval]['flow']))
            row['flow'] = prior[interval]['flow']
    # Unmerged intervals must carry byte-identical text and flow values. Only
    # paths change to reflect the new receiver segment indices.
    write_json(run/'metadata_tx.json', rows)
    write_json(run/'flow_reuse.json', dict(unchanged_scores_reused=True,
        recomputed_unchanged_max_difference=max(differences, default=0.)))


def packet(j):
    import cv2
    import numpy as np
    from PIL import Image
    from .wire import pack, unpack
    from .packets import accounting
    from .bsu_sequence_refinement import sender_candidates, choose_packet
    run = Path(j['run']); old = Path(j['case']['original_run']); tx = run/'transmitter'; tx.mkdir()
    if j.get('reuse_conditions'):
        condition = Path(j['reuse_conditions'])
        rows = read(condition/'metadata_tx.json')
        write_json(run/'metadata_tx.json', rows)
    else:
        rows = read(run/'metadata_tx.json')
    raw, symbols = subset_packet((old/'transmitter/metadata.bin').read_bytes(),
        (old/'transmitter/visual.c64').read_bytes(), j['keys'], rows, j['seed'])
    (tx/'metadata.bin').write_bytes(raw); (tx/'visual.c64').write_bytes(symbols)
    header, _ = unpack(raw)
    base_uses = len(symbols)//8 + accounting(len(raw), bits_per_symbol=4)['complex_channel_uses']
    maximum = math.floor(base_uses*.05)
    reuse = (Path(j['reuse_conditions'])/'transmitter/trajectory.bin') if j.get('reuse_conditions') else None
    if j['keys'] == j['case']['original_keys']:
        reuse = Path(j['case']['original_trajectory'])
    if reuse is not None:
        trajectory = None
        if reuse.exists():
            h, payload = unpack(reuse.read_bytes()); h['base_metadata_sha256'] = sha256(tx/'metadata.bin')
            trajectory = pack(h, payload)
        audit = dict(reused_geometry=True)
    else:
        cv2.setNumThreads(2); cv2.setRNGSeed(2025); candidates = []
        for a, b in zip(j['keys'], j['keys'][1:]):
            frames = []
            for i in range(a, b+1):
                with Image.open(run/f'data/frames/sample/{i}.png') as image:
                    frames.append(cv2.resize(np.asarray(image.convert('RGB')), (288, 160), interpolation=cv2.INTER_AREA))
            candidates.append(sender_candidates(np.stack(frames), a, b))
        trajectory, audit = choose_packet(candidates, header['video'], sha256(tx/'metadata.bin'), maximum, 4)
    extra = accounting(len(trajectory), bits_per_symbol=4)['complex_channel_uses'] if trajectory else 0
    if extra > maximum:
        raise ValueError('trajectory exceeded reduced-base cap')
    if trajectory:
        (tx/'trajectory.bin').write_bytes(trajectory)
    write_json(tx/'accounting.json', dict(base_channel_uses=base_uses, trajectory_channel_uses=extra,
        channel_uses=base_uses+extra, maximum_trajectory_uses=maximum,
        retained_codec_symbols_exact=True, audit=audit))


def channel(j):
    from .metadata_channel import transmit
    from .wire import unpack
    from .bsu_sequence_refinement import decode_packet
    from .transmission_accounting import channel_breakdown
    from . import generation_noise as noise, tail_reference as tail, precision_text_cache as cache
    from . import condition_collision as collision, short_schedule
    run = Path(j['run']); old = Path(j['case']['original_run'])
    rx = run/'received'; rx.mkdir(); reports = {}
    for name, seed in [('metadata', 42), ('trajectory', 43)]:
        path = run/f'transmitter/{name}.bin'
        if not path.exists():
            continue
        original = path.read_bytes(); restored, report = transmit(original, 10, seed, bits_per_symbol=4)
        if restored != original or report['bit_errors']:
            raise ValueError('digital channel was not exact')
        (rx/f'{name}.bin').write_bytes(restored); reports[name] = report
    header, _ = unpack((rx/'metadata.bin').read_bytes())
    if (rx/'trajectory.bin').exists():
        decode_packet((rx/'trajectory.bin').read_bytes(), (rx/'metadata.bin').read_bytes())
    # Frame-seeded AWGN and independently decoded keys permit exact cache reuse.
    _, retained = subset_packet((old/'received/metadata.bin').read_bytes(),
        (old/'received/visual.c64').read_bytes(), j['keys'], header['segments'], j['seed'])
    (rx/'visual.c64').write_bytes(retained)
    receiver = run/'receiver'; receiver.mkdir()
    for k in j['keys']:
        link(old/f'receiver/frames/sample/key_frames_received/{k}.png', receiver/f'frames/sample/key_frames_received/{k}.png')
    write_csv(receiver/'metadata.csv', header['segments'])
    write_json(receiver/'decoder_inputs.json', dict(video=header['video'], decoder=header['decoder'], indices=j['keys']))
    contract = dict(policy=noise.POLICY, seed=j['seed'], segments=len(j['keys'])-1, steps=30,
        inputs=fingerprint({n: sha256(run/n) for n in ['received/metadata.bin', 'received/visual.c64']}),
        code=j['protocol'], reference_policy=tail.POLICY)
    write_json(run/'receiver_policy.json', dict(policy=tail.POLICY, reference_frames=17, reference_latents=5,
        alignment=5, text_encoder_policy=cache.POLICY, t5_precision='fp32', noise_contract=contract,
        noise_reference=None, condition_collision_policy=collision.POLICY, short_schedule_policy=short_schedule.POLICY))
    ledger = channel_breakdown((rx/'metadata.bin').read_bytes(), len(retained)//8, header['video'], reports['metadata'])
    counts = read(run/'transmitter/accounting.json')
    actual = ledger['complex_channel_uses']['total'] + reports.get('trajectory', {}).get('complex_channel_uses', 0)
    if actual != counts['channel_uses']:
        raise ValueError('packet accounting differs from actual channel')
    write_json(run/'channel_accounting.json', dict(reports=reports, ledger=ledger,
        channel_uses=actual, retained_received_symbols_and_rgb_exact=True))


def reconstruct(j):
    from .fc_lgvsc_qwen35 import reconstruct as decode
    from . import tail_reference, precision_text_cache, generation_noise, short_schedule, condition_collision
    run = Path(j['run']); decode(run)
    precision_text_cache.validate_usage(run)
    tail_reference.validate_trace(read(run/'receiver/tail_reference_trace.json'), len(j['keys'])-2)
    generation_noise.validate_trace(read(run/generation_noise.TRACE), read(run/'receiver_policy.json')['noise_contract'],
                                    len(j['keys'])-1, 30)
    short_schedule.validate_trace(read(run/short_schedule.TRACE), j['keys'], 30)
    condition_collision.validate_trace(read(run/condition_collision.TRACE),
        condition_collision.preview(j['keys'], condition_collision.upstream_functions(REPO)))


def refine(j):
    from .bsu_sequence_refinement import decode_packet
    from .bsu_receiver import refine_windows, encode_frames
    from .wire import unpack
    run = Path(j['run']); output = Path(j['output'])/'refined'
    packet = (run/'received/metadata.bin').read_bytes(); h, _ = unpack(packet)
    trajectory = run/'received/trajectory.bin'
    tracks = decode_packet(trajectory.read_bytes(), packet) if trajectory.exists() else {}
    paths = sorted((run/'receiver/reconstruction/sample_0000_frames').glob('*.png'))
    if len(paths) != h['video']['frames']:
        raise ValueError('incomplete generated timeline')
    audit = refine_windows(paths, j['keys'], run/'receiver/frames/sample/key_frames_received', tracks,
                           output/'frames', mode='combined_context')
    encode_frames(output/'frames', output/'reconstruction.mp4', len(paths), h['video']['fps'])
    write_json(output/'audit.json', dict(windows=audit, receiver_reads_source=False))


def evaluate(j):
    import cv2
    import numpy as np
    from PIL import Image
    from .official_quality import OfficialMetrics, summarize
    from .research_quality import read_video
    from .stutter_metrics import evaluate as temporal
    from .bsu_receiver_compare import boundary_diagnostics
    from .etri_60s_check import pts_audit
    cv2.setNumThreads(2)
    case = j['case']; root = Path(j['output']); old = j.get('baseline_reuse', False)
    movie = Path(case['original_movie']) if old else root/'refined/reconstruction.mp4'
    source = read_video(case['source']['normalized_video']); generated = read_video(movie)
    pts_audit(movie, case['source']['frames'], 24)
    quality, rows = OfficialMetrics().evaluate(source, generated)
    fixed = set(case['original_keys']); nonkeys = [r for r in rows if r['frame'] not in fixed]
    boundaries = boundary_diagnostics(source, generated, case['original_keys'])
    if not old:
        for k in j['keys']:
            a = np.asarray(Image.open(root/f'refined/frames/{k:05d}.png').convert('RGB'))
            b = np.asarray(Image.open(Path(j['run'])/f'receiver/frames/sample/key_frames_received/{k}.png').convert('RGB'))
            if not np.array_equal(a, b):
                raise ValueError('received endpoint RGB changed')
    counts = case['original_channel_uses'] if old else read(Path(j['run'])/'channel_accounting.json')['channel_uses']
    value = dict(quality=quality, nonkeys=summarize(nonkeys, (len(nonkeys), 320, 576, 3)),
        temporal=temporal(source, generated), boundaries=boundaries,
        boundary_error=statistics.mean(x['mean_change_error'] for x in boundaries),
        channel_uses=counts, keys=j['keys'], movie=str(movie), movie_sha256=sha256(movie),
        seed=j['seed'], fixed_nonkey_indices=[r['frame'] for r in nonkeys],
        source_movie_sha256=sha256(case['source']['normalized_video']))
    write_json(root/'quality.json', value)
    with (root/'quality.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def trial(root, p, case, name, keys, seed, reuse_conditions=None, baseline_reuse=False):
    out = root/case['id']/name; run = out/'run'
    j = dict(protocol=p['signature'], protocol_path=str(root/'protocol.json'), case=case,
        output=str(out), run=str(run), keys=keys, seed=seed, cache=str(root/case['id']/'caption_cache'),
        reuse_conditions=str(reuse_conditions) if reuse_conditions else None, baseline_reuse=baseline_reuse)
    freeze(out/'job.json', j); job = out/'job.json'; stages = Stages(out, fingerprint(j))
    def step(name, owned, required=None, callback=None):
        event(current=f'{case["id"]}/{out.name}', stage=name)
        stages.step(name, owned, required or owned,
                    callback or (lambda log: execute_worker(p, name, job, log)))
    if not baseline_reuse:
        step('initialize', ['run'], ['run/run_config.json', 'run/keyframes.json', 'run/data/normalized.mp4', 'run/data/frames/sample'],
             lambda _: initialize(j))
        if not reuse_conditions:
            step('caption', ['run/captions.json', 'run/caption_provenance.json'])
            step('flow', ['run/metadata_tx.json', 'run/flow_reuse.json', 'run/flow_sampling.json',
                          'run/semantic_clips_audit.json', 'run/data/clips'])
        step('packet', ['run/transmitter'] + (['run/metadata_tx.json'] if reuse_conditions else []))
        step('channel', ['run/received', 'run/receiver', 'run/receiver_policy.json', 'run/channel_accounting.json'],
             ['run/received', 'run/receiver/frames', 'run/receiver/metadata.csv', 'run/receiver/decoder_inputs.json',
              'run/receiver_policy.json', 'run/channel_accounting.json'])
        from . import precision_text_cache as cache
        step('text', ['run/'+cache.REPORT])
        # Explicit required receiver inputs above exclude subsequently appended
        # generation outputs, preserving the earlier receipt during resume.
        from . import generation_noise, condition_collision, short_schedule, decoder_memory
        traces = ['receiver/reference_trace.json', 'receiver/tail_reference_trace.json', cache.TRACE,
                  generation_noise.TRACE, condition_collision.TRACE, short_schedule.TRACE, decoder_memory.TRACE,
                  'receiver/decoder_config.py']
        step('reconstruct', ['run/receiver/reconstruction', *['run/'+n for n in traces]])
        step('refine', ['refined'])
    step('evaluate', ['quality.json', 'quality.csv'])
    return read(out/'quality.json')


def write_action_manifest(path, rows, separator, num_classes=400):
    """Match Kinetics' literal split parser; CSV quoting is not supported."""
    if not rows or not separator or any(c in separator for c in '\r\n'):
        raise ValueError('nonempty action rows and a valid path-label separator required')
    lines = []
    for row in rows:
        video, label = str(row['path']), row['label']
        if not video or separator in video or any(c in video for c in '\r\n'):
            raise ValueError('video path cannot be represented by the TimeSformer split parser')
        if type(label) is not int or not 0 <= label < num_classes:
            raise ValueError('action label outside the classifier class range')
        lines.append(f'{video}{separator}{label}\n')
    # Validate all rows before changing an existing manifest.
    Path(path).write_text(''.join(lines), encoding='utf-8')


def action(j):
    from . import kinetics_timesformer as base
    import torch
    base.validate_runtime(j['runtime']); directory = Path(j['output']); directory.mkdir(parents=True, exist_ok=True)
    for row in j['rows']:
        if sha256(row['path']) != row['sha256']:
            raise ValueError('action input changed')
    cfg = base.configure(j['runtime'], directory)
    write_action_manifest(directory/'test.csv', j['rows'], cfg.DATA.PATH_LABEL_SEPARATOR, cfg.MODEL.NUM_CLASSES)
    from timesformer.datasets.kinetics import Kinetics
    dataset = Kinetics(cfg, 'test'); torch.manual_seed(2025)
    model = base.load_model(j['runtime'], 'cuda'); records = []
    for i, row in enumerate(j['rows']):
        logits, ranking = base.classify(model, dataset, i, row['label'], 'cuda')
        records.append(dict(input=row, top5=ranking, summed_logits=logits,
            top1_correct=ranking[0] == row['label'], top5_correct=row['label'] in ranking))
    summaries = {}
    for arm in sorted({r['variant'] for r in j['rows']}):
        group = [r for r in records if r['input']['variant'] == arm]
        summaries[arm] = dict(videos=len(group), top1_correct=sum(r['top1_correct'] for r in group),
                             top5_correct=sum(r['top5_correct'] for r in group))
    write_json(directory/'RESULT.json', dict(records=records, summary=summaries, labels_used_for_key_selection=False))


def run_experiment(root):
    from .bsu_receiver_compare import lock
    from .etri_60s_check import lock as gpu_lock
    from .auto_extraction import unseal
    import numpy as np
    from PIL import Image
    with lock(root), gpu_lock(REPO/'.local/etri_60s_check.lock'):
        p = prepare(root); decisions = []; baselines = {}; final = {}; confirmations = []
        for ci, case in enumerate(p['cases']):
            event(done=ci, total=len(p['cases']), current=case['id'], stage='송신 후보 선택')
            directory = root/case['id']; directory.mkdir(exist_ok=True)
            selection = directory/'selection.json'
            def select(_):
                def frame(i):
                    with Image.open(Path(case['source']['source_frames'])/f'{i}.png') as image:
                        return np.asarray(image.convert('RGB'))
                saved = unseal(Path(case['original_base'])/'selection.json')
                write_json(selection, proposals(case['original_keys'], saved['decisions'], frame))
            Stages(directory, p['signature']).step('selection', ['selection.json'], ['selection.json'], select)
            choice = read(selection)
            baseline = trial(root, p, case, 'baseline_2025', case['original_keys'], 2025, baseline_reuse=True)
            baselines[case['id']] = baseline; final[case['id']] = baseline
            record = dict(id=case['id'], original_keys=case['original_keys'], trials=[], accepted=False)
            selected_name = None; selected_keys = None
            variants = [('motion20', choice['aggressive']), ('single', choice['single'])]
            seen = {tuple(case['original_keys'])}
            for name, keys in variants:
                if tuple(keys) in seen:
                    continue
                seen.add(tuple(keys))
                quality = trial(root, p, case, name+'_2025', keys, 2025)
                verdict = guard(baseline, quality)
                record['trials'].append(dict(name=name, keys=keys, verdict=verdict, quality=quality))
                if verdict['passed']:
                    selected_name, selected_keys = name, keys
                    break
            if selected_name:
                # Confirm the entire selected configuration. Never adapt again
                # to confirmation noise or classifier target labels.
                old = Path(case['original_run'])
                a = trial(root, p, case, 'baseline_2026', case['original_keys'], 2026, reuse_conditions=old)
                prior = root/case['id']/(selected_name+'_2025')/'run'
                b = trial(root, p, case, selected_name+'_2026', selected_keys, 2026, reuse_conditions=prior)
                verdict = guard(a, b)
                record['confirmation'] = dict(baseline=a, candidate=b, verdict=verdict)
                record['accepted'] = verdict['passed']
                confirmations.append(dict(id=case['id'], baseline=a, candidate=b))
                if verdict['passed']:
                    final[case['id']] = record['trials'][-1]['quality']
                    record['accepted_keys'] = selected_keys
            decisions.append(record); write_json(directory/'decision.json', record)
        # Action labels become available only after all removal decisions.
        videos = read(BASE/'benchmark/protocol.json')['videos']; labels = {v['id']: v['label'] for v in videos}
        rows = []
        for case in p['cases']:
            for variant, value in [('baseline_2025', baselines[case['id']]), ('selected_2025', final[case['id']])]:
                rows.append(dict(id=case['id'], variant=variant, label=labels[case['id']],
                                 path=value['movie'], sha256=value['movie_sha256']))
        for pair in confirmations:
            for variant in ['baseline', 'candidate']:
                value = pair[variant]
                rows.append(dict(id=pair['id'], variant=variant+'_2026', label=labels[pair['id']],
                                 path=value['movie'], sha256=value['movie_sha256']))
        job = dict(runtime=p['action_runtime'], rows=rows, output=str(root/'action'))
        freeze(root/'action_job.json', job)
        Stages(root, p['signature']).step('action', ['action'], ['action/RESULT.json'],
            lambda log: execute_worker(p, 'action', root/'action_job.json', log))
        def aggregate(values):
            values = list(values)
            return dict(videos=len(values), keys=sum(len(v['keys']) for v in values),
                channel_uses=sum(v['channel_uses'] for v in values),
                quality={k: statistics.mean(v['quality'][k] for v in values)
                         for k in ['psnr_db', 'ssim', 'lpips_vgg', 'clip', 'dists']},
                temporal={k: statistics.mean(v['temporal'][k] for v in values)
                          for k in ['flow_error_mean', 'cadence_error_mean', 'false_freeze_rate']})
        a, b = aggregate(baselines.values()), aggregate(final.values())
        result = dict(status='COMPLETE_KEY_REDUCTION_EXPERIMENT', signature=p['signature'],
            baseline=a, selected=b, accepted_videos=sum(d['accepted'] for d in decisions),
            saved_channel_uses=a['channel_uses']-b['channel_uses'],
            saved_percent=100*(1-b['channel_uses']/a['channel_uses']), decisions=decisions,
            action=read(root/'action/RESULT.json')['summary'],
            limitations=[p['selection_scope'], p['noise_pairing'], 'temporal metrics are optical-flow proxies',
                        'independent semantic review pending; no production promotion'],
            production_defaults_changed=False)
        write_json(root/'RESULT.json', result)
        write_json(root/'completion_receipt.json', dict(status='PASSED', protocol_sha256=sha256(root/'protocol.json'),
            artifacts={str(x.relative_to(root)): sha256(x) for x in root.rglob('*.json')
                       if 'failed_attempts' not in x.parts and x.name != 'completion_receipt.json'}))
        event(done=len(p['cases']), total=len(p['cases']), stage='전체 실험 완료')
        print(json.dumps({k: v for k, v in result.items() if k != 'decisions'}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--worker', choices=['caption', 'flow', 'packet', 'channel', 'text', 'reconstruct', 'refine', 'evaluate', 'action'])
    parser.add_argument('--job', type=Path)
    args = parser.parse_args()
    if not args.worker:
        run_experiment(args.root.resolve()); return
    j = read(args.job)
    if args.worker == 'action':
        action(j); return
    verify_protocol(read(j['protocol_path']))
    if args.worker == 'text':
        from .precision_text_cache import prepare as prepare_text
        from .tail_reference import decoder_config
        run = Path(j['run']); prepare_text(run, REPO, decoder_config(run), 'fp32')
    else:
        globals()[{'caption': 'captions'}.get(args.worker, args.worker)](j)


if __name__ == '__main__':
    main()
