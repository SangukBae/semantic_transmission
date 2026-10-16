"""Matched full BSU reconstruction after paid DCVC key-image transmission."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .artifacts import sha256, write_json
from .webvid5 import fingerprint
from .webvid_ablation import Stages
from .qwen35_pairs_video import freeze
from .key_reduction_selection import CRITERIA, guard
from . import bsu_key_reduction as legacy

REPO = Path(__file__).resolve().parents[2]
MODULE = 'semantic_transmission.bsu_dcvc_validation'
read = lambda p: json.loads(Path(p).read_text())


def portable_text_report(old, run):
    """Rebind moved cache inventory only after hashing weights and exact tensors."""
    from . import precision_text_cache as cache, text_embedding_cache as base
    report = read(old/cache.REPORT)
    original_signature = report['signature']
    contract = report['contract']
    if fingerprint(contract) != original_signature:
        raise ValueError('original text preparation signature changed')
    inventory = base.model_inventory(contract['model_path'])
    if set(inventory) != set(contract['model_inventory']):
        raise ValueError('T5 file set changed')
    for name, digest in contract['model_sha256'].items():
        if sha256(Path(contract['model_path'])/name) != digest:
            raise ValueError('T5 weight/tokenizer bytes changed')
    for name, digest in contract['code'].items():
        if sha256(REPO/name) != digest:
            raise ValueError('T5 preparation code changed')
    for entry in report['entries']:
        cache.load(entry, contract, entry['prompt'])
    contract['model_inventory'] = inventory
    report['signature'] = fingerprint(contract)
    for entry in report['entries']:
        entry['contract'] = report['signature']
    report['inventory_rebinding'] = dict(original_report=str(old/cache.REPORT),
        original_sha256=sha256(old/cache.REPORT), original_signature=original_signature,
        reason='cross-machine file targets and ctimes differ; all model and embedding bytes verified',
        tensors_recomputed=False, tensors_changed=False)
    write_json(run/cache.REPORT, report)


def worker(j, stage):
    if stage == 'action':
        legacy.action(j)
        return
    run, old = Path(j['run']), Path(j['case']['original_run'])
    if stage == 'channel':
        from .metadata_channel import transmit
        from .lossless_transport import transmit as side_transmit
        from .bsu_lossless_validation import audit_case
        screen = Path(j['screen'])
        source = (screen/'keys.bin').read_bytes()
        if sha256(screen/'keys.bin') != j['screen_record']['packet_sha256']:
            raise ValueError('screened key packet changed')
        case = old.parents[3]
        baseline_row = next(v for v in read(j['baseline_result'])['videos'] if v['id'] == j['case']['id'])
        metadata, trajectory, _ = audit_case(case, baseline_row)
        run.mkdir(parents=True, exist_ok=True)
        rx = run/'received'; rx.mkdir(parents=True, exist_ok=True)
        reports = []
        for seed in [42, 1042, 2042]:
            packet, report = transmit(source, 10, seed, bits_per_symbol=4)
            if packet != source or report['bit_errors']:
                raise ValueError('key packet failed channel integrity')
            restored, side = side_transmit(metadata, trajectory, 10, seed+1, method='xz', bundle=True,
                                           bits_per_symbol=4)
            count = report['complex_channel_uses'] + side['complex_channel_uses']
            if count != j['screen_record']['total_uses']:
                raise ValueError('actual channel cost differs from screening')
            reports.append(dict(seed=seed, keys=report, side=side, channel_uses=count))
            if seed == 42:
                (rx/'keys.bin').write_bytes(packet)
                for name, data in restored.items():
                    (rx/f'{name}.bin').write_bytes(data)
        write_json(run/'channel_accounting.json', dict(channel_uses=count, reports=reports,
            successful_channel_seeds=3, all_payload_bytes_exact=True, retransmission_uses=0))
    elif stage == 'decode':
        from PIL import Image
        from .bsu_dcvc_keys import Codec
        codec = Codec(REPO/'.local/vendor/DCVC-RT', REPO/'.local/checkpoints/dcvc_rt')
        keys, images = codec.decode((run/'received/keys.bin').read_bytes())
        if keys != j['keys']:
            raise ValueError('received keys changed')
        dest = run/'receiver/frames/sample/key_frames_received'; dest.mkdir(parents=True)
        for k, image in zip(keys, images):
            Image.fromarray(image).save(dest/f'{k}.png')
            if sha256(dest/f'{k}.png') != j['screen_record']['decoded_hashes'][str(k)]:
                raise ValueError('independent receiver decode differs from screen')
    elif stage == 'initialize':
        from . import generation_noise as noise, precision_text_cache as cache
        from .wire import unpack
        cfg = read(old/'run_config.json'); cfg['seed'] = j['seed']
        write_json(run/'run_config.json', cfg)
        write_json(run/'keyframes.json', dict(indices=j['keys']))
        h, _ = unpack((run/'received/metadata.bin').read_bytes())
        legacy.write_csv(run/'receiver/metadata.csv', h['segments'])
        decoder = copy.deepcopy(h['decoder']); decoder['seed'] = j['seed']
        write_json(run/'receiver/decoder_inputs.json', dict(video=h['video'], decoder=decoder, indices=j['keys']))
        # Original FP32 embeddings are reused only for byte-identical captions/flow.
        if sha256(run/'receiver/metadata.csv') != sha256(old/'receiver/metadata.csv'):
            raise ValueError('caption/flow changed')
        portable_text_report(old, run)
        policy = read(old/'receiver_policy.json')
        policy['noise_contract'].update(seed=j['seed'], code=j['protocol'],
            inputs=fingerprint({p:sha256(run/p) for p in ['received/metadata.bin', 'received/keys.bin']}))
        policy['noise_reference'] = None
        write_json(run/'receiver_policy.json', policy)
        legacy.link(old/'data/normalized.mp4', run/'data/normalized.mp4')
    elif stage == 'reconstruct':
        legacy.reconstruct(j)
    elif stage == 'refine':
        legacy.refine(j)
    elif stage == 'evaluate':
        legacy.evaluate(j)
    else:
        raise ValueError(stage)


def execute(p, stage, job, log):
    key = 'channel' if stage == 'channel' else 'codec' if stage == 'decode' else 'action' if stage == 'action' else 'python'
    env = dict(os.environ, PYTHONPATH=str(REPO/'src'), PYTHONNOUSERSITE='1',
               PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    if stage == 'channel':
        env['CUDA_VISIBLE_DEVICES'] = '-1'
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('x') as stream:
        proc = subprocess.run([p['runtimes'][key], '-m', p.get('worker_module', MODULE), '--worker', stage, '--job', str(job)],
                              cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if proc.returncode:
        print(log.read_text(errors='replace')[-8000:], flush=True)
        raise RuntimeError(f'{stage} failed ({proc.returncode}): {log}')


def trial(root, p, case, candidate, seed, original=False):
    out = root/case['id']/f'{"baseline" if original else "dcvc"}_{seed}'
    run = out/'run'
    j = dict(protocol=p['signature'], case=case, keys=case['original_keys'], seed=seed,
             output=str(out), run=str(run), screen=candidate['directory'],
             screen_record=candidate['record'], baseline_result=p['baseline_result'])
    out.mkdir(parents=True, exist_ok=True)
    freeze(out/'job.json', j)
    stages = Stages(out, fingerprint(j))
    if original:
        # A matched original seed-2026 reconstruction, with all received keys unchanged.
        def prepare(_):
            old = Path(case['original_run']); run.mkdir(parents=True, exist_ok=True)
            for path in ['received/metadata.bin', 'receiver/frames', 'receiver/metadata.csv',
                         'data/normalized.mp4']:
                legacy.link(old/path, run/path)
            portable_text_report(old, run)
            trajectory = Path(case['original_trajectory'])
            if trajectory.exists():
                legacy.link(trajectory, run/'received/trajectory.bin')
            cfg = read(old/'run_config.json'); cfg['seed'] = seed
            write_json(run/'run_config.json', cfg); write_json(run/'keyframes.json', dict(indices=j['keys']))
            inputs = read(old/'receiver/decoder_inputs.json'); inputs['decoder']['seed'] = seed
            write_json(run/'receiver/decoder_inputs.json', inputs)
            policy = read(old/'receiver_policy.json'); policy['noise_contract'].update(seed=seed, code=p['signature'])
            policy['noise_reference'] = None
            write_json(run/'receiver_policy.json', policy)
            write_json(run/'channel_accounting.json', dict(channel_uses=case['original_channel_uses']))
        stages.step('original_inputs', ['run'], ['run/run_config.json', 'run/receiver_policy.json',
                    'run/receiver/decoder_inputs.json', 'run/receiver/frames'], prepare)
    else:
        for stage, owned, required in [
            ('channel', ['run/received','run/channel_accounting.json'], ['run/received','run/channel_accounting.json']),
            ('decode', ['run/receiver/frames'], ['run/receiver/frames']),
            ('initialize', ['run/run_config.json','run/keyframes.json','run/receiver_policy.json',
                            'run/receiver/metadata.csv','run/receiver/decoder_inputs.json',
                            'run/receiver/text_embeddings.json','run/data'],
                           ['run/run_config.json','run/keyframes.json','run/receiver_policy.json',
                            'run/receiver/metadata.csv','run/receiver/decoder_inputs.json','run/receiver/text_embeddings.json']),
        ]:
            stages.step(stage, owned, required, lambda log,s=stage: execute(p,s,out/'job.json',log))
    for stage, owned, required in [
        ('reconstruct', ['run/receiver/reconstruction','run/receiver/generation_noise.json'],
         ['run/receiver/reconstruction/sample_0000.mp4','run/receiver/generation_noise.json']),
        ('refine', ['refined'], ['refined/reconstruction.mp4','refined/audit.json']),
        ('evaluate', ['quality.json','quality.csv'], ['quality.json','quality.csv']),
    ]:
        legacy.event(current=f'{case["id"]}/{out.name}', stage=stage)
        stages.step(stage, owned, required, lambda log,s=stage: execute(p,s,out/'job.json',log))
    return read(out/'quality.json'), run


def run(args):
    from .etri_60s_check import lock
    root, screen, old_experiment = args.root.resolve(), args.screen.resolve(), args.previous.resolve()
    old = read(old_experiment/'protocol.json')
    selected = []
    rows = read(screen/'RESULT.json')['rows']
    for case in old['cases']:
        options = [r for r in rows if r['id']==case['id'] and r['every_key_noninferior'] and r['lower_rate']]
        if options:
            row = min(options, key=lambda r:r['total_uses'])
            directory = screen/row['id']/f'qp_{row["qp"]}'
            selected.append(dict(case=case, directory=str(directory), record=read(directory/'result.json')))
    p = dict(criteria=CRITERIA, candidates=selected, seed_selection=2025, seed_confirmation=2026,
             screen_sha256=sha256(screen/'RESULT.json'), baseline_result=str(REPO/'outputs/bsu_kinetics_context/RESULT.json'),
             runtimes=dict(python=str(REPO/'.local/bsu_codec_eval_env/bin/python'),
                           codec=str(REPO/'.local/dcvc_rt_env/bin/python'), channel=old['runtimes']['channel'],
                           action=old['runtimes']['action']),
             action_runtime=old['action_runtime'],
             code={str(f.relative_to(REPO)):sha256(f) for f in (REPO/'src/semantic_transmission').glob('*.py')},
             selection='minimum actual paid cost among every-key PSNR/SSIM noninferior candidates; no final-video tuning',
             cohort='previously seen 14-video development cohort; not held-out generalization',
             fallback='retain original visual symbols; losslessly bundle original digital metadata',
             production_defaults_changed=False)
    p['signature'] = fingerprint(p)
    with lock(REPO/'.local/etri_60s_check.lock'):
        freeze(root/'protocol.json',p)
        decisions, action_rows = [], []
        labels = {v['id']:v['label'] for v in read(REPO/'outputs/bsu_kinetics_context/benchmark/protocol.json')['videos']}
        for i,candidate in enumerate(selected):
            case = candidate['case']
            legacy.event(done=i,total=len(selected),current=case['id'],stage='full reconstruction')
            prior = old_experiment/case['id']/'baseline_2025/quality.json'
            baseline = read(prior)
            if sha256(baseline['movie']) != baseline['movie_sha256']:
                raise ValueError('baseline quality/movie binding changed')
            quality, run_dir = trial(root,p,case,candidate,2025)
            noise = read(run_dir/'receiver/generation_noise.json')
            original_noise = read(Path(case['original_run'])/'receiver/generation_noise.json')
            matched = noise['records'] == original_noise['records'] and noise['runtime'] == original_noise['runtime']
            if not matched:
                raise ValueError('original and DCVC generation noise/runtime differ')
            verdict = guard(baseline,quality)
            decision = dict(id=case['id'],qp=candidate['record']['qp'],saving_percent=candidate['record']['saving_percent'],
                            noise_records_exact=True,seed2025=verdict,baseline=baseline,candidate=quality,accepted=False)
            pairs = [(2025,baseline,quality)]
            if verdict['passed']:
                # Fresh paired seed, fixed codec/QP and guard thresholds.
                a, arun = trial(root,p,case,candidate,2026,original=True)
                b, brun = trial(root,p,case,candidate,2026)
                an = read(arun/'receiver/generation_noise.json'); bn = read(brun/'receiver/generation_noise.json')
                if an['records'] != bn['records'] or an['runtime'] != bn['runtime']:
                    raise ValueError('confirmation generation noise/runtime differs')
                decision['seed2026'] = guard(a,b)
                decision['accepted'] = decision['seed2026']['passed']
                pairs.append((2026,a,b))
            for seed,a,b in pairs:
                for variant,value in [('baseline',a),('dcvc',b)]:
                    action_rows.append(dict(id=case['id'],variant=f'{variant}_{seed}', label=labels[case['id']],
                                            path=value['movie'],sha256=value['movie_sha256']))
            decisions.append(decision)
            write_json(root/case['id']/'decision.json',decision)
        job = dict(runtime=p['action_runtime'],rows=action_rows,output=str(root/'action'))
        freeze(root/'action_job.json',job)
        Stages(root,p['signature']).step('action',['action'],['action/RESULT.json'],
                                       lambda log:execute(p,'action',root/'action_job.json',log))
        action = read(root/'action/RESULT.json')
        for d in decisions:
            action_guards=[]
            for seed in [2025,2026] if 'seed2026' in d else [2025]:
                pair={r['input']['variant']:r for r in action['records'] if r['input']['id']==d['id']}
                a,b=pair[f'baseline_{seed}'],pair[f'dcvc_{seed}']
                action_guards.append(all(not a[k] or b[k] for k in ['top1_correct','top5_correct']))
            d['action_noninferior']=all(action_guards)
            d['accepted']=d['accepted'] and d['action_noninferior']
        write_json(root/'RESULT.json',dict(status='COMPLETE_DCVC_BSU_CANDIDATE_VALIDATION',decisions=decisions,
            accepted_videos=sum(d['accepted'] for d in decisions),tested_videos=len(decisions),
            criteria=CRITERIA,action=action['summary'],production_defaults_changed=False,
            limitation='seen development videos; independent visual semantic review and held-out validation pending'))
        legacy.event(done=len(selected),total=len(selected),stage='completed')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=REPO/'outputs/bsu_dcvc_validation_20261015_v3')
    p.add_argument('--screen',type=Path,default=REPO/'outputs/bsu_dcvc_screen_20261015_v1')
    p.add_argument('--previous',type=Path,default=REPO/'outputs/bsu_key_reduction_20261015_v2')
    p.add_argument('--worker');p.add_argument('--job',type=Path)
    args=p.parse_args()
    if args.worker: worker(read(args.job),args.worker)
    else: run(args)


if __name__=='__main__':main()
