"""Source-reviewed, capped-key sender pilot with a paired motion ablation.

Reuses only hash-verified exact-interval captions. Does not mutate parent runs.
Selector changes require new interval captions; this is explicitly an end-to-end
sender comparison, not a caption-fixed isolated selector causal experiment.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from . import auto_extraction as ae, qwen35_pairs_comparison as pc
from . import qwen35_pairs_video as video, sender_repairs as repair
from .artifacts import sha256, write_json
from .validation_progress import emit

REPO = pc.REPO
MODULE = 'semantic_transmission.sender_repair_experiment'
ARMS = ('selector', 'selector_flow')
DEFAULT_PARENT = REPO/'outputs/lgvsc_qwen_zea_16s_20261008/qwen'
DEFAULT_OUTPUT = REPO/'outputs/sender_repair_20261008'
CODE = ('sender_repairs.py', 'sender_repair_experiment.py', 'sender_repair_evaluation.py')


def make_plan(parent, output, cap=None, max_gap=24):
    pc.forbid_overlap(output, [parent.parent, pc.SNAPSHOT, REPO/'outputs/qwen35_evidence_repair_v3'])
    p = pc.load_protocol(parent)
    pc.validate_bundle(parent, 'event_pairs', p)
    source = p['source']
    if sha256(source['normalized_video']) != source['normalized_sha256']:
        raise ValueError('source video changed')
    selection = ae.unseal(parent/'selection.json')
    paths = [parent/'protocol.json', parent/'selection.json', parent/'ledger.json',
        parent/'captions/event_pairs.json', parent/'reconstruction/event_pairs/run/receiver/reconstruction/sample_0000.mp4']
    paths += sorted((parent/'calls/event_pairs').rglob('*.json'))
    code = dict(p['code'])
    for name in CODE:
        path = REPO/'src/semantic_transmission'/name
        code[str(path.relative_to(REPO))] = sha256(path)
    script = REPO/'scripts/experiment_sender_repair.sh'
    code[str(script.relative_to(REPO))] = sha256(script)
    value = dict(policy='source_reviewed_capped_sender_pilot_v1', parent=str(parent),
        source=source, original_keys=selection['keys'],
        cap=len(selection['keys']) if cap is None else cap, max_gap=max_gap,
        event_prompt=repair.EVENT_PROMPT, event_tokens=256, caption=p['caption'],
        receiver=p['receiver'], interpreters=p['interpreters'], base_model_files=p['base_model_files'],
        code=code, inputs={str(path): sha256(path) for path in paths}, arms=list(ARMS),
        training_executed=False, independent_semantic_review='PENDING',
        default_policy_changed=False,
        comparison='selector vs parent changes selected intervals and their required captions; '
                   'selector_flow vs selector changes only motion sampling/score and resulting encoded text condition',
        pilot_gate=dict(flow_error_ratio_max=.95, mad_p95_ratio_max=1.,
            nonkey_psnr_drop_max_db=.3, nonkey_lpips_increase_max=.01, transmission_ratio_max=1.))
    # Validate the ceiling and max-gap without touching GPU or fitting to outputs.
    import math
    if max_gap < 1 or not math.ceil((source['frames']-1)/max_gap)+1 <= value['cap'] <= source['frames']:
        raise ValueError('infeasible requested budget')
    value['signature'] = ae.digest(value)
    return value


def load_plan(root):
    p = ae.read(root/'plan.json')
    identity = dict(p); signature = identity.pop('signature')
    if ae.digest(identity) != signature: raise ValueError('plan signature changed')
    for name, digest in p['code'].items():
        if sha256(REPO/name) != digest: raise ValueError(f'frozen code changed: {name}')
    for name, digest in p['inputs'].items():
        if sha256(name) != digest: raise ValueError(f'frozen input changed: {name}')
    return p


def call(vlm, root, p, role, name, ids, prompt, validator, tokens, extra=None):
    from PIL import Image
    paths = [ae.check_frame(p['source'], i) for i in ids]
    identity = dict(protocol=p['signature'], task=role, source_indices=ids,
        frame_hashes=[sha256(path) for path in paths], **(extra or {}))
    images = [Image.open(path).convert('RGB') for path in paths]
    try:
        return pc.infer(vlm, root/'calls'/name, identity, images,
            [f'Source frame {i}, time {i/p["source"]["fps"]:.6f} seconds' for i in ids],
            prompt, validator, tokens)
    finally:
        for im in images: im.close()


def extract(root):
    from .qwen35_backend import LocalQwen35
    p = load_plan(root)
    parent = Path(p['parent'])
    old = ae.unseal(parent/'selection.json')
    original = pc.validate_bundle(parent, 'event_pairs', ae.read(parent/'protocol.json'))
    vlm = LocalQwen35(p['caption'])
    reviews, started = [], time.monotonic()
    for j, pair in enumerate(old['pairs']):
        a, b = pair['before'], pair['after']
        ids = sorted(set([a, a+(b-a)//3, a+2*(b-a)//3, b]))
        result = call(vlm, root, p, 'visual_review', f'events/{j:05d}.json', ids,
            repair.EVENT_PROMPT + json.dumps(pair['origins'], ensure_ascii=False),
            repair.validate_event, p['event_tokens'], dict(pair=[a,b]))
        reviews.append(result)
        video.progress('사건 원본 재검토', j+1, len(old['pairs']), started)
    pairs = repair.evidence_pairs(old['pairs'], reviews)
    keys, solver = repair.capped_keys(p['source']['frames'], p['cap'], pairs,
        old['prior_keys'], old['details']['pixel_novelty'], p['max_gap'])
    selection = dict(protocol=p['signature'], keys=keys, proposal_pairs=old['pairs'],
        source_reviews=reviews, retained_pairs=pairs, solver=solver,
        rejected_pair_indices=[j for j, r in enumerate(reviews) if r['classification'] in {'none','camera_only'}],
        independent_verification=False)
    video.freeze(root/'selection.json', selection)
    lookup = {(r['start'],r['end_exclusive'],tuple(r['source_indices'])): r for r in original['captions']}
    rows, started = [], time.monotonic()
    for j, (a,b) in enumerate(zip(keys, keys[1:])):
        ids = pc.sample_indices(a,b)
        prior = lookup.get((a,b,tuple(ids)))
        row = dict(segment=j, start=a, end_exclusive=b, source_indices=ids)
        if prior is not None:
            row.update({k: prior[k] for k in ('caption','uncertainties','removed_claims')})
            row['provenance'] = dict(mode='exact_source_interval_reuse',
                bundle=str(parent/'captions/event_pairs.json'), bundle_sha256=sha256(parent/'captions/event_pairs.json'),
                original_segment=prior['segment'])
        else:
            draft = call(vlm, root, p, 'draft', f'captions/draft/{j:05d}.json', ids,
                p['caption']['prompts']['caption'], ae.caption_validator(p['caption']), 384,
                dict(segment=j, start=a, end_exclusive=b))
            reviewed = call(vlm, root, p, 'visual_review', f'captions/review/{j:05d}.json', ids,
                p['caption']['prompts']['review']+draft['caption'], ae.caption_validator(p['caption'], True), 384,
                dict(segment=j, start=a, end_exclusive=b, draft_sha256=ae.digest(draft)))
            row.update(reviewed)
            row['provenance'] = dict(mode='new_source_inference', draft=f'calls/captions/draft/{j:05d}.json',
                review=f'calls/captions/review/{j:05d}.json')
        rows.append(row)
        video.progress('구간 캡션 검증', j+1, len(keys)-1, started)
    video.freeze(root/'captions.json', dict(protocol=p['signature'], captions=rows,
        author=pc.MODEL_ID, model_revision=pc.REVISION, independent_verification=False))


def arm_protocol(root, arm):
    p = load_plan(root)
    s, caps = ae.read(root/'selection.json'), ae.read(root/'captions.json')
    if s['protocol'] != p['signature'] or caps['protocol'] != p['signature']:
        raise ValueError('extraction identity mismatch')
    cfg = copy.deepcopy(p['receiver'])
    cfg.update(profile=p['policy'], selector='source_reviewed_capped_pairs',
        flow_sampling_policy='distinct_duration_normalized_v1' if arm == 'selector_flow' else 'official_release')
    value = dict(policy=p['policy'], config=cfg, keys=s['keys'], arm=arm,
        extraction_sha256={name: sha256(root/name) for name in ('plan.json','selection.json','captions.json')},
        call_sha256={str(path.relative_to(root)): sha256(path) for path in sorted((root/'calls').rglob('*.json'))},
        condition_plan=pc.collision.preview(s['keys'], pc.collision.upstream_functions(REPO)))
    value['signature'] = ae.digest(value)
    video.freeze(root/arm/'execution_protocol.json', value)
    return p, value


def import_captions(root, run):
    p = load_plan(root)
    keys = ae.read(run/'keyframes.json')['indices']
    bundle = ae.read(root/'captions.json')
    if len(bundle['captions']) != len(keys)-1: raise ValueError('caption count mismatch')
    rows, sampling = [], []
    for j, (row,a,b) in enumerate(zip(bundle['captions'],keys,keys[1:])):
        ids = pc.sample_indices(a,b)
        if (row['segment'],row['start'],row['end_exclusive'],row['source_indices']) != (j,a,b,ids):
            raise ValueError('caption interval mismatch')
        ae.caption_validator(p['caption'], True)(row)
        for i in ids: ae.check_frame(p['source'],i)
        clip = f'clips/sample/{j:05d}.mp4'
        rows.append(dict(path=clip,text=row['caption'],flow=0.))
        sampling.append(dict(clip=str(run/'data'/clip), decoded_frames=b-a,
            indices=[i-a for i in ids], source_indices=ids, provenance=row['provenance']))
    write_json(run/'captions.json',rows)
    write_json(run/'caption_sampling.json',sampling)
    write_json(run/'caption_provenance.json',dict(provider='local_qwen35_caption_repair',
        model_id=pc.MODEL_ID, model_revision=pc.REVISION, caption_adapter=p['caption']['caption_adapter'],
        bundle_sha256=sha256(root/'captions.json'), assistant_authored=False,
        independent_verification=False, selector='source_reviewed_capped_pairs'))


def worker(root, stage, arm):
    if stage == 'extract': return extract(root)
    p, protocol = arm_protocol(root, arm)
    run = root/arm/'run'
    if stage == 'caption': import_captions(root,run)
    elif stage == 'flow': repair.flow_worker(protocol['config'], REPO, run)
    elif stage == 'channel': pc.paired_channel(run)
    elif stage == 'prepare-text': video.prepare_text(run)
    elif stage == 'reconstruct': pc.fc.reconstruct(run)
    elif stage == 'output-audit':
        pc.fc.audit(run)
        value = ae.read(run/'output_audit.json')
        value.update(selector='source_reviewed_capped_pairs', automatic_keyframe_insertion=False,
            automatic_keyframe_insertion_stage=None, prepared_source_reconstructed=True)
        write_json(run/'output_audit.json',value)
    else: raise ValueError(stage)


def launch(cmd, log, env, resource):
    ipc = resource.with_suffix('.progress.json')
    ipc.unlink(missing_ok=True)
    log.parent.mkdir(parents=True,exist_ok=True)
    start = time.monotonic()
    with log.open('x') as stream:
        process = subprocess.Popen(cmd,cwd=REPO,env=dict(env,QUALITY_PROGRESS_FILE=str(ipc)),
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while True:
                try:
                    code = process.wait(timeout=20); break
                except subprocess.TimeoutExpired:
                    elapsed = time.monotonic()-start
                    if ipc.exists():
                        state = ae.read(ipc); n,total = state['done'],state['total']
                        eta = f' ETA={(elapsed/n*(total-n)):.0f}s' if n else ' ETA unknown'
                        print(f'{log.stem}: {n}/{total} ({100*n/total:.1f}%) elapsed={elapsed:.0f}s{eta}',flush=True)
                    else:
                        print(f'{log.stem}: initializing/running, measured progress unavailable; elapsed={elapsed:.0f}s',flush=True)
                    if log.stem == 'extract':
                        lines = [s for s in log.read_text(errors='replace').splitlines() if '% (' in s]
                        if lines: print(lines[-1],flush=True)
            if code:
                print(log.read_text(errors='replace')[-6500:],flush=True)
                raise RuntimeError(f'worker exit {code}: {log}')
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL); process.wait()
            raise
        finally:
            write_json(resource,dict(command=cmd,seconds=time.monotonic()-start,returncode=process.returncode))


def execute(root, p):
    root.mkdir(parents=True,exist_ok=True)
    with pc.lock(REPO/'.local/etri_60s_check.lock'), pc.lock(root/'.execution.lock'):
        video.freeze(root/'plan.json',p)
        env = dict(pc.environment(2025),PYTHONPATH=str(REPO/'src'),PYTHONUNBUFFERED='1')
        phases = pc.Stages(root,p['signature'])
        products = ['selection.json','captions.json','calls','resources/extract.json']
        # Successful inference calls survive a partial extraction and are individually checked.
        phases.step('extract',['resources/extract.json'],products,
            lambda log: launch([p['interpreters']['qwen'],'-m',MODULE,'--output',str(root),'--worker','extract'],
                               log,env,root/'resources/extract.json'))
        for number, arm in enumerate(ARMS):
            _, protocol = arm_protocol(root,arm)
            dest = root/arm
            stages = pc.Stages(dest,protocol['signature'])
            stages.step('initialize',['run'],pc.initialization_products(p['source']['frames']),
                lambda _: pc.initialize(dest,p['source'],dict(indices=protocol['keys'],selector='source_reviewed_capped_pairs'),protocol['config']))
            jobs = list(pc.reconstruction_jobs())
            for j,(stage,module,products) in enumerate(jobs):
                done,total = number*len(jobs)+j,len(ARMS)*len(jobs)
                print(f'전체 수신·복원 {done}/{total} ({100*done/total:.1f}%, 단계 수 기준) | {arm}/{stage}',flush=True)
                python = p['interpreters']['channel_python' if stage=='channel' else 'python']
                if module == pc.MODULE or stage == 'flow' and arm == 'selector_flow':
                    cmd = [python,'-m',MODULE,'--output',str(root),'--worker',stage,'--arm',arm]
                elif module == pc.fc.ETRI:
                    cmd = [python,'-m',module,'--worker',stage,'--run-dir',str(dest/'run')]
                else:
                    cmd = [python,'-m',module,stage,str(dest/'run')]
                required = [*[f'run/{x}' for x in products],f'resources/{stage}.json']
                e = dict(env,ETRI_RUN_SIGNATURE=protocol['signature'])
                if stage == 'channel': e['CUDA_VISIBLE_DEVICES']='-1'
                stages.step(stage,required,required,
                    lambda log,c=cmd,e=e,s=stage: launch(c,log,e,dest/f'resources/{s}.json'))
        print('수신·복원 22/22 완료 (100%, 단계 수 기준); 비교평가 시작',flush=True)
        cmd = [p['interpreters']['python'],'-m','semantic_transmission.sender_repair_evaluation',str(root)]
        products=['results.json','report.html','comparison.mp4','samples','resources/comparison.json']
        phases.step('comparison',products,products,
            lambda log: launch(cmd,log,env,root/'resources/comparison.json'))
        print(json.dumps(ae.read(root/'results.json'),ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,default=DEFAULT_PARENT)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--cap',type=int)
    parser.add_argument('--max-gap',type=int,default=24)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--worker')
    parser.add_argument('--arm',choices=ARMS)
    args=parser.parse_args()
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    root=args.output.resolve()
    if args.worker: return worker(root,args.worker,args.arm)
    print('사전 검사: 원본·부모 결과·코드·캡션 출처 검증 중 (ETA unknown)',flush=True)
    p=make_plan(args.parent.resolve(),root,args.cap,args.max_gap)
    if args.check:
        print(json.dumps(dict(status='READY',frames=p['source']['frames'],cap=p['cap'],max_gap=p['max_gap'],arms=p['arms'])))
        return
    try: execute(root,p)
    except BaseException as exc:
        write_json(root/'status.json',dict(status='FAILED',error=str(exc)))
        raise


if __name__=='__main__': main()
