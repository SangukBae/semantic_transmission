"""Sequential, resumable local validation of BSU sender/generator improvements."""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .artifacts import sha256,write_json
from .validation_progress import emit

REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'outputs/bsu_motion_validation_20261015_v1'
IMPORTED=REPO/'outputs/bsu_webvid_completed_20261015'
CASES=REPO/'outputs/bsu_receiver_improvement_20261015_v1/protocol.json'
AUDIT=REPO/'outputs/bsu_sender_information_audit_20261015/packet_audit.json'
PYTHON=REPO.parent/'anaconda3/envs/lgvsc/bin/python'
CHANNEL_PYTHON=REPO.parent/'anaconda3/envs/lgvsc-channel/bin/python'
PILOT_NUMBERS=(17,19,22,28,30)


def read(p):return json.loads(Path(p).read_text())


def cases():return read(CASES)['cases']


def source_directory(case):
    return Path(case['source_movie']).parent/'source_frames'


def progress(stage,done,total):
    emit(done,total)
    print(f'{stage} {done}/{total} ({100*done/total:.1f}%)',flush=True)


def stage_complete(root,name,inputs,outputs,extra=None):
    write_json(root/f'{name}/COMPLETE.json',dict(status='COMPLETE',
        inputs={str(p):sha256(p) for p in inputs},outputs={str(p):sha256(p) for p in outputs},**(extra or {})))


def verified(root,name):
    p=root/f'{name}/COMPLETE.json'
    if not p.exists():return False
    d=read(p)
    if any(not Path(p).exists() or sha256(p)!=h for group in ('inputs','outputs') for p,h in d[group].items()):
        raise ValueError(f'{name}: frozen inputs/code/output changed')
    return True


def flow(root):
    from .motion_sampling import UniMatchScorer,distributed_pairs,normalized_pair_score,read_frames,POLICY
    from .wire import pack,unpack
    from .packets import accounting
    import numpy as np
    import cv2
    if verified(root,'flow'):return
    cv2.setNumThreads(2)
    scorer=UniMatchScorer(REPO);rows=[];start=time.monotonic();outputs=[]
    primary=[c for c in cases() if c['snr']==10]
    prior={r['id']:r for r in read(AUDIT)['rows'] if r['snr']==10}
    for ci,c in enumerate(primary):
        dest=root/'flow'/c['id'];dest.mkdir(parents=True,exist_ok=True)
        old_run=Path(c['receiver']['base_packet']).parent.parent
        out=dest/'scores.json'
        if out.exists():
            saved=read(out)
            if saved['input_packet_sha256']!=sha256(c['receiver']['base_packet']):raise ValueError('flow input changed')
            rows.extend(saved['segments']);outputs.append(out);progress('FLOW sources',ci+1,len(primary));continue
        results=[]
        for i,s in enumerate(prior[c['id']]['segments']):
            n=s['end']-s['start'];new_pairs=distributed_pairs(n,24)
            old_ids=[min(j,n-1) for j in (0,10,20,30)]
            old_pairs=list(zip(old_ids,old_ids[1:]));wanted=sorted({t for p in new_pairs+old_pairs for t in p})
            clip=old_run/f'data/clips/sample/{i:05d}.mp4'
            if not clip.exists():raise FileNotFoundError(clip)
            images=read_frames(clip,wanted)
            legacy=scorer.measure(images,old_pairs);values=scorer.measure(images,new_pairs)
            score=normalized_pair_score(values,new_pairs,24)
            legacy_score=float(np.mean(legacy))
            result=dict(id=c['id'],segment=i,start=s['start'],end=s['end'],policy=POLICY,
                old_pairs=old_pairs,new_pairs=new_pairs,old_flow=s['scalar_flow'],legacy_recomputed=legacy_score,
                legacy_absolute_difference=abs(legacy_score-s['scalar_flow']),new_displacements=values,
                new_flow=0. if score is None else score,motion_observed=score is not None,
                prompt_changed=f"{s['scalar_flow']:.1f}"!=f"{(score or 0.):.1f}",clip_sha256=sha256(clip))
            results.append(result)
        write_json(out,dict(input_packet_sha256=sha256(c['receiver']['base_packet']),segments=results))
        rows.extend(results);outputs.append(out);progress('FLOW sources',ci+1,len(primary))
    # Account the complete base metadata, including all unmodified visual side information.
    budgets=[]
    for c in cases():
        header,payload=unpack(Path(c['receiver']['base_packet']).read_bytes())
        replacement=copy.deepcopy(header)
        selected=[r for r in rows if r['id']==c['id']]
        for s,r in zip(replacement['segments'],selected):s['flow']=r['new_flow']
        packet=pack(replacement,payload);bps=2 if c['snr'] in (6,8) else 4
        old=accounting(Path(c['receiver']['base_packet']).stat().st_size,bits_per_symbol=bps)
        new=accounting(len(packet),bits_per_symbol=bps)
        budgets.append(dict(id=c['id'],snr=c['snr'],old=old,new=new,
            extra_channel_uses=new['complex_channel_uses']-old['complex_channel_uses']))
    result=dict(status='SENDER_MEASUREMENT_COMPLETE_RECONSTRUCTION_PENDING',policy=POLICY,
        sources=len(primary),segments=len(rows),changed_prompts=sum(r['prompt_changed'] for r in rows),
        legacy_max_absolute_difference=max(r['legacy_absolute_difference'] for r in rows),
        old_duplicate_pairs=sum(a==b for r in rows for a,b in r['old_pairs']),
        new_duplicate_pairs=sum(a==b for r in rows for a,b in r['new_pairs']),
        budgets=budgets,seconds=time.monotonic()-start,rows=rows)
    summary=root/'flow/RESULT.json';write_json(summary,result);outputs.append(summary)
    stage_complete(root,'flow',[CASES,AUDIT,REPO/'src/semantic_transmission/motion_sampling.py'],outputs)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','budgets')},ensure_ascii=False),flush=True)


def pilot_cases():
    return [c for c in cases() if c['snr']==10 and int(c['id'].split('_')[1]) in PILOT_NUMBERS]


def subprocess_stage(args, log, python=PYTHON):
    log.parent.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(REPO/'src'),PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1')
    started=time.monotonic()
    with log.open('w') as f:
        p=subprocess.run([str(python),'-m','semantic_transmission.bsu_motion_validation',*args],
            cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
    write_json(log.with_suffix('.resource.json'),dict(returncode=p.returncode,seconds=time.monotonic()-started))
    if p.returncode:
        print(log.read_text(errors='replace')[-7000:],flush=True)
        raise RuntimeError(f'worker failed: {log}')


def run_directory(root,c,arm):return root/'flow_reconstruction'/c['id']/arm/'run'


def initialize_flow_runs(root):
    from .wire import pack,unpack
    from .generation_noise import POLICY as noise_policy
    from . import tail_reference,precision_text_cache,condition_collision,short_schedule
    score_rows=read(root/'flow/RESULT.json')['rows'];jobs=[]
    for c in pilot_cases():
        original=Path(c['receiver']['base_packet']).parent.parent
        scores=[r for r in score_rows if r['id']==c['id']]
        for arm in ('baseline','distributed_flow'):
            run=run_directory(root,c,arm);run.mkdir(parents=True,exist_ok=True)
            for name in ('run_config.json','keyframes.json','receiver/decoder_inputs.json'):
                target=run/name;target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():shutil.copyfile(original/name,target)
            keys=run/'receiver/frames/sample/key_frames_received'
            if not keys.exists():shutil.copytree(original/'receiver/frames/sample/key_frames_received',keys)
            (run/'received').mkdir(exist_ok=True)
            visual=run/'received/visual.c64'
            if not visual.exists():visual.symlink_to(original/'received/visual.c64')
            header,payload=unpack(Path(c['receiver']['base_packet']).read_bytes())
            for s,r in zip(header['segments'],scores):
                s['flow']=r['new_flow'] if arm=='distributed_flow' else r['legacy_recomputed']
            packet=pack(header,payload)
            (run/'transmitter').mkdir(exist_ok=True)
            (run/'transmitter/metadata.bin').write_bytes(packet)
            # The inherited receiver packet format is preserved. Rebind the
            # separate paid trajectory packet to the new base metadata digest.
            if c['receiver']['trajectory']:
                th,tp=unpack(Path(c['receiver']['trajectory']).read_bytes())
                th['base_metadata_sha256']=sha256(run/'transmitter/metadata.bin')
                (run/'transmitter/trajectory.bin').write_bytes(pack(th,tp))
            cfg=read(run/'run_config.json')
            contract=dict(policy=noise_policy,seed=cfg['seed'],segments=len(c['keys'])-1,
                steps=cfg['steps'],comparison='same-source-keys-noise-flow-only',reference_policy=tail_reference.POLICY)
            write_json(run/'receiver_policy.json',dict(policy=tail_reference.POLICY,
                reference_frames=17,reference_latents=5,alignment=5,text_encoder_policy=precision_text_cache.POLICY,
                t5_precision='fp32',noise_contract=contract,noise_reference=None,
                condition_collision_policy=condition_collision.POLICY,short_schedule_policy=short_schedule.POLICY))
            jobs.append(dict(id=c['id'],arm=arm,run=str(run),snr=10,bits_per_symbol=4,
                base_packet_sha256=sha256(c['receiver']['base_packet']),
                received_key_hashes={str(k):sha256(p) for k,p in zip(c['keys'],c['receiver']['key_paths'])}))
    plan=dict(status='PREREGISTERED',jobs=jobs,sources=[c['id'] for c in pilot_cases()],
        arms=['baseline','distributed_flow'],seed=2025,steps=30,snr_db=10,
        selection='five previously diagnosed development cases; not unseen-video evidence',
        baseline='legacy three pairs remeasured locally with the same UniMatch as the candidate; stored remote scores are diagnostic only',
        acceptance=dict(max_psnr_loss_db=.1,max_lpips_relative_increase=.01,
            max_flow_relative_increase=.02,max_cadence_relative_increase=.02,
            max_boundary_relative_increase=.05,min_psnr_gain_db=.1,
            min_lpips_relative_gain=.02,min_flow_relative_gain=.05),
        inputs={str(p):sha256(p) for p in [CASES,AUDIT,root/'flow/RESULT.json']},
        code={str(p.relative_to(REPO)):sha256(p) for p in [Path(__file__),REPO/'src/semantic_transmission/motion_sampling.py']})
    path=root/'flow_reconstruction/plan.json'
    if path.exists() and read(path)!=plan:raise ValueError('flow reconstruction plan changed')
    write_json(path,plan)
    return plan


def channel_packets(root):
    from .metadata_channel import transmit
    from .wire import unpack
    from .bsu_sequence_refinement import decode_packet
    jobs=read(root/'flow_reconstruction/plan.json')['jobs']
    for ji,j in enumerate(jobs):
        run=Path(j['run']);report={}
        for name,seed in [('metadata',42),('trajectory',43)]:
            src=run/f'transmitter/{name}.bin'
            if not src.exists():continue
            packet=src.read_bytes();received,info=transmit(packet,10,seed,bits_per_symbol=4)
            if packet!=received:raise ValueError(f'channel corruption: {j["id"]}/{j["arm"]}/{name}')
            (run/f'received/{name}.bin').write_bytes(received)
            info.update(exact_match=True,sha256=sha256(src));report[name]=info
        header,_=unpack((run/'received/metadata.bin').read_bytes())
        if (run/'received/trajectory.bin').exists():decode_packet((run/'received/trajectory.bin').read_bytes(),(run/'received/metadata.bin').read_bytes())
        with (run/'receiver/metadata.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['path','text','flow']);w.writeheader();w.writerows(header['segments'])
        write_json(run/'channel_validation.json',report)
        progress('PAIRED CHANNEL',ji+1,len(jobs))


def prepare_pilot_text(root):
    from . import precision_text_cache as cache,tail_reference
    from mmengine import Config
    jobs=read(root/'flow_reconstruction/plan.json')['jobs']
    batch=root/'flow_reconstruction/text_batch';(batch/'receiver').mkdir(parents=True,exist_ok=True)
    combined=[]
    for j in jobs:
        with (Path(j['run'])/'receiver/metadata.csv').open() as f:combined.extend(csv.DictReader(f))
    with (batch/'receiver/metadata.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','text','flow']);w.writeheader();w.writerows(combined)
    config=tail_reference.decoder_config(Path(jobs[0]['run']))
    cache.prepare(batch,REPO,config,'fp32')
    all_report=read(batch/cache.REPORT);entries={e['prompt']:e for e in all_report['entries']}
    for j in jobs:
        run=Path(j['run']);cfg=Config.fromstring(tail_reference.decoder_config(run),file_format='.py')
        prompts=cache.legacy.prepare_prompts(run/'receiver/metadata.csv',cfg,REPO)
        report=dict(all_report,prompts=prompts,entries=[entries[p] for p in dict.fromkeys(prompts)],
            prompt_count=len(prompts),metadata_sha256=sha256(run/'receiver/metadata.csv'))
        write_json(run/cache.REPORT,report)
    print('T5 paired condition cache complete',flush=True)


def reconstruct_run(run):
    from .decoder_runner import run as decode
    from . import tail_reference,condition_collision,short_schedule
    config=run/'receiver/decoder_config.py';config.write_text(tail_reference.decoder_config(run))
    for p in ('receiver/reference_trace.json','receiver/tail_reference_trace.json',condition_collision.TRACE,short_schedule.TRACE):write_json(run/p,[])
    env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(REPO/'src'),str(REPO/'.local/vendor/Open-Sora')]),
        ETRI_REFERENCE_TRACE=str(run/'receiver/reference_trace.json'),
        ETRI_TAIL_REFERENCE_TRACE=str(run/'receiver/tail_reference_trace.json'),ETRI_PRECISION_RUN=str(run))
    target=run/'receiver/reconstruction'
    if target.exists():target.rename(target.with_name('reconstruction_failed_'+str(time.time_ns())))
    decode(run/'receiver/metadata.csv',target,'key_frames_received',run/'receiver',
        decoder=REPO/'scripts/etri_schedule_fix_decoder.py',config=config,environment=env)
    from .precision_text_cache import validate_usage
    validate_usage(run)


def refine_run(run):
    from .bsu_receiver import refine_windows,encode_frames
    from .bsu_sequence_refinement import decode_packet
    inputs=read(run/'receiver/decoder_inputs.json');keys=inputs['indices'];n=inputs['video']['frames']
    folder=run/'receiver/reconstruction/sample_0000_frames';paths=[folder/f'{i:05d}.png' for i in range(n)]
    if not all(p.exists() for p in paths):raise ValueError('incomplete generator frames')
    tracks=decode_packet((run/'received/trajectory.bin').read_bytes(),(run/'received/metadata.bin').read_bytes()) if (run/'received/trajectory.bin').exists() else {}
    dest=run.parent/'refined';dest.mkdir(exist_ok=True)
    result=refine_windows(paths,keys,run/'receiver/frames/sample/key_frames_received',tracks,dest/'frames')
    encode_frames(dest/'frames',dest/'reconstruction.mp4',n,24)
    write_json(dest/'refinement.json',result)


def evaluate_flow(root):
    from .bsu_receiver_improvement import PersistentMetrics
    from .research_quality import read_video
    from .stutter_metrics import evaluate as temporal
    from .bsu_receiver_compare import boundary_diagnostics
    from .generation_noise import TRACE
    import statistics as st
    import numpy as np
    from PIL import Image
    metrics=PersistentMetrics();results=[]
    for ci,c in enumerate(pilot_cases()):
        source=read_video(c['source_movie']);traces=[]
        for arm in ('baseline','distributed_flow'):
            run=run_directory(root,c,arm);traces.append(read(run/TRACE)['records'])
            dest=run.parent/'refined';q=dest/'quality.json'
            if q.exists():results.append(read(q));continue
            generated=read_video(dest/'reconstruction.mp4');quality,rows=metrics.evaluate(source,generated)
            movement=temporal(source,generated)
            for key,path in zip(c['keys'],c['receiver']['key_paths']):
                if not np.array_equal(np.asarray(Image.open(path)),np.asarray(Image.open(dest/f'frames/{key:05d}.png'))):raise ValueError('received key RGB changed')
            result=dict(id=c['id'],arm=arm,quality=quality,temporal=movement,
                boundaries=boundary_diagnostics(source,generated,c['keys']),key_rgb_exact=True,
                movie_sha256=sha256(dest/'reconstruction.mp4'))
            write_json(q,result);results.append(result)
        if traces[0]!=traces[1]:raise ValueError('paired generation noise tensors differ')
        progress('PAIRED full-video evaluation',ci+1,len(pilot_cases()))
    means={}
    for arm in ('baseline','distributed_flow'):
        selected=[r for r in results if r['arm']==arm]
        means[arm]=dict(**{k:st.mean(r['quality'][k] for r in selected) for k in ('psnr_db','ssim','lpips_vgg','clip','dists')},
            **{k:st.mean(r['temporal'][k] for r in selected) for k in ('flow_error_mean','cadence_error_mean','false_freeze_rate')},
            boundary_error=st.mean(st.mean(b['maximum_change_error'] for b in r['boundaries']) for r in selected))
    a,b=means['baseline'],means['distributed_flow'];g=read(root/'flow_reconstruction/plan.json')['acceptance']
    guard=dict(psnr=b['psnr_db']>=a['psnr_db']-g['max_psnr_loss_db'],
        lpips=b['lpips_vgg']<=a['lpips_vgg']*(1+g['max_lpips_relative_increase']),
        flow=b['flow_error_mean']<=a['flow_error_mean']*(1+g['max_flow_relative_increase']),
        cadence=b['cadence_error_mean']<=a['cadence_error_mean']*(1+g['max_cadence_relative_increase']),
        boundary=b['boundary_error']<=a['boundary_error']*(1+g['max_boundary_relative_increase']))
    gain=(b['psnr_db']>=a['psnr_db']+g['min_psnr_gain_db'] or b['lpips_vgg']<=a['lpips_vgg']*(1-g['min_lpips_relative_gain']) or b['flow_error_mean']<=a['flow_error_mean']*(1-g['min_flow_relative_gain']))
    result=dict(status='COMPLETE_LOCAL_PAIRED_FLOW_PILOT',means=means,guardrails=guard,improvement=gain,
        passed=all(guard.values()) and gain,noise_tensor_identity=True,rows=results,
        generalization='not established: five previously diagnosed development videos')
    write_json(root/'flow_reconstruction/RESULT.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False),flush=True)


def flow_reconstruction(root):
    if verified(root,'flow_reconstruction'):return
    if not verified(root,'flow'):raise ValueError('flow stage required')
    plan=initialize_flow_runs(root)
    subprocess_stage(['channel_packets','--root',str(root)],root/'flow_reconstruction/logs/channel.log',CHANNEL_PYTHON)
    subprocess_stage(['prepare_pilot_text','--root',str(root)],root/'flow_reconstruction/logs/text.log')
    outputs=[]
    for i,j in enumerate(plan['jobs']):
        run=Path(j['run']);receipt=run.parent/'generation_complete.json'
        if receipt.exists():
            if any(sha256(p)!=h for p,h in read(receipt)['outputs'].items()):raise ValueError('completed generation changed')
        else:
            subprocess_stage(['reconstruct_run','--run',str(run)],run.parent/'logs/reconstruct.log')
            subprocess_stage(['refine_run','--run',str(run)],run.parent/'logs/refine.log')
            files=[run/'receiver/reconstruction/sample_0000.mp4',run.parent/'refined/reconstruction.mp4',run/'receiver/generation_noise.json']
            write_json(receipt,dict(outputs={str(p):sha256(p) for p in files}))
        outputs.append(receipt);progress('PAIRED reconstruction',i+1,len(plan['jobs']))
    subprocess_stage(['evaluate_flow','--root',str(root)],root/'flow_reconstruction/logs/evaluation.log')
    outputs.append(root/'flow_reconstruction/RESULT.json')
    stage_complete(root,'flow_reconstruction',[root/'flow/RESULT.json',root/'flow_reconstruction/plan.json'],outputs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['flow','flow_reconstruction','channel_packets','prepare_pilot_text','reconstruct_run','refine_run','evaluate_flow'])
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--run',type=Path)
    a=parser.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    globals()[a.stage](a.run if a.stage in ('reconstruct_run','refine_run') else a.root)


if __name__=='__main__':main()
