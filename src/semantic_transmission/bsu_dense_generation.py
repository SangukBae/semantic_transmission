"""Controlled-generation ablation of the paid BSU dense-video model."""
import argparse
import copy
import os
from pathlib import Path
import subprocess

from .artifacts import sha256,write_json
from .qwen35_pairs_video import freeze
from .webvid5 import fingerprint
from .webvid_ablation import Stages
from .bsu_dense_validation import REPO,read,evaluate,save_movie,progress

MODULE='semantic_transmission.bsu_dense_generation'


def worker(j,stage):
    if stage=='action':
        from .bsu_key_reduction import action
        return action(j)
    run=Path(j['run']);out=Path(j['output'])
    if stage=='initialize':
        from .dense_video_transport import decode_packet
        from .bsu_key_reduction import write_csv,link
        from .bsu_dcvc_validation import portable_text_report
        from .dense_generation_runtime import POLICY
        old=Path(j['old_run']);context,_,_=decode_packet(Path(j['packet']).read_bytes())
        cfg=read(old/'run_config.json');cfg['seed']=2025
        # Decoder config needs geometry/models; original image/video locations
        # are deliberately not copied into this new receiver run.
        cfg['input']='received_dense_guide';cfg.pop('input_sha256',None)
        write_json(run/'run_config.json',cfg);write_json(run/'keyframes.json',dict(indices=context['indices']))
        write_csv(run/'receiver/metadata.csv',context['segments'])
        write_json(run/'receiver/decoder_inputs.json',dict(video=context['video'],decoder=context['decoder'],indices=context['indices']))
        if sha256(run/'receiver/metadata.csv')!=sha256(old/'receiver/metadata.csv'):
            raise ValueError('paid captions/flow differ from reused text conditions')
        portable_text_report(old,run)
        policy=read(old/'receiver_policy.json');policy['noise_reference']=None
        policy['noise_contract'].update(seed=2025,code=j['signature'],inputs=sha256(j['packet']))
        write_json(run/'receiver_policy.json',policy)
        guide=Path(j['guide_frames'])
        for k in context['indices']:
            link(guide/f'{k:05d}.png',run/f'receiver/frames/sample/key_frames_received/{k}.png')
        write_json(run/'dense_runtime.json',dict(policy=POLICY,strength=.12,guide_frames=str(guide),
            guide_sha256={p.name:sha256(p) for p in sorted(guide.glob('*.png'))}))
    elif stage=='reconstruct':
        from .decoder_runner import run as decode
        from . import tail_reference as tail,precision_text_cache as cache,generation_noise as noise
        from . import condition_collision as collision,short_schedule
        keys=read(run/'keyframes.json')['indices']
        cfg=run/'receiver/decoder_config.py';cfg.write_text(tail.decoder_config(run))
        for f in ['receiver/reference_trace.json','receiver/tail_reference_trace.json',collision.TRACE,short_schedule.TRACE]:
            write_json(run/f,[])
        env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(REPO/'src'),str(REPO/'.local/vendor/Open-Sora')]),
            ETRI_PRECISION_RUN=str(run),ETRI_REFERENCE_TRACE=str(run/'receiver/reference_trace.json'),
            ETRI_TAIL_REFERENCE_TRACE=str(run/'receiver/tail_reference_trace.json'))
        decode(run/'receiver/metadata.csv',run/'receiver/reconstruction','key_frames_received',run/'receiver',
            decoder=REPO/'scripts/etri_dense_condition_decoder.py',config=cfg,environment=env)
        cache.validate_usage(run)
        tail.validate_trace(read(run/'receiver/tail_reference_trace.json'),len(keys)-2)
        noise.validate_trace(read(run/noise.TRACE),read(run/'receiver_policy.json')['noise_contract'],len(keys)-1,30)
        short_schedule.validate_trace(read(run/short_schedule.TRACE),keys,30)
    elif stage=='projection':
        import numpy as np
        from PIL import Image
        from .dense_video_receiver import detail_projection
        raw=run/'receiver/reconstruction/sample_0000_frames';guide=Path(j['guide_frames'])
        names=[p.name for p in sorted(guide.glob('*.png'))]
        if names!=[p.name for p in sorted(raw.glob('*.png'))]:raise ValueError('generation timeline changed')
        frames=lambda d:np.stack([np.asarray(Image.open(d/n).convert('RGB')) for n in names])
        output,audit=detail_projection(frames(guide),frames(raw),strength=.25,max_delta=12.)
        save_movie(output,out/'projected',24);write_json(out/'projected/audit.json',audit)
    elif stage=='evaluate':
        for arm,movie in [('raw',run/'receiver/reconstruction/sample_0000.mp4'),('projected',out/'projected/reconstruction.mp4')]:
            dest=out/arm;dest.mkdir(parents=True,exist_ok=True)
            evaluate(j['case'],movie,dest,j['channel_uses'])
    else:raise ValueError(stage)


def run(root,dense,case_ids):
    from .etri_60s_check import lock
    from .key_reduction_selection import guard
    with lock(REPO/'.local/etri_60s_check.lock'):
        base=read(dense/'protocol.json')
        if read(dense/'RESULT.json')['status']!='COMPLETE_DENSE_RATE_QUALITY_ACTION_COMPARISON':
            raise ValueError('complete dense control inputs required')
        cases=[c for c in base['cases'] if not case_ids or c['id'] in case_ids]
        p=dict(cases=cases,dense=str(dense),dense_result_sha256=sha256(dense/'RESULT.json'),
            strength=.12,detail_strength=.25,detail_max_delta=12.,seed=2025,steps=30,
            code={f:sha256(REPO/f) for f in ['src/semantic_transmission/bsu_dense_generation.py',
                'src/semantic_transmission/dense_generation_runtime.py','src/semantic_transmission/dense_video_receiver.py',
                'scripts/etri_dense_condition_decoder.py']},
            note='same paid dense packet; received SwinIR frames are guide and virtual keys; no original RGB keys',
            labels_or_output_quality_used_to_select_guidance=False)
        p['signature']=fingerprint(p);root.mkdir(parents=True,exist_ok=True);freeze(root/'protocol.json',p)
        records=[];action_rows=[]
        labels={v['id']:v['label'] for v in read(REPO/'outputs/bsu_kinetics_context/RESULT.json')['videos']}
        def execute(stage,job,log):
            env=dict(os.environ,PYTHONPATH=str(REPO/'src'),PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',
                OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
                HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            runtime=base['runtimes']['action' if stage=='action' else 'python']
            log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('x') as stream:
                code=subprocess.run([runtime,'-m',MODULE,'--worker',stage,'--job',str(job)],cwd=REPO,env=env,
                                    stdout=stream,stderr=subprocess.STDOUT).returncode
            if code:
                print(log.read_text(errors='replace')[-7000:],flush=True);raise RuntimeError(f'{stage} failed: {log}')
        for ci,case in enumerate(cases):
            progress('dense_generation',ci,len(cases),case['id'])
            out=root/case['id'];run_dir=out/'run';out.mkdir(parents=True,exist_ok=True)
            src=dense/case['id'];j=dict(output=str(out),run=str(run_dir),old_run=case['original_run'],case=case,
                packet=str(src/'dense/received/packet.bin'),guide_frames=str(src/'dense_swinir/frames'),
                channel_uses=read(src/'dense/received/channel.json')['channel_uses'],signature=p['signature'])
            freeze(out/'job.json',j);stages=Stages(out,p['signature'])
            for stage,owned,required in [
                ('initialize',['run'],['run/run_config.json','run/receiver/metadata.csv','run/receiver/frames',
                                     'run/receiver/text_embeddings.json','run/dense_runtime.json','run/receiver_policy.json']),
                ('reconstruct',['run/receiver/reconstruction','run/receiver/dense_condition_trace.json'],
                 ['run/receiver/reconstruction','run/receiver/dense_condition_trace.json','run/receiver/generation_noise.json']),
                ('projection',['projected'],['projected/reconstruction.mp4','projected/audit.json','projected/frames']),
                ('evaluate',['raw/quality.json','raw/quality.csv','projected/quality.json','projected/quality.csv'],
                 ['raw/quality.json','raw/quality.csv','projected/quality.json','projected/quality.csv'])]:
                stages.step(stage,owned,required,lambda log,s=stage:execute(s,out/'job.json',log))
            baseline=read(REPO/'outputs/bsu_key_reduction_20261015_v2'/case['id']/'baseline_2025/quality.json')
            control=read(src/'dense_swinir/quality.json')
            qualities={a:read(out/a/'quality.json') for a in ('raw','projected')}
            records.append(dict(id=case['id'],baseline=baseline,control=control,qualities=qualities,
                strict_guards={a:guard(baseline,q) for a,q in qualities.items()}))
            for a,q in dict(baseline=baseline,control=control,**qualities).items():
                action_rows.append(dict(id=case['id'],variant=a,label=labels[case['id']],path=q['movie'],sha256=q['movie_sha256']))
        freeze(root/'action_job.json',dict(output=str(root/'action'),runtime=base['action_runtime'],rows=action_rows))
        Stages(root,p['signature']).step('action',['action'],['action/RESULT.json'],
            lambda log:execute('action',root/'action_job.json',log))
        write_json(root/'RESULT.json',dict(status='COMPLETE_DENSE_CONTROLLED_GENERATION_ABLATION',records=records,
            action=read(root/'action/RESULT.json')['summary'],generalization_proven=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=REPO/'outputs/bsu_dense_generation_20261015_v1')
    parser.add_argument('--dense',type=Path,default=REPO/'outputs/bsu_dense_512_20261015_v1')
    parser.add_argument('--case',action='append');parser.add_argument('--worker');parser.add_argument('--job',type=Path)
    args=parser.parse_args()
    if args.worker:return worker(read(args.job),args.worker)
    run(args.root.resolve(),args.dense.resolve(),args.case)


if __name__=='__main__':main()
