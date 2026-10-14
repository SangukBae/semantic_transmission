"""Matched motion-preserving key-context experiment, using frozen local inputs."""
import argparse
import os
from pathlib import Path
import shutil
import statistics as st
import subprocess
import sys

from . import bsu_key_context_experiment as parent
from . import bsu_receiver_compare as common
from . import bsu_compare_transfer as transfer
from .artifacts import sha256, write_json

REPO=parent.REPO
ROOT=REPO/'outputs/bsu_motion_context_20261013_v1'
MODULE='semantic_transmission.bsu_motion_context_experiment'
ARMS=('current','context_only','motion_context')
CODE=(*parent.CODE,'receiver_motion_context.py','bsu_motion_context_experiment.py')
read=common.read
event=common.event


def folder_for(root,case,arm):return root/'comparison'/case['name']/arm


def prepare(root,previous,device,strength):
    from .receiver_motion_context import PARAMETERS
    old=read(previous/'protocol.json');done=read(previous/'RESULT.json')
    if done['completed']!=96 or done['pending']:raise ValueError('parent incomplete')
    for name,digest in old['code'].items():
        if sha256(REPO/'src/semantic_transmission'/name)!=digest:raise ValueError('parent code changed')
    inputs={**old['inputs'],str(previous/'protocol.json'):sha256(previous/'protocol.json')}
    development=REPO/'outputs/bsu_motion_context_20261013_pilot1'
    evidence={}
    for name in ('protocol.json','RESULT.json','selection.json','pilot.py','receiver_motion_context.py'):
        path=development/name
        if path.exists():evidence[str(path)]=sha256(path)
    inputs.update(evidence)
    for case in old['cases']:
        for arm in ARMS[:2]:
            folder=folder_for(previous,case,arm)
            for kind in ('generation','evaluation'):
                if not common.verify_receipt(folder,kind,old['signature']):raise ValueError('missing parent receipt')
                for name,digest in read(folder/f'{kind}.json')['artifacts'].items():inputs[str(folder/name)]=digest
    runtime=common.runtime_identity(device)
    if runtime!=old['runtime']:raise ValueError('evaluation runtime differs from parent')
    p=dict(policy='motion_preserving_key_context_v1',parent=str(previous),parent_signature=old['signature'],
        cases=old['cases'],arms=list(ARMS),primary_arm='motion_context',strength=strength,align=True,parameters=PARAMETERS,
        inputs=inputs,code={name:sha256(REPO/'src/semantic_transmission'/name) for name in CODE},runtime=runtime,
        scope=old['scope'],development_evidence=evidence,
        development='two previously identified WebVid failure videos at SNR 10 were used for development diagnostics; all eight sources previously inspected',
        source_only_for_evaluation=True,new_training=False,additional_transmission=0,generator_reused=True,
        criteria_scope='engineering development screening, not ETRI thresholds or independent statistical validation',
        criteria=dict(min_quality_gain_retention=.75,max_lpips_relative_increase_vs_context=.01,
            max_flow_relative_increase_vs_current=.02,max_cadence_relative_increase_vs_current=.02,
            max_boundary_relative_increase_vs_context=.20,min_pedestrian_excess_freeze_reduction=.50,
            max_source_freeze_increase_vs_current=.005,received_keys_exact=True),
        independent_semantic_review='PENDING',automatically_adopted=False)
    p['signature']=transfer.fingerprint(p)
    file=root/'protocol.json'
    if file.exists() and read(file)!=p:raise ValueError('frozen inputs/code/options changed; choose new --root')
    progress=common.tick('동결 입력 해시 검사')
    for i,(name,digest) in enumerate(inputs.items(),1):
        if sha256(Path(name))!=digest:raise ValueError('input changed: '+name)
        if i%100==0 or i==len(inputs):progress(i,len(inputs))
    if not file.exists():
        write_json(file,p)
        for name in CODE:
            target=root/'frozen_code'/name;target.parent.mkdir(exist_ok=True);shutil.copyfile(REPO/'src/semantic_transmission'/name,target)
    return p


def generate(job,folder,signature):
    import cv2
    import numpy as np
    from PIL import Image
    from .receiver_motion_context import preserve_motion_context
    from .bsu_sequence_refinement import validate_keys
    from .etri_60s_check import pts_audit
    from .wire import unpack
    allowed={'frames','fps','keys','frames_paths','key_paths','base_packet','strength','align'}
    if set(job)!=allowed:raise ValueError('receiver must contain received inputs only')
    validate_keys(job['keys'],job['frames']);header,_=unpack(Path(job['base_packet']).read_bytes())
    if header['video']!=dict(frames=job['frames'],fps=24,width=576,height=320):raise ValueError('timeline differs')
    if [r['index'] for r in header['keyframes']]!=job['keys']:raise ValueError('received keys differ')
    if len(job['frames_paths'])!=job['frames'] or len(job['key_paths'])!=len(job['keys']):raise ValueError('input length differs')
    cv2.setNumThreads(2);cv2.setRNGSeed(2025)
    output=folder/'frames';output.mkdir(parents=True,exist_ok=True);audits=[]
    for index,(a,b) in enumerate(zip(job['keys'],job['keys'][1:])):
        frames=common.pixels(job['frames_paths'][a:b+1],common.tick('생성 프레임 읽기'))
        keys=common.pixels(job['key_paths'][index:index+2])
        result,audit=preserve_motion_context(frames,keys,common.tick('키 연결과 움직임 복원'),job['strength'],job['align'])
        audits.append(dict(start=a,end=b,**audit));progress=common.tick('PNG 저장')
        for j,array in enumerate(result):
            path=output/f'{a+j:05d}.png'
            if index and j==0:
                if not np.array_equal(np.asarray(Image.open(path).convert('RGB')),array):raise ValueError('shared boundary differs')
                continue
            temporary=path.with_suffix('.tmp.png');temporary.unlink(missing_ok=True)
            if np.array_equal(array,frames[j]):os.link(job['frames_paths'][a+j],temporary)
            else:Image.fromarray(array).save(temporary)
            temporary.replace(path);progress(j+1,len(result))
    movie=folder/'reconstruction.mp4';temporary=folder/'encoding.tmp.mp4'
    event(stage='MP4 인코딩 (진행률 미확정)')
    subprocess.run(['ffmpeg','-y','-v','error','-nostdin','-framerate','24','-i',str(output/'%05d.png'),
        '-c:v','libx264','-crf','18','-preset','fast','-pix_fmt','yuv420p','-threads','2',str(temporary)],check=True)
    timeline=pts_audit(temporary,job['frames'],24);temporary.replace(movie)
    paths=[str(output/f'{i:05d}.png') for i in range(job['frames'])]
    info=folder/'generation_info.json';write_json(info,dict(arm='motion_context',frames_paths=paths,movie=str(movie),windows=audits,timeline=timeline))
    common.receipt(folder,'generation',signature,[*[Path(v) for v in paths],movie,info])


def report(root,p):
    rows=[];pending=[]
    for case in p['cases']:
        for arm in ARMS:
            folder=folder_for(root,case,arm)
            if not common.verify_receipt(folder,'evaluation',p['signature']):pending.append([case['name'],arm]);continue
            q=read(folder/'quality.json')
            rows.append(dict(case=case['name'],id=case['id'],group=case['group'],snr=case['snr'],arm=arm,
                channel_uses=case['channel_uses'],**{b:{**q[b],'temporal':{k:v for k,v in q[b]['temporal'].items() if k!='per_transition'}}
                    for b in ('lossless_frames','delivered_mp4')}))
    def mean(selected,b):
        return dict(**{k:st.mean(r[b]['quality'][k] for r in selected) for k in parent.METRICS},
            **{k:st.mean(r[b]['temporal'][k] for r in selected) for k in ('flow_error_mean','cadence_error_mean','false_freeze_rate')},
            nonkey_psnr=st.mean(r[b]['partitions']['nonkeys']['psnr_db'] for r in selected),
            boundary_error=st.mean(st.mean(k['maximum_change_error'] for k in r[b]['key_boundary_changes']) for r in selected))
    groups=[];sources=[];contrasts=[];checks=[]
    completed=[c for c in p['cases'] if all(any(r['case']==c['name'] and r['arm']==a for r in rows) for a in ARMS)]
    for group in dict.fromkeys(c['group'] for c in completed):
        names=[c['name'] for c in completed if c['group']==group]
        for boundary in ('lossless_frames','delivered_mp4'):
            means={a:mean([r for r in rows if r['case'] in names and r['arm']==a],boundary) for a in ARMS}
            groups.append(dict(group=group,boundary=boundary,cases=names,means=means))
            for base in ARMS[:2]:
                contrasts.append(dict(group=group,boundary=boundary,baseline=base,
                    delta={k:means['motion_context'][k]-v for k,v in means[base].items()},
                    relative_percent={k:100*(means['motion_context'][k]/v-1) if v else None for k,v in means[base].items()}))
            if boundary=='delivered_mp4':
                c,old,base=(means[a] for a in ('motion_context','context_only','current'));bounds=p['criteria']
                flags={k:c[k]>=base[k]+bounds['min_quality_gain_retention']*(old[k]-base[k]) for k in ('psnr_db','nonkey_psnr')}
                flags.update({short:c[k]<=old[k]*(1+bounds[bound]) for short,k,bound in (
                    ('lpips','lpips_vgg','max_lpips_relative_increase_vs_context'),
                    ('boundary','boundary_error','max_boundary_relative_increase_vs_context'))})
                flags.update(flow=c['flow_error_mean']<=base['flow_error_mean']*(1+bounds['max_flow_relative_increase_vs_current']),
                             cadence=c['cadence_error_mean']<=base['cadence_error_mean']*(1+bounds['max_cadence_relative_increase_vs_current']))
                checks.append(dict(group=group,checks=flags,passed=all(flags.values())))
    for identity in dict.fromkeys(c['id'] for c in completed):
        selected=[r for r in rows if r['id']==identity and r['case'] in [c['name'] for c in completed]]
        means={a:mean([r for r in selected if r['arm']==a],'delivered_mp4') for a in ARMS}
        freeze={a:m['false_freeze_rate'] for a,m in means.items()}
        sources.append(dict(id=identity,group=selected[0]['group'],means=means,
            freeze_guardrail=freeze['motion_context']<=freeze['current']+p['criteria']['max_source_freeze_increase_vs_current'],
            pedestrian_excess_reduced=(freeze['motion_context']-freeze['current']<=(1-p['criteria']['min_pedestrian_excess_freeze_reduction'])*max(0,freeze['context_only']-freeze['current'])) if identity.startswith('webvid_002') else None))
    key_checks=[k['exact_received_rgb'] for r in rows if r['arm']=='motion_context' for k in r['lossless_frames']['key_fidelity']]
    value=dict(status='COMPLETE_QUANTITATIVE_REVIEW_PENDING' if not pending else 'PARTIAL',signature=p['signature'],
        completed=len(rows),total=len(p['cases'])*len(ARMS),pending=pending,rows=rows,groups=groups,contrasts=contrasts,sources=sources,
        engineering_screen=checks,key_png_exact=dict(preserved=sum(key_checks),total=len(key_checks)),
        independent_semantic_review='PENDING',automatically_adopted=False)
    write_json(root/'RESULT.json',value);return value


def run(args):
    root=args.root.resolve();previous=args.parent.resolve()
    with common.lock(root):
        event(overall='준비: 입력·정책 동결',stage='검증된 부모 결과 확인 (진행률 미확정)')
        p=prepare(root,previous,args.device,args.strength);signature=p['signature']
        if args.action=='check':print('CHECK: frozen 8 sources / 48 conditions; 32 baselines reused and 16 new outputs',flush=True);return
        result=report(root,p);done=result['completed']
        try:
            for case in p['cases']:
                for arm in ARMS:
                    folder=folder_for(root,case,arm);folder.mkdir(parents=True,exist_ok=True)
                    event(overall=f'{done}/48 ({100*done/48:.1f}%) 평가 완료',current=f'{case["name"]} / {arm}',state='실행 중')
                    if arm!='motion_context':
                        old=folder_for(previous,case,arm)
                        if not common.verify_receipt(folder,'generation',signature):
                            info=folder/'generation_info.json';write_json(info,dict(**read(old/'generation_info.json'),imported_from=str(old)))
                            common.receipt(folder,'generation',signature,[info])
                        if not common.verify_receipt(folder,'evaluation',signature):
                            paths=[folder/name for name in ('quality.json','quality_lossless_frames.csv','quality_delivered_mp4.csv')]
                            for path in paths:shutil.copyfile(old/path.name,path)
                            common.receipt(folder,'evaluation',signature,paths,imported_from=str(old))
                    else:
                        job=case['receiver']
                        if not common.verify_receipt(folder,'generation',signature):
                            receiver=dict(frames=case['frames'],fps=24,keys=case['keys'],frames_paths=job['current_frames'],
                                key_paths=job['key_paths'],base_packet=job['base_packet'],strength=p['strength'],align=p['align'])
                            path=root/'jobs'/case['name']/'receiver.json';write_json(path,receiver)
                            worker('generate',path,folder,signature,args.device)
                        if not common.verify_receipt(folder,'evaluation',signature):
                            generated=read(folder/'generation_info.json');ref_folder=folder_for(previous,case,'context_only')
                            ref=read(ref_folder/'generation_info.json')
                            donor=dict(frames_paths=ref['frames_paths'],movie=ref['movie'],quality=str(ref_folder/'quality.json'),layout='comparison',
                                scores={b:str(ref_folder/f'quality_{b}.csv') for b in ('lossless_frames','delivered_mp4')})
                            evaluation=dict(arm=arm,source_frames=case['source_frames'],source_movie=case['source_movie'],frames=case['frames'],
                                keys=case['keys'],key_paths=job['key_paths'],frames_paths=generated['frames_paths'],movie=generated['movie'],donor=donor)
                            path=root/'jobs'/case['name']/'evaluation.json';write_json(path,evaluation)
                            worker('evaluate',path,folder,signature,args.device)
                    result=report(root,p);done=result['completed']
            event(overall='48/48 (100.0%) 평가 완료',stage='동일 조건 비교 완료',state='정량 완료 / 의미 검수 대기')
            print('REPORT: '+str(root/'RESULT.json'),flush=True)
        except BaseException as error:
            write_json(root/'last_failure.json',dict(error=str(error),completed=done,total=48,resume='repeat command'));raise


def worker(stage,job,folder,signature,device):
    subprocess.run([sys.executable,'-m',MODULE,'--worker',stage,'--job',str(job),'--output',str(folder),
        '--signature',signature,'--device',device],check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',nargs='?',default='run',choices=('run','check'))
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--parent',type=Path,default=parent.ROOT)
    parser.add_argument('--strength',type=float,default=1.);parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--driver',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--worker',choices=('generate','evaluate'),help=argparse.SUPPRESS)
    parser.add_argument('--job',type=Path,help=argparse.SUPPRESS);parser.add_argument('--output',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--signature',help=argparse.SUPPRESS);args=parser.parse_args()
    if not 0<=args.strength<=1:parser.error('--strength must be in [0,1]')
    if args.worker:
        if args.worker=='generate':generate(read(args.job),args.output,args.signature)
        else:
            # Reuse the unchanged evaluator and register the exact-key invariant for this policy.
            parent.FACTORS['motion_context']=(False,'context')
            parent.evaluate(read(args.job),args.output,args.signature,args.device)
    elif args.driver:run(args)
    else:
        from .terminal_progress import run as display
        command=[sys.executable,'-m',MODULE,*sys.argv[1:],'--driver']
        raise SystemExit(display(command,REPO/'.local/terminal_logs/bsu_motion_context',os.environ.get('BSU_PROGRESS','auto')))


if __name__=='__main__':main()
