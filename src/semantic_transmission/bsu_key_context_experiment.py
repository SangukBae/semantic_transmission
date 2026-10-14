"""Frozen 2x3 receiver validation: correspondence gate x key context policy."""
import argparse
import base64
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .artifacts import sha256, write_json
from . import bsu_receiver_compare as common
from . import bsu_compare_transfer as transfer

REPO=Path(__file__).resolve().parents[2]
PARENT=common.ROOT
EXTRA=REPO/'outputs/bsu_adaptive_visibility_20261012_v1'
ROOT=REPO/'outputs/bsu_key_context_20261013_v1'
MODULE='semantic_transmission.bsu_key_context_experiment'
ARMS=('current','guard_only','key_only','guard_key','context_only','guard_context')
FACTORS={'current':(False,'none'),'guard_only':(True,'none'),
         'key_only':(False,'legacy'),'guard_key':(True,'legacy'),
         'context_only':(False,'context'),'guard_context':(True,'context')}
CODE=(*common.CODE,'receiver_key_context.py','bsu_key_context_experiment.py')
METRICS=('psnr_db','ssim','lpips_vgg','clip','dists')
read=common.read
event=common.event
tick=common.tick


def import_requests(parent):
    manifest=read(parent/'transfer_manifest.json');requests={}
    for case in manifest['cases']:
        result=read(parent/'inputs'/case['result_path'])
        for boundary in ('lossless_frames','delivered_mp4'):
            name='refined/quality_'+boundary+'.csv'
            relative=(Path(case['result_path']).parent/name).as_posix()
            requests[relative]=result['artifacts'][name]
    return manifest,requests


def missing_imports(root,parent):
    manifest,requests=import_requests(parent)
    return manifest,{name:digest for name,digest in requests.items()
        if not (root/'metric_imports'/name).is_file() or sha256(root/'metric_imports'/name)!=digest}


def fetch_csv(root,parent):
    manifest,requests=missing_imports(root,parent)
    if requests:
        script=('import pathlib,json,base64,hashlib\nitems='+repr(requests)+'\nrepo=pathlib.Path('+repr(manifest['remote_repo'])+')\n'
                'out={}\nfor name,expected in items.items():\n data=(repo/name).read_bytes()\n'
                ' assert hashlib.sha256(data).hexdigest()==expected,name\n'
                ' out[name]=base64.b64encode(data).decode()\nprint(json.dumps(out))\n')
        event(stage='기존 프레임별 지표 CSV만 가져오기 (진행률 미확정)')
        proc=subprocess.run([*transfer.ssh_args(manifest['host']),'-o','BatchMode=yes',manifest['host'],'python3 -'],
                            input=script,text=True,capture_output=True,check=True)
        content=json.loads(proc.stdout)
        if set(content)!=set(requests):raise ValueError('metric import file set mismatch')
        for name,encoded in content.items():
            target=transfer.local_file(root/'metric_imports',name);target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(base64.b64decode(encoded,validate=True))
            if sha256(target)!=requests[name]:raise ValueError('metric import digest mismatch')


def csv_rows(path):
    if Path(path).suffix=='.json':return read(Path(path))
    with Path(path).open() as stream:
        return [{k:int(v) if k=='frame' else float(v) for k,v in row.items()} for row in csv.DictReader(stream)]


def prepare(root,parent,extra,device):
    from .receiver_key_context import PARAMETERS
    from .receiver_key_guard import ANCHOR
    from .guarded_trajectory_receiver import GUARD
    fetch_csv(root,parent)
    m=read(parent/'transfer_manifest.json');transfer.validate_manifest(m)
    old=read(parent/'protocol.json');inputs={str(parent/'protocol.json'):sha256(parent/'protocol.json'),
        str(parent/'transfer_manifest.json'):sha256(parent/'transfer_manifest.json')}
    for name,identity in m['files'].items():inputs[str(parent/'inputs'/name)]=identity['sha256']
    cases=[]
    for c in m['cases']:
        source=next(s for s in m['sources'] if s['id']==c['id'])
        refs={}
        for arm,previous in (('current','current_combined'),('guard_key','key_guarded')):
            folder,frames,movie=common.arm_paths(parent,parent/'inputs',c,previous)
            if not common.verify_receipt(folder,'evaluation',old['signature']):raise ValueError('parent evaluation missing')
            if arm=='guard_key' and not common.verify_receipt(folder,'generation',old['signature']):raise ValueError('parent generation missing')
            qpath=folder/'quality.json';inputs[str(qpath)]=sha256(qpath)
            scores={}
            for boundary in ('lossless_frames','delivered_mp4'):
                path=(root/'metric_imports'/Path(c['result_path']).parent/f'refined/quality_{boundary}.csv'
                      if arm=='current' else folder/f'quality_{boundary}.csv')
                scores[boundary]=str(path);inputs[str(path)]=sha256(path)
            if arm=='guard_key':
                for name,digest in read(folder/'generation.json')['artifacts'].items():inputs[str(folder/name)]=digest
            refs[arm]=dict(frames_paths=frames,movie=movie,quality=str(qpath),scores=scores,layout='comparison')
        receiver=common.receiver_inputs(parent/'inputs',c)
        receiver.update(current_frames=refs['current']['frames_paths'],packet_format='separate_trajectory')
        cases.append(dict(name=common.case_name(c),id=c['id'],snr=c['snr'],group='webvid_development',
            source_sha256=source['source_sha256'],source_frames=[str(parent/'inputs'/p) for p in source['frames_paths']],source_movie=None,
            receiver=receiver,refs=refs,frames=c['frames'],keys=c['keys'],channel_uses=c['base_channel_uses']+c['trajectory_channel_uses']))
    ep=read(extra/'protocol.json');inputs[str(extra/'protocol.json')]=sha256(extra/'protocol.json')
    for c in ep['cases']:
        if not c['name'].startswith('transfer_'):continue
        received=extra/'received'/c['name'];baseline=extra/'generation'/c['name']/'baseline';current=extra/'generation'/c['name']/'current_combined'
        for folder,filename in ((received,'receipt.json'),(baseline,'result.json'),(current,'result.json')):
            record=read(folder/filename)
            if record['signature']!=ep['signature']:raise ValueError('extra signature mismatch')
            inputs[str(folder/filename)]=sha256(folder/filename)
            for name,digest in record['artifacts'].items():
                if name!='visual.c64':inputs[str(folder/name)]=digest
        if not read(received/'receipt.json')['exact_match']:raise ValueError('extra metadata inexact')
        quality={};scores={}
        for b in ('lossless_frames','delivered_mp4'):
            q=current/f'evaluation_{b}.json';f=current/f'per_frame_{b}.json'
            quality[b]=str(q);scores[b]=str(f);inputs[str(q)]=sha256(q);inputs[str(f)]=sha256(f)
        inputs[c['source_video']]=c['source_sha256']
        images=lambda folder:[str(folder/f'frames/{i:05d}.png') for i in range(c['frames'])]
        receiver=dict(frames=c['frames'],fps=24,keys=c['keys'],raw_frames=images(baseline),current_frames=images(current),
            key_paths=[str(received/f'key{i}.png') for i in range(len(c['keys']))],base_packet=str(received/'metadata.bin'),
            trajectory=None,packet_format='embedded_trajectory')
        refs=dict(current=dict(frames_paths=images(current),movie=str(current/'reconstruction.mp4'),quality=quality,scores=scores,layout='adaptive'))
        cases.append(dict(name=c['name']+'/snr_10',id=c['source_id'],snr=10,group='davis_transfer',source_sha256=c['source_sha256'],
            source_frames=None,source_movie=c['source_video'],receiver=receiver,refs=refs,frames=c['frames'],keys=c['keys'],
            channel_uses=c['channel']['candidate_uses']))
    if len({c['source_sha256'] for c in cases})!=8 or len(cases)!=16:raise ValueError('expected 8 sources and 16 cases')
    p=dict(policy='bsu_key_context_factorial_v1',cases=cases,arms=list(ARMS),factors={k:list(v) for k,v in FACTORS.items()},
        parameters=dict(context=PARAMETERS,legacy_anchor=ANCHOR,gate=GUARD),inputs=inputs,
        code={name:sha256(REPO/'src/semantic_transmission'/name) for name in CODE},runtime=common.runtime_identity(device),
        receiver_reads_source=False,training=False,generator_reused=True,channel_reused=True,
        primary_arm='context_only',primary_boundary='delivered_mp4',
        preregistered_criteria=dict(key_png_exact=True,boundary_error_reduction_vs_key_only=.20,
            max_psnr_loss_db=.10,max_nonkey_psnr_loss_db=.10,max_lpips_relative_increase=.01,
            max_flow_relative_increase=.02,max_cadence_relative_increase=.02,max_false_freeze_increase=.02),
        criteria_scope='engineering screening bounds, not ETRI acceptance thresholds or statistical noninferiority; each dataset group separately',
        tuning='shared settings fixed before outputs; no per-video tuning or source-based receiver decisions',
        scope=dict(webvid='four inspected development sources, three SNRs',davis='four 96-frame transfer sources, one SNR; historical LGVSC and adaptive receiver use, not globally unseen'),
        independent_semantic_review='PENDING')
    p['signature']=transfer.fingerprint(p)
    path=root/'protocol.json'
    if path.exists() and read(path)!=p:raise ValueError('frozen inputs/code/options changed; choose new --root')
    callback=tick('동결 입력 해시 검사')
    for i,(name,digest) in enumerate(inputs.items(),1):
        if sha256(Path(name))!=digest:raise ValueError('input changed: '+name)
        if i%100==0 or i==len(inputs):callback(i,len(inputs))
    if not path.exists():
        write_json(path,p)
        for name in CODE:
            target=root/'frozen_code'/name;target.parent.mkdir(exist_ok=True);shutil.copyfile(REPO/'src/semantic_transmission'/name,target)
    return p


def folder_for(root,case,arm):return root/'comparison'/case['name']/arm


def generate(job,folder,signature,arm):
    import cv2
    import numpy as np
    from PIL import Image
    from .guarded_trajectory_receiver import refine_references
    from .trajectory_receiver import refine_temporal
    from .receiver_key_guard import preserve_keys
    from .receiver_key_context import preserve_context
    from .bsu_sequence_refinement import decode_packet,validate_keys
    from .sparse_trajectory import FIELD,decode
    from .wire import unpack
    from .etri_60s_check import pts_audit
    allowed={'frames','fps','keys','raw_frames','current_frames','key_paths','base_packet','trajectory','packet_format','guard_frames'}
    if set(job)!=allowed:raise ValueError('receiver job has unexpected/source fields')
    validate_keys(job['keys'],job['frames']);guard,anchor=FACTORS[arm]
    cv2.setNumThreads(2);cv2.setRNGSeed(2025)
    packet=Path(job['base_packet']).read_bytes();header,_=unpack(packet)
    if header['video']!=dict(frames=job['frames'],fps=24,width=576,height=320):raise ValueError('packet timeline differs')
    if [k['index'] for k in header['keyframes']]!=job['keys']:raise ValueError('packet keys differ')
    tracks=(decode_packet(Path(job['trajectory']).read_bytes(),packet) if job['trajectory'] else {})
    if job['packet_format']=='embedded_trajectory' and FIELD in header:tracks={(0,job['frames']-1):decode(header[FIELD],job['frames'])}
    paths=(job['raw_frames'] if arm=='guard_only' else job['guard_frames'] if guard else job['current_frames'])
    if not paths or len(paths)!=job['frames']:raise ValueError('base frames unavailable')
    output=folder/'frames';output.mkdir(parents=True,exist_ok=True);audits=[]
    for index,(a,b) in enumerate(zip(job['keys'],job['keys'][1:])):
        frames=common.pixels(paths[a:b+1],tick(f'구간 {index+1} PNG 읽기'))
        keys=[np.asarray(Image.open(p).convert('RGB')) for p in job['key_paths'][index:index+2]]
        audit=dict(start=a,end=b)
        if arm=='guard_only':
            result,audit['gate']=refine_references(frames,keys,tracks.get((a,b)),tick('윤곽 검사'))
            result,audit['cadence']=refine_temporal(result,tick('시간 보정'))
        elif anchor=='legacy':result,audit['anchor']=preserve_keys(frames,keys,tick('기존 키 보존'))
        elif anchor=='context':result,audit['anchor']=preserve_context(frames,keys,tick('키 전후 구간 연결'))
        else:raise ValueError('baseline must be imported')
        callback=tick('PNG 저장')
        for j,array in enumerate(result):
            path=output/f'{a+j:05d}.png'
            if index and j==0:
                if not np.array_equal(np.asarray(Image.open(path).convert('RGB')),array):raise ValueError('shared boundary mismatch')
                continue
            temp=path.with_suffix('.tmp.png');temp.unlink(missing_ok=True)
            if np.array_equal(array,frames[j]):os.link(paths[a+j],temp)
            else:Image.fromarray(array).save(temp)
            temp.replace(path);callback(j+1,len(result))
        audits.append(audit)
    movie=folder/'reconstruction.mp4';tmp=folder/'encoding.tmp.mp4'
    event(stage='MP4 인코딩 (진행률 미확정)')
    subprocess.run(['ffmpeg','-y','-v','error','-nostdin','-threads','2','-framerate','24','-i',str(output/'%05d.png'),
                    '-c:v','libx264','-crf','18','-preset','fast','-pix_fmt','yuv420p','-threads','2',str(tmp)],check=True)
    audit=pts_audit(tmp,job['frames'],24);tmp.replace(movie)
    frames_paths=[str(output/f'{i:05d}.png') for i in range(job['frames'])]
    info=folder/'generation_info.json';write_json(info,dict(arm=arm,frames_paths=frames_paths,movie=str(movie),windows=audits,timeline=audit))
    common.receipt(folder,'generation',signature,[*[Path(p) for p in frames_paths],movie,info])


def reference_quality(ref,boundary):
    return read(Path(ref['quality']))[boundary] if ref['layout']=='comparison' else read(Path(ref['quality'][boundary]))


def evaluate(job,folder,signature,device):
    import cv2
    import numpy as np
    from .official_quality import OfficialMetrics,summarize
    from .research_quality import read_video
    from .stutter_metrics import evaluate as temporal
    from .etri_60s_check import pts_audit
    cv2.setNumThreads(2)
    source=(common.pixels(job['source_frames'],tick('원본 평가 입력')) if job['source_frames'] else read_video(job['source_movie']))
    keys=common.pixels(job['key_paths']);n=len(source)
    if n!=job['frames']:raise ValueError('source length mismatch')
    pts_audit(Path(job['movie']),n,24)
    q=dict(arm=job['arm'],source_only_used_for_evaluation=True,frames=n)
    artifacts=[]
    for boundary in ('lossless_frames','delivered_mp4'):
        generated=common.pixels(job['frames_paths'],tick('출력 PNG 읽기')) if boundary=='lossless_frames' else read_video(job['movie'])
        ref=job['donor'];donor=common.pixels(ref['frames_paths']) if boundary=='lossless_frames' else read_video(ref['movie'])
        if generated.shape!=source.shape or donor.shape!=source.shape:raise ValueError('metric timeline mismatch')
        old=csv_rows(ref['scores'][boundary]);old_quality=reference_quality(ref,boundary)['quality']
        if len(old)!=n or [r['frame'] for r in old]!=list(range(n)):raise ValueError('metric donor rows mismatch')
        if any(abs(summarize(old,source.shape)[k]-old_quality[k])>1e-8 for k in METRICS):raise ValueError('donor summary mismatch')
        same=np.all(generated==donor,axis=(1,2,3));changed=np.flatnonzero(~same)
        # Validate source identity for every reused row, not just cache names.
        for i in np.flatnonzero(same):
            if abs(cv2.PSNR(source[i],donor[i])-old[i]['psnr_db'])>1e-8:raise ValueError('cached source/pixels differ')
        rows=[dict(r) for r in old]
        event(stage=f'{boundary}: 동일 픽셀 {int(same.sum())}/{n} 재사용, {len(changed)}프레임 평가')
        if len(changed):
            _,new=OfficialMetrics(device).evaluate(source[changed],generated[changed],tick(boundary+' 화질 지표'))
            for index,row in zip(changed,new):rows[int(index)]=dict(row,frame=int(index))
        scores=summarize(rows,source.shape)
        # Recalculate temporal diagnostics including transitions across changed boundaries.
        motion=temporal(source,generated,tick(boundary+' 시간 지표'))
        fidelity=[dict(frame=k,output_vs_source_psnr=float(cv2.PSNR(source[k],generated[k])),
            output_vs_received_psnr=float(cv2.PSNR(key,generated[k])),exact_received_rgb=bool(np.array_equal(key,generated[k])))
            for k,key in zip(job['keys'],keys)]
        if FACTORS[job['arm']][1]!='none' and boundary=='lossless_frames' and not all(k['exact_received_rgb'] for k in fidelity):raise ValueError('received keys not exact')
        parts={'keys':[r for r in rows if r['frame'] in job['keys']],
               'nonkeys':[r for r in rows if r['frame'] not in job['keys']]}
        q[boundary]=dict(quality=scores,temporal=motion,key_fidelity=fidelity,
            key_boundary_changes=common.boundary_diagnostics(source,generated,job['keys']),
            partitions={name:summarize(values,[len(values),*source.shape[1:]]) for name,values in parts.items()},
            exact_pixel_metric_reuse=int(same.sum()),recomputed_frames=len(changed))
        path=folder/f'quality_{boundary}.csv'
        with path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
        artifacts.append(path);del generated,donor
    path=folder/'quality.json';write_json(path,q);artifacts.append(path)
    common.receipt(folder,'evaluation',signature,artifacts)


def report(root,p):
    import statistics as st
    rows=[];pending=[]
    for case in p['cases']:
        for arm in ARMS:
            folder=folder_for(root,case,arm)
            if not common.verify_receipt(folder,'evaluation',p['signature']):pending.append([case['name'],arm]);continue
            q=read(folder/'quality.json')
            rows.append(dict(case=case['name'],id=case['id'],group=case['group'],snr=case['snr'],arm=arm,
                             channel_uses=case['channel_uses'],**{b:q[b] for b in ('lossless_frames','delivered_mp4')}))
    groups=[];contrasts=[];decisions=[]
    def stats(selected,b):
        return dict(**{k:st.mean(r[b]['quality'][k] for r in selected) for k in METRICS},
            **{k:st.mean(r[b]['temporal'][k] for r in selected) for k in ('flow_error_mean','cadence_error_mean','false_freeze_rate','max_luma_impulse')},
            nonkey_psnr=st.mean(r[b]['partitions']['nonkeys']['psnr_db'] for r in selected),
            key_psnr=st.mean(r[b]['partitions']['keys']['psnr_db'] for r in selected),
            boundary_error=st.mean(st.mean(k['maximum_change_error'] for k in r[b]['key_boundary_changes']) for r in selected))
    for group in ('webvid_development','davis_transfer'):
        complete=[c['name'] for c in p['cases'] if c['group']==group and all(any(r['case']==c['name'] and r['arm']==a for r in rows) for a in ARMS)]
        if not complete:continue
        for b in ('lossless_frames','delivered_mp4'):
            aggregate={a:stats([r for r in rows if r['case'] in complete and r['arm']==a],b) for a in ARMS}
            groups.append(dict(group=group,boundary=b,cases=complete,means=aggregate))
            pairs=[('guard_only','current'),('key_only','current'),('guard_key','guard_only'),('guard_key','key_only'),
                   ('context_only','current'),('context_only','key_only'),('guard_context','guard_only'),('guard_context','guard_key'),('guard_context','context_only')]
            for a,base in pairs:
                contrasts.append(dict(group=group,boundary=b,arm=a,baseline=base,delta={k:aggregate[a][k]-v for k,v in aggregate[base].items()},
                    relative_percent={k:100*(aggregate[a][k]/v-1) if v else None for k,v in aggregate[base].items()}))
            if b=='delivered_mp4':
                c,base,legacy=aggregate['context_only'],aggregate['current'],aggregate['key_only'];bounds=p['preregistered_criteria']
                checks=dict(psnr=c['psnr_db']>=base['psnr_db']-bounds['max_psnr_loss_db'],
                    nonkey=c['nonkey_psnr']>=base['nonkey_psnr']-bounds['max_nonkey_psnr_loss_db'],
                    lpips=c['lpips_vgg']<=base['lpips_vgg']*(1+bounds['max_lpips_relative_increase']),
                    flow=c['flow_error_mean']<=base['flow_error_mean']*(1+bounds['max_flow_relative_increase']),
                    cadence=c['cadence_error_mean']<=base['cadence_error_mean']*(1+bounds['max_cadence_relative_increase']),
                    freeze=c['false_freeze_rate']<=base['false_freeze_rate']+bounds['max_false_freeze_increase'],
                    boundary=c['boundary_error']<=legacy['boundary_error']*(1-bounds['boundary_error_reduction_vs_key_only']),
                    keys=all(k['exact_received_rgb'] for r in rows if r['case'] in complete and r['arm']=='context_only' for k in r['lossless_frames']['key_fidelity']))
                decisions.append(dict(group=group,checks=checks,passed=all(checks.values())))
    compact=[]
    for r in rows:
        r={**r,**{b:{**r[b],'temporal':{k:v for k,v in r[b]['temporal'].items() if k!='per_transition'}} for b in ('lossless_frames','delivered_mp4')}}
        compact.append(r)
    value=dict(status='COMPLETE_QUANTITATIVE_REVIEW_PENDING' if not pending else 'PARTIAL',signature=p['signature'],
        completed=len(rows),total=len(p['cases'])*len(ARMS),pending=pending,rows=compact,groups=groups,contrasts=contrasts,
        primary_candidate_screen=decisions,independent_semantic_review='PENDING',automatically_adopted=False)
    write_json(root/'RESULT.json',value)
    return value


def worker(stage,job,folder,signature,arm,device):
    subprocess.run([sys.executable,'-m',MODULE,'--worker',stage,'--job',str(job),'--output',str(folder),
                    '--signature',signature,'--arm',arm,'--device',device],check=True)


def run(args):
    root=args.root.resolve();parent=args.parent.resolve();extra=args.extra.resolve()
    with common.lock(root):
        event(overall='준비: 입력·정책 동결',stage='기존 산출물 검증 (진행률 미확정)')
        p=prepare(root,parent,extra,args.device);signature=p['signature']
        result=report(root,p);total=result['total'];done=result['completed']
        if args.action=='check':print(f'CHECK: 8 sources, 16 cases x 6 arms = {total}; frozen, no generation',flush=True);return
        try:
            for case in p['cases']:
                for arm in ARMS:
                    folder=folder_for(root,case,arm);folder.mkdir(parents=True,exist_ok=True)
                    event(overall=f'{done}/{total} ({100*done/total:.1f}%) 평가 완료',current=f'{case["name"]} / {arm}',state='실행 중')
                    if not common.verify_receipt(folder,'generation',signature):
                        if arm in case['refs']:
                            path=folder/'generation_info.json';write_json(path,dict(arm=arm,**case['refs'][arm],imported=True))
                            common.receipt(folder,'generation',signature,[path])
                        else:
                            guard_folder=folder_for(root,case,'guard_only')
                            job={**case['receiver'],'guard_frames':([str(guard_folder/f'frames/{i:05d}.png') for i in range(case['frames'])]
                                if FACTORS[arm][0] and arm!='guard_only' else None)}
                            path=root/'jobs'/case['name']/(arm+'_receiver.json');write_json(path,job)
                            worker('generate',path,folder,signature,arm,args.device)
                    if common.verify_receipt(folder,'evaluation',signature):event(state='검증된 완료 결과 재사용');continue
                    generated=read(folder/'generation_info.json')
                    if FACTORS[arm][0] and 'guard_key' in case['refs']:donor=case['refs']['guard_key']
                    elif FACTORS[arm][0] and arm!='guard_only':
                        ref_folder=folder_for(root,case,'guard_only');ref=read(ref_folder/'generation_info.json')
                        donor=dict(frames_paths=ref['frames_paths'],movie=ref['movie'],quality=str(ref_folder/'quality.json'),
                            scores={b:str(ref_folder/f'quality_{b}.csv') for b in ('lossless_frames','delivered_mp4')},layout='comparison')
                    else:donor=case['refs']['current']
                    job=dict(arm=arm,source_frames=case['source_frames'],source_movie=case['source_movie'],frames=case['frames'],
                        keys=case['keys'],key_paths=case['receiver']['key_paths'],frames_paths=generated['frames_paths'],movie=generated['movie'],donor=donor)
                    path=root/'jobs'/case['name']/(arm+'_evaluation.json');write_json(path,job)
                    worker('evaluate',path,folder,signature,arm,args.device)
                    result=report(root,p);done=result['completed']
            event(overall=f'{done}/{total} (100.0%) 평가 완료',stage='동일 조건 대조 집계 완료',state='정량 완료 / 의미 검수 대기')
            print('REPORT: '+str(root/'RESULT.json'),flush=True)
        except BaseException as error:
            write_json(root/'last_failure.json',dict(error=str(error),type=type(error).__name__,completed=done,total=total,resume='repeat command'))
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',nargs='?',default='run',choices=('run','check'))
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--parent',type=Path,default=PARENT);p.add_argument('--extra',type=Path,default=EXTRA)
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda');p.add_argument('--driver',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--worker',choices=('generate','evaluate'),help=argparse.SUPPRESS);p.add_argument('--job',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--output',type=Path,help=argparse.SUPPRESS);p.add_argument('--signature',help=argparse.SUPPRESS);p.add_argument('--arm',choices=ARMS,help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.worker:
        if args.worker=='generate':generate(read(args.job),args.output,args.signature,args.arm)
        else:evaluate(read(args.job),args.output,args.signature,args.device)
    elif args.driver:run(args)
    else:
        manifest,missing=missing_imports(args.root.resolve(),args.parent.resolve())
        if missing:
            print('기존 지표 CSV 인증: 원격 읽기 연결을 준비합니다.',flush=True);transfer.authenticate(manifest['host'])
        from .terminal_progress import run as display
        command=[sys.executable,'-m',MODULE,*sys.argv[1:],'--driver']
        raise SystemExit(display(command,REPO/'.local/terminal_logs/bsu_key_context',os.environ.get('BSU_PROGRESS','auto')))


if __name__=='__main__':main()
