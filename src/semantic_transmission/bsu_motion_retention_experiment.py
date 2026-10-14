"""Guard the frozen residual candidate using receiver-only motion retention."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
from . import bsu_motion_context_experiment as runner
from .receiver_motion_retention import PARAMETERS
from .artifacts import sha256,write_json

ROOT=runner.REPO/'outputs/bsu_motion_retention_20261013_v1'
RESIDUAL=runner.ROOT
MODULE='semantic_transmission.bsu_motion_retention_experiment'
CODE=(*runner.CODE,'receiver_motion_retention.py','bsu_motion_retention_experiment.py')
read=runner.read


def prepare(root,previous,device,strength):
    p0=read(previous/'protocol.json');pr=read(RESIDUAL/'protocol.json')
    if pr['parent_signature']!=p0['signature'] or strength!=pr['strength']:raise ValueError('residual policy differs')
    if read(RESIDUAL/'RESULT.json')['completed']!=48:raise ValueError('residual experiment incomplete')
    for name,digest in pr['code'].items():
        if sha256(runner.REPO/'src/semantic_transmission'/name)!=digest:raise ValueError('residual code changed')
    inputs={**pr['inputs'],str(RESIDUAL/'protocol.json'):sha256(RESIDUAL/'protocol.json')}
    for case in p0['cases']:
        folder=runner.folder_for(RESIDUAL,case,'motion_context')
        for kind in ('generation','evaluation'):
            if not runner.common.verify_receipt(folder,kind,pr['signature']):raise ValueError('residual receipt missing')
            for name,digest in read(folder/f'{kind}.json')['artifacts'].items():inputs[str(folder/name)]=digest
    development={}
    for directory in ('bsu_motion_gate_20261013_pilot1','bsu_motion_retention_20261013_pilot1'):
        for name in ('protocol.json','RESULT.json'):
            path=runner.REPO/'outputs'/directory/name;development[str(path)]=sha256(path)
    inputs.update(development)
    runtime=runner.common.runtime_identity(device)
    if runtime!=pr['runtime']:raise ValueError('evaluation runtime differs')
    p=dict(policy='receiver_motion_retention_v1',parent=str(previous),parent_signature=p0['signature'],
        residual_parent=str(RESIDUAL),residual_signature=pr['signature'],cases=p0['cases'],arms=list(runner.ARMS),
        primary_arm='motion_context',strength=strength,align=True,parameters=PARAMETERS,inputs=inputs,
        code={name:sha256(runner.REPO/'src/semantic_transmission'/name) for name in CODE},runtime=runtime,
        scope=p0['scope'],development='All eight previously inspected sources used in development. PNG pilot revealed harm from a permissive gate; final loss threshold tightened from 25 to 90 percent motion retention.',
        development_evidence=development,criteria=pr['criteria'],criteria_scope=pr['criteria_scope'],
        source_only_for_evaluation=True,new_training=False,additional_transmission=0,generator_reused=True,
        independent_semantic_review='PENDING',automatically_adopted=False)
    p['signature']=runner.transfer.fingerprint(p);path=root/'protocol.json'
    if path.exists() and read(path)!=p:raise ValueError('frozen configuration changed; choose new --root')
    tick=runner.common.tick('동결 입력 해시 검사')
    for i,(name,digest) in enumerate(inputs.items(),1):
        if sha256(Path(name))!=digest:raise ValueError('input changed: '+name)
        if i%100==0 or i==len(inputs):tick(i,len(inputs))
    if not path.exists():
        write_json(path,p)
        for name in CODE:
            target=root/'frozen_code'/name;target.parent.mkdir(exist_ok=True);shutil.copyfile(runner.REPO/'src/semantic_transmission'/name,target)
    return p


def generate(job,folder,signature):
    import cv2
    import numpy as np
    from PIL import Image
    from .receiver_motion_retention import gate_segment
    from .etri_60s_check import pts_audit
    from .bsu_sequence_refinement import validate_keys
    from .wire import unpack
    if set(job)!={'frames','fps','keys','context_paths','motion_paths','windows','key_paths','base_packet'}:
        raise ValueError('receiver job contains unexpected/source fields')
    if (job['fps']!=24 or len(job['context_paths'])!=job['frames'] or len(job['motion_paths'])!=job['frames']
        or len(job['key_paths'])!=len(job['keys']) or len(job['windows'])!=len(job['keys'])-1):
        raise ValueError('receiver timeline or input length differs')
    validate_keys(job['keys'],job['frames']);header,_=unpack(Path(job['base_packet']).read_bytes())
    if header['video']!=dict(frames=job['frames'],fps=24,width=576,height=320):raise ValueError('timeline differs')
    if [k['index'] for k in header['keyframes']]!=job['keys']:raise ValueError('keys differ')
    cv2.setNumThreads(2);output=folder/'frames';output.mkdir(parents=True,exist_ok=True);audits=[]
    for i,(a,b) in enumerate(zip(job['keys'],job['keys'][1:])):
        context=runner.common.pixels(job['context_paths'][a:b+1]);motion=runner.common.pixels(job['motion_paths'][a:b+1])
        keys=runner.common.pixels(job['key_paths'][i:i+2])
        if not np.array_equal(context[[0,-1]],keys):raise ValueError('received RGB differs')
        result,audit=gate_segment(context,motion,job['windows'][i]['endpoints'],runner.common.tick('움직임 감소 구간 제외'))
        audits.append(dict(start=a,end=b,**audit))
        tick=runner.common.tick('검증된 PNG 연결')
        for j,array in enumerate(result):
            path=output/f'{a+j:05d}.png'
            if i and j==0:
                if not np.array_equal(np.asarray(Image.open(path).convert('RGB')),array):raise ValueError('shared key differs')
                continue
            origin=job['context_paths'][a+j] if np.array_equal(array,context[j]) else job['motion_paths'][a+j]
            temporary=path.with_suffix('.tmp.png');temporary.unlink(missing_ok=True);os.link(origin,temporary);temporary.replace(path);tick(j+1,len(result))
    temporary=folder/'encoding.tmp.mp4';movie=folder/'reconstruction.mp4'
    runner.event(stage='MP4 인코딩 (진행률 미확정)')
    subprocess.run(['ffmpeg','-y','-v','error','-nostdin','-framerate','24','-i',str(output/'%05d.png'),'-c:v','libx264',
        '-crf','18','-preset','fast','-pix_fmt','yuv420p','-threads','2',str(temporary)],check=True)
    timeline=pts_audit(temporary,job['frames'],24);temporary.replace(movie)
    paths=[str(output/f'{i:05d}.png') for i in range(job['frames'])];info=folder/'generation_info.json'
    write_json(info,dict(arm='motion_context',policy='receiver_motion_retention_v1',frames_paths=paths,movie=str(movie),windows=audits,timeline=timeline))
    runner.common.receipt(folder,'generation',signature,[*[Path(v) for v in paths],movie,info])


def worker(stage,job,folder,signature,device):
    if stage=='generate':
        root=folder.parents[3];p=read(root/'protocol.json');name='/'.join(folder.parts[-3:-1])
        case=next(c for c in p['cases'] if c['name']==name)
        context=read(runner.folder_for(Path(p['parent']),case,'context_only')/'generation_info.json')
        motion=read(runner.folder_for(Path(p['residual_parent']),case,'motion_context')/'generation_info.json')
        write_json(job,dict(frames=case['frames'],fps=24,keys=case['keys'],context_paths=context['frames_paths'],
            motion_paths=motion['frames_paths'],windows=motion['windows'],key_paths=case['receiver']['key_paths'],base_packet=case['receiver']['base_packet']))
    subprocess.run([sys.executable,'-m',MODULE,'--worker',stage,'--job',str(job),'--output',str(folder),
        '--signature',signature,'--device',device],check=True)


def main():
    # Reuse the verified execution, evaluation and reporting harness unchanged on disk.
    runner.ROOT=ROOT;runner.MODULE=MODULE;runner.CODE=CODE
    runner.prepare=prepare;runner.generate=generate;runner.worker=worker
    runner.main()


if __name__=='__main__':main()
