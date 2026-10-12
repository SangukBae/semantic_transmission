"""Apply the BSU cadence option, or opt into guarded trajectory refinement.

Cadence needs only received metadata.bin and completed BSU generation/frames.
guarded_combined also needs received key0.png, key1.png and paid trajectories.
Sender originals are not inputs. The default cadence option adds no payload.
"""
import argparse
import fcntl
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from .artifacts import sha256,write_json
from .webvid5 import read_json,fingerprint
from .wire import unpack
from .sparse_trajectory import FIELD,decode
from .trajectory_receiver import PARAMETERS,refine_temporal
from .guarded_trajectory_receiver import GUARD,refine_references
from .research_quality import read_frames
from .receiver_joint_experiment import save_pixels


def execute(received,generation,output,mode='cadence'):
    if mode not in ['cadence','guarded_combined']:raise ValueError('unsupported refinement mode')
    received,generation,output=map(lambda x:Path(x).resolve(),(received,generation,output))
    if any(output==p or output in p.parents or p in output.parents for p in [received,generation]):
        raise ValueError('output must be separate from received inputs and baseline generation')
    packet=(received/'metadata.bin').read_bytes();header,_=unpack(packet)
    count=header['video']['frames'];fps=header['video']['fps']
    if [k['index'] for k in header['keyframes']]!=[0,count-1]:
        raise ValueError('requires one two-endpoint BSU window; split multi-key runs into accounted windows')
    frame_paths=sorted((generation/'frames').glob('*.png'))
    if len(frame_paths)!=count:raise ValueError('received/generated timeline mismatch')
    paths=[received/'metadata.bin',*frame_paths]
    if mode=='guarded_combined':paths.extend([received/'key0.png',received/'key1.png'])
    code_paths=[Path(__file__),*[Path(__file__).with_name(n) for n in ['trajectory_receiver.py','guarded_trajectory_receiver.py','sparse_trajectory.py']]]
    p=dict(policy='bsu_optional_receiver_refinement_v1',mode=mode,inputs={str(p):sha256(p) for p in paths},
           code={str(p):sha256(p) for p in code_paths},parameters=PARAMETERS,guard=GUARD,
           source_images_read=False,additional_receiver_transmission=0,
           trajectory_transport='not consumed' if mode=='cadence' else 'already in the supplied received packet',
           opencv=cv2.__version__,numpy=np.__version__,frames=count,fps=fps)
    p['signature']=fingerprint(p)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (output/'protocol.json').exists() and read_json(output/'protocol.json')!=p:
            raise ValueError('existing output has different inputs/code/options')
        if (output/'RESULT.json').exists():
            result=read_json(output/'RESULT.json')
            if result['signature']!=p['signature'] or any(sha256(output/k)!=v for k,v in result['artifacts'].items()):
                raise ValueError('completed refinement changed')
            print('COMPLETE_REUSED 1/1 (100%); all artifacts verified, no refinement repeated',flush=True)
            return result
        write_json(output/'protocol.json',p);cv2.setNumThreads(2);cv2.setRNGSeed(2025)
        base=read_frames(generation/'frames')
        if base.shape!=(count,header['video']['height'],header['video']['width'],3):raise ValueError('RGB geometry differs from received metadata')
        operations=[]
        if mode=='guarded_combined':
            keys=[np.asarray(Image.open(received/f'key{j}.png').convert('RGB')) for j in range(2)]
            trajectory=decode(header[FIELD],count) if FIELD in header else None
            operations.append(('reference',lambda images,tick:refine_references(images,keys,trajectory,tick)))
        operations.append(('cadence',lambda images,tick:refine_temporal(images,tick)))
        overall=time.monotonic();reports=[]
        for i,(label,operation) in enumerate(operations):
            start=time.monotonic()
            def tick(done,total):
                if done==1 or done%32==0 or done==total:
                    elapsed=time.monotonic()-start
                    eta=f'{elapsed/done*(total-done):.1f}s' if done>=32 else 'unknown'
                    print(f'OVERALL stage {i+1}/{len(operations)} | {label} {done}/{total} ({100*done/total:.1f}%) elapsed={elapsed:.1f}s ETA={eta}',flush=True)
            current,audit=operation(base if i==0 else current,tick);reports.append(audit)
        if not np.array_equal(current[[0,-1]],base[[0,-1]]):raise ValueError('endpoint changed')
        print('EXPORT PNG/MP4; encoding ETA unknown',flush=True);save_pixels(output,current,fps)
        artifacts=[*sorted((output/'frames').glob('*.png')),output/'reconstruction.mp4']
        result=dict(status='COMPLETE',signature=p['signature'],mode=mode,frames=count,fps=fps,seconds=time.monotonic()-overall,
                    audits=reports,endpoint_pixels_preserved=True,source_images_read=False,
                    artifacts={str(path.relative_to(output)):sha256(path) for path in artifacts})
        write_json(output/'RESULT.json',result)
        print(f'COMPLETE 1/1 (100%) elapsed={result["seconds"]:.1f}s',flush=True)
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['received','generation','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--mode',choices=['cadence','guarded_combined'],default='cadence')
    args=parser.parse_args();execute(args.received,args.generation,args.output,args.mode)


if __name__=='__main__':main()
