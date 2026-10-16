"""Paired LK/TAPNext++ synthetic truth and paid BSU receiver comparison."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from . import bsu_motion_validation as common
from .artifacts import sha256,write_json


def lk_queries(frames,points):
    import cv2
    from .sparse_trajectory import gray
    previous=gray(frames[0]);p=points[:,None].copy();alive=np.ones(len(points),bool)
    xy=[points.copy()];valid=[alive.copy()]
    for frame in frames[1:]:
        image=gray(frame);n,status,_=cv2.calcOpticalFlowPyrLK(previous,image,p,None,winSize=(21,21),maxLevel=3)
        b,ok,_=cv2.calcOpticalFlowPyrLK(image,previous,n,None,winSize=(21,21),maxLevel=3)
        alive &= (status[:,0]!=0)&(ok[:,0]!=0)&(np.linalg.norm(b[:,0]-p[:,0],axis=1)<1.)
        pos=n[:,0];alive &= np.isfinite(pos).all(axis=1)&(pos[:,0]>=2)&(pos[:,0]<286)&(pos[:,1]>=2)&(pos[:,1]<158)
        p=np.nan_to_num(n).clip([0,0],[287,159]).astype(np.float32);previous=image
        xy.append(p[:,0].copy());valid.append(alive.copy())
    return np.asarray(xy),np.asarray(valid)


def synthetic(tracker,root):
    import cv2
    from PIL import Image
    from .sparse_trajectory import gray
    rows=[]
    for ci,c in enumerate(common.pilot_cases()[:3]):
        frame=cv2.resize(np.asarray(Image.open(common.source_directory(c)/'0.png').convert('RGB')),(288,160))
        mask=np.zeros((160,288),np.uint8);mask[35:120,45:230]=255
        corners=cv2.goodFeaturesToTrack(gray(frame),maxCorners=32,qualityLevel=.015,minDistance=10,mask=mask)
        if corners is None:continue
        points=corners[:,0]
        for mode in ('translation','reversal','occlusion'):
            frames=[];truth=[];visible=[]
            for t in range(49):
                dx=24*np.sin(np.pi*t/48) if mode!='translation' else t/2
                dy=8*np.sin(2*np.pi*t/48)
                pixels=cv2.warpAffine(frame,np.array([[1,0,dx],[0,1,dy]],np.float32),(288,160),borderMode=cv2.BORDER_REFLECT)
                pos=points+[dx,dy];v=np.ones(len(points),bool)
                if mode=='occlusion' and 16<=t<32:
                    pixels[:,105:185]=64;v &= (pos[:,0]<105)|(pos[:,0]>=185)
                frames.append(pixels);truth.append(pos);visible.append(v)
            frames=np.stack(frames);truth=np.stack(truth);visible=np.stack(visible)
            for name,fn in [('lk',lk_queries),('tapnextpp',tracker.queries)]:
                start=time.monotonic();xy,v=fn(frames,points);error=np.linalg.norm(xy-truth,axis=-1)
                row=dict(id=c['id'],mode=mode,tracker=name,seconds=time.monotonic()-start,
                    visible_epe=float(error[visible].mean()),correct_within_3px=float(((error<3)&v)[visible].mean()),
                    visibility_accuracy=float((v==visible).mean()),visible_recall=float(v[visible].mean()),
                    redetection_recall=float(v[32:][visible[32:]].mean()) if mode=='occlusion' else None)
                rows.append(row)
            common.progress('TRACKER known-geometry sequences',len(rows)//2,9)
    write_json(root/'tracking/synthetic.json',dict(scope='analytically transformed real textures; controlled diagnostic, not real-world ground truth',rows=rows))


def prepare(root):
    import cv2
    from .learned_trajectory import TapNextTracker,candidates_from_tracks
    from .bsu_sequence_refinement import choose_packet,decode_packet
    from .wire import unpack
    from .research_quality import read_video
    from .packets import accounting
    cv2.setNumThreads(2);dest=root/'tracking';dest.mkdir(parents=True,exist_ok=True)
    tracker=TapNextTracker(common.REPO)
    if not (dest/'synthetic.json').exists():synthetic(tracker,root)
    jobs=[]
    lookup={(r['id'],r['snr']):r for r in common.read(common.AUDIT)['rows']}
    for ci,c in enumerate(common.pilot_cases()):
        d=dest/c['id'];d.mkdir(exist_ok=True);cached=d/'sender.json'
        if cached.exists():jobs.append(common.read(cached));continue
        source=read_video(c['source_movie']);source=np.stack([cv2.resize(f,(288,160),interpolation=cv2.INTER_AREA) for f in source])
        candidates=[];diagnostics=[]
        for si,(a,b) in enumerate(zip(c['keys'],c['keys'][1:])):
            cache=d/f'tracks_{si:03d}.npz'
            start=time.monotonic()
            if cache.exists():
                data=np.load(cache);tracks=[(data[f'xy{i}'],data[f'valid{i}']) for i in range(2)]
            else:
                tracks=[tracker.track(source[a:b+1],reverse=v) for v in (False,True)]
                np.savez_compressed(cache,**{f'{key}{i}':track[j] for i,track in enumerate(tracks) for j,key in enumerate(('xy','valid'))})
            candidates.append(candidates_from_tracks(tracks,a,b))
            diagnostics.append(dict(start=a,end=b,seconds=time.monotonic()-start,
                points=[t[0].shape[1] for t in tracks],valid_fraction=[float(t[1].mean()) if t[1].size else 0. for t in tracks]))
        old=lookup[c['id'],10]['sender'];packet=Path(c['receiver']['base_packet']).read_bytes();h,_=unpack(packet)
        budget=old['extra_channel_uses']
        new,info=choose_packet(candidates,h['video'],sha256(c['receiver']['base_packet']),budget,4)
        if new:
            (d/'trajectory_tx.bin').write_bytes(new);decode_packet(new,packet)
        job=dict(case=c,folder=str(d),packet_present=bool(new),packet_info=info,
            maximum_channel_uses=budget,actual_channel_uses=accounting(len(new))['complex_channel_uses'] if new else 0,
            diagnostics=diagnostics)
        write_json(cached,job);jobs.append(job);common.progress('TRACKER WebVid sources',ci+1,5)
    write_json(dest/'plan.json',dict(jobs=jobs,tracker='TAPNext++ public 256 checkpoint',
        checkpoint_sha256=sha256(common.REPO/'.local/checkpoints/tapnextpp/model.pt'),
        vendor_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=common.REPO/'.local/vendor/tapnet',text=True).strip(),
        packet_policy='same legacy selection/quantization/knots; cap at previous actual trajectory channel uses',
        synthetic_ground_truth='known geometric transform; independent from tracker',
        criterion=common.read(root/'flow_reconstruction/plan.json')['acceptance']))


def channel(root):
    from .metadata_channel import transmit
    from .bsu_sequence_refinement import decode_packet
    for j in common.read(root/'tracking/plan.json')['jobs']:
        d=Path(j['folder']);info=dict(packet_present=j['packet_present'])
        if j['packet_present']:
            original=(d/'trajectory_tx.bin').read_bytes();received,r=transmit(original,10,43,bits_per_symbol=4)
            if original!=received:raise ValueError('learned trajectory corrupted')
            decode_packet(received,Path(j['case']['receiver']['base_packet']).read_bytes())
            (d/'trajectory_rx.bin').write_bytes(received);info.update(r,exact_match=True)
        write_json(d/'channel.json',info)


def compare(root):
    from .bsu_receiver import refine_windows,encode_frames
    from .bsu_sequence_refinement import decode_packet
    from .research_quality import read_video
    from .bsu_receiver_improvement import PersistentMetrics
    from .stutter_metrics import evaluate
    from .bsu_receiver_compare import boundary_diagnostics
    import statistics as st
    metrics=PersistentMetrics();rows=[]
    for i,j in enumerate(common.read(root/'tracking/plan.json')['jobs']):
        c=j['case'];d=Path(j['folder']);source=read_video(c['source_movie'])
        for arm in ('lk','tapnextpp'):
            dest=d/arm;dest.mkdir(exist_ok=True);q=dest/'quality.json'
            if q.exists():rows.append(common.read(q));continue
            path=Path(c['receiver']['trajectory']) if arm=='lk' else d/'trajectory_rx.bin'
            tracks=decode_packet(path.read_bytes(),Path(c['receiver']['base_packet']).read_bytes()) if path.exists() else {}
            refine_windows([Path(p) for p in c['receiver']['raw_frames']],c['keys'],Path(c['receiver']['key_paths'][0]).parent,tracks,dest/'frames')
            encode_frames(dest/'frames',dest/'reconstruction.mp4',c['frames'],24)
            current=read_video(dest/'reconstruction.mp4');quality,_=metrics.evaluate(source,current)
            row=dict(id=c['id'],arm=arm,quality=quality,temporal=evaluate(source,current),boundaries=boundary_diagnostics(source,current,c['keys']),
                extra_channel_uses=j['maximum_channel_uses'] if arm=='lk' else j['actual_channel_uses'])
            write_json(q,row);rows.append(row)
        common.progress('TRACKER reconstruction comparison',i+1,5)
    means={arm:dict(**{k:st.mean(r['quality'][k] for r in rows if r['arm']==arm) for k in ('psnr_db','lpips_vgg','ssim','clip','dists')},
        **{k:st.mean(r['temporal'][k] for r in rows if r['arm']==arm) for k in ('flow_error_mean','cadence_error_mean','false_freeze_rate')}) for arm in ('lk','tapnextpp')}
    result=dict(status='COMPLETE_LK_TAPNEXT_PAIRED_PILOT',means=means,rows=rows,
        synthetic=common.read(root/'tracking/synthetic.json'),frozen_generator=True,source_used_by_receiver=False,
        scope='five development videos; received trajectory use remains postgeneration; no claim of unseen-source generalization')
    write_json(root/'tracking/RESULT.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('rows','synthetic')},ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['run','prepare','channel','compare']);p.add_argument('--root',type=Path,default=common.ROOT)
    a=p.parse_args()
    if a.stage!='run':return globals()[a.stage](a.root)
    for stage in ('prepare','channel','compare'):
        interpreter=common.CHANNEL_PYTHON if stage=='channel' else sys.executable
        subprocess.run([str(interpreter),'-m',__package__+'.bsu_tracking_validation',stage,'--root',str(a.root)],check=True)


if __name__=='__main__':main()
