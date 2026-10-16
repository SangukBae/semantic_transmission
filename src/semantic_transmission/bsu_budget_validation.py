"""Validate partial trajectory allocation on the six frozen no-fit conditions."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from . import bsu_motion_validation as common
from .artifacts import sha256,write_json
from .wire import unpack


def prepare(root):
    from .bsu_sequence_refinement import sender_candidates,choose_packet,decode_packet
    from .trajectory_allocation import allocate
    from .packets import accounting
    from .research_quality import read_video
    import cv2
    cv2.setNumThreads(2)
    if common.verified(root,'budget_prepare'):return
    audit=common.read(common.AUDIT)['rows']
    eligible={(r['id'],r['snr']):r for r in audit if r['sender']['audit'].get('reason','').startswith('no useful full-video')}
    jobs=[];cache={};start=time.monotonic()
    for c in common.cases():
        if (c['id'],c['snr']) not in eligible:continue
        old=eligible[c['id'],c['snr']]['sender'];packet=Path(c['receiver']['base_packet']).read_bytes();header,_=unpack(packet)
        if c['id'] not in cache:
            source=read_video(c['source_movie'])
            source=np.stack([cv2.resize(f,(288,160),interpolation=cv2.INTER_AREA) for f in source])
            cache[c['id']]=[sender_candidates(source[a:b+1],a,b) for a,b in zip(c['keys'],c['keys'][1:])]
        candidates=cache[c['id']]
        legacy,legacy_info=choose_packet(candidates,header['video'],sha256(c['receiver']['base_packet']),old['maximum_extra_channel_uses'],old['bits_per_symbol'])
        if legacy is not None:raise ValueError('local legacy no-fit case did not reproduce')
        new,info=allocate(candidates,header['video'],sha256(c['receiver']['base_packet']),old['maximum_extra_channel_uses'],old['bits_per_symbol'])
        dest=root/'budget'/c['id']/f'snr_{c["snr"]}';dest.mkdir(parents=True,exist_ok=True)
        if new:
            (dest/'trajectory_tx.bin').write_bytes(new);decode_packet(new,packet)
        job=dict(case=c,info=info,old_sender=old,new_packet=bool(new),folder=str(dest),
            channel_uses=accounting(len(new),bits_per_symbol=old['bits_per_symbol'])['complex_channel_uses'] if new else 0)
        write_json(dest/'sender.json',job);jobs.append(job)
        common.progress('BUDGET replay',len(jobs),len(eligible))
    p=root/'budget/plan.json';write_json(p,dict(jobs=jobs,scope='same budget ceiling; actual trajectory cost increases from zero',
        preserved_existing_packets=83,below_minimum_block_unchanged=19,
        seconds=time.monotonic()-start,criteria=common.read(root/'flow_reconstruction/plan.json')['acceptance']))
    common.stage_complete(root,'budget_prepare',[common.AUDIT,common.REPO/'src/semantic_transmission/trajectory_allocation.py'],[p,*[Path(j['folder'])/'sender.json' for j in jobs]])


def channel(root):
    from .metadata_channel import transmit
    from .bsu_sequence_refinement import decode_packet
    jobs=common.read(root/'budget/plan.json')['jobs']
    for i,j in enumerate(jobs):
        d=Path(j['folder']);c=j['case'];record=dict(packet_present=j['new_packet'])
        if j['new_packet']:
            original=(d/'trajectory_tx.bin').read_bytes()
            received,r=transmit(original,c['snr'],43,bits_per_symbol=j['old_sender']['bits_per_symbol'])
            if original!=received:raise ValueError('trajectory channel not exact')
            decode_packet(received,Path(c['receiver']['base_packet']).read_bytes())
            (d/'trajectory_rx.bin').write_bytes(received);record.update(r,exact_match=True)
        write_json(d/'channel.json',record);common.progress('BUDGET channel',i+1,len(jobs))


def compare(root):
    from .bsu_receiver import refine_windows,encode_frames
    from .bsu_sequence_refinement import decode_packet
    from .research_quality import read_video
    from .bsu_receiver_improvement import PersistentMetrics
    from .stutter_metrics import evaluate
    from .bsu_receiver_compare import boundary_diagnostics
    metrics=PersistentMetrics();rows=[]
    for i,j in enumerate(common.read(root/'budget/plan.json')['jobs']):
        c=j['case'];d=Path(j['folder']);source=read_video(c['source_movie'])
        for arm in ('baseline','allocated'):
            dest=d/arm;dest.mkdir(exist_ok=True);q=dest/'quality.json'
            if q.exists():rows.append(common.read(q));continue
            packet=Path(c['receiver']['base_packet']).read_bytes()
            tracks=decode_packet((d/'trajectory_rx.bin').read_bytes(),packet) if arm=='allocated' and j['new_packet'] else {}
            paths=[Path(p) for p in c['receiver']['raw_frames']]
            keydir=Path(c['receiver']['key_paths'][0]).parent
            refine_windows(paths,c['keys'],keydir,tracks,dest/'frames')
            encode_frames(dest/'frames',dest/'reconstruction.mp4',c['frames'],24)
            current=read_video(dest/'reconstruction.mp4');quality,_=metrics.evaluate(source,current)
            row=dict(id=c['id'],snr=c['snr'],arm=arm,quality=quality,temporal=evaluate(source,current),
                boundaries=boundary_diagnostics(source,current,c['keys']),
                extra_channel_uses=j['channel_uses'] if arm=='allocated' else 0,
                budget_cap=j['old_sender']['maximum_extra_channel_uses'])
            write_json(q,row);rows.append(row)
        common.progress('BUDGET reconstruction evaluation',i+1,6)
    import statistics as st
    means={arm:{k:st.mean(r['quality'][k] for r in rows if r['arm']==arm) for k in ('psnr_db','lpips_vgg','ssim','clip','dists')} for arm in ('baseline','allocated')}
    for arm in means:means[arm]['flow_error_mean']=st.mean(r['temporal']['flow_error_mean'] for r in rows if r['arm']==arm)
    result=dict(status='COMPLETE_BUDGET_ALLOCATION_PILOT',conditions=6,rows=rows,means=means,
        scope='same declared cap; additional actual transmission counted, not a same-rate quality improvement claim',
        new_packets=sum(j['new_packet'] for j in common.read(root/'budget/plan.json')['jobs']))
    write_json(root/'budget/RESULT.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['run','prepare','channel','compare']);p.add_argument('--root',type=Path,default=common.ROOT)
    a=p.parse_args()
    if a.stage!='run':return globals()[a.stage](a.root)
    import subprocess,sys
    for stage in ('prepare','channel','compare'):
        python=common.CHANNEL_PYTHON if stage=='channel' else sys.executable
        subprocess.run([str(python),'-m',__package__+'.bsu_budget_validation',stage,'--root',str(a.root)],check=True)


if __name__=='__main__':main()
