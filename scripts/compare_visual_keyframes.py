"""Source-grounded assistant keyframe selection versus a preserved SKEM run.

This is an offline, single-development-video diagnostic, not a trained selector
or independent semantic ground truth. Selection is frozen before reconstruction.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "outputs/etri_visual_keys_20260929"
BASE = REPO / "outputs/etri_60s_tv_low_08_42057b2ee8ed"
sys.path.insert(0, str(REPO / "src"))
from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.webvid5 import read_json


def sheet(root, indices, name, width=288, cols=6):
    from PIL import Image, ImageDraw, ImageFont
    height = round(width * 320 / 576)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    canvas = Image.new('RGB', (cols*width, 30+math.ceil(len(indices)/cols)*(height+24)), '#202020')
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), f'tv_low_08 SOURCE | {name} | 24 fps | zero-based frames', font=font, fill='white')
    for j, i in enumerate(indices):
        x, y = j % cols * width, 30 + j // cols * (height+24)
        with Image.open(BASE / f'baseline/data/frames/sample/{i}.png') as im:
            canvas.paste(im.convert('RGB').resize((width, height)), (x, y))
        draw.text((x+3, y+height+2), f'f{i:04d}  {i/24:.3f}s', font=font, fill='white')
    path = root / f'sheets/{name}.jpg'
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=95)
    write_json(path.with_suffix('.json'), {'indices':indices, 'sha256':sha256(path), 'viewed':False})
    return str(path)


def prepare(root):
    import cv2
    import numpy as np
    if (root/'protocol.json').exists(): raise ValueError('Protocol already exists')
    started = time.time()
    cfg = read_json(BASE/'baseline/run_config.json')
    assert sha256(Path(cfg['input'])) == cfg['input_sha256']
    protocol = {'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'selection_started_unix':started, 'input':cfg['input'], 'input_sha256':cfg['input_sha256'],
        'baseline':str(BASE), 'baseline_keys_sha256':sha256(BASE/'baseline/keyframes.json'),
        'baseline_result_sha256':sha256(BASE/'RESULT.json'),
        'frames':1440, 'fps':24, 'max_gap_frames':24, 'overview_stride':4,
        'criteria':['first and last', 'cut before and after', 'object entry/exit/occlusion',
                    'action or state transition', 'substantial accumulated pose/viewpoint change',
                    'insert time anchors so no gap exceeds 24 frames'],
        'scope':'Offline assistant visual selection from source sheets, one development source; prior source/baseline diagnoses are known. Not blinded or independent truth.',
        'inspection':'6fps overview plus native-rate focus windows and selected full-resolution views; not exhaustive semantic annotation of all 1440 frames',
        'quality_plan':'Full 60s reconstruction, unchanged decoder and AWGN 10dB; preserve exact received data for common keys, generate seeded AWGN for new keys; different segments do not share identical diffusion noise.',
        'adoption':'Compare delivered and interior quality, source-grounded error samples, actual total channel uses, and timing separately; no default replacement from one video.'}
    write_json(root/'protocol.json',protocol)
    before, scores = None, []
    hashes = {}
    for i in range(1440):
        path=BASE/f'baseline/data/frames/sample/{i}.png'
        frame=cv2.imread(str(path))
        assert frame.shape[:2] == (320,576)
        hashes[str(i)]=sha256(path)
        small=cv2.resize(frame,(144,80)).astype(np.float32)/255
        if before is not None: scores.append({'frame':i,'mad':float(np.abs(small-before).mean())})
        before=small
    write_json(root/'source_frame_hashes.json',hashes)
    write_json(root/'motion_scan.json',{'scope':'Pixel-change proposals only, not semantic labels','scores':scores,
               'top20':sorted(scores,key=lambda r:r['mad'],reverse=True)[:20]})
    for page in range(10):
        indices=list(range(page*144,(page+1)*144,4))
        if page==9: indices.append(1439)
        sheet(root,indices,f'overview_{page:02d}')
    write_json(root/'prepare_time.json',{'seconds':time.time()-started})
    print(json.dumps({'prepared':str(root),'pages':10,'top_changes':sorted(scores,key=lambda r:r['mad'],reverse=True)[:12]}))


def verify_selection(root):
    selection=read_json(root/'assistant_selection.json')
    frozen=read_json(root/'selection_freeze.json')
    assert sha256(root/'assistant_selection.json') == frozen['selection_sha256']
    assert sha256(root/'protocol.json') == selection['protocol_sha256']
    assert sha256(BASE/'baseline/keyframes.json') == read_json(root/'protocol.json')['baseline_keys_sha256']
    assert sha256(Path(read_json(root/'protocol.json')['input'])) == selection['input_sha256']
    for row in selection['viewed_sheets']: assert sha256(root/row['path']) == row['sha256']
    for name,digest in read_json(root/'source_frame_hashes.json').items():
        assert sha256(BASE/f'baseline/data/frames/sample/{name}.png') == digest
    for name,digest in frozen['frames'].items(): assert sha256(root/'selected_frames'/name)==digest
    return selection


def initialize(root):
    import csv
    chosen=verify_selection(root)
    run=root/'assistant'
    frames=run/'data/frames/sample'
    frames.mkdir(parents=True)
    for i in range(1440):
        (frames/f'{i}.png').symlink_to(BASE/f'baseline/data/frames/sample/{i}.png')
    with (frames/'frames.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=['frame_path']); writer.writeheader()
        writer.writerows({'frame_path':str(frames/f'{i}.png')} for i in range(1440))
    (run/'data/normalized.mp4').symlink_to(BASE/'baseline/data/normalized.mp4')
    cfg=read_json(BASE/'baseline/run_config.json')
    cfg.pop('selector_checkpoint',None)
    cfg.update(profile='etri_assistant_visual_selection_v1',selector='assistant_visual',
        visual_channel='baseline_common_replay_new_frame_awgn',
        caption_checkpoint=str(root/'checkpoints/caption.json'))
    write_json(run/'run_config.json',cfg)
    write_json(run/'keyframes.json',{'indices':chosen['indices'],'selector':chosen['selector']})


def paired_channel(root):
    import numpy as np
    from semantic_transmission.metadata_channel import transmit
    from semantic_transmission.wire import unpack
    from semantic_transmission.transmission_accounting import channel_breakdown
    run=root/'assistant'; base=BASE/'baseline'
    cfg=read_json(run/'run_config.json')
    packet=(run/'transmitter/metadata.bin').read_bytes()
    restored,report=transmit(packet,cfg['snr_db'],cfg['channel_seed'])
    assert restored==packet and not report['bit_errors']
    header,payload=unpack(restored)
    old_header,old_payload=unpack((base/'received/metadata.bin').read_bytes())
    old_items={r['index']:r for r in old_header['keyframes']}
    sent=np.fromfile(run/'transmitter/visual.c64',dtype='<c8')
    old_sent=np.fromfile(base/'transmitter/visual.c64',dtype='<c8')
    old_rx=np.fromfile(base/'received/visual.c64',dtype='<c8')
    values=sent.copy(); reused=[]; fresh=[]
    sigma=math.sqrt(1/(2*10**(cfg['snr_db']/10)))
    for item in header['keyframes']:
        a,n=item['complex_offset'],item['complex_count']; index=item['index']
        if index in old_items:
            prev=old_items[index]; b=prev['complex_offset']
            assert n==prev['complex_count']
            assert np.array_equal(sent[a:a+n],old_sent[b:b+n]),f'encoded common frame changed: {index}'
            ra,rb=item['rate_offset'],prev['rate_offset']
            assert payload[ra:ra+item['rate_bytes']]==old_payload[rb:rb+prev['rate_bytes']]
            values[a:a+n]=old_rx[b:b+n]
            reused.append(index)
        else:
            rng=np.random.default_rng(cfg['channel_seed']+100000+index)
            noise=(rng.normal(0,sigma,n)+1j*rng.normal(0,sigma,n)).astype('<c8')
            values[a:a+n]+=noise; fresh.append(index)
    out=run/'received'; out.mkdir()
    (out/'metadata.bin').write_bytes(restored); values.tofile(out/'visual.c64')
    digital=report['complex_channel_uses']
    report.update(status='PASSED',metadata_exact_match=True,visual_complex_channel_uses=len(sent),
        digital_complex_channel_uses=digital,total_complex_channel_uses=len(sent)+digital,
        cbr_complex_uses_per_source_scalar=(len(sent)+digital)/(3*cfg['width']*cfg['height']*cfg['frames']),
        complete_sample_dependent_model_input_accounting=True,physical_link_overhead_included=False,
        visual_awgn_rng='Exact historical received symbols for common keys; new keys PCG64(channel_seed+100000+frame)',
        channel_seed=cfg['channel_seed'],common_frames_exact=reused,new_noise_frames=fresh,
        transmission_breakdown=channel_breakdown(packet,len(sent),header['video'],report),
        received_files={p.name:{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in out.iterdir()})
    write_json(run/'channel_accounting.json',report)


def run_pipeline(root):
    from semantic_transmission.cli import settings
    from semantic_transmission.etri_60s_check import lock
    from semantic_transmission.webvid_ablation import Stages,environment
    from semantic_transmission.webvid5 import fingerprint
    verify_selection(root)
    local=settings(REPO); env=environment(2025)
    identity={'selection_sha256':sha256(root/'assistant_selection.json'),
        'code':{str(p.relative_to(REPO)):sha256(p) for p in [Path(__file__).resolve(),
          REPO/'src/semantic_transmission/workers.py', REPO/'src/semantic_transmission/codec_transport.py',
          REPO/'src/semantic_transmission/etri_60s.py', REPO/'04_semantic_decoder/scripts/mydemo_new_align_sh.py']}}
    signature=fingerprint(identity)
    receipt=root/'execution_protocol.json'
    if receipt.exists(): assert read_json(receipt)['signature']==signature
    else: write_json(receipt,dict(identity,signature=signature))
    with lock(REPO/'.local/etri_60s_check.lock'), lock(root/'.execution.lock'):
        stages=Stages(root,signature)
        stages.step('initialize',['assistant'],['assistant/run_config.json','assistant/keyframes.json','assistant/data'],lambda _:initialize(root))
        run=root/'assistant'
        plan=[('input-audit','etri_60s',['input_audit.json']),
              ('semantic-clips','etri_60s',['semantic_clips_audit.json','data/clips']),
              ('caption','workers',['captions.json','caption_sampling.json']),
              ('flow','workers',['metadata_tx.json','flow_sampling.json']),
              ('send','codec_transport',['transmitter','sender_accounting.json']),
              ('channel',None,['received','channel_accounting.json']),
              ('receive','codec_transport',['receiver/frames','receiver/metadata.csv','receiver/decoder_inputs.json','receiver_accounting.json']),
              ('reconstruct','etri_60s',['receiver/decoder_config.py','receiver/reconstruction','receiver/reference_trace.json']),
              ('output-audit','etri_60s',['output_audit.json']),
              ('evaluate','research_quality',['quality.json','quality_delivered_mp4.csv','quality_lossless_frames.csv'])]
        for stage,module,products in plan:
            py=local['channel_python'] if stage=='channel' else local['python']
            if module=='etri_60s': cmd=[py,'-m','semantic_transmission.etri_60s','--worker',stage,'--run-dir',str(run)]
            elif module: cmd=[py,'-m',f'semantic_transmission.{module}',stage,str(run)]
            else: cmd=[py,str(Path(__file__).resolve()),'channel','--output',str(root)]
            def launch(log,cmd=cmd,stage=stage):
                begin=time.perf_counter(); log.parent.mkdir(parents=True,exist_ok=True)
                # Wait on process completion; no repeated model-side status queries.
                with log.open('x') as stream:
                    completed=subprocess.run(cmd,cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT)
                write_json(root/f'resources/{stage}.json',{'command':cmd,'seconds':time.perf_counter()-begin,'returncode':completed.returncode})
                completed.check_returncode()
            required=[f'assistant/{p}' for p in products]+[f'resources/{stage}.json']
            stages.step(stage,required,required,launch)
        write_json(root/'pipeline_complete.json',{'status':'FULL_60S_ASSISTANT_RECONSTRUCTION_COMPLETE',
            'stages_seconds':{k:v['seconds'] for k,v in stages.completed.items()},
            'selection_sha256':sha256(root/'assistant_selection.json'),
            'independent_semantic_review':'PENDING'})
        print('FULL_60S_ASSISTANT_RECONSTRUCTION_COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','focus','run','channel'])
    p.add_argument('--output',type=Path,default=ROOT)
    p.add_argument('--indices',type=int,nargs='+')
    p.add_argument('--name',default='focus')
    args=p.parse_args()
    root=args.output.resolve()
    if args.action=='prepare': prepare(root)
    elif args.action=='run': run_pipeline(root)
    elif args.action=='channel': paired_channel(root)
    else: print(sheet(root,args.indices,args.name))


if __name__=='__main__': main()
