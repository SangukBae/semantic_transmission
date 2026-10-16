"""BSU sender with learned predictive dense latents, actual DCVC entropy stream.

All timeline frames are coded. Only the first image initializes the learned
reference; later packets carry entropy-coded temporal context/residuals. No
free source frames or source-trained per-video weights are available to RX.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import zlib

from .artifacts import sha256,write_json
from .wire import pack,unpack
from .packets import accounting
from .webvid5 import fingerprint
from .webvid_ablation import Stages
from .qwen35_pairs_video import freeze
from .bsu_dense_validation import REPO,read,evaluate,save_movie,progress

POLICY='bsu_predictive_dense_latents_v1'
MODULE='semantic_transmission.bsu_dense_neural'


def packet(context,video):
    from .dense_video_transport import validate_context
    validate_context(context)
    encoded=zlib.compress(json.dumps(context,sort_keys=True,separators=(',',':')).encode(),9)
    import hashlib
    return pack(dict(schema=POLICY,context_bytes=len(encoded),context_codec='zlib',
        video_sha256=hashlib.sha256(video).hexdigest()),encoded+video)


def unpack_packet(data):
    from .dense_video_transport import validate_context
    import hashlib
    h,p=unpack(data)
    if h.get('schema')!=POLICY or h.get('context_codec')!='zlib':raise ValueError('wrong learned dense packet')
    n=h['context_bytes']
    if type(n) is not int or not 0<n<len(p):raise ValueError('invalid dense context length')
    decoder=zlib.decompressobj();raw=decoder.decompress(p[:n],4*1024*1024+1)
    if len(raw)>4*1024*1024 or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError('invalid compressed context')
    context=json.loads(raw);validate_context(context)
    video=p[n:]
    if hashlib.sha256(video).hexdigest()!=h['video_sha256']:raise ValueError('neural stream changed')
    return context,video


def worker(j,stage):
    if stage=='action':
        from .bsu_key_reduction import action
        return action(j)
    output=Path(j['output'])
    if stage=='sender':
        import numpy as np
        import cv2
        from PIL import Image
        from .bsu_dcvc_keys import Codec
        case=j['case'];old=Path(case['original_run'])
        h,_=unpack((old/'transmitter/metadata.bin').read_bytes())
        context=dict(video=h['video'],indices=case['original_keys'],segments=h['segments'],decoder=h['decoder'])
        source=Path(case['source']['source_frames']);images=[]
        for i in range(case['source']['frames']):
            path=source/f'{i}.png'
            if sha256(path)!=case['source']['source_frame_hashes'][str(i)]:raise ValueError('source changed')
            a=np.asarray(Image.open(path).convert('RGB'))
            if j['scale']==2:a=cv2.resize(a,(a.shape[1]//2,a.shape[0]//2),interpolation=cv2.INTER_CUBIC)
            images.append(a)
        codec=Codec(REPO/'.local/vendor/DCVC-RT',REPO/'.local/checkpoints/dcvc_rt')
        video=codec.encode(np.stack(images),list(range(len(images))),j['qp'])
        data=packet(context,video);dest=output/'sender';dest.mkdir(parents=True)
        (dest/'packet.bin').write_bytes(data)
        write_json(dest/'sender.json',dict(policy=POLICY,channel_uses=accounting(len(data))['complex_channel_uses'],
            packet_bytes=len(data),packet_sha256=sha256(dest/'packet.bin'),qp=j['qp'],scale=j['scale'],
            independently_coded_frames=1,predictive_frames=len(images)-1,source_pixel_count=int(np.stack(images).size),
            per_video_training=False,estimated_entropy_used_for_accounting=False))
    elif stage=='channel':
        from .metadata_channel import transmit
        source=(output/'sender/packet.bin').read_bytes();dest=output/'received';dest.mkdir(parents=True)
        reports=[]
        for seed in (42,1042,2042):
            data,report=transmit(source,10,seed,bits_per_symbol=4)
            if data!=source or report['bit_errors']:raise ValueError('neural dense channel failed; no source fallback')
            unpack_packet(data);reports.append(report)
            if seed==42:(dest/'packet.bin').write_bytes(data)
        if reports[0]['complex_channel_uses']!=read(output/'sender/sender.json')['channel_uses']:
            raise ValueError('channel accounting differs')
        write_json(dest/'channel.json',dict(channel_uses=reports[0]['complex_channel_uses'],reports=reports,
            all_bytes_exact=True,retransmission_uses=0))
    elif stage=='decode':
        from .bsu_dcvc_keys import Codec
        from PIL import Image
        context,video=unpack_packet(Path(j['packet']).read_bytes())
        codec=Codec(REPO/'.local/vendor/DCVC-RT',REPO/'.local/checkpoints/dcvc_rt')
        indices,frames=codec.decode(video)
        if indices!=list(range(context['video']['frames'])):raise ValueError('dense neural timeline differs')
        out=output/'decoded';out.mkdir(parents=True)
        for i,a in enumerate(frames):Image.fromarray(a).save(out/f'{i:05d}.png')
        write_json(output/'decoded.json',dict(context=context,shape=list(frames.shape),packet_sha256=sha256(j['packet']),
            receiver_inputs=['received packet','shared pretrained DCVC weights'],encoder_state_reused=False))
    elif stage=='restore':
        import numpy as np
        from PIL import Image
        from .dense_video_receiver import restore
        from .bsu_dense_validation import MODEL
        decoded=Path(j['decoded']);meta=read(decoded.parent/'decoded.json')
        frames=np.stack([np.asarray(Image.open(f).convert('RGB')) for f in sorted(decoded.glob('*.png'))])
        video=meta['context']['video'];size=(video['width'],video['height'])
        if j['mode']=='swinir':
            frames=restore(frames,size,'swinir',model_root=REPO/'.local/vendor/SwinIR',checkpoint=MODEL)
        elif frames.shape[1:3]!=(size[1],size[0]):
            frames=restore(frames,size,'bicubic')
        save_movie(frames,output,video['fps'])
        write_json(output/'receiver.json',dict(input_packet_sha256=meta['packet_sha256'],policy=POLICY,
            source_at_receiver=False,restore=j['mode'],frames=len(frames)))
    elif stage=='evaluate':
        evaluate(j['case'],output/'reconstruction.mp4',output,j['channel_uses'])
    else:raise ValueError(stage)


def run(root,qp,scale,case_ids=None):
    from .etri_60s_check import lock
    from .key_reduction_selection import guard
    from .official_quality import KEYS
    import statistics
    with lock(REPO/'.local/etri_60s_check.lock'):
        previous=read(REPO/'outputs/bsu_key_reduction_20261015_v2/protocol.json')
        cases=[c for c in previous['cases'] if not case_ids or c['id'] in case_ids]
        p=dict(policy=POLICY,cases=cases,qp=qp,scale=scale,channel=dict(snr_db=10,seeds=[42,1042,2042],ldpc=[6144,9216],qam=16),
            qp_selection='one constant predefined quality index for all clips; actual rate reported, not forced equal',
            runtimes={**previous['runtimes'],'python':str(REPO/'.local/bsu_codec_eval_env/bin/python'),
                'codec':str(REPO/'.local/dcvc_rt_env/bin/python')},action_runtime=previous['action_runtime'],
            code={f'src/semantic_transmission/{n}.py':sha256(REPO/f'src/semantic_transmission/{n}.py') for n in
                ['bsu_dense_neural','bsu_dcvc_keys','dense_video_receiver','dense_video_transport','bsu_dense_validation']},
            source_at_receiver=False,model_training=False,old_cohort_is_development=True)
        p['signature']=fingerprint(p);root.mkdir(parents=True,exist_ok=True);freeze(root/'protocol.json',p)
        labels={v['id']:v['label'] for v in read(REPO/'outputs/bsu_kinetics_context/RESULT.json')['videos']}
        rows=[];action_rows=[]
        def execute(stage,job,log):
            key='codec' if stage in ('sender','decode') else 'channel' if stage=='channel' else 'action' if stage=='action' else 'python'
            env=dict(os.environ,PYTHONPATH=str(REPO/'src'),PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',
                     OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
            if stage=='channel':env['CUDA_VISIBLE_DEVICES']='-1'
            log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('x') as stream:
                code=subprocess.run([p['runtimes'][key],'-m',MODULE,'--worker',stage,'--job',str(job)],
                    cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
            if code:
                print(log.read_text(errors='replace')[-7000:],flush=True);raise RuntimeError(f'{stage} failed: {log}')
        for ci,case in enumerate(cases):
            progress('learned_dense',ci,len(cases),case['id'])
            out=root/case['id'];out.mkdir(parents=True,exist_ok=True)
            freeze(out/'sender_job.json',dict(output=str(out),case=case,qp=qp,scale=scale))
            freeze(out/'channel_job.json',dict(output=str(out)))
            freeze(out/'receiver_job.json',dict(output=str(out),packet=str(out/'received/packet.bin')))
            stages=Stages(out,p['signature'])
            for stage,owned,job in [('sender',['sender'],'sender_job.json'),('channel',['received'],'channel_job.json'),
                                    ('decode',['decoded','decoded.json'],'receiver_job.json')]:
                stages.step(stage,owned,owned,lambda log,s=stage,j=job:execute(s,out/j,log))
            modes=['swinir','bicubic'] if scale==2 else ['native']
            for mode in modes:
                dest=out/mode;dest.mkdir(parents=True,exist_ok=True)
                freeze(dest/'receiver_job.json',dict(output=str(dest),decoded=str(out/'decoded'),mode=mode))
                counts=read(out/'received/channel.json')['channel_uses']
                freeze(dest/'evaluation_job.json',dict(output=str(dest),case=case,channel_uses=counts))
                steps=Stages(dest,p['signature'])
                steps.step('restore',['frames','reconstruction.mp4','receiver.json'],['frames','reconstruction.mp4','receiver.json'],
                    lambda log:execute('restore',dest/'receiver_job.json',log))
                steps.step('evaluate',['quality.json','quality.csv'],['quality.json','quality.csv'],
                    lambda log:execute('evaluate',dest/'evaluation_job.json',log))
                q=read(dest/'quality.json');baseline=read(REPO/'outputs/bsu_key_reduction_20261015_v2'/case['id']/'baseline_2025/quality.json')
                if sha256(baseline['movie'])!=baseline['movie_sha256'] or baseline['source_movie_sha256']!=q['source_movie_sha256']:
                    raise ValueError('baseline provenance changed')
                rows.append(dict(id=case['id'],mode=mode,quality=q,baseline=baseline,strict_guard=guard(baseline,q)))
                action_rows.append(dict(id=case['id'],variant=mode,label=labels[case['id']],path=q['movie'],sha256=q['movie_sha256']))
            action_rows.append(dict(id=case['id'],variant='baseline',label=labels[case['id']],path=baseline['movie'],sha256=baseline['movie_sha256']))
        freeze(root/'action_job.json',dict(output=str(root/'action'),runtime=p['action_runtime'],rows=action_rows))
        Stages(root,p['signature']).step('action',['action'],['action/RESULT.json'],lambda log:execute('action',root/'action_job.json',log))
        summaries={}
        for mode in modes:
            group=[r for r in rows if r['mode']==mode];uses=sum(r['quality']['channel_uses'] for r in group)
            total=sum(r['baseline']['channel_uses'] for r in group)
            summaries[mode]=dict(channel_uses=uses,bsu_saving_percent=100*(1-uses/total),
                mean_quality={k:statistics.mean(r['quality']['quality'][k] for r in group) for k in KEYS},
                strict_all_metric_passes=sum(r['strict_guard']['passed'] for r in group),
                action=read(root/'action/RESULT.json')['summary'][mode])
        write_json(root/'RESULT.json',dict(status='COMPLETE_NEURAL_DENSE_RATE_QUALITY_ACTION',summary=summaries,rows=rows,
            action=read(root/'action/RESULT.json')['summary'],source_at_receiver=False,
            learned_components='official pretrained DCVC-RT; optional SwinIR x2; no training on these clips',
            generalization_proven=False,physical_link_overhead_included=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=REPO/'outputs/bsu_dense_neural_qp32_20261015_v1')
    parser.add_argument('--qp',type=int,default=32);parser.add_argument('--scale',type=int,choices=[1,2],default=1)
    parser.add_argument('--case',action='append');parser.add_argument('--worker');parser.add_argument('--job',type=Path)
    args=parser.parse_args()
    if args.worker:return worker(read(args.job),args.worker)
    run(args.root.resolve(),args.qp,args.scale,args.case)


if __name__=='__main__':main()
