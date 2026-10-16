"""Predictive latent transmission at short intervals plus learned interpolation."""
import argparse
import os
from pathlib import Path
import subprocess
from .artifacts import sha256,write_json
from .webvid5 import fingerprint
from .webvid_ablation import Stages
from .qwen35_pairs_video import freeze
from .wire import pack,unpack
from .packets import accounting
from .bsu_dense_validation import REPO,read,evaluate,save_movie,progress
from . import bsu_dense_neural as dense
from . import dense_temporal_receiver as temporal
POLICY=temporal.POLICY
MODULE='semantic_transmission.bsu_dense_temporal'


def worker(j,stage):
    if stage not in ('sender','decode','restore'):
        return dense.worker(j,stage)
    import numpy as np
    from PIL import Image
    output=Path(j['output'])
    if stage=='sender':
        from .bsu_dcvc_keys import Codec
        case=j['case'];old=Path(case['original_run'])
        h,_=unpack((old/'transmitter/metadata.bin').read_bytes())
        context=dict(video=h['video'],indices=case['original_keys'],segments=h['segments'],decoder=h['decoder'])
        images=[]
        for i in range(case['source']['frames']):
            path=Path(case['source']['source_frames'])/f'{i}.png'
            if sha256(path)!=case['source']['source_frame_hashes'][str(i)]:raise ValueError('source changed')
            images.append(np.asarray(Image.open(path).convert('RGB')))
        images=np.stack(images)
        protected=case['original_keys'] if j.get('protect_semantic_keys',False) else []
        indices=temporal.select_indices(images,j['stride'],protected)
        codec=Codec(REPO/'.local/vendor/DCVC-RT',REPO/'.local/checkpoints/dcvc_rt')
        encoded=codec.encode(images[indices],list(range(len(indices))),j['qp'])
        header,bits=unpack(encoded);header['indices']=indices
        header['temporal']=dict(policy=POLICY,stride=j['stride'],frames=len(images),
            weights=temporal.WEIGHT_SHA,network=temporal.NETWORK_SHA,warp=temporal.WARP_SHA)
        if j.get('protect_semantic_keys',False):
            header['temporal']['protected_semantic_keys']=True
        data=dense.packet(context,pack(header,bits));dest=output/'sender';dest.mkdir(parents=True)
        (dest/'packet.bin').write_bytes(data)
        write_json(dest/'sender.json',dict(channel_uses=accounting(len(data))['complex_channel_uses'],
            qp=j['qp'],stride=j['stride'],independently_coded_frames=1,predictive_frames=len(indices)-1,
            indices=indices,protected_semantic_keys=protected,
            all_timestamps_and_model_identity_paid=True,source_quality_selected=False))
    elif stage=='decode':
        from .bsu_dcvc_keys import Codec
        context,video=dense.unpack_packet(Path(j['packet']).read_bytes());h,_=unpack(video);t=h['temporal']
        if (t['policy']!=POLICY or t['weights']!=temporal.WEIGHT_SHA or
            t['network']!=temporal.NETWORK_SHA or t['warp']!=temporal.WARP_SHA or
            t['frames']!=context['video']['frames']):raise ValueError('wrong temporal decoder')
        temporal.validate_indices(h['indices'],t['frames'],t['stride'])
        if t.get('protected_semantic_keys',False) and not set(context['indices']) <= set(h['indices']):
            raise ValueError('a protected semantic key would be interpolated')
        codec=Codec(REPO/'.local/vendor/DCVC-RT',REPO/'.local/checkpoints/dcvc_rt')
        indices,frames=codec.decode(video);out=output/'decoded';out.mkdir(parents=True)
        for i,a in zip(indices,frames):Image.fromarray(a).save(out/f'{i:05d}.png')
        write_json(output/'decoded.json',dict(context=context,indices=indices,temporal=t,
            packet_sha256=sha256(j['packet']),encoder_state_reused=False))
    else:
        decoded=Path(j['decoded']);meta=read(decoded.parent/'decoded.json')
        frames=np.stack([np.asarray(Image.open(f).convert('RGB')) for f in sorted(decoded.glob('*.png'))])
        v=meta['context']['video']
        frames=temporal.interpolate(frames,meta['indices'],v['frames'],meta['temporal']['stride'],
            REPO/'.local/vendor/Practical-RIFE',REPO/'.local/checkpoints/rife/v4.25')
        save_movie(frames,output,v['fps'])
        write_json(output/'receiver.json',dict(policy=POLICY,source_at_receiver=False,
            input_packet_sha256=meta['packet_sha256'],interpolated_frames=len(frames)-len(meta['indices']),
            anchors_preserved=True,weights=temporal.WEIGHT_SHA))


def run(root,qp,scale,case_ids=None,stride=2,protect_semantic_keys=False):
    from .etri_60s_check import lock
    from .key_reduction_selection import guard
    from .official_quality import KEYS
    import statistics
    with lock(REPO/'.local/etri_60s_check.lock'):
        previous=read(REPO/'outputs/bsu_key_reduction_20261015_v2/protocol.json')
        cases=[c for c in previous['cases'] if not case_ids or c['id'] in case_ids]
        p=dict(policy=POLICY,cases=cases,qp=qp,scale=scale,stride=stride,protect_semantic_keys=protect_semantic_keys,
            channel=dict(snr_db=10,seeds=[42,1042,2042],ldpc=[6144,9216],qam=16),
            qp_selection='one constant predefined quality index for all clips; actual rate reported, not forced equal',
            runtimes={**previous['runtimes'],'python':str(REPO/'.local/bsu_codec_eval_env/bin/python'),
                'codec':str(REPO/'.local/dcvc_rt_env/bin/python')},action_runtime=previous['action_runtime'],
            code={f'src/semantic_transmission/{n}.py':sha256(REPO/f'src/semantic_transmission/{n}.py') for n in
                ['bsu_dense_temporal','dense_temporal_receiver','bsu_dense_neural','bsu_dcvc_keys','dense_video_receiver','dense_video_transport','bsu_dense_validation']},
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
            freeze(out/'sender_job.json',dict(output=str(out),case=case,qp=qp,scale=scale,stride=stride,
                protect_semantic_keys=protect_semantic_keys))
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
        write_json(root/'RESULT.json',dict(status='COMPLETE_TEMPORAL_NEURAL_RATE_QUALITY_ACTION',summary=summaries,rows=rows,
            action=read(root/'action/RESULT.json')['summary'],source_at_receiver=False,
            learned_components='official pretrained DCVC-RT and Practical-RIFE 4.25; no training on these clips',
            generalization_proven=False,physical_link_overhead_included=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=REPO/'outputs/bsu_dense_temporal_qp32_s2_20261016_v1')
    parser.add_argument('--qp',type=int,default=32);parser.add_argument('--scale',type=int,choices=[1],default=1)
    parser.add_argument('--stride',type=int,choices=[2,3,4],default=2)
    parser.add_argument('--protect-semantic-keys',action='store_true')
    parser.add_argument('--case',action='append');parser.add_argument('--worker');parser.add_argument('--job',type=Path)
    args=parser.parse_args()
    if args.worker:return worker(read(args.job),args.worker)
    run(args.root.resolve(),args.qp,args.scale,args.case,args.stride,args.protect_semantic_keys)


if __name__=='__main__':main()
