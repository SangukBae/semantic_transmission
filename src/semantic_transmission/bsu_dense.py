"""Run the new dense BSU transport on an existing semantic extraction.

Reuses BSU's captions and interval boundaries, but sends no independent NTSCC
RGB keys or sparse trajectory packet. Every dense latent/guide byte is paid.
"""
import argparse
import os
from pathlib import Path
import subprocess

from .artifacts import sha256,write_json
from .qwen35_pairs_video import freeze
from .webvid5 import fingerprint
from .webvid_ablation import Stages
from .bsu_dense_validation import REPO,read,progress,MODEL,MODEL_SHA
from .wire import unpack


def execute(args):
    from .etri_60s_check import lock
    old=args.semantic_run.resolve();source=args.source_frames.resolve();root=args.output.resolve()
    if root==old or old in root.parents or root in old.parents or source==root or root in source.parents or source in root.parents:
        raise ValueError('output must be separate from the original BSU run and source frames')
    h,_=unpack((old/'transmitter/metadata.bin').read_bytes());v=h['video']
    if v['width']%4 or v['height']%4:raise ValueError('current half-resolution path requires dimensions divisible by four')
    keys=[k['index'] for k in h['keyframes']]
    hashes={str(i):sha256(source/f'{i}.png') for i in range(v['frames'])}
    case=dict(id=root.name,original_run=str(old),original_keys=keys,
        source=dict(source_frames=str(source),source_frame_hashes=hashes,frames=v['frames'],fps=v['fps'],width=v['width'],height=v['height']))
    local=read(REPO/'.local/settings.json')
    main=REPO/'.local/bsu_codec_eval_env/bin/python'
    runtimes=dict(python=str(main) if main.exists() else local['python'],channel=local['channel_python'],
                  codec=str(REPO/'.local/dcvc_rt_env/bin/python'))
    if args.backend=='neural' and not Path(runtimes['codec']).is_file():
        raise ValueError('DCVC runtime missing; use the validated remote semantic-pc installation')
    mode='swinir' if args.scale==2 and args.restoration=='swinir' else 'bicubic'
    if mode=='swinir' and sha256(MODEL)!=MODEL_SHA:raise ValueError('pinned SwinIR weights required')
    files=['bsu_dense','bsu_dense_neural','bsu_dense_validation','dense_video_transport','dense_video_receiver','bsu_dcvc_keys']
    p=dict(backend=args.backend,qp=args.qp,scale=args.scale,uses_per_frame=args.uses_per_frame,restoration=mode,
        case=case,metadata_sha256=sha256(old/'transmitter/metadata.bin'),snr_db=10,seeds=[42,1042,2042],
        runtimes=runtimes,code={n:sha256(REPO/f'src/semantic_transmission/{n}.py') for n in files},
        semantic_intervals=len(keys)-1,independent_ntscc_key_images=0,
        model='BSU captions/intervals plus predictive dense representation and receiver restoration',
        generator_enabled=False)
    p['signature']=fingerprint(p);root.mkdir(parents=True,exist_ok=True)
    with lock(root/'execution.lock'),lock(REPO/'.local/etri_60s_check.lock'):
        freeze(root/'protocol.json',p)
        module='semantic_transmission.bsu_dense_neural' if args.backend=='neural' else 'semantic_transmission.bsu_dense_validation'
        sender=dict(output=str(root),case=case,qp=args.qp,scale=args.scale,uses_per_frame=args.uses_per_frame,
                    width=v['width']//args.scale,height=v['height']//args.scale)
        freeze(root/'sender_job.json',sender);freeze(root/'channel_job.json',dict(output=str(root)))
        freeze(root/'receiver_job.json',dict(output=str(root/'restored') if args.backend=='hevc' else str(root),
                                           packet=str(root/'received/packet.bin'),mode=mode))
        freeze(root/'restore_job.json',dict(output=str(root/'restored'),decoded=str(root/'decoded'),mode=mode))
        def launch(stage,job,log):
            key='channel' if stage=='channel' else 'codec' if args.backend=='neural' and stage in ('sender','decode') else 'python'
            env=dict(os.environ,PYTHONPATH=str(REPO/'src'),PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',
                     OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
            if stage=='channel' or (args.backend=='hevc' and stage=='sender'):env['CUDA_VISIBLE_DEVICES']='-1'
            log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('x') as stream:
                code=subprocess.run([runtimes[key],'-m',module,'--worker',stage,'--job',str(root/job)],
                    cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
            if code:
                print(log.read_text(errors='replace')[-6000:],flush=True);raise RuntimeError(f'{stage} failed: {log}')
        plan=[('sender',['sender'],'sender_job.json'),('channel',['received'],'channel_job.json')]
        if args.backend=='neural':plan += [('decode',['decoded','decoded.json'],'receiver_job.json'),('restore',['restored'],'restore_job.json')]
        else:plan += [('receiver',['restored'],'receiver_job.json')]
        stages=Stages(root,p['signature'])
        for i,(stage,owned,job) in enumerate(plan):
            progress(stage,i,len(plan),root.name)
            stages.step(stage,owned,owned,lambda log,s=stage,j=job:launch(s,j,log))
        movie=root/'restored/reconstruction.mp4';channel=read(root/'received/channel.json')
        write_json(root/'RESULT.json',dict(status='COMPLETE_DENSE_BSU_TRANSMISSION_AND_RECONSTRUCTION',
            signature=p['signature'],backend=args.backend,channel_uses=channel['channel_uses'],
            receiver_video=str(movie),receiver_video_sha256=sha256(movie),frames=v['frames'],fps=v['fps'],
            all_three_channel_seeds_exact=channel['all_bytes_exact'],quality_evaluation_executed=False,
            independently_transmitted_ntscc_keys=0,semantic_intervals=len(keys)-1,generator_enabled=False))
        progress('complete',len(plan),len(plan),root.name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--semantic-run',type=Path,required=True,help='existing BSU run with transmitter/metadata.bin')
    parser.add_argument('--source-frames',type=Path,required=True,help='sender PNG directory: 0.png, 1.png, ...')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--backend',choices=['neural','hevc'],required=True)
    parser.add_argument('--scale',type=int,choices=[1,2],default=1)
    parser.add_argument('--qp',type=int,default=32,help='DCVC quality index; larger means higher quality and rate')
    parser.add_argument('--uses-per-frame',type=int,default=512,help='HEVC hard cap including context/FEC/padding')
    parser.add_argument('--restoration',choices=['bicubic','swinir'],default='swinir')
    execute(parser.parse_args())


if __name__=='__main__':main()
