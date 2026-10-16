"""Executable cost-routed BSU receiver using paid, self-describing packets.

Inputs are existing semantic encodings and either sender PNGs or a predictive
packet. The selected receiver is run afresh; the
legacy branch decodes actual noisy NTSCC symbols and runs combined_context.
"""
import argparse
import os
from pathlib import Path
import subprocess

from .artifacts import sha256, write_json
from .webvid5 import fingerprint
from .qwen35_pairs_video import freeze
from .webvid_ablation import Stages
from .bsu_dense_validation import REPO, read, progress
from .bsu_dense_router import choose, receive, POLICY
from .wire import unpack

MODULE = 'semantic_transmission.bsu_adaptive_dense'


def prepare_sender(args):
    """Encode a candidate locally at TX; only the selected stream is sent."""
    from .etri_60s_check import lock
    old=args.semantic_run.resolve();source=args.source_frames.resolve();root=args.output.resolve()
    for original in (old,source):
        if root==original or root in original.parents or original in root.parents:
            raise ValueError('use an independent output directory')
    h,_=unpack((old/'transmitter/metadata.bin').read_bytes());v=h['video']
    from PIL import Image
    for i in range(v['frames']):
        with Image.open(source/f'{i}.png') as image:
            if image.size != (v['width'],v['height']):
                raise ValueError('source frame dimensions differ from semantic metadata')
    case=dict(id=root.name,original_run=str(old),original_keys=[k['index'] for k in h['keyframes']],
        source=dict(source_frames=str(source),frames=v['frames'],fps=v['fps'],width=v['width'],height=v['height'],
            source_frame_hashes={str(i):sha256(source/f'{i}.png') for i in range(v['frames'])}))
    candidate=root/'predictive_sender'
    j=dict(output=str(candidate),case=case,qp=args.qp,scale=1)
    module='semantic_transmission.bsu_dense_neural'
    if args.temporal_stride > 1:
        j.update(stride=args.temporal_stride,protect_semantic_keys=args.protect_semantic_keys)
        module='semantic_transmission.bsu_dense_temporal'
    p=dict(sender=j,metadata_sha256=sha256(old/'transmitter/metadata.bin'),
        sender_module=module,
        code={n:sha256(REPO/f'src/semantic_transmission/{n}.py') for n in
            ['bsu_adaptive_dense','bsu_dense_neural','bsu_dcvc_keys',
             'bsu_dense_temporal','dense_temporal_receiver']})
    p['signature']=fingerprint(p)
    with lock(REPO/'.local/etri_60s_check.lock'):
        root.mkdir(parents=True,exist_ok=True);candidate.mkdir(exist_ok=True)
        with lock(root/'execution.lock'):
            freeze(root/'sender_protocol.json',p);freeze(candidate/'sender_job.json',j)
            def launch(log):
                env=dict(os.environ,PYTHONPATH=str(REPO/'src'),PYTHONNOUSERSITE='1',
                    OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
                log.parent.mkdir(parents=True,exist_ok=True)
                with log.open('x') as stream:
                    code=subprocess.run([str(REPO/'.local/dcvc_rt_env/bin/python'),'-m',
                        module,'--worker','sender','--job',
                        str(candidate/'sender_job.json')],cwd=REPO,env=env,stdout=stream,
                        stderr=subprocess.STDOUT).returncode
                if code:
                    print(log.read_text(errors='replace')[-6000:],flush=True)
                    raise RuntimeError(f'predictive sender failed: {log}')
            progress('predictive_sender',0,1,root.name)
            Stages(candidate,p['signature']).step('sender',['sender'],['sender'],launch)
    return candidate


def worker(j, stage):
    root = Path(j['output']); run = root/'run'
    if stage == 'send':
        old = Path(j['semantic_run']); trajectory = Path(j['trajectory']) if j['trajectory'] else None
        tx = old/'transmitter'; visual = tx/'visual.c64'
        if visual.stat().st_size % 8:
            raise ValueError('malformed original NTSCC stream')
        packet, selection = choose((tx/'metadata.bin').read_bytes(),
            trajectory.read_bytes() if trajectory else None, visual.stat().st_size//8,
            (Path(j['neural_run'])/'sender/packet.bin').read_bytes(), j['minimum_original_keys'])
        dest = root/'transmitter'; dest.mkdir(parents=True)
        (dest/'packet.bin').write_bytes(packet)
        if selection['mode'] == 'bsu':
            (dest/'visual.c64').symlink_to(visual.resolve())
        write_json(root/'selection.json', selection)
    elif stage == 'channel':
        from .metadata_channel import transmit
        packet = (root/'transmitter/packet.bin').read_bytes(); dest = root/'received'; dest.mkdir()
        reports = []
        for seed in (42,1042,2042):
            data, report = transmit(packet,10,seed,bits_per_symbol=4)
            if data != packet or report['bit_errors']:
                raise ValueError('routed digital packet failed integrity')
            receive(data); reports.append(report)
            if seed == 42:
                (dest/'packet.bin').write_bytes(data)
        mode, restored = receive(packet); visual_count = 0
        if mode == 'bsu':
            import numpy as np
            from .qwen35_pairs_comparison import frame_noise
            h, _ = unpack(restored['metadata'])
            values = np.fromfile(root/'transmitter/visual.c64', dtype='<c8')
            visual_count = len(values); cursor = 0
            if not np.isfinite(values).all():
                raise ValueError('nonfinite transmitted symbols')
            for key in h['keyframes']:
                offset, count = key['complex_offset'], key['complex_count']
                if offset != cursor:
                    raise ValueError('noncontiguous analog stream')
                values[offset:offset+count] += frame_noise(key['index'],count,j['channel_seed'],10)
                cursor += count
            if cursor != len(values):
                raise ValueError('analog symbol count mismatch')
            values.tofile(dest/'visual.c64')
        count = reports[0]['complex_channel_uses']+visual_count
        if count != read(root/'selection.json')['channel_uses']:
            raise ValueError('routed total cost changed')
        write_json(root/'channel.json',dict(channel_uses=count,digital_reports=reports,
            visual_channel_uses=visual_count,visual_seed=j['channel_seed'],snr_db=10))
    elif stage == 'initialize':
        mode, restored = receive((root/'received/packet.bin').read_bytes())
        dest = run/'received'; dest.mkdir(parents=True)
        for name,data in restored.items():
            (dest/f'{name}.bin').write_bytes(data)
        if mode == 'bsu':
            (dest/'visual.c64').symlink_to((root/'received/visual.c64').resolve())
            h,_=unpack(restored['metadata']);cfg=read(Path(j['semantic_run'])/'run_config.json')
            cfg['input']='received_routed_packet';cfg.pop('input_sha256',None)
            write_json(run/'run_config.json',cfg)
            write_json(run/'keyframes.json',dict(indices=[k['index'] for k in h['keyframes']]))
    elif stage == 'bsu_receive':
        from .codec_transport import receive as decode
        from . import tail_reference as tail, precision_text_cache as cache, generation_noise as noise
        from . import short_schedule, condition_collision
        from .bsu_dcvc_validation import portable_text_report
        cfg=read(run/'run_config.json');decode(cfg,REPO,run)
        keys=read(run/'keyframes.json')['indices']
        contract=dict(policy=noise.POLICY,seed=cfg['seed'],segments=len(keys)-1,steps=cfg['steps'],
                      inputs=sha256(root/'received/packet.bin'),code=j['signature'],reference_policy=tail.POLICY)
        write_json(run/'receiver_policy.json',dict(policy=tail.POLICY,reference_frames=17,reference_latents=5,
            alignment=5,text_encoder_policy=cache.POLICY,t5_precision='fp32',noise_contract=contract,
            noise_reference=None,condition_collision_policy=condition_collision.POLICY,
            short_schedule_policy=short_schedule.POLICY))
        old=Path(j['semantic_run'])
        if (old/cache.REPORT).exists() and sha256(old/'receiver/metadata.csv')==sha256(run/'receiver/metadata.csv'):
            portable_text_report(old,run)
        else:
            cache.prepare(run,REPO,tail.decoder_config(run),'fp32')
    elif stage == 'bsu_reconstruct':
        from .bsu_key_reduction import reconstruct,refine
        job=dict(run=str(run),output=str(root),keys=read(run/'keyframes.json')['indices'])
        reconstruct(job);refine(job)
    else:
        raise ValueError(stage)


def execute(args):
    from .etri_60s_check import lock
    root=args.output.resolve();old=args.semantic_run.resolve();neural=args.neural_run.resolve()
    if any(root==p or p in root.parents or root in p.parents for p in [old,neural]):
        raise ValueError('use a separate output directory')
    local=read(REPO/'.local/settings.json');main=REPO/'.local/bsu_codec_eval_env/bin/python'
    runtimes=dict(python=str(main) if main.exists() else local['python'],channel=local['channel_python'],
                  codec=str(REPO/'.local/dcvc_rt_env/bin/python'))
    cfg=read(old/'run_config.json')
    sender_job=read(neural/'sender_job.json')
    if Path(sender_job['case']['original_run']).resolve()!=old:
        raise ValueError('neural sender provenance belongs to another semantic run')
    if cfg['snr_db']!=10 or cfg['visual_channel']!='source_frame_seeded_numpy_awgn_v1':
        raise ValueError('current matched runner requires 10dB and frame-seeded AWGN')
    trajectory=args.trajectory.resolve() if args.trajectory else old.parents[3]/'sender/trajectory.bin'
    if args.trajectory and not trajectory.exists():raise FileNotFoundError(trajectory)
    source_files=[old/'transmitter/metadata.bin',old/'transmitter/visual.c64',neural/'sender/packet.bin']
    if trajectory.exists():source_files.append(trajectory)
    p=dict(policy=POLICY,semantic_run=str(old),neural_run=str(neural),output=str(root),
        minimum_original_keys=args.min_original_keys,
        trajectory=str(trajectory) if trajectory.exists() else None,channel_seed=cfg['channel_seed'],
        inputs={str(f):sha256(f) for f in source_files},runtimes=runtimes,
        code={n:sha256(REPO/f'src/semantic_transmission/{n}.py') for n in
            ['bsu_adaptive_dense','bsu_dense_router','bsu_dense_neural','bsu_key_reduction','codec_transport',
             'bsu_dense_temporal','dense_temporal_receiver']})
    p['signature']=fingerprint(p);root.mkdir(parents=True,exist_ok=True)
    with lock(REPO/'.local/etri_60s_check.lock'),lock(root/'execution.lock'):
        freeze(root/'protocol.json',p);freeze(root/'job.json',p)
        def launch(stage,job,log,module=MODULE,runtime='python'):
            env=dict(os.environ,PYTHONPATH=str(REPO/'src'),PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',
                OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            if runtime=='channel':env['CUDA_VISIBLE_DEVICES']='-1'
            log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('x') as stream:
                code=subprocess.run([runtimes[runtime],'-m',module,'--worker',stage,'--job',str(job)],
                    cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
            if code:
                print(log.read_text(errors='replace')[-6000:],flush=True);raise RuntimeError(f'{stage} failed: {log}')
        stages=Stages(root,p['signature'])
        for index,(stage,owned,required) in enumerate([
            ('send',['transmitter','selection.json'],['transmitter','selection.json']),
            ('channel',['received','channel.json'],['received','channel.json']),
            ('initialize',['run'],['run/received'])]):
            progress(stage,index,5)
            stages.step(stage,owned,required,lambda log,s=stage:launch(s,root/'job.json',log,
                runtime='channel' if s=='channel' else 'python'))
        mode=read(root/'selection.json')['mode']
        if mode=='bsu':
            stages.step('bsu_receive',['run/receiver','run/receiver_policy.json'],['run/receiver/frames',
                'run/receiver/metadata.csv','run/receiver/text_embeddings.json','run/receiver_policy.json'],
                lambda log:launch('bsu_receive',root/'job.json',log))
            stages.step('bsu_reconstruct',['run/receiver/reconstruction','refined'],
                ['refined/reconstruction.mp4','refined/audit.json'],lambda log:launch('bsu_reconstruct',root/'job.json',log))
        else:
            from .bsu_dense_neural import unpack_packet
            context,video=unpack_packet((root/'run/received/neural.bin').read_bytes());h,_=unpack(video)
            mode_sr='swinir' if h['width']*2==context['video']['width'] else 'native'
            freeze(root/'decode_job.json',dict(output=str(root/'run'),packet=str(root/'run/received/neural.bin')))
            freeze(root/'restore_job.json',dict(output=str(root/'refined'),decoded=str(root/'run/decoded'),mode=mode_sr))
            mod='semantic_transmission.bsu_dense_neural'
            if 'temporal' in h:
                if mode_sr != 'native':raise ValueError('temporal decoder requires native resolution')
                mod='semantic_transmission.bsu_dense_temporal'
            stages.step('decode',['run/decoded','run/decoded.json'],['run/decoded','run/decoded.json'],
                lambda log:launch('decode',root/'decode_job.json',log,module=mod,runtime='codec'))
            stages.step('restore',['refined'],['refined/reconstruction.mp4','refined/receiver.json'],
                lambda log:launch('restore',root/'restore_job.json',log,module=mod))
        movie=root/'refined/reconstruction.mp4'
        write_json(root/'RESULT.json',dict(status='COMPLETE_ADAPTIVE_DENSE_BSU',mode=mode,
            channel_uses=read(root/'channel.json')['channel_uses'],movie=str(movie),movie_sha256=sha256(movie),
            receiver_ran_afresh=True,source_pixels_at_receiver=False,quality_selected_routing=False))
        progress('complete',5,5)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--semantic-run',type=Path)
    inputs=parser.add_mutually_exclusive_group()
    inputs.add_argument('--neural-run',type=Path,help='reuse an existing paid predictive sender packet')
    inputs.add_argument('--source-frames',type=Path,help='encode sender PNGs before routing: 0.png,1.png,...')
    parser.add_argument('--qp',type=int,choices=range(64),default=32,
        metavar='0..63',help='predictive quality index; default 32 is the adopted development profile')
    parser.add_argument('--temporal-stride',type=int,choices=[1,2,3,4],default=2,
        help='default 2 is validated; 1 codes every frame; 3-4 are experimental')
    parser.add_argument('--min-original-keys',type=int,default=5,
        help='preserve the original BSU path below this sender key count')
    parser.add_argument('--protect-semantic-keys',action=argparse.BooleanOptionalAction,default=False,
        help='include original BSU event timestamps among the paid predictive anchors')
    parser.add_argument('--output',type=Path);parser.add_argument('--trajectory',type=Path)
    parser.add_argument('--worker');parser.add_argument('--job',type=Path)
    args=parser.parse_args()
    if args.worker:return worker(read(args.job),args.worker)
    if not all([args.semantic_run,args.neural_run or args.source_frames,args.output]):
        parser.error('semantic-run, output and either neural-run or source-frames are required')
    outer=args.output.resolve() if args.source_frames else None
    if outer:
        args.neural_run=prepare_sender(args);args.output=outer/'delivery'
    execute(args)
    if outer:
        write_json(outer/'RESULT.json',dict(**read(args.output/'RESULT.json'),
            sender_protocol_sha256=sha256(outer/'sender_protocol.json'),
            delivery_result=str(args.output/'RESULT.json')))


if __name__=='__main__':main()
