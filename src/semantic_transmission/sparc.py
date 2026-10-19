"""SPARC: semantic anchors + adaptive support + predictive DCVC + received-only RIFE."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import threading
import time

from . import bsu_predict as base
from . import dense_temporal_receiver as temporal
from . import sparc_selection as selection
from . import sparc_transport as transport
from .artifacts import sha256, write_json
from .packets import accounting
from .webvid5 import fingerprint

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / 'configs/sparc.json'
MODULE = 'semantic_transmission.sparc'
COMPLETE = 'COMPLETE_SPARC'
STAGES = ('prepare', 'semantic', 'select', 'sender', 'channel', 'decode', 'restore')
NUMERICS = dict(seed=0, deterministic_algorithms=True, cudnn_benchmark=False,
    cudnn_deterministic=True, matmul_tf32=False, cudnn_tf32=False, cublas_workspace_config=':4096:8')


def profile(mode='adaptive'):
    value = base.read(CONFIG)
    if (value['name'] != 'SPARC' or value['policy'] != 'sparc_semantic_predictive_v1' or
            value['semantic_selector'] != 'qwen35_sequential_semantic_v1' or
            value['codec'] != 'DCVC-RT' or value['interpolator'] != 'Practical-RIFE4.25' or
            value['generative_fallback'] is not False or value['new_training_required'] is not False or
            value['metadata_compression'] != 'auto' or value['snr_db'] != 10 or value['channel_seed'] != 42):
        raise ValueError('unsupported SPARC profile')
    for name, low, high in [('max_gap', 2, 4), ('candidate_stride', 1, 24), ('qp', 0, 63)]:
        if type(value[name]) is not int or not low <= value[name] <= high:
            raise ValueError('invalid SPARC profile field: ' + name)
    for name in ('proxy_mae_threshold', 'scene_cut_mad_threshold'):
        if type(value[name]) not in (float, int) or not math.isfinite(value[name]) or not 0 <= value[name] <= 255:
            raise ValueError('invalid source proxy threshold')
    if value['proxy_size'] != [64, 36] or mode not in ('adaptive', 'uniform-matched', 'fixed-stride2'):
        raise ValueError('unsupported source proxy or anchor policy')
    return dict(value, anchor_policy=mode, restore_numerics=NUMERICS)


def preflight(args):
    from .predictive_codec import WEIGHTS
    from .qwen35_backend import SNAPSHOT
    config = profile(args.anchor_policy); output = args.output.resolve()
    source = (args.video or args.source_frames).resolve(strict=True)
    protected = [source]
    if args.semantic_replay: protected.append(args.semantic_replay.resolve(strict=True).parent)
    if any(output == p or output in p.parents or p in output.parents for p in protected):
        raise ValueError('output overlaps a protected input/replay')
    if not math.isfinite(args.fps) or args.fps <= 0: raise ValueError('fps must be positive')
    if args.video:
        if not source.is_file() or min(args.width, args.height) < 2 or args.width % 2 or args.height % 2:
            raise ValueError('video requires a file and positive even normalization dimensions')
        inputs = dict(video=str(source), sha256=sha256(source), fps=args.fps, width=args.width, height=args.height)
    else:
        inputs = base.inspect_frames(source, args.fps)
    runtimes = base.runtime_paths(args)
    runtimes['selector'] = str(args.selector_python or REPO / '.local/qwen35_env/bin/python') if not args.semantic_replay else runtimes['receiver']
    for name, path in runtimes.items():
        if not path or not Path(path).is_file() or not os.access(path, os.X_OK):
            raise ValueError(f'{name} runtime missing: {path}; supply --{name}-python')
        runtimes[name] = str(Path(path).absolute())
    assets = {REPO / f'.local/checkpoints/dcvc_rt/dcvc_rt_{k}.tar':v for k,v in WEIGHTS.items()}
    assets.update({REPO / '.local/checkpoints/rife/v4.25/flownet.pkl':temporal.WEIGHT_SHA,
        REPO / '.local/checkpoints/rife/v4.25/IFNet_HDv3.py':temporal.NETWORK_SHA,
        REPO / '.local/vendor/Practical-RIFE/model/warplayer.py':temporal.WARP_SHA})
    for path, digest in assets.items():
        if not path.is_file() or sha256(path) != digest: raise ValueError('pinned asset changed/missing: ' + str(path))
    if not args.semantic_replay:
        weights = list(SNAPSHOT.glob('*.safetensors'))
        if not weights or not (SNAPSHOT / 'config.json').is_file(): raise ValueError('pinned Qwen3.5 snapshot missing')
        assets.update({p:sha256(p) for p in sorted(SNAPSHOT.iterdir()) if p.is_file()})
    vendor = REPO / '.local/vendor/DCVC-RT'
    if not (vendor / 'src/models/video_model.py').is_file(): raise ValueError('DCVC-RT source missing')
    binaries = {}
    for name in ('ffmpeg', 'ffprobe'):
        path = shutil.which(name)
        if not path: raise ValueError(name + ' is required')
        binaries[name] = dict(path=path, sha256=sha256(path))
    code = list((REPO / 'src/semantic_transmission').glob('*.py'))
    code += [CONFIG, REPO / 'scripts/run_sparc.sh', REPO / 'scripts/qwen35_caption.py', REPO / 'scripts/qwen35_advanced_caption.py']
    replay = None
    if args.semantic_replay:
        path = args.semantic_replay.resolve(strict=True)
        if 'directory' in inputs:
            from .sparc_semantics import validate_manifest
            validate_manifest(path, inputs, config)
        replay = dict(path=str(path), sha256=sha256(path))
    record = dict(profile=config, output=str(output), source=inputs, runtimes=runtimes,
        assets={str(p):d for p,d in assets.items()}, binaries=binaries, semantic_replay=replay,
        code={str(p.relative_to(REPO)):sha256(p) for p in code},
        vendor_code={str(p.relative_to(vendor)):sha256(p) for p in sorted(vendor.rglob('*.py'))})
    record['signature'] = fingerprint(record)
    if (output / 'protocol.json').exists() and base.read(output / 'protocol.json') != record:
        raise ValueError('frozen SPARC input/code/assets changed; use a new output')
    return record


def progress(root, stage, done, total):
    write_json(root / 'worker_progress.json', dict(stage=stage, done=done, total=total))


def configure_numerics():
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False


def worker(job, stage):
    root = Path(job['output']); config = job.get('profile')
    if stage == 'semantic':
        from .sparc_semantics import select
        return select(job['source'], config, root, job.get('replay'),
                      lambda n,t:progress(root, stage, n,t))
    if stage == 'select':
        source = job['source']; semantic = base.read(root / 'semantic.json')
        proxies, cuts = selection.signals(source, config, lambda n,t:progress(root,stage,n,t))
        plan = selection.plan(proxies, semantic['keys'], cuts, config, config['anchor_policy'])
        write_json(root / 'selection.json', dict(plan, semantic_sha256=sha256(root/'semantic.json')))
        return
    if stage == 'sender':
        import numpy as np
        from PIL import Image
        from .predictive_codec import Codec
        source = job['source']; plan = base.read(root / 'selection.json')
        groups = selection.scene_groups(plan['indices'], plan['cuts'])
        model = Codec(REPO/'.local/vendor/DCVC-RT', REPO/'.local/checkpoints/dcvc_rt')
        packets = []; done = 0
        for indices in groups:
            images = []
            for index in indices:
                path = Path(source['directory']) / f'{index}.png'
                if sha256(path) != source['hashes'][path.name]: raise ValueError('sender source changed')
                with Image.open(path) as frame: images.append(np.asarray(frame.convert('RGB')))
            packets.append(model.encode(np.stack(images), indices, config['qp']))
            done += len(indices); progress(root,stage,done,len(plan['indices']))
        original = transport.make(source['video'], packets, plan['cuts'], config['max_gap'])
        wire = transport.compress(original)
        if transport.decompress(wire) != original: raise ValueError('sender compression round-trip failed')
        dest = root / 'sender'; dest.mkdir()
        (dest/'packet_original.bin').write_bytes(original); (dest/'packet.bin').write_bytes(wire)
        write_json(dest/'sender.json', dict(indices=plan['indices'], scene_groups=groups,
            intra_frames=len(groups), predictive_frames=len(plan['indices'])-len(groups),
            qp=config['qp'], packet_bytes=len(wire), original_packet_bytes=len(original),
            dcvc_entropy_bytes=len(transport.parse(original)[1]),
            channel_uses=accounting(len(wire))['complex_channel_uses'],
            metadata_method='raw' if wire[:4] == b'LGV1' else 'zlib_headers',
            original_packet_sha256=sha256(dest/'packet_original.bin'), packet_sha256=sha256(dest/'packet.bin')))
        return
    if stage == 'channel':
        from .metadata_channel import transmit
        packet = Path(job['packet']).read_bytes(); transport.decompress(packet)
        received, report = transmit(packet, 10, 42, bits_per_symbol=4)
        if received != packet or report['bit_errors']: raise ValueError('SPARC packet channel integrity failure')
        if any(report[k] != v for k,v in accounting(len(packet)).items()): raise ValueError('channel accounting mismatch')
        restored = transport.decompress(received)
        dest = root/'received'; dest.mkdir()
        (dest/'packet_wire.bin').write_bytes(received); (dest/'packet.bin').write_bytes(restored)
        write_json(dest/'channel.json', dict(report, original_packet_restored_exactly=True,
            physical_link_overhead_included=False, received_packet_sha256=sha256(dest/'packet.bin')))
        return
    if stage == 'decode':
        import numpy as np
        from PIL import Image
        from .predictive_codec import Codec
        packet = Path(job['packet']); data = packet.read_bytes(); header, _ = transport.parse(data)
        model = Codec(REPO/'.local/vendor/DCVC-RT', REPO/'.local/checkpoints/dcvc_rt')
        dest = root/'decoded'; dest.mkdir(); indices = []
        for number, coded in enumerate(transport.decoder_packets(data), 1):
            ids, frames = model.decode(coded)
            expected = header['chunks'][number-1]['indices']; v = header['video']
            if ids != expected or frames.shape != (len(ids),v['height'],v['width'],3):
                raise ValueError('decoded scene dimensions/timeline mismatch')
            for i, frame in zip(ids,frames): Image.fromarray(frame).save(dest/f'{i:05d}.png')
            indices.extend(ids); progress(root,stage,number,len(header['chunks']))
        write_json(root/'decoded.json',dict(video=header['video'], indices=indices, cuts=header['cuts'],
            max_gap=header['max_gap'], packet_sha256=sha256(packet), source_pixels_at_receiver=False,
            encoder_state_reused=False))
        return
    if stage == 'restore':
        import numpy as np
        from PIL import Image
        import subprocess
        from fractions import Fraction
        configure_numerics()
        folder = Path(job['decoded']); info = base.read(folder.parent/'decoded.json'); v = info['video']
        selection.validate_anchors(info['indices'],v['frames'],info['max_gap'],info['cuts'])
        anchors = np.stack([np.asarray(Image.open(folder/f'{i:05d}.png').convert('RGB')) for i in info['indices']])
        frames = temporal.interpolate(anchors,info['indices'],v['frames'],info['max_gap'],
            REPO/'.local/vendor/Practical-RIFE',REPO/'.local/checkpoints/rife/v4.25')
        if frames.shape != (v['frames'],v['height'],v['width'],3) or not np.array_equal(frames[info['indices']],anchors):
            raise ValueError('receiver timeline/anchors changed')
        dest=root/'reconstruction'; images=dest/'frames'; images.mkdir(parents=True)
        for i,frame in enumerate(frames):
            Image.fromarray(frame).save(images/f'{i:05d}.png'); progress(root,stage,i+1,len(frames))
        movie=dest/'reconstruction.mp4'
        subprocess.run(['ffmpeg','-v','error','-nostdin','-framerate',str(v['fps']),'-i',str(images/'%05d.png'),
            '-frames:v',str(v['frames']),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(movie)],check=True)
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
            '-show_entries','stream=width,height,nb_read_frames,avg_frame_rate','-of','json',str(movie)]))['streams'][0]
        if ((int(probe['nb_read_frames']),probe['height'],probe['width']) != (v['frames'],v['height'],v['width'])
                or not math.isclose(float(Fraction(probe['avg_frame_rate'])),v['fps'],rel_tol=1e-6)):
            raise ValueError('delivered MP4 timeline changed')
        write_json(dest/'receiver.json',dict(video=v,probe=probe,source_pixels_at_receiver=False,
            packet_sha256=info['packet_sha256'],received_anchors_preserved_before_mp4=True,
            scene_cut_interpolation=False,generative_fallback=False,numerics=NUMERICS))
        return
    raise ValueError('unknown SPARC stage')


def launch(stage, job, log, runtime):
    from .webvid_ablation import command
    env = dict(os.environ,PYTHONPATH=str(REPO/'src'),PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    env.pop('LD_LIBRARY_PATH',None)
    if stage in ('channel','select'): env['CUDA_VISIBLE_DEVICES']='-1'
    if stage == 'semantic': env['TOKENIZERS_PARALLELISM']='false'
    root=job.parent; stop=threading.Event()
    def monitor():
        last=None
        while not stop.wait(1):
            try:
                value=base.read(root/'worker_progress.json')
                if value['stage']==stage and value!=last:
                    n,total=value['done'],value['total']
                    print('COMPARE_PROGRESS '+json.dumps(dict(stage=f'{stage}: {n}/{total} ({100*n/total:.1f}%)',
                        current=root.name,state='RUNNING')),flush=True)
                    last=value
            except (FileNotFoundError,ValueError,KeyError): pass
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try: command(REPO,[runtime,'-m',MODULE,'--worker',stage,'--job',str(job)],log,env)
    finally: stop.set();thread.join(timeout=2)


def execute(record):
    from .etri_60s_check import lock
    from .qwen35_pairs_video import freeze
    from .webvid_ablation import Stages
    root=Path(record['output']);root.mkdir(parents=True,exist_ok=True)
    (REPO/'.local').mkdir(exist_ok=True)
    with lock(REPO/'.local/etri_60s_check.lock'),lock(root/'execution.lock'):
        freeze(root/'protocol.json',record);stages=Stages(root,record['signature']);started=time.time()
        current,done='prepare',0
        def state(stage,n,status='RUNNING'):
            write_json(root/'status.json',dict(status=status,stage=stage,completed_stages=n,total_stages=7,
                elapsed_seconds=time.time()-started,model='SPARC'))
            print('COMPARE_PROGRESS '+json.dumps(dict(stage=stage,current=root.name,state=status,
                overall=f'{n}/7 ({100*n/7:.1f}%) stages complete')),flush=True)
        try:
            state(current,done)
            owned=['source.json','prepared'] if isinstance(record['source']['video'],str) else ['source.json']
            stages.step('prepare',owned,owned,lambda _:base.prepare(record,root))
            source=base.read(root/'source.json');config=record['profile'];replay=record['semantic_replay']
            if replay and sha256(replay['path'])!=replay['sha256']: raise ValueError('semantic replay changed')
            jobs=dict(semantic=dict(output=str(root),source=source,profile=config,replay=replay['path'] if replay else None),
                select=dict(output=str(root),source=source,profile=config),
                sender=dict(output=str(root),source=source,profile=config),
                channel=dict(output=str(root),packet=str(root/'sender/packet.bin')),
                decode=dict(output=str(root),packet=str(root/'received/packet.bin')),
                restore=dict(output=str(root),decoded=str(root/'decoded')))
            specs=[('semantic',['semantic.json'],['semantic.json','semantic_cache'],'selector'),
                ('select',['selection.json'],['selection.json'],'receiver'),
                ('sender',['sender'],['sender'],'codec'),('channel',['received'],['received'],'channel'),
                ('decode',['decoded','decoded.json'],['decoded','decoded.json'],'codec'),
                ('restore',['reconstruction'],['reconstruction'],'receiver')]
            for done,(current,owned,required,runtime) in enumerate(specs,1):
                state(current,done);job=root/f'{current}_job.json';freeze(job,jobs[current])
                stages.step(current,owned,required,lambda log,s=current,j=job,r=runtime:launch(s,j,log,record['runtimes'][r]))
            movie=root/'reconstruction/reconstruction.mp4'
            write_json(root/'RESULT.json',dict(status=COMPLETE,model='SPARC',anchor_policy=config['anchor_policy'],
                movie=str(movie),movie_sha256=sha256(movie),protocol_sha256=sha256(root/'protocol.json'),
                channel_uses=base.read(root/'received/channel.json')['complex_channel_uses'],
                semantic_keys=base.read(root/'semantic.json')['keys'],indices=base.read(root/'selection.json')['indices'],
                source_pixels_at_receiver=False,generative_fallback=False,new_training_performed=False,
                quality_evaluation_performed=False,independent_semantic_review='NOT_PERFORMED'))
            state('complete',7,COMPLETE)
        except BaseException as error:
            result=root/'RESULT.json'
            if result.exists():
                archive=root/'failed_attempts'/f'result_{time.time_ns()}.json';archive.parent.mkdir(parents=True,exist_ok=True)
                result.rename(archive)
            state(current,done,'INTERRUPTED' if isinstance(error,KeyboardInterrupt) else 'FAILED')
            raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',nargs='?',default='run',choices=('profile','check','run','status'))
    inputs=p.add_mutually_exclusive_group();inputs.add_argument('--video',type=Path);inputs.add_argument('--source-frames',type=Path)
    p.add_argument('--fps',type=float,default=24);p.add_argument('--width',type=int,default=576);p.add_argument('--height',type=int,default=320)
    p.add_argument('--output',type=Path);p.add_argument('--semantic-replay',type=Path,
        help='completed SPARC semantic.json with original call cache; no new Qwen calls')
    p.add_argument('--anchor-policy',choices=('adaptive','uniform-matched','fixed-stride2'),default='adaptive')
    for runtime in ('selector','codec','receiver','channel'):p.add_argument(f'--{runtime}-python',type=Path)
    p.add_argument('--worker',choices=STAGES[1:],help=argparse.SUPPRESS);p.add_argument('--job',type=Path,help=argparse.SUPPRESS)
    return p


def main(argv=None):
    p=parser();args=p.parse_args(argv)
    if args.worker:
        if not args.job:p.error('--job is required')
        job=base.read(args.job);record=base.read(Path(job['output'])/'protocol.json')
        if record['signature']!=fingerprint({k:v for k,v in record.items() if k!='signature'}):raise ValueError('SPARC protocol changed')
        for path,digest in record['code'].items():
            if sha256(REPO/path)!=digest:raise ValueError('frozen SPARC code changed: '+path)
        for path,digest in record['vendor_code'].items():
            if sha256(REPO/'.local/vendor/DCVC-RT'/path)!=digest:raise ValueError('DCVC source changed')
        return worker(job,args.worker)
    if args.action=='profile':print(json.dumps(profile(args.anchor_policy),indent=2));return
    if not args.output:p.error('--output is required')
    if args.action=='status':
        path=args.output/'status.json';print(json.dumps(base.read(path) if path.exists() else dict(status='NOT_STARTED'),indent=2));return
    if not(args.video or args.source_frames):p.error('--video or --source-frames is required')
    record=preflight(args)
    if args.action=='check':
        print(json.dumps(dict(status='READY_FILES_VERIFIED',profile=record['profile'],model_inference_started=False,
            runtime_imports_and_gpu_inference_verified=False),indent=2));return
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    execute(record)


if __name__=='__main__':main()
