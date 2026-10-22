"""Training-free, resumable SPARC/DCVC-UF pilot on frozen selected frames."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import threading

from . import sparc
from . import sparc_uf_transport as transport
from .artifacts import sha256, write_json
from .bsu_predict import read
from .dcvc_uf_codec import FILES
from .packets import accounting
from .qwen35_pairs_video import freeze
from .webvid5 import fingerprint
from .webvid_ablation import Stages, command, exclusive_lock

REPO = sparc.REPO
MODULE = 'semantic_transmission.sparc_uf'
COMPLETE = 'COMPLETE_SPARC_UF_PILOT'
WEIGHTS = {
    'cvpr2026_image.pth.tar': 'b3b900de23f30e4fc437010ddffe6a5413a56d6cfd97fb6235adbcd2b6973302',
    'cvpr2026_video_hts.pth.tar': '565607d84943ee79072a5760d34a200d5724b4bb72c6bc24bfab41e8ae4f5f47',
}


def preflight(args):
    from .sparc_motion_cache import audit_reference
    ref, root = args.reference.resolve(), args.output.resolve()
    source, semantic = audit_reference(ref)
    for protected in (ref, Path(source['directory']).resolve()):
        if root == protected or root in protected.parents or protected in root.parents:
            raise ValueError('output overlaps protected reference/source')
    old = read(ref/'protocol.json')
    # Preserve exactly the baseline decoder, channel and metric implementations.
    for name in ('sparc', 'dense_temporal_receiver', 'official_quality', 'metadata_channel',
                 'packets', 'wire', 'research_quality'):
        path = 'src/semantic_transmission/'+name+'.py'
        if old['code'][path] != sha256(REPO/path):
            raise ValueError('baseline implementation changed: '+name)
    for path, digest in old['assets'].items():
        if 'qwen35' not in path and sha256(path) != digest:
            raise ValueError('baseline asset changed: '+path)
    vendor, weights = REPO/'.local/dcvc_uf_vendor', REPO/'.local/dcvc_uf_weights'
    revision = subprocess.check_output(['git', '-C', str(vendor), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != transport.UPSTREAM or subprocess.check_output(['git', '-C', str(vendor), 'diff', 'HEAD']):
        raise ValueError('UF upstream revision changed')
    for name in FILES:
        if sha256(weights/name) != WEIGHTS[name]:
            raise ValueError('official UF checkpoint changed: '+name)
    runtimes = dict(codec=str(REPO/'.local/dcvc_uf_runtime/bin/python'),
                    receiver=old['runtimes']['receiver'], channel=old['runtimes']['channel'])
    runtime = subprocess.check_output([runtimes['codec'], '-c',
        'import torch,inference_extensions_cuda as c,MLCodec_extensions_cpp as e,json,platform;'
        'print(json.dumps(dict(python=platform.python_version(),torch=torch.__version__,'
        'cuda=torch.version.cuda,extension=c.__file__,entropy=e.__file__)))'], text=True)
    runtime = json.loads(runtime)
    runtime['binaries'] = {runtime[k]:sha256(runtime[k]) for k in ('extension', 'entropy')}
    plan = read(ref/'selection.json')
    gate = read(REPO/'configs/sparc_auxiliary_search.json')
    record = dict(policy=transport.SCHEMA, source=source, selection=plan,
        reference=str(ref), output=str(root), weights=WEIGHTS, vendor=str(vendor),
        weights_directory=str(weights), runtime=runtime, runtimes=runtimes,
        reference_files={str(ref/p):sha256(ref/p) for p in
            ('protocol.json', 'RESULT.json', 'selection.json', 'semantic.json', 'reconstruction/reconstruction.mp4')},
        code={str(p.relative_to(REPO)):sha256(p) for p in
              sorted((REPO/'src/semantic_transmission').glob('*.py'))},
        script_sha256=sha256(REPO/'scripts/run_sparc_uf.sh'),
        modes=['scene', 'interval'], qps=args.qps, quality_gate=gate,
        new_training=False, snr_db=10, channel_seed=42,
        limits=['single development video', 'no independent semantic review',
                'UF HT-S uses causal 8-frame chunks, not a custom two-endpoint codec',
                'interval mode repeats each short group tail to 8; all coded bytes are paid',
                'physical link overhead outside existing LDPC/QAM accounting is excluded'])
    record['signature'] = fingerprint(record)
    return record


def worker(stage, job):
    root = Path(job['output'])
    if stage == 'restore':
        return sparc.worker(job, stage)
    if stage == 'quality':
        import cv2
        import numpy as np
        from PIL import Image
        from .official_quality import OfficialMetrics
        from .research_quality import read_video
        from .sparc_auxiliary_selection import Signals
        cv2.setNumThreads(2)
        source = job['source']
        originals = np.stack([np.asarray(Image.open(Path(source['directory'])/f'{i}.png').convert('RGB'))
                              for i in range(source['video']['frames'])])
        decoded = read_video(Path(job['movie']))
        metrics = OfficialMetrics()
        quality, rows = metrics.evaluate(originals, decoded,
            lambda n,t:sparc.progress(root, stage, n,t))
        signals = Signals(originals, job['gate'])
        small = np.stack([cv2.resize(f, (144,80), interpolation=cv2.INTER_AREA) for f in decoded]).astype(np.float32)
        source_small = signals.frames.astype(np.float32)
        errors = np.abs(small-source_small).mean(-1)
        roi = [float(e[m].mean() if m.any() else e.mean()) for e,m in zip(errors,signals.masks)]
        temporal = np.abs(np.diff(small,axis=0)-np.diff(source_small,axis=0)).mean(axis=(1,2,3))
        quality.update(roi_mae=float(np.mean(roi)),temporal_mae=float(temporal.mean()))
        per = {key:[row[key] for row in rows] for key in ('psnr_db','lpips_vgg')}
        per.update(roi_mae=roi, temporal_mae=temporal.tolist())
        write_json(root/'quality.json',dict(quality=quality,per_frame=per,official_per_frame=rows,
            movie_sha256=sha256(job['movie']),resources=metrics.resources))
        return
    weights = job['weights']
    if stage == 'channel':
        from .metadata_channel import transmit
        packet = Path(job['packet']).read_bytes()
        transport.parse(packet, weights)
        received, report = transmit(packet, 10, 42, bits_per_symbol=4)
        if received != packet or report['bit_errors']:
            raise ValueError('UF channel packet integrity failure')
        if any(report[k] != v for k,v in accounting(len(packet)).items()):
            raise ValueError('UF channel accounting mismatch')
        folder = root/'received'
        folder.mkdir()
        (folder/'packet.bin').write_bytes(received)
        write_json(folder/'channel.json',dict(report,received_packet_sha256=sha256(folder/'packet.bin'),
                                              physical_link_overhead_included=False))
        return
    from .dcvc_uf_codec import Codec
    import numpy as np
    from PIL import Image
    model = Codec(Path(job['vendor']), Path(job['weights_directory']), weights)
    if stage == 'sender':
        source, plan = job['source'], job['selection']
        selected = transport.groups(plan['indices'], plan['semantic_keys'], job['mode'])
        coded, records = [], []
        for k, indices in enumerate(selected):
            images = []
            for i in indices:
                path = Path(source['directory'])/f'{i}.png'
                if sha256(path) != source['hashes'][path.name]:
                    raise ValueError('sender source changed')
                images.append(np.asarray(Image.open(path).convert('RGB')))
            stream, details = model.encode(np.stack(images), job['qp'])
            coded.append((indices, stream))
            records.append(dict(indices=indices, records=details, stream_bytes=len(stream)))
            sparc.progress(root, stage, k+1, len(selected))
        packet = transport.make(source['video'], plan['semantic_keys'], job['mode'], coded, weights, job['qp'])
        dest = root/'sender'
        dest.mkdir()
        (dest/'packet.bin').write_bytes(packet)
        write_json(dest/'sender.json',dict(groups=records,
            entropy_bytes=sum(r['entropy_bytes'] for g in records for r in g['records']),
            stream_bytes=sum(len(s) for _,s in coded), intra_frames=len(coded),
            padded_frames=sum(transport.padding(len(ids)) for ids,_ in coded),
            packet_sha256=sha256(dest/'packet.bin'),**accounting(len(packet))))
        return
    if stage == 'decode':
        packet = Path(job['packet'])
        header, payload = transport.parse(packet.read_bytes(), weights)
        dest = root/'decoded'
        dest.mkdir()
        indices = []
        for n, chunk in enumerate(header['chunks']):
            stream = payload[chunk['offset']:chunk['offset']+chunk['bytes']]
            frames = model.decode(stream,len(chunk['indices']),header['video'],header['qp'])
            for i, frame in zip(chunk['indices'],frames):
                Image.fromarray(frame).save(dest/f'{i:05d}.png')
            indices.extend(chunk['indices'])
            sparc.progress(root, stage, n+1,len(header['chunks']))
        write_json(root/'decoded.json',dict(video=header['video'],indices=indices,max_gap=4,cuts=[],
            packet_sha256=sha256(packet),source_pixels_at_receiver=False,encoder_state_reused=False))
        return
    raise ValueError('unknown UF worker stage')


def launch(stage, job, log, runtime):
    env = dict(os.environ,PYTHONPATH=str(REPO/'src'),PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    env.pop('LD_LIBRARY_PATH',None)
    if stage == 'channel':
        env['CUDA_VISIBLE_DEVICES']='-1'
    stop = threading.Event()
    def monitor():
        previous = None
        while not stop.wait(1):
            try:
                value = read(job.parent/'worker_progress.json')
                if value['stage'] == stage and value != previous:
                    print('COMPARE_PROGRESS '+json.dumps(dict(stage=stage,state='RUNNING',
                        current=job.parent.name,overall=f"{value['done']}/{value['total']}")),flush=True)
                    previous = value
            except (OSError,ValueError,KeyError):
                pass
    thread = threading.Thread(target=monitor,daemon=True)
    thread.start()
    try:
        command(REPO,[runtime,'-m',MODULE,'worker','--stage',stage,'--job',str(job)],log,env)
    finally:
        stop.set()
        thread.join()


def run(p):
    from .sparc_auxiliary_selection import quality_gate
    root, ref = Path(p['output']), Path(p['reference'])
    root.mkdir(parents=True,exist_ok=True)
    freeze(root/'protocol.json',p)
    baseline = root/'baseline'
    baseline.mkdir(exist_ok=True)
    stages = Stages(baseline,p['signature'])
    job = baseline/'quality_job.json'
    freeze(job,dict(output=str(baseline),source=p['source'],movie=str(ref/'reconstruction/reconstruction.mp4'),
                    gate=p['quality_gate']))
    stages.step('quality',['quality.json'],['quality.json'],
                lambda log:launch('quality',job,log,p['runtimes']['receiver']))
    base_quality = read(baseline/'quality.json')
    baseline_uses = read(ref/'RESULT.json')['channel_uses']
    results = []
    for mode in p['modes']:
        for qp in p['qps']:
            name = f'{mode}_q{qp:02d}'
            folder = root/name
            folder.mkdir(exist_ok=True)
            stages = Stages(folder,fingerprint(dict(parent=p['signature'],mode=mode,qp=qp)))
            common = {k:p[k] for k in ('weights','vendor','weights_directory')}
            jobs = [
                ('sender',dict(**common,source=p['source'],selection=p['selection'],mode=mode,qp=qp),['sender'],'codec'),
                ('channel',dict(weights=p['weights'],packet=str(folder/'sender/packet.bin')),['received'],'channel'),
                ('decode',dict(**common,packet=str(folder/'received/packet.bin')),['decoded','decoded.json'],'codec'),
                ('restore',dict(decoded=str(folder/'decoded')),['reconstruction'],'receiver'),
                ('quality',dict(source=p['source'],movie=str(folder/'reconstruction/reconstruction.mp4'),
                                gate=p['quality_gate']),['quality.json'],'receiver'),
            ]
            for stage, extra, products, runtime in jobs:
                write_json(root/'status.json',dict(status='RUNNING',trial=name,stage=stage,pid=os.getpid(),
                    completed_trials=len(results),total_trials=len(p['modes'])*len(p['qps'])))
                print('COMPARE_PROGRESS '+json.dumps(dict(stage=stage,state='RUNNING',current=name,
                    overall=f"{len(results)}/{len(p['modes'])*len(p['qps'])} trials")),flush=True)
                path = folder/f'{stage}_job.json'
                freeze(path,dict(output=str(folder),**extra))
                stages.step(stage,products,products,
                    lambda log,s=stage,j=path,r=runtime:launch(s,j,log,p['runtimes'][r]))
            quality = read(folder/'quality.json')
            # shape/frame fields in OfficialMetrics are metadata, not scalar scores.
            gate_input = dict(quality={k:quality['quality'][k] for k in
                ('psnr_db','lpips_vgg','roi_mae','temporal_mae')},per_frame=quality['per_frame'])
            base_input = dict(quality={k:base_quality['quality'][k] for k in gate_input['quality']},
                              per_frame=base_quality['per_frame'])
            gate = quality_gate(gate_input,base_input,p['quality_gate'])
            channel = read(folder/'received/channel.json')
            sender = read(folder/'sender/sender.json')
            results.append(dict(name=name,mode=mode,qp=qp,quality=quality['quality'],gate=gate,
                channel_uses=channel['complex_channel_uses'],
                reduction_percent=100*(1-channel['complex_channel_uses']/baseline_uses),
                packet_bytes=sender['packet_bytes'],entropy_bytes=sender['entropy_bytes'],
                padded_frames=sender['padded_frames'],intra_frames=sender['intra_frames'],
                movie=str(folder/'reconstruction/reconstruction.mp4')))
            write_json(root/'partial_results.json',results)
    eligible = [r for r in results if r['gate']['passed'] and r['channel_uses'] < baseline_uses]
    result = dict(status=COMPLETE,reference=str(ref),baseline_quality=base_quality['quality'],
        baseline_channel_uses=baseline_uses,trials=results,
        best_observed=min(eligible,key=lambda r:r['channel_uses']) if eligible else None,
        quality_tolerance=p['quality_gate'],new_training_performed=False,
        selected_frames_unchanged=True,source_pixels_at_receiver=False,
        protocol_sha256=sha256(root/'protocol.json'),limits=p['limits'])
    write_json(root/'RESULT.json',result)
    write_json(root/'status.json',dict(status=COMPLETE,completed_trials=len(results)))
    print(json.dumps(result,ensure_ascii=False),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['check','run','worker'])
    parser.add_argument('--reference',type=Path,default=REPO/'outputs/kinetics_001_adaptive_v2_20261019')
    parser.add_argument('--output',type=Path,default=REPO/'outputs/kinetics_001_dcvc_uf_20261020')
    parser.add_argument('--qps',type=int,nargs='+',default=[0,16,32,48,63])
    parser.add_argument('--stage')
    parser.add_argument('--job',type=Path)
    args = parser.parse_args()
    if args.command == 'worker':
        return worker(args.stage,read(args.job))
    if len(set(args.qps)) != len(args.qps) or any(not 0 <= q <= 63 for q in args.qps):
        parser.error('unique QPs in [0,63] required')
    p = preflight(args)
    if args.command == 'check':
        print(json.dumps(dict(status='PREFLIGHT_PASSED',signature=p['signature'],runtime=p['runtime'])))
        return
    root = Path(p['output'])
    root.mkdir(parents=True,exist_ok=True)
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:
        with exclusive_lock(root/'.run.lock'):
            run(p)
    except BaseException as error:
        write_json(root/'status.json',dict(status='INTERRUPTED' if isinstance(error,KeyboardInterrupt) else 'FAILED',
                                          error=str(error),resume='rerun the same command; completed stages are hash verified'))
        raise


if __name__ == '__main__':
    main()
