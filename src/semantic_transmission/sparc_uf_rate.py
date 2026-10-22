"""Four-video UF rate experiment: fixed QPs and distortion-guided chunk QPs."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from . import bsu_predict as base, sparc, sparc_uf as pilot
from . import sparc_uf_rate_transport as transport
from .artifacts import sha256, write_json
from .dcvc_uf_codec import Codec
from .packets import accounting
from .qwen35_pairs_video import freeze
from .webvid5 import fingerprint
from .webvid_ablation import Stages, command, exclusive_lock, snapshot

REPO = sparc.REPO
MODULE = 'semantic_transmission.sparc_uf_rate'
DEFAULT = REPO/'outputs/kinetics_001_004_uf_rate_20261020'
IDS = ['kinetics_001_a393e9d3','kinetics_002_5e11db4c',
       'kinetics_003_5d588509','kinetics_004_afa8dee0']
QPS = [30,31,32]
STRENGTHS = [2,4]
COMPLETE = 'COMPLETE_SPARC_UF_RATE_001_004'


def audit_reference(root):
    """The existing stage contract, including clips with detected scene cuts."""
    p = base.read(root/'protocol.json')
    if p['signature'] != fingerprint({k:v for k,v in p.items() if k != 'signature'}):
        raise ValueError('reference protocol changed')
    dependencies = {}
    for stage in sparc.STAGES:
        receipt = base.read(root/f'stages/{stage}.json')
        if (receipt['status'] != 'PASSED' or receipt['identity'] != p['signature'] or
                receipt['dependencies'] != dependencies or
                snapshot(root,receipt['required']) != receipt['artifacts']):
            raise ValueError('reference receipt changed: '+stage)
        dependencies[stage] = fingerprint(receipt)
    result = base.read(root/'RESULT.json')
    if (result['status'] != sparc.COMPLETE or result['protocol_sha256'] != sha256(root/'protocol.json') or
            result['movie_sha256'] != sha256(root/'reconstruction/reconstruction.mp4')):
        raise ValueError('reference completion changed')
    source = base.read(root/'source.json')
    if base.inspect_frames(Path(source['directory']),source['video']['fps']) != source:
        raise ValueError('reference pixels changed')
    return source, base.read(root/'selection.json')


def preflight(args):
    import cv2
    import numpy as np
    from PIL import Image
    lg = args.lgvsc.resolve()
    root = args.output.resolve()
    if root == lg or root in lg.parents or lg in root.parents:
        raise ValueError('output overlaps LGVSC evidence')
    lr = base.read(lg/'RESULT.json')
    lp = base.read(lg/'protocol.json')
    if lr['status'] != 'COMPLETE_RATE_QUALITY_ACTION_COMPARISON' or lp['snr_db'] != 10:
        raise ValueError('LGVSC baseline incomplete or different SNR')
    old_ref = REPO/'outputs/kinetics_001_adaptive_v2_20261019'
    old_source, _ = audit_reference(old_ref)
    old = base.read(old_ref/'protocol.json')
    if old['profile'] != sparc.profile():
        raise ValueError('SPARC selection/receiver profile differs from frozen 001')
    for name in ('sparc','sparc_selection','sparc_semantics','dense_temporal_receiver',
                 'official_quality','metadata_channel','packets'):
        path = 'src/semantic_transmission/'+name+'.py'
        if old['code'][path] != sha256(REPO/path):
            raise ValueError('SPARC baseline implementation changed: '+name)
    clips = []
    for index, name in enumerate(IDS):
        row = next(v for v in lr['videos'] if v['id'] == name)
        info = next(v for v in lp['videos'] if v['id'] == name)
        config = info['config']
        if (config['channel_seed'] != 42 or config['snr_db'] != 10 or
                row['quality']['evaluation_profile'] != 'lgvsc_official_metrics_v1'):
            raise ValueError('LGVSC channel/metric conditions changed')
        movie = Path(row['video'])
        normalized = Path(row['quality']['reference'])
        if (sha256(movie) != row['quality']['video_sha256'] or
                sha256(normalized) != row['quality']['reference_sha256']):
            raise ValueError('LGVSC movies changed')
        run = movie.parents[2]
        for file, key in [('metadata.bin','packet_sha256'),('visual.c64','visual_sha256')]:
            if sha256(run/'transmitter'/file) != row['rate'][key]:
                raise ValueError('LGVSC transmitted evidence changed')
        inputs = lg.parents[1]/'outputs/bsu_kinetics_context/inputs/kinetics'/name
        source_path = Path(old_source['directory']) if index == 0 else inputs/'source_frames'
        source = base.inspect_frames(source_path,24)
        if source['video'] != dict(frames=240,width=576,height=320,fps=24):
            raise ValueError('unexpected clip timeline')
        if root == source_path or root in source_path.parents or source_path in root.parents:
            raise ValueError('output overlaps source')
        cap, count = cv2.VideoCapture(str(normalized)), 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                image = np.asarray(Image.open(source_path/f'{count}.png').convert('RGB'))
                if not np.array_equal(image,cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)):
                    raise ValueError('LGVSC and UF source RGB differ')
                count += 1
        finally:
            cap.release()
        if count != 240:
            raise ValueError('LGVSC source timeline mismatch')
        ref = old_ref if index == 0 else root/'references'/name
        reference_record = None
        if index:
            opts = sparc.parser().parse_args(['check','--source-frames',str(source_path),
                '--fps','24','--output',str(ref)])
            reference_record = sparc.preflight(opts)
        clips.append(dict(id=name,source=source,reference=str(ref),reference_record=reference_record,
            reference_protocol_sha256=sha256(ref/'protocol.json') if index == 0 else None,
            lgvsc_channel_uses=row['rate']['total_channel_uses'],
            lgvsc_quality=row['quality']['delivered_mp4'],lgvsc_movie_sha256=sha256(movie),
            source_rgb_identical_frames=count))
    weights, vendor = REPO/'.local/dcvc_uf_weights', REPO/'.local/dcvc_uf_vendor'
    for name, digest in pilot.WEIGHTS.items():
        if sha256(weights/name) != digest:
            raise ValueError('UF weight changed')
    if (subprocess.check_output(['git','-C',str(vendor),'rev-parse','HEAD'],text=True).strip() !=
            pilot.transport.UPSTREAM or subprocess.check_output(['git','-C',str(vendor),'diff','HEAD'])):
        raise ValueError('UF vendor changed')
    runtimes = dict(codec=str(REPO/'.local/dcvc_uf_runtime/bin/python'),
                    receiver=old['runtimes']['receiver'],channel=old['runtimes']['channel'])
    runtime = json.loads(subprocess.check_output([runtimes['codec'],'-c',
        'import torch,inference_extensions_cuda as c,MLCodec_extensions_cpp as e,json,platform;'
        'assert torch.cuda.is_available();'
        'print(json.dumps(dict(python=platform.python_version(),torch=torch.__version__,'
        'cuda=torch.version.cuda,extension=c.__file__,entropy=e.__file__)))'],text=True))
    runtime['binaries'] = {runtime[k]:sha256(runtime[k]) for k in ('extension','entropy')}
    p = dict(policy='sparc_uf_rate_001_004_v1',output=str(root),clips=clips,runtimes=runtimes,
        runtime=runtime,weights=pilot.WEIGHTS,weights_directory=str(weights),vendor=str(vendor),
        qps=QPS,strengths=STRENGTHS,total_trials=36,quality_gate=base.read(REPO/'configs/sparc_auxiliary_search.json'),
        allocator=dict(calibration_qp=31,score='mean(normalized_MSE + 0.01 * LPIPS) on selected frames',
                       centering='entropy-byte-weighted zero-mean chunk priority; rounded QP offsets',
                       intra_qp='fixed at uniform center',maximum_qp_offset=4,
                       exact_rate_prediction=False,final_measurement='full reencode, channel, independent decode, RIFE, final MP4'),
        snr_db=10,channel_seed=42,new_training=False,
        lgvsc_result_sha256=sha256(lg/'RESULT.json'),lgvsc_protocol_sha256=sha256(lg/'protocol.json'),
        code={str(f.relative_to(REPO)):sha256(f) for f in sorted((REPO/'src/semantic_transmission').glob('*.py'))},
        scripts={str(f.relative_to(REPO)):sha256(f) for f in
                 [REPO/'scripts/run_sparc_uf_rate.sh',REPO/'scripts/watch_sparc_uf_rate.sh']})
    p['signature'] = fingerprint(p)
    return p


def adaptive_schedules(plan, calibration, sender):
    """Measured errors guide candidates; actual full streams decide their cost."""
    import numpy as np
    desc = transport.descriptors(plan['indices'],plan['cuts'])
    records = [r for g in sender['groups'] for r in g['records']]
    if len(records) != len(desc):
        raise ValueError('calibration chunk layout differs')
    scores, byte_weights = [], []
    for chunk, record in zip(desc,records):
        ids = chunk['indices']
        if record['position'] != chunk['position'] or record['valid_frames'] != len(ids) or record['qp'] != 31:
            raise ValueError('calibration is not uniform QP31')
        scores.append(float(np.mean([10**(-calibration['per_frame']['psnr_db'][i]/10)+
                                     .01*calibration['per_frame']['lpips_vgg'][i] for i in ids])))
        byte_weights.append(record['entropy_bytes'])
    scores, weights = np.asarray(scores), np.asarray(byte_weights,dtype=float)
    active = np.asarray([not c['intra'] for c in desc])
    priority = np.zeros(len(desc))
    if active.sum() > 1 and np.ptp(scores[active]) > 1e-12:
        z = scores[active]-np.median(scores[active])
        z /= max(float(np.quantile(np.abs(z),.75)),1e-12)
        z = np.clip(z,-1,1)
        z -= np.average(z,weights=weights[active])
        z /= max(1.,float(np.abs(z).max()))
        priority[active] = z
    proposals = []
    for center in QPS:
        for strength in STRENGTHS:
            qps = np.clip(center+np.rint(strength*priority),0,63).astype(int).tolist()
            proposals.append(dict(name=f'adaptive_q{center}_s{strength}',kind='adaptive',
                qps=qps,center=center,strength=strength,
                changed_chunks=sum(q != center for q in qps),
                calibration_is_sender_only=True))
    return dict(chunks=desc,scores=scores.tolist(),priority=priority.tolist(),proposals=proposals)


def worker(stage, job):
    root = Path(job['output'])
    if stage in ('quality','restore'):
        return pilot.worker(stage,job)
    if stage == 'channel':
        from .metadata_channel import transmit
        packet = Path(job['packet']).read_bytes()
        transport.parse(packet,job['weights'])
        received, report = transmit(packet,10,42,bits_per_symbol=4)
        if received != packet or report['bit_errors'] or any(report[k] != v for k,v in accounting(len(packet)).items()):
            raise ValueError('channel integrity or accounting failed')
        dest = root/'received'; dest.mkdir()
        (dest/'packet.bin').write_bytes(received)
        write_json(dest/'channel.json',dict(report,packet_sha256=sha256(dest/'packet.bin'),
                                          physical_link_overhead_included=False))
        return
    import numpy as np
    from PIL import Image
    codec = Codec(Path(job['vendor']),Path(job['weights_directory']),job['weights'])
    if stage == 'sender':
        plan, source = job['selection'], job['source']
        groups = transport.scene_groups(plan['indices'],plan['cuts'])
        coded, details, cursor = [], [], 0
        for n, ids in enumerate(groups):
            images = []
            for i in ids:
                path = Path(source['directory'])/f'{i}.png'
                if sha256(path) != source['hashes'][path.name]:
                    raise ValueError('sender source changed')
                images.append(np.asarray(Image.open(path).convert('RGB')))
            count = 1+(len(ids)-1+7)//8
            qps = job['qps'][cursor:cursor+count]
            stream, records = codec.encode(np.stack(images),qps)
            coded.append((ids,qps,stream)); details.append(dict(indices=ids,records=records))
            cursor += count
            sparc.progress(root,stage,n+1,len(groups))
        if cursor != len(job['qps']):
            raise ValueError('unused QP entries')
        packet = transport.make(source['video'],plan['semantic_keys'],plan['cuts'],coded,job['weights'],
                                max_gap=plan.get('max_gap',4))
        dest = root/'sender'; dest.mkdir()
        (dest/'packet.bin').write_bytes(packet)
        write_json(dest/'sender.json',dict(groups=details,qps=job['qps'],
            entropy_bytes=sum(r['entropy_bytes'] for g in details for r in g['records']),
            padded_frames=sum(transport.old.padding(len(ids)) for ids in groups),
            intra_frames=len(groups),packet_sha256=sha256(dest/'packet.bin'),**accounting(len(packet))))
        return
    if stage == 'decode':
        packet = Path(job['packet']); header,payload = transport.parse(packet.read_bytes(),job['weights'])
        dest = root/'decoded'; dest.mkdir(); indices = []
        for n, chunk in enumerate(header['chunks']):
            frames = codec.decode(payload[chunk['offset']:chunk['offset']+chunk['bytes']],
                                  len(chunk['indices']),header['video'],chunk['qps'])
            for i, frame in zip(chunk['indices'],frames):
                Image.fromarray(frame).save(dest/f'{i:05d}.png')
            indices.extend(chunk['indices']); sparc.progress(root,stage,n+1,len(header['chunks']))
        write_json(root/'decoded.json',dict(video=header['video'],indices=indices,cuts=header['cuts'],max_gap=header['max_gap'],
            packet_sha256=sha256(packet),encoder_state_reused=False,source_pixels_at_receiver=False))
        return
    raise ValueError('unknown rate worker stage')


def launch(stage, job, log, runtime):
    env = dict(os.environ,PYTHONPATH=str(REPO/'src'),PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',
               OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    env.pop('LD_LIBRARY_PATH',None)
    if stage == 'channel':
        env['CUDA_VISIBLE_DEVICES']='-1'
    command(REPO,[runtime,'-m',MODULE,'worker','--stage',stage,'--job',str(job)],log,env)


def quality_scalar(value):
    return {k:value[k] for k in ('psnr_db','ssim','lpips_vgg','clip','dists','roi_mae','temporal_mae')}


def objective(quality):
    return 10**(-quality['psnr_db']/10)+.01*quality['lpips_vgg']


class Experiment:
    def __init__(self,p):
        self.p,self.root = p,Path(p['output'])
        self.completed = 0
        self.start = base.read(self.root/'started.json')['unix']

    def state(self,clip,trial,stage,active=None,log=None,status='RUNNING',**extra):
        write_json(self.root/'status.json',dict(status=status,pid=os.getpid(),started_unix=self.start,
            updated_unix=time.time(),clip=clip,trial=trial,stage=stage,
            completed_trials=self.completed,total_trials=self.p['total_trials'],
            active_root=str(active) if active else None,log=str(log) if log else None,
            resume='bash scripts/run_sparc_uf_rate.sh run',**extra))
        print('COMPARE_PROGRESS '+json.dumps(dict(state=status,current=f'{clip} {trial}',stage=stage,
            overall=f'{self.completed}/{self.p["total_trials"]} completed conditions')),flush=True)

    def evaluate(self,clip,source,plan,proposal):
        name = proposal['name']; folder = self.root/clip['id']/name; folder.mkdir(parents=True,exist_ok=True)
        signature = fingerprint(dict(parent=self.p['signature'],clip=clip['id'],selection=plan,proposal=proposal))
        freeze(folder/'trial.json',dict(signature=signature,proposal=proposal))
        stages = Stages(folder,signature)
        common = {k:self.p[k] for k in ('vendor','weights_directory','weights')}
        jobs = [
            ('sender',dict(**common,source=source,selection=plan,qps=proposal['qps']),['sender'],'codec'),
            ('channel',dict(weights=self.p['weights'],packet=str(folder/'sender/packet.bin')),['received'],'channel'),
            ('decode',dict(**common,packet=str(folder/'received/packet.bin')),['decoded','decoded.json'],'codec'),
            ('restore',dict(decoded=str(folder/'decoded')),['reconstruction'],'receiver'),
            ('quality',dict(source=source,movie=str(folder/'reconstruction/reconstruction.mp4'),
                            gate=self.p['quality_gate']),['quality.json'],'receiver'),
        ]
        for stage, extra, products, role in jobs:
            path = folder/f'{stage}_job.json'; freeze(path,dict(output=str(folder),**extra))
            self.state(clip['id'],name,stage,folder,folder/f'logs/{stage}.log')
            stages.step(stage,products,products,lambda log,s=stage,j=path,r=role:
                        launch(s,j,log,self.p['runtimes'][r]))
        sender, measured = base.read(folder/'sender/sender.json'), base.read(folder/'quality.json')
        uses = base.read(folder/'received/channel.json')['complex_channel_uses']
        if (folder/'sender/packet.bin').read_bytes() != (folder/'received/packet.bin').read_bytes():
            raise ValueError('paid packet changed')
        result = dict(**proposal,channel_uses=uses,packet_bytes=sender['packet_bytes'],
            quality=quality_scalar(measured['quality']),movie_sha256=measured['movie_sha256'],
            movie=str(folder/'reconstruction/reconstruction.mp4'),padded_frames=sender['padded_frames'],
            sender_metadata_paid=True,source_pixels_at_receiver=False)
        if clip.get('lgvsc_channel_uses') is not None:
            result.update(lgvsc_budget=clip['lgvsc_channel_uses'],
                          within_lgvsc_budget=uses <= clip['lgvsc_channel_uses'])
        write_json(folder/'RESULT.json',result)
        self.completed += 1
        return result

    def run(self):
        summaries = []
        for clip in self.p['clips']:
            ref = Path(clip['reference'])
            if clip['reference_record']:
                self.state(clip['id'],'SPARC reference','reference',ref,ref/'logs/semantic.log')
                # This builds exactly the existing SPARC selection/reference, once.
                sparc.execute(clip['reference_record'])
            elif sha256(ref/'protocol.json') != clip['reference_protocol_sha256']:
                raise ValueError('reused 001 reference changed')
            source, plan = audit_reference(ref)
            if source != clip['source']:
                raise ValueError('source changed after preflight')
            folder = self.root/clip['id']; folder.mkdir(parents=True,exist_ok=True)
            freeze(folder/'selection.json',plan)
            count = len(transport.descriptors(plan['indices'],plan['cuts']))
            rows = []
            for qp in QPS:
                rows.append(self.evaluate(clip,source,plan,dict(name=f'uniform_q{qp}',kind='uniform',
                    center=qp,strength=0,qps=[qp]*count,changed_chunks=0)))
            calibration_folder = folder/'uniform_q31'
            schedule = adaptive_schedules(plan,base.read(calibration_folder/'quality.json'),
                                           base.read(calibration_folder/'sender/sender.json'))
            schedule.update(quality_sha256=sha256(calibration_folder/'quality.json'),
                            sender_sha256=sha256(calibration_folder/'sender/sender.json'))
            freeze(folder/'allocation.json',schedule)
            for proposal in schedule['proposals']:
                rows.append(self.evaluate(clip,source,plan,proposal))
            uniform32 = next(r for r in rows if r['name']=='uniform_q32')
            for row in rows:
                row['within_uniform32_budget'] = row['channel_uses'] <= uniform32['channel_uses']
                row['psnr_delta_vs_uniform32'] = row['quality']['psnr_db']-uniform32['quality']['psnr_db']
                row['lpips_delta_vs_uniform32'] = row['quality']['lpips_vgg']-uniform32['quality']['lpips_vgg']
            def best(kind,limit):
                eligible = [r for r in rows if r['kind']==kind and r['channel_uses'] <= limit]
                return min(eligible,key=lambda r:objective(r['quality'])) if eligible else None
            summary = dict(id=clip['id'],status='COMPLETE_CLIP',indices=plan['indices'],
                semantic_keys=plan['semantic_keys'],cuts=plan['cuts'],selected_frames_unchanged=True,
                lgvsc_budget=clip['lgvsc_channel_uses'],lgvsc_quality=clip['lgvsc_quality'],trials=rows,
                best_uniform_within_lgvsc_budget=best('uniform',clip['lgvsc_channel_uses']),
                best_adaptive_within_lgvsc_budget=best('adaptive',clip['lgvsc_channel_uses']),
                best_adaptive_within_uniform32_budget=best('adaptive',uniform32['channel_uses']),
                selection_objective='10**(-mean_PSNR/10) + 0.01*mean_LPIPS',
                independent_semantic_evaluation=False)
            write_json(folder/'RESULT.json',summary); summaries.append(summary)
            write_json(self.root/'partial_results.json',summaries)
        write_json(self.root/'RESULT.json',dict(status=COMPLETE,clips=summaries,
            protocol_sha256=sha256(self.root/'protocol.json'),new_training_performed=False,
            limits=['four development videos, single SNR/seed',
                    'adaptive QP uses sender-side reconstructed errors; extra encoder computation',
                    'candidate costs measured by full reencoding, not additive chunk estimates',
                    'a null best candidate means no tested candidate meets that video budget',
                    'independent semantic/action evaluation not performed']))
        self.state('001–004','complete','complete',status=COMPLETE,exit_code=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['check','run','worker'])
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--lgvsc',type=Path,default=REPO.parent/'semantic_transmission/outputs/lgvsc_kinetics_matched')
    parser.add_argument('--stage')
    parser.add_argument('--job',type=Path)
    args = parser.parse_args()
    if args.action == 'worker':
        return worker(args.stage,base.read(args.job))
    if args.action == 'check':
        p = preflight(args)
        print(json.dumps(dict(status='PREFLIGHT_PASSED',clips=[c['id'] for c in p['clips']],
                             trials=p['total_trials'],signature=p['signature'])))
        return
    root = args.output.resolve(); root.mkdir(parents=True,exist_ok=True)
    with exclusive_lock(root/'execution.lock'):
        if not (root/'started.json').exists():
            write_json(root/'started.json',dict(unix=time.time()))
        write_json(root/'status.json',dict(status='PREFLIGHT',pid=os.getpid(),stage='preflight',
            clip='001–004',trial='source/model/resume verification',completed_trials=0,total_trials=36,
            started_unix=base.read(root/'started.json')['unix']))
        def stop(*_):
            raise KeyboardInterrupt('SIGTERM')
        signal.signal(signal.SIGTERM,stop)
        try:
            p = preflight(args)
            freeze(root/'protocol.json',p)
            experiment = Experiment(p)
            experiment.run()
        except BaseException as error:
            status = base.read(root/'status.json') if (root/'status.json').exists() else {}
            status.update(status='INTERRUPTED' if isinstance(error,KeyboardInterrupt) else 'FAILED',
                          error=str(error),exit_code=130 if isinstance(error,KeyboardInterrupt) else 1,
                          updated_unix=time.time())
            write_json(root/'status.json',status)
            raise


if __name__ == '__main__':
    main()
