"""SPARC experiment: cached semantic keys + paid sparse trajectories/visibility.

Runs a keys-only baseline, received-motion correction, visibility ablation and
a few-extra-anchor control under the motion arm's total channel-use cap.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from . import bsu_predict as base, sparc
from . import sparc_motion_cache as cache
from . import sparc_motion_transport as transport
from .artifacts import sha256, write_json
from .packets import accounting
from .webvid5 import fingerprint

REPO=sparc.REPO
CONFIG=REPO/'configs/sparc_motion.json'
STAGES=('prepare','encode','channel','decode','restore','evaluate')
ARMS=('keys','motion','motion_no_visibility','auxiliary')
COMPLETE='COMPLETE_SPARC_MOTION_PILOT'


def profile():
    value=base.read(CONFIG)
    if value != dict(policy=transport.POLICY,maximum_key_gap=64,side_information_ratio_cap=.10,
            trajectory_levels=[[16,9],[12,7],[8,5],[8,3]],auxiliary_control_candidates=4,
            qp=32,snr_db=10,channel_seed=42,trajectory_channel_seed=43,reference_strength=.65,
            scene_policy='require_no_detected_cuts',new_training=False,selection_uses_reconstruction_quality=False):
        raise ValueError('register a new profile for changed pilot settings')
    return value


def preflight(args):
    root=args.output.resolve();reference=args.reference.resolve(strict=True)
    legacy=args.trajectory_run.resolve(strict=True);prepared=args.trajectory_source.resolve(strict=True)
    protected=[reference,legacy,prepared.parent]
    if args.historical_comparison:protected.append(args.historical_comparison.resolve(strict=True).parent)
    if any(root==p or root in p.parents or p in root.parents for p in protected):
        raise ValueError('experiment output overlaps protected evidence')
    source,semantic=cache.audit_reference(reference)
    windows,reuse=cache.audit_legacy(source,semantic['keys'],legacy,prepared)
    # Reuse the complete asset/runtime validation without starting a selector.
    options=sparc.parser().parse_args(['check','--source-frames',source['directory'],
        '--fps',str(source['video']['fps']),'--semantic-replay',str(reference/'semantic.json'),
        '--output',str(root/'asset_preflight')])
    record=sparc.preflight(options)
    record.pop('signature');record.update(output=str(root),profile=profile(),reference=str(reference),
        reference_semantic_sha256=sha256(reference/'semantic.json'),
        trajectory_reuse=reuse,legacy_windows=windows,reference_protocol_sha256=sha256(reference/'protocol.json'))
    for path in (CONFIG,REPO/'scripts/run_sparc_motion.sh'):
        record['code'][str(path.relative_to(REPO))]=sha256(path)
    record['historical']=None
    if args.historical_comparison:
        path=args.historical_comparison.resolve(strict=True);old=base.read(path)
        if (old['status']!='COMPLETE_SPARC_MATCHED_COMPARISON' or old['video']!=source['video'] or
                old['source_frame_hashes']!=source['hashes']):raise ValueError('historical comparison source differs')
        records={}
        for label in ('fixed','adaptive'):
            result=old['results'][label]
            matches=[p.parent for p in reference.parent.glob('*/protocol.json') if sha256(p)==result['protocol_sha256']]
            if len(matches)!=1:raise ValueError('historical protocol cannot be resolved')
            case=matches[0];other_source,_=cache.audit_reference(case)
            old_profile=base.read(case/'protocol.json')['profile']
            if other_source!=source or any(old_profile[k]!=profile()[k] for k in ('qp','snr_db','channel_seed')):
                raise ValueError('historical comparison profile differs')
            for module in ('official_quality.py','predictive_codec.py','dense_temporal_receiver.py'):
                key='src/semantic_transmission/'+module
                if base.read(case/'protocol.json')['code'][key]!=record['code'][key]:
                    raise ValueError('historical inference/evaluation code differs')
            rows=list(csv.DictReader((path.parent/f'{label}_per_frame.csv').open()))
            if len(rows)!=source['video']['frames']:raise ValueError('historical metric timeline differs')
            for metric in ('psnr_db','ssim','lpips_vgg','clip','dists'):
                if abs(sum(float(row[metric]) for row in rows)/len(rows)-result['quality'][metric])>1e-9:
                    raise ValueError('historical metric aggregation differs')
            movie=case/'reconstruction/reconstruction.mp4'
            if sha256(movie)!=result['movie_sha256']:raise ValueError('historical movie changed')
            records[label]=dict(result,movie=str(movie),per_frame_csv_sha256=sha256(path.parent/f'{label}_per_frame.csv'))
        record['historical']=dict(path=str(path),sha256=sha256(path),results=records)
    record['signature']=fingerprint(record)
    if (root/'protocol.json').exists() and base.read(root/'protocol.json')!=record:
        raise ValueError('frozen experiment changed; use a new output')
    return record


def progress(root,stage,n,total):sparc.progress(root,stage,n,total)


def prepare(record):
    from .sparc_semantics import select
    root=Path(record['output']);source=record['source'];reference=Path(record['reference'])
    if sha256(reference/'semantic.json')!=record['reference_semantic_sha256']:raise ValueError('semantic replay changed')
    for path,digest in record['trajectory_reuse']['provenance'].items():
        if sha256(path)!=digest:raise ValueError('trajectory replay changed')
    write_json(root/'source.json',source)
    semantic=select(source,sparc.profile(),root,reference/'semantic.json')
    if semantic['new_model_calls']:raise ValueError('unexpected new Qwen call')
    write_json(root/'legacy_windows.json',record['legacy_windows'])
    write_json(root/'reuse.json',dict(record['trajectory_reuse'],semantic_keys=semantic['keys'],
        qwen_new_calls=0,qwen_reused_calls=semantic['replayed_model_calls'],
        dcvc_cache_reused=False,dcvc_reason='Changing temporal reference frames requires new I/P coding'))


def worker(stage,job):
    if stage!='channel':
        import numpy as np
        from PIL import Image
    root=Path(job['output'])
    if stage=='encode':
        from .predictive_codec import Codec
        source=job['source'];video=source['video'];keys=base.read(root/'semantic.json')['keys'];config=job['profile']
        def images(indices):
            values=[]
            for i in indices:
                path=Path(source['directory'])/f'{i}.png'
                if sha256(path)!=source['hashes'][path.name]:raise ValueError('sender source changed')
                with Image.open(path) as image:values.append(np.asarray(image.convert('RGB')))
            return np.stack(values)
        model=Codec(REPO/'.local/vendor/DCVC-RT',REPO/'.local/checkpoints/dcvc_rt')
        packet=transport.make_keys(video,model.encode(images(keys),keys,config['qp']))
        motion,audit=cache.choose_motion(packet,base.read(root/'legacy_windows.json'),config)
        key_uses=accounting(len(packet))['complex_channel_uses'];motion_uses=accounting(len(motion))['complex_channel_uses']
        cap=key_uses+motion_uses;chosen=packet;chosen_keys=keys;trials=[]
        dest=root/'sender';dest.mkdir()
        for n in range(1,config['auxiliary_control_candidates']+1):
            indices=cache.auxiliary_schedule(keys,n)
            candidate=transport.make_keys(video,model.encode(images(indices),indices,config['qp']))
            uses=accounting(len(candidate))['complex_channel_uses']
            (dest/f'trial_{n}.bin').write_bytes(candidate)
            trials.append(dict(extra_anchors=len(indices)-len(keys),indices=indices,channel_uses=uses,bytes=len(candidate)))
            if uses<=cap:chosen=candidate;chosen_keys=indices
            progress(root,stage,n,config['auxiliary_control_candidates'])
        for name,data in [('keys',packet),('motion',motion),('auxiliary',chosen)]:
            (dest/f'{name}.bin').write_bytes(data)
        write_json(dest/'sender.json',dict(indices=keys,auxiliary_indices=chosen_keys,base_channel_uses=key_uses,
            motion_channel_uses=motion_uses,motion_budget=audit,total_motion_channel_uses=cap,
            auxiliary_channel_uses=accounting(len(chosen))['complex_channel_uses'],auxiliary_trials=trials,
            auxiliary_selection='largest tested extra-anchor count within actual motion-arm channel budget',
            search_uses_quality=False,packet_sha256={name:sha256(dest/f'{name}.bin') for name in ('keys','motion','auxiliary')}))
        return
    if stage=='channel':
        from .metadata_channel import transmit
        source=Path(job['sender']);dest=root/'received';dest.mkdir();reports={}
        packets={name:(source/f'{name}.bin').read_bytes() for name in ('keys','motion','auxiliary')}
        transport.parse_keys(packets['keys']);transport.parse_keys(packets['auxiliary'])
        transport.parse_motion(packets['motion'],packets['keys'])
        for i,(name,data) in enumerate(packets.items(),1):
            if name=='auxiliary' and data==packets['keys']:
                received=(dest/'keys.bin').read_bytes();report=dict(reports['keys'],reused_identical_transmission=True)
            else:received,report=transmit(data,10,43 if name=='motion' else 42,bits_per_symbol=4)
            if received!=data or report['bit_errors'] or any(report[k]!=v for k,v in accounting(len(data)).items()):
                raise ValueError('motion experiment channel integrity/accounting failed')
            (dest/f'{name}.bin').write_bytes(received);reports[name]=report
            progress(root,stage,i,3)
        write_json(dest/'channel.json',dict(packets=reports,all_packets_exact=True,physical_link_overhead_included=False))
        return
    if stage=='decode':
        from .predictive_codec import Codec
        received=Path(job['received']);dest=root/'decoded';dest.mkdir()
        model=Codec(REPO/'.local/vendor/DCVC-RT',REPO/'.local/checkpoints/dcvc_rt')
        metadata={};seen={}
        for i,name in enumerate(('keys','auxiliary'),1):
            packet=(received/f'{name}.bin').read_bytes();header,coded,indices=transport.parse_keys(packet)
            digest=sha256(received/f'{name}.bin')
            if digest in seen:frames=seen[digest]
            else:
                ids,frames=model.decode(coded)
                if ids!=indices or frames.shape!=(len(indices),header['video']['height'],header['video']['width'],3):
                    raise ValueError('DCVC receiver timeline differs')
                seen[digest]=frames
            folder=dest/name;folder.mkdir()
            for index,frame in zip(indices,frames):Image.fromarray(frame).save(folder/f'{index:05d}.png')
            metadata[name]=dict(video=header['video'],indices=indices,packet_sha256=digest)
            progress(root,stage,i,2)
        write_json(dest/'decoded.json',metadata)
        return
    if stage=='restore':
        from . import sparc_motion_receiver as receiver
        sparc.configure_numerics()
        received=Path(job['received']);decoded=Path(job['decoded']);info=base.read(decoded/'decoded.json')
        anchors={name:np.stack([np.asarray(Image.open(decoded/name/f'{i:05d}.png').convert('RGB'))
                  for i in meta['indices']]) for name,meta in info.items()}
        for name,meta in info.items():
            if meta['packet_sha256']!=sha256(received/f'{name}.bin'):raise ValueError('decoded packet binding differs')
        trajectories=transport.parse_motion((received/'motion.bin').read_bytes(),(received/'keys.bin').read_bytes())
        images=receiver.interpolate(anchors['keys'],info['keys']['indices'],info['keys']['video'],
            REPO/'.local/vendor/Practical-RIFE',REPO/'.local/checkpoints/rife/v4.25',lambda n,t:progress(root,stage,n,t))
        dest=root/'reconstruction';dest.mkdir();audits={}
        for number,name in enumerate(ARMS,1):
            if name=='keys':frames=images;audit=dict(reused_rife=True,source_pixels_used=False)
            elif name=='auxiliary':
                same=info['keys']['packet_sha256']==info['auxiliary']['packet_sha256']
                frames=images if same else receiver.interpolate(anchors[name],info[name]['indices'],info[name]['video'],
                    REPO/'.local/vendor/Practical-RIFE',REPO/'.local/checkpoints/rife/v4.25')
                audit=dict(reused_rife=same,source_pixels_used=False)
            else:frames,audit=receiver.correct(images,anchors['keys'],info['keys']['indices'],trajectories,
                visibility=name=='motion')
            indices=info['auxiliary' if name=='auxiliary' else 'keys']['indices']
            expected=anchors['auxiliary' if name=='auxiliary' else 'keys']
            if not np.array_equal(frames[indices],expected):raise ValueError('restoration changed received anchors')
            export(frames,info['keys']['video'],dest/name)
            audits[name]=dict(audit,anchors_preserved_before_mp4=True,movie_sha256=sha256(dest/name/'reconstruction.mp4'))
            progress(root,stage,number,len(ARMS))
        write_json(dest/'receiver.json',dict(arms=audits,source_pixels_at_receiver=False,numerics=sparc.NUMERICS,
            valid_trajectory_samples=sum(int(t['valid'].sum()) for t in trajectories.values()),
            invalid_trajectory_samples=sum(int((~t['valid']).sum()) for t in trajectories.values()),
            trajectory_windows=len(trajectories),new_rife_inferences=1+(info['keys']!=info['auxiliary'])))
        return
    if stage=='evaluate':
        from .official_quality import OfficialMetrics
        from .research_quality import read_video
        from .stutter_metrics import evaluate as motion_metrics
        import cv2
        cv2.setNumThreads(2)
        source=job['source'];reference=np.stack([np.asarray(Image.open(Path(source['directory'])/f'{i}.png').convert('RGB'))
            for i in range(source['video']['frames'])])
        for name,digest in source['hashes'].items():
            if sha256(Path(source['directory'])/name)!=digest:raise ValueError('evaluation source changed')
        sender=base.read(root/'sender/sender.json');channel=base.read(root/'received/channel.json')['packets'];reports={}
        total=len(ARMS)*source['video']['frames']*4
        for number,name in enumerate(ARMS):
            movie=root/f'reconstruction/{name}/reconstruction.mp4';frames=read_video(movie)
            quality,rows=OfficialMetrics().evaluate(reference,frames,
                lambda n,t:progress(root,stage,number*t+n,total))
            with (root/f'reconstruction/{name}/quality_per_frame.csv').open('x') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            uses=channel['auxiliary' if name=='auxiliary' else 'keys']['complex_channel_uses']
            if name.startswith('motion'):uses+=channel['motion']['complex_channel_uses']
            reports[name]=dict(channel_uses=uses,anchor_count=len(sender['auxiliary_indices'] if name=='auxiliary' else sender['indices']),
                quality=quality,motion_diagnostics=motion_metrics(reference,frames),movie_sha256=sha256(movie))
        historical={}
        if job.get('historical'):
            old=job['historical']
            if sha256(old['path'])!=old['sha256']:raise ValueError('historical comparison changed')
            for name,result in old['results'].items():
                if sha256(result['movie'])!=result['movie_sha256']:raise ValueError('historical movie changed')
                historical[name]=dict(result,motion_diagnostics=motion_metrics(reference,read_video(result['movie'])),
                                     quality_reused=True,new_model_inference=False)
        write_json(root/'comparison.json',dict(results=reports,historical=historical,
            same_key_bitstream_for_keys_and_motion=True,same_total_budget_motion_auxiliary=True,
            equal_actual_channel_uses=reports['motion']['channel_uses']==reports['auxiliary']['channel_uses'],
            metric_profile='lgvsc_official_metrics_v1',evaluation_boundary='decoded final MP4',
            independent_semantic_action_evaluation=False,receiver_quality_used_for_selection=False,
            limitations=['one source, one SNR and seed','long-gap RIFE is experimental',
                'visibility is tracked-point validity, not a dense segmentation mask',
                'auxiliary control searches only zero through four longest-gap midpoint additions',
                'motion_no_visibility receives the same packet but ignores its validity bits']))
        return
    raise ValueError('unknown pilot stage')


def export(frames,video,dest):
    from PIL import Image
    from fractions import Fraction
    images=dest/'frames';images.mkdir(parents=True)
    if len(frames)!=video['frames']:raise ValueError('export frame count differs')
    for i,frame in enumerate(frames):Image.fromarray(frame).save(images/f'{i:05d}.png')
    movie=dest/'reconstruction.mp4'
    subprocess.run(['ffmpeg','-v','error','-nostdin','-framerate',str(video['fps']),'-i',str(images/'%05d.png'),
        '-frames:v',str(len(frames)),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(movie)],check=True)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=width,height,nb_read_frames,avg_frame_rate','-of','json',str(movie)]))['streams'][0]
    if (int(probe['nb_read_frames'])!=len(frames) or int(probe['width'])!=video['width'] or
            int(probe['height'])!=video['height'] or abs(float(Fraction(probe['avg_frame_rate']))-video['fps'])>1e-6):
        raise ValueError('exported MP4 timeline differs')


def launch(stage,job,log,runtime):
    from .webvid_ablation import command
    env=dict(os.environ,PYTHONPATH=str(REPO/'src'),PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    env.pop('LD_LIBRARY_PATH',None)
    if stage=='channel':env['CUDA_VISIBLE_DEVICES']='-1'
    stop=threading.Event();root=job.parent
    def monitor():
        last=None
        while not stop.wait(1):
            try:
                value=base.read(root/'worker_progress.json')
                if value['stage']==stage and value!=last:
                    n,t=value['done'],value['total'];last=value
                    print('COMPARE_PROGRESS '+json.dumps(dict(stage=f'{stage}: {n}/{t} ({100*n/t:.1f}%)',
                        current=root.name,state='RUNNING')),flush=True)
            except (FileNotFoundError,ValueError,KeyError):pass
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:command(REPO,[runtime,'-m','semantic_transmission.sparc_motion','--worker',stage,'--job',str(job)],log,env)
    finally:stop.set();thread.join(timeout=2)


def execute(record):
    from .etri_60s_check import lock
    from .qwen35_pairs_video import freeze
    from .webvid_ablation import Stages
    root=Path(record['output']);root.mkdir(parents=True,exist_ok=True);(REPO/'.local').mkdir(exist_ok=True)
    with lock(REPO/'.local/etri_60s_check.lock'),lock(root/'execution.lock'):
        freeze(root/'protocol.json',record);stages=Stages(root,record['signature']);started=time.time();current='prepare';done=0
        def state(status='RUNNING'):
            write_json(root/'status.json',dict(status=status,stage=current,completed_stages=done,total_stages=len(STAGES),
                elapsed_seconds=time.time()-started))
            print('COMPARE_PROGRESS '+json.dumps(dict(stage=current,current=root.name,state=status,
                overall=f'{done}/{len(STAGES)} ({100*done/len(STAGES):.1f}%) stages complete')),flush=True)
        try:
            state();required=['source.json','semantic.json','semantic_cache','legacy_windows.json','reuse.json']
            stages.step('prepare',[x for x in required if x!='semantic_cache'],required,lambda _:prepare(record))
            jobs=dict(encode=dict(source=record['source'],profile=record['profile']),channel=dict(sender=str(root/'sender')),
                decode=dict(received=str(root/'received')),restore=dict(received=str(root/'received'),decoded=str(root/'decoded')),
                evaluate=dict(source=record['source'],historical=record['historical']))
            for done,(current,owned,runtime) in enumerate([
                    ('encode',['sender'],'codec'),('channel',['received'],'channel'),('decode',['decoded'],'codec'),
                    ('restore',['reconstruction'],'receiver'),('evaluate',['comparison.json'],'receiver')],1):
                state();job=root/f'{current}_job.json';freeze(job,dict(output=str(root),**jobs[current]))
                required=owned if current!='restore' else [f'reconstruction/{name}/{part}' for name in ARMS
                    for part in ('frames','reconstruction.mp4')]+['reconstruction/receiver.json']
                if current=='evaluate':
                    owned=required=['comparison.json',*[f'reconstruction/{name}/quality_per_frame.csv' for name in ARMS]]
                stages.step(current,owned,required,lambda log,s=current,j=job,r=runtime:launch(s,j,log,record['runtimes'][r]))
            result=dict(status=COMPLETE,protocol_sha256=sha256(root/'protocol.json'),comparison_sha256=sha256(root/'comparison.json'),
                reuse=base.read(root/'reuse.json'),results=base.read(root/'comparison.json')['results'],new_training=False)
            write_json(root/'RESULT.json',result);current='complete';done=len(STAGES);state(COMPLETE)
        except BaseException as error:
            if (root/'RESULT.json').exists():
                target=root/'failed_attempts'/f'result_{time.time_ns()}.json';target.parent.mkdir(parents=True,exist_ok=True)
                (root/'RESULT.json').rename(target)
            state('INTERRUPTED' if isinstance(error,KeyboardInterrupt) else 'FAILED');raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',nargs='?',default='run',choices=('profile','check','run','status'))
    for name in ('reference','trajectory-run','trajectory-source','historical-comparison','output','job'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--worker',choices=STAGES[1:],help=argparse.SUPPRESS);args=p.parse_args(argv)
    if args.worker:
        job=base.read(args.job);record=base.read(Path(job['output'])/'protocol.json')
        if record['signature']!=fingerprint({k:v for k,v in record.items() if k!='signature'}):raise ValueError('protocol changed')
        for path,digest in record['code'].items():
            if sha256(REPO/path)!=digest:raise ValueError('frozen worker code changed')
        return worker(args.worker,job)
    if args.action=='profile':print(json.dumps(profile(),indent=2));return
    if not args.output:p.error('--output is required')
    if args.action=='status':
        print(json.dumps(base.read(args.output/'status.json') if (args.output/'status.json').exists() else dict(status='NOT_STARTED'),indent=2));return
    if not all((args.reference,args.trajectory_run,args.trajectory_source)):p.error('reference and trajectory provenance paths are required')
    record=preflight(args)
    if args.action=='check':
        print(json.dumps(dict(status='READY_REUSE_VERIFIED',reuse=record['trajectory_reuse'],profile=record['profile']),indent=2));return
    def interrupted(*_):raise KeyboardInterrupt('termination requested; stage receipts retained')
    signal.signal(signal.SIGTERM,interrupted)
    execute(record)


if __name__=='__main__':main()
