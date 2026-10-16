"""Resumable, rate-matched evaluation of the BSU dense-guide architecture."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

from .artifacts import sha256, write_json
from .webvid5 import fingerprint
from .webvid_ablation import Stages
from .qwen35_pairs_video import freeze
from . import dense_video_transport as transport

REPO = Path(__file__).resolve().parents[2]
MODULE = 'semantic_transmission.bsu_dense_validation'
MODEL = REPO/'.local/checkpoints/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth'
MODEL_SHA = '193b229909ca89cd8b55de9c9e7fce146ae759d59dfcd78d8feb9dd1d6fa0fd7'
read = lambda p: json.loads(Path(p).read_text())


def progress(stage, done, total, current=''):
    print('COMPARE_PROGRESS '+json.dumps(dict(stage=stage, done=done, total=total, current=current,
        overall=f'{done}/{total} ({100*done/total:.1f}%) 영상·조건 완료')), flush=True)


def save_movie(frames, directory, fps):
    from PIL import Image
    from .bsu_receiver import encode_frames
    directory = Path(directory); paths = directory/'frames'; paths.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(paths/f'{i:05d}.png')
    encode_frames(paths, directory/'reconstruction.mp4', len(frames), fps)


def evaluate(case, movie, output, channel_uses):
    from .official_quality import OfficialMetrics, summarize
    from .research_quality import read_video
    from .stutter_metrics import evaluate as temporal
    from .bsu_receiver_compare import boundary_diagnostics
    from .etri_60s_check import pts_audit
    source = Path(case['source']['normalized_video'])
    if sha256(source) != case['source']['normalized_sha256']:
        raise ValueError('evaluation source changed')
    a, b = read_video(source), read_video(movie)
    pts_audit(movie, case['source']['frames'], case['source']['fps'])
    quality, rows = OfficialMetrics().evaluate(a, b)
    fixed = set(case['original_keys']); nonkeys = [r for r in rows if r['frame'] not in fixed]
    boundaries = boundary_diagnostics(a, b, case['original_keys'])
    value = dict(quality=quality, nonkeys=summarize(nonkeys, (len(nonkeys),*a.shape[1:])),
        temporal=temporal(a,b), boundaries=boundaries,
        boundary_error=statistics.mean(x['mean_change_error'] for x in boundaries),
        channel_uses=channel_uses, keys=case['original_keys'], movie=str(movie), movie_sha256=sha256(movie),
        source_movie_sha256=sha256(source), fixed_nonkey_indices=[r['frame'] for r in nonkeys],
        independent_rgb_keys=0, source_used_at_receiver=False)
    write_json(output/'quality.json', value)
    with (output/'quality.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def worker(j, stage):
    if stage == 'action':
        from .bsu_key_reduction import action
        return action(j)
    output = Path(j['output'])
    if stage == 'sender':
        case = j['case']; old = Path(case['original_run'])
        from .wire import unpack
        h, _ = unpack((old/'transmitter/metadata.bin').read_bytes())
        context = dict(video=h['video'], indices=case['original_keys'], segments=h['segments'], decoder=h['decoder'])
        for i, digest in case['source']['source_frame_hashes'].items():
            if sha256(Path(case['source']['source_frames'])/f'{i}.png') != digest:
                raise ValueError('source frame changed')
        transport.send(case['source']['source_frames'], context, output/'sender',
            uses_per_frame=j['uses_per_frame'], width=j['width'], height=j['height'],
            allow_budget_miss=j.get('allow_budget_miss',False))
    elif stage == 'channel':
        from .metadata_channel import transmit
        source = (output/'sender/packet.bin').read_bytes(); reports = []
        destination = output/'received'; destination.mkdir(parents=True)
        for seed in (42,1042,2042):
            packet, report = transmit(source,10,seed,bits_per_symbol=4)
            if source != packet or report['bit_errors']:
                raise ValueError('dense control channel failed integrity; no source fallback')
            transport.decode_packet(packet)
            reports.append(report)
            if seed == 42:
                (destination/'packet.bin').write_bytes(packet)
        actual = reports[0]['complex_channel_uses']
        if actual != read(output/'sender/sender.json')['channel_uses']:
            raise ValueError('actual channel accounting mismatch')
        write_json(destination/'channel.json',dict(channel_uses=actual, reports=reports,
            all_bytes_exact=True, retransmission_uses=0, seeds=[42,1042,2042]))
    elif stage == 'receiver':
        # The receiver job contains ONLY received packet/output/model paths.
        from .dense_video_receiver import restore
        context, stream, header = transport.decode_packet(Path(j['packet']).read_bytes())
        guide = transport.decode_video(stream, output, context, header)
        frames = restore(guide, (context['video']['width'], context['video']['height']),j['mode'],
                         model_root=REPO/'.local/vendor/SwinIR', checkpoint=MODEL)
        save_movie(frames,output,context['video']['fps'])
        write_json(output/'receiver.json',dict(packet_sha256=sha256(j['packet']),mode=j['mode'],
            policy=transport.POLICY, receiver_inputs=['received packet','fixed pretrained weights'],
            source_frames_available_in_job=False, frame_count=len(frames), independent_rgb_keys=0,
            virtual_keyframes=len(context['indices']),guide_width=header['guide_width'],
            guide_height=header['guide_height'],pretrained_sha256=sha256(MODEL) if j['mode']=='swinir' else None))
    elif stage == 'evaluate':
        evaluate(j['case'], output/'reconstruction.mp4', output, j['channel_uses'])
    else:
        raise ValueError(stage)


def execute(p, stage, job, log):
    runtime = p['runtimes']['channel' if stage=='channel' else 'action' if stage=='action' else 'python']
    env = dict(os.environ,PYTHONPATH=str(REPO/'src'),PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    if stage in ('sender','channel'):
        env['CUDA_VISIBLE_DEVICES']='-1'
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('x') as stream:
        code = subprocess.run([runtime,'-m',MODULE,'--worker',stage,'--job',str(job)],
                              cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
    if code:
        print(log.read_text(errors='replace')[-7000:],flush=True)
        raise RuntimeError(f'{stage} failed ({code}): {log}')


def protocol(root, uses_per_frame, reuse=None):
    old = read(REPO/'outputs/bsu_key_reduction_20261015_v2/protocol.json')
    if sha256(MODEL) != MODEL_SHA:
        raise ValueError('pretrained SwinIR weight mismatch')
    files = ['dense_video_transport','dense_video_receiver','bsu_dense_validation','official_quality',
             'stutter_metrics','metadata_channel','packets','wire','bsu_receiver','bsu_key_reduction']
    p = dict(policy=transport.POLICY,cases=old['cases'],uses_per_frame=uses_per_frame,
        rates='same target cap; native HEVC may exceed it at CRF 51 and must report actual cost',
        source_result_sha256=sha256(REPO/'outputs/bsu_kinetics_context/RESULT.json'),
        runtimes={**old['runtimes'],'python':str(REPO/'.local/bsu_codec_eval_env/bin/python')},
        action_runtime=old['action_runtime'],
        code={f'src/semantic_transmission/{n}.py':sha256(REPO/f'src/semantic_transmission/{n}.py') for n in files},
        model_sha256=MODEL_SHA,model_code_sha256=sha256(REPO/'.local/vendor/SwinIR/models/network_swinir.py'),
        ffmpeg_version=subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0],
        channel=dict(snr_db=10,seeds=[42,1042,2042],ldpc_k=6144,ldpc_n=9216,bits_per_symbol=4),
        arms=['dense_bicubic','dense_swinir','native_hevc'],
        source_selection='same previously examined 14-video development cohort; not held out',
        decision='report all arms; no classifier/quality-selected per-clip fallback',
        generator='separate controlled-generation ablation; not used in these three arms',
        unchanged_defaults=True,physical_link_overhead_included=False,
        success_target='lower paid rate; compare full-frame, fixed-nonkey, temporal and action metrics without relaxing prior strict guard',
        reuse_dense=str(reuse) if reuse else None,
        reuse_protocol_sha256=sha256(reuse/'protocol.json') if reuse else None)
    p['signature']=fingerprint(p);root.mkdir(parents=True,exist_ok=True)
    freeze(root/'protocol.json',p)
    return p


def summarize(root,p):
    from .key_reduction_selection import guard
    from .official_quality import KEYS
    rows=[]; summaries={}
    for case in p['cases']:
        baseline=read(REPO/'outputs/bsu_key_reduction_20261015_v2'/case['id']/'baseline_2025/quality.json')
        for arm in p['arms']:
            current=read(root/case['id']/arm/'quality.json')
            rows.append(dict(id=case['id'],arm=arm,quality=current,baseline=baseline,
                             strict_guard=guard(baseline,current)))
    for arm in p['arms']:
        group=[r for r in rows if r['arm']==arm]
        uses=sum(r['quality']['channel_uses'] for r in group)
        original=sum(r['baseline']['channel_uses'] for r in group)
        summaries[arm]=dict(videos=len(group),channel_uses=uses,bsu_saving_percent=100*(1-uses/original),
            ratio_to_lgvsc=uses/1407617,strict_all_metric_passes=sum(r['strict_guard']['passed'] for r in group),
            mean_quality={k:statistics.mean(r['quality']['quality'][k] for r in group) for k in KEYS},
            action=read(root/'action/RESULT.json')['summary'][arm])
        summaries[arm]['budget_miss_videos']=sum(
            r['quality']['channel_uses']>(r['quality']['quality']['frames']*p['uses_per_frame']//2304)*2304 for r in group)
    write_json(root/'RESULT.json',dict(status='COMPLETE_DENSE_RATE_QUALITY_ACTION_COMPARISON',
        signature=p['signature'],summary=summaries,rows=rows,original_action=read(root/'action/RESULT.json')['summary']['baseline'],
        generalization_proven=False,learned_module='official pretrained SwinIR-S x2, no fine tuning',
        generation_used=False,source_at_receiver=False))


def reuse_dense(previous,case,p):
    """Import only completed dense stages after hashes and exact old contract checks.

    The v2 change only permits the classical native-resolution comparator to
    report an infeasible rate. Existing dense packet/pixels/metrics are unchanged.
    """
    from .webvid_ablation import snapshot
    import shutil
    old=read(previous/'protocol.json')
    if old['uses_per_frame']!=p['uses_per_frame'] or old['model_sha256']!=p['model_sha256']:
        raise ValueError('reuse runtime/model/rate differs')
    if next(c for c in old['cases'] if c['id']==case['id'])!=case:
        raise ValueError('reuse source differs')
    src=previous/case['id']
    paths=[('dense',['sender','channel']),('dense_bicubic',['receiver','evaluate']),('dense_swinir',['receiver','evaluate'])]
    if not all((src/arm/'stages'/f'{s}.json').exists() for arm,stages in paths for s in stages):
        return None
    receipts={}
    for arm,stages in paths:
        for stage in stages:
            receipt=src/arm/'stages'/f'{stage}.json';v=read(receipt)
            if v['status']!='PASSED' or v['identity']!=old['signature']:
                return None
            if snapshot(src/arm,v['required'])!=v['artifacts']:
                raise ValueError('completed reuse artifact changed')
            receipts[str(receipt)]=sha256(receipt)
    # New output uses read-only symlinks to fully completed old arm directories;
    # no stage runner mutates the old experiment or rewrites old receipts.
    return dict(source=src,receipts=receipts)


def run(root, uses_per_frame, reuse=None):
    from .etri_60s_check import lock
    with lock(REPO/'.local/etri_60s_check.lock'):
        p=protocol(root,uses_per_frame,reuse); action_rows=[]
        labels={v['id']:v['label'] for v in read(REPO/'outputs/bsu_kinetics_context/RESULT.json')['videos']}
        for ci,case in enumerate(p['cases']):
            progress('dense_guide',ci,len(p['cases']),case['id'])
            reused=reuse_dense(reuse,case,p) if reuse else None
            for family,width,height in [('dense',288,160),('native',576,320)]:
                if family=='dense' and reused:
                    (root/case['id']).mkdir(parents=True,exist_ok=True)
                    for arm in ['dense','dense_bicubic','dense_swinir']:
                        dst=root/case['id']/arm;src=reused['source']/arm
                        if not dst.exists():dst.symlink_to(src,target_is_directory=True)
                    write_json(root/case['id']/'reuse.json',dict(source=str(reused['source']),receipts=reused['receipts'],
                        reason='completed dense arms unchanged by native-comparator rate-cap failure fix'))
                    for arm in ['dense_bicubic','dense_swinir']:
                        movie=root/case['id']/arm/'reconstruction.mp4'
                        action_rows.append(dict(id=case['id'],variant=arm,label=labels[case['id']],path=str(movie),sha256=sha256(movie)))
                    continue
                output=root/case['id']/family;output.mkdir(parents=True,exist_ok=True)
                j=dict(output=str(output),case=case,uses_per_frame=uses_per_frame,width=width,height=height,
                       allow_budget_miss=family=='native')
                freeze(output/'job.json',j); stages=Stages(output,p['signature'])
                for stage,owned in [('sender',['sender']),('channel',['received'])]:
                    stages.step(stage,owned,owned,lambda log,s=stage:execute(p,s,output/'job.json',log))
                for mode in (['bicubic','swinir'] if family=='dense' else ['bicubic']):
                    arm=f'dense_{mode}' if family=='dense' else 'native_hevc'
                    dest=root/case['id']/arm;dest.mkdir(parents=True,exist_ok=True)
                    rx=dict(output=str(dest),packet=str(output/'received/packet.bin'),mode=mode)
                    freeze(dest/'receiver_job.json',rx)
                    ev=dict(output=str(dest),case=case,channel_uses=read(output/'received/channel.json')['channel_uses'])
                    freeze(dest/'evaluation_job.json',ev)
                    steps=Stages(dest,p['signature'])
                    required=['reconstruction.mp4','frames','receiver.json','guide.mkv']
                    steps.step('receiver',required,required,lambda log:execute(p,'receiver',dest/'receiver_job.json',log))
                    steps.step('evaluate',['quality.json','quality.csv'],['quality.json','quality.csv'],
                               lambda log:execute(p,'evaluate',dest/'evaluation_job.json',log))
                    movie=dest/'reconstruction.mp4'
                    action_rows.append(dict(id=case['id'],variant=arm,label=labels[case['id']],path=str(movie),sha256=sha256(movie)))
            movie=Path(case['original_movie'])
            baseline=read(REPO/'outputs/bsu_key_reduction_20261015_v2'/case['id']/'baseline_2025/quality.json')
            if sha256(movie)!=baseline['movie_sha256'] or baseline['source_movie_sha256']!=case['source']['normalized_sha256']:
                raise ValueError('baseline reuse provenance changed')
            action_rows.append(dict(id=case['id'],variant='baseline',label=labels[case['id']],path=str(movie),sha256=sha256(movie)))
        j=dict(output=str(root/'action'),runtime=p['action_runtime'],rows=action_rows)
        freeze(root/'action_job.json',j)
        progress('action',len(p['cases']),len(p['cases']))
        Stages(root,p['signature']).step('action',['action'],['action/RESULT.json'],
                                       lambda log:execute(p,'action',root/'action_job.json',log))
        summarize(root,p)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',nargs='?',choices=['run','status'],default='run')
    parser.add_argument('--root',type=Path,default=REPO/'outputs/bsu_dense_512_20261015_v2')
    parser.add_argument('--reuse-dense',type=Path)
    parser.add_argument('--uses-per-frame',type=int,default=512)
    parser.add_argument('--worker');parser.add_argument('--job',type=Path)
    args=parser.parse_args()
    if args.worker:return worker(read(args.job),args.worker)
    if args.command=='status':
        path=args.root/'RESULT.json'
        print(json.dumps(read(path)['summary'] if path.exists() else dict(status='NOT_COMPLETE',root=str(args.root)),indent=2));return
    run(args.root.resolve(),args.uses_per_frame,args.reuse_dense.resolve() if args.reuse_dense else None)


if __name__=='__main__':main()
