"""Matched received-condition LTX/Open-Sora comparison plus the LGVSC baseline.

All 14 existing Kinetics clips are included in manifest order. Baseline movies,
source pixels, receipts and cost ledgers must verify before reuse. This is a
receiver replacement experiment, not an equal-rate LGVSC comparison.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from .artifacts import sha256, write_json
from . import ltx_receiver as ltx
from .webvid5 import fingerprint
from .webvid_ablation import Stages, environment, snapshot
from .qwen35_pairs_video import freeze
from .wire import unpack

REPO = Path(__file__).resolve().parents[2]
MODULE = 'semantic_transmission.bsu_ltx_comparison'
KEYS = ('psnr_db', 'ssim', 'lpips_vgg', 'clip', 'dists')


def progress(done, total, current, stage, state='실행 중'):
    print('COMPARE_PROGRESS '+json.dumps(dict(
        overall=f'{done}/{total} ({100*done/total:.1f}%) 영상 완료',
        current=current,stage=stage,state=state),ensure_ascii=False),flush=True)


def read(path):
    return json.loads(Path(path).read_text())


def verify_files(files):
    for path, digest in files.items():
        if sha256(path) != digest:
            raise ValueError(f'comparison artifact changed: {path}')


def baseline_audit(bsu_root, lgvsc_root):
    from .transmission_accounting import channel_breakdown
    bsu_root, lgvsc_root = Path(bsu_root).resolve(), Path(lgvsc_root).resolve()
    bp = read(bsu_root/'benchmark/protocol.json'); lp = read(lgvsc_root/'protocol.json')
    if read(bsu_root/'RESULT.json')['status'] != 'COMPLETE_RECONSTRUCTION_QUALITY_ACTION_EVALUATION':
        raise ValueError('BSU reference cohort is incomplete')
    if read(lgvsc_root/'RESULT.json')['status'] != 'COMPLETE_RATE_QUALITY_ACTION_COMPARISON':
        raise ValueError('LGVSC reference cohort is incomplete')
    if [v['id'] for v in bp['videos']] != [v['id'] for v in lp['videos']] or len(bp['videos']) != 14:
        raise ValueError('expected the same complete 14-video cohort')
    files, rows = {}, []
    def keep(path, expected=None):
        path = Path(path).resolve(); digest = sha256(path)
        if expected is not None and digest != expected:
            raise ValueError(f'baseline artifact changed: {path}')
        files[str(path)] = digest
    for p in (bsu_root/'RESULT.json', bsu_root/'benchmark/protocol.json',
              lgvsc_root/'RESULT.json', lgvsc_root/'protocol.json'):
        keep(p)
    def receipts(root, names):
        for name in names:
            path = root/'stages'/f'{name}.json'; r = read(path); keep(path)
            if r['status'] != 'PASSED':
                raise ValueError(f'incomplete baseline stage: {path}')
            for relative, digest in r['artifacts'].items():
                keep(root/relative, digest)
    for v in bp['videos']:
        condition = bsu_root/'benchmark/snr_10'/v['id']; b = read(condition/'RESULT.json')
        dest = lgvsc_root/'videos'/v['id']; g = read(dest/'RESULT.json')
        source = read(v['prepared']); keep(v['prepared'], v['prepared_sha256'])
        keep(source['source_path'], source['source_sha256'])
        keep(source['normalized_video'], source['normalized_sha256'])
        for index, digest in source['source_frame_hashes'].items():
            keep(Path(source['source_frames'])/f'{index}.png', digest)
        if (v['frames'],v['fps'],v['width'],v['height']) != (source['frames'],24,576,320):
            raise ValueError('source geometry mismatch')
        if b['mode'] != 'combined_context' or b['frames'] != v['frames'] or g['frames'] != v['frames']:
            raise ValueError('baseline receiver or duration mismatch')
        for relative, digest in b['artifacts'].items(): keep(condition/relative, digest)
        for path, digest in read(condition/'job.json')['base_artifacts'].items(): keep(path,digest)
        base = Path(b['base']); run = base/'reconstruction/event_pairs/run'; grun = dest/'run'
        receipts(run.parent, ('channel','receive','reconstruct','evaluate','output-audit'))
        receipts(dest, ('channel','receive','reconstruct','evaluate','output-audit','rate'))
        bc, gc = read(run/'run_config.json'), read(grun/'run_config.json')
        for k, expected in dict(seed=2025,channel_seed=42,steps=30,snr_db=10,
                evaluation_profile='lgvsc_official_metrics_v1',
                visual_channel='source_frame_seeded_numpy_awgn_v1',benchmark_digital_bits_per_symbol=4).items():
            if bc[k] != expected or gc[k] != expected:
                raise ValueError(f'baseline conditions differ: {k}')
        if b['refined_quality']['source_sha256'] != source['normalized_sha256'] or g['quality']['source_sha256'] != source['normalized_sha256']:
            raise ValueError('baseline quality used different reference pixels')
        for q in (b['refined_quality'],g['quality']):
            if q['evaluation_profile'] != 'lgvsc_official_metrics_v1' or any(q[t]['frames'] != v['frames'] for t in ('lossless_frames','delivered_mp4')):
                raise ValueError('baseline metric profile or frame coverage differs')
        for root in (run,grun):
            audit = read(root/'output_audit.json')
            if audit['status'] != 'PASSED' or audit['source_indices'] != list(range(v['frames'])):
                raise ValueError('baseline drops source frames')
        keep(b['video'], b['refined_quality']['video_sha256']); keep(g['video'],g['video_sha256'])
        packet = (run/'transmitter/metadata.bin').read_bytes(); header,_ = unpack(packet)
        if packet != (run/'received/metadata.bin').read_bytes():
            raise ValueError('BSU metadata channel is not intact')
        ledger = channel_breakdown(packet,(run/'transmitter/visual.c64').stat().st_size//8,
                                   header['video'],read(run/'channel_accounting.json'))
        if ledger != b['base_transmission_ledger']:
            raise ValueError('baseline transmission ledger differs from actual packets')
        extra = read(condition/'received/channel.json')['complex_channel_uses']
        if b['total_channel_uses'] != ledger['complex_channel_uses']['total'] + extra:
            raise ValueError('BSU total rate excludes transmitted data')
        gp = (grun/'transmitter/metadata.bin').read_bytes(); gh,_ = unpack(gp)
        gl = channel_breakdown(gp,(grun/'transmitter/visual.c64').stat().st_size//8,
                              gh['video'],read(grun/'channel_accounting.json'))
        if gl != g['rate']['ledger']:
            raise ValueError('LGVSC ledger differs from actual packets')
        keep(condition/'RESULT.json'); keep(dest/'RESULT.json')
        rows.append(dict(id=v['id'],frames=v['frames'],prepared=v['prepared'],base=str(base),
            received_packet=str(run/'received/metadata.bin'),
            received_keys=str(run/'receiver/frames/sample/key_frames_received'),
            trajectory=str(condition/'received'),
            bsu_video=b['video'],lgvsc_video=g['video'],
            bsu_quality=b['refined_quality'],bsu_base_quality=b['base_quality'],lgvsc_quality=g['quality'],
            bsu_channel_uses=b['total_channel_uses'],lgvsc_channel_uses=gl['complex_channel_uses']['total']))
        print(f'BASELINE AUDIT {len(rows)}/14 {v["id"]}: verified',flush=True)
    # Reused metric outputs must come from the same metric implementation.
    for name in ('official_quality.py','research_quality.py','stutter_metrics.py'):
        relative = 'src/semantic_transmission/'+name
        if lp['code'][relative] != sha256(REPO/relative):
            raise ValueError(f'evaluator changed since baseline: {name}')
    return dict(status='BASELINES_VERIFIED',rows=rows,inputs=files,frames=sum(r['frames'] for r in rows),
        exact_paper_lgvsc_reproduction=False,equal_rate_lgvsc_comparison=False,
        ltx_same_received_conditions_and_rate_as_opensora=True)


def prepare(args):
    from .qwen35_pairs_comparison import forbid_overlap
    root = args.output.resolve()
    forbid_overlap(root,[args.bsu_root,args.lgvsc_root,REPO/'src',REPO/'.local',REPO/'configs'])
    if root.exists() and any(root.iterdir()) and not (root/'baseline_audit.json').exists():
        raise ValueError('unrecognized nonempty comparison output')
    if (root/'baseline_audit.json').exists():
        audit = read(root/'baseline_audit.json'); verify_files(audit['inputs'])
    else:
        audit = baseline_audit(args.bsu_root,args.lgvsc_root)
        root.mkdir(parents=True,exist_ok=True); freeze(root/'baseline_audit.json',audit)
    if not (root/'status.json').exists():
        write_json(root/'status.json',dict(status='BASELINES_VERIFIED_LTX_NOT_RUN',completed=0,total=len(audit['rows'])))
    return audit


def copy_conditions(row, dest):
    dest.mkdir(parents=True)
    shutil.copyfile(row['received_packet'],dest/'metadata.bin')
    shutil.copytree(row['received_keys'],dest/'keys')
    shutil.copytree(row['trajectory'],dest/'trajectory')
    freeze(dest/'request.json',ltx.request(dest/'metadata.bin',dest/'keys'))


def refine(root):
    from .bsu_receiver import refine_windows, encode_frames
    from .bsu_sequence_refinement import decode_packet
    packet=(root/'conditions/metadata.bin').read_bytes(); header,_=unpack(packet)
    tracks=root/'conditions/trajectory/trajectory.bin'
    trajectories=decode_packet(tracks.read_bytes(),packet) if tracks.exists() else {}
    paths=sorted((root/'raw/frames').glob('*.png')); keys=[k['index'] for k in header['keyframes']]
    if len(paths)!=header['video']['frames']: raise ValueError('LTX frame coverage mismatch')
    output=root/'refined'
    audits=refine_windows(paths,keys,root/'conditions/keys',trajectories,output/'frames','combined_context')
    timeline=encode_frames(output/'frames',output/'reconstruction.mp4',len(paths),header['video']['fps'])
    write_json(output/'refinement.json',dict(mode='combined_context',windows=audits,timeline=timeline,
        receiver_reads_source=False,received_key_endpoints_exact=True))


def code_identity():
    # Small source tree only; exclude local environments and unrelated outputs.
    paths=list((REPO/'src/semantic_transmission').glob('*.py'))+[ltx.CONFIG,
        REPO/'scripts/run_bsu_gen_ltx.sh',REPO/'scripts/compare_bsu_gen.sh']
    return {str(p):sha256(p) for p in paths}


def command(cmd,log,env):
    """Retain full logs while feeding worker progress to the fixed-row display."""
    log.parent.mkdir(parents=True,exist_ok=True)
    cmd=[str(x) for x in cmd]
    with log.open('w') as stream:
        proc=subprocess.Popen(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              text=True,bufsize=1)
        try:
            for line in proc.stdout:
                stream.write(line);stream.flush();print(line,end='',flush=True)
            code=proc.wait()
        except BaseException:
            proc.terminate()
            try: proc.wait(timeout=15)
            except subprocess.TimeoutExpired: proc.kill();proc.wait()
            raise
        if code:
            raise subprocess.CalledProcessError(code,cmd)


def run(args):
    from .cli import settings
    from .etri_60s_check import lock
    root=args.output.resolve(); audit=prepare(args)
    runtime=ltx.runtime(args.model_dir,args.vendor)
    environment_file=args.ltx_python.parent.parent/'bsu-environment.txt'
    cfg=dict(profile=ltx.profile(),baseline_audit_sha256=sha256(root/'baseline_audit.json'),
        runtime=runtime,code=code_identity(),ltx_python=str(args.ltx_python.resolve()),
        ltx_environment_sha256=sha256(environment_file),
        metrics_python=settings(REPO)['python'])
    freeze(root/'protocol.json',cfg); verify_files(cfg['code'])
    signature=fingerprint(cfg); start=time.monotonic()
    env=dict(environment(2025),PYTHONPATH=str(REPO/'src'))
    def launch(cmd,log):
        return command(cmd,log,env)
    results=[]
    with lock(root/'execution.lock'),lock(REPO/'.local/etri_60s_check.lock'):
        (root/'RESULT.json').unlink(missing_ok=True)
        try:
            for row in audit['rows']:
                progress(len(results),len(audit['rows']),row['id'],'수신 조건 확인 / LTX 복원 준비')
                dest=root/'videos'/row['id']; dest.mkdir(parents=True,exist_ok=True)
                stages=Stages(dest,signature)
                write_json(root/'status.json',dict(status='RUNNING',stage='conditions/generation/refinement/metrics',
                    video=row['id'],completed=len(results),total=len(audit['rows']),elapsed_seconds=time.monotonic()-start))
                stages.step('conditions',['conditions'],['conditions'],lambda _:copy_conditions(row,dest/'conditions'))
                # Keep verified partial intervals across interruption of the generation worker.
                stages.step('generate',[],['raw/frames','raw/reconstruction.mp4','raw/generation.json'],
                    lambda log:launch([args.ltx_python,'-m','semantic_transmission.ltx_receiver',
                        '--request',dest/'conditions/request.json','--output',dest/'raw',
                        '--model-dir',args.model_dir,'--vendor',args.vendor],log))
                stages.step('refine',['refined'],['refined/frames','refined/reconstruction.mp4','refined/refinement.json'],
                    lambda log:launch([cfg['metrics_python'],'-m',MODULE,'_refine','--output',dest],log))
                for boundary in ('raw','refined'):
                    products=[f'{boundary}/quality.json',f'{boundary}/quality_lossless_frames.csv',f'{boundary}/quality_delivered_mp4.csv']
                    stages.step('evaluate_'+boundary,products,products,
                        lambda log,b=boundary:launch([cfg['metrics_python'],'-m',MODULE,'_evaluate',
                            '--output',dest/b,'--base',row['base']],log))
                q=read(dest/'refined/quality.json')
                if q['source_sha256']!=row['bsu_quality']['source_sha256'] or q['delivered_mp4']['frames']!=row['frames']:
                    raise ValueError('LTX quality reference/coverage differs from baselines')
                result=dict(id=row['id'],status='COMPLETE_LTX_RECONSTRUCTION_QUALITY',frames=row['frames'],
                    quality=q,raw_quality=read(dest/'raw/quality.json'),total_channel_uses=row['bsu_channel_uses'],
                    video=str(dest/'refined/reconstruction.mp4'),signature=signature,
                    artifacts=snapshot(dest,['conditions','raw','refined']))
                write_json(dest/'RESULT.json',result);results.append(result)
                progress(len(results),len(audit['rows']),row['id'],'복원·보정·평가 완료')
                print(f'VIDEOS {len(results)}/{len(audit["rows"])} ({100*len(results)/len(audit["rows"]):.1f}%) '
                      f'elapsed={time.monotonic()-start:.1f}s ETA=unknown',flush=True)
            verify_files(audit['inputs'])
            report(root,audit,results,signature)
        except BaseException as e:
            write_json(root/'status.json',dict(status='INTERRUPTED' if isinstance(e,KeyboardInterrupt) else 'FAILED',
                error=str(e),completed=len(results),total=len(audit['rows']),resume='Repeat the same command.'))
            raise


def report(root,audit,results,signature):
    if [r['id'] for r in results]!=[r['id'] for r in audit['rows']]:
        raise ValueError('incomplete comparison cohort')
    rows=audit['rows']; arms={}
    if any(r['total_channel_uses']!=b['bsu_channel_uses'] or r['frames']!=b['frames']
           for r,b in zip(results,rows)):
        raise ValueError('LTX must retain the complete BSU received timeline and rate')
    for name,qualities,costs in (
        ('LGVSC',[r['lgvsc_quality'] for r in rows],[r['lgvsc_channel_uses'] for r in rows]),
        ('BSU-Gen/Open-Sora',[r['bsu_quality'] for r in rows],[r['bsu_channel_uses'] for r in rows]),
        ('BSU-Gen/LTX-2.5',[r['quality'] for r in results],[r['total_channel_uses'] for r in results])):
        arms[name]=dict(total_channel_uses=sum(costs),delivered_mp4_video_mean={
            k:sum(q['delivered_mp4'][k] for q in qualities)/len(rows) for k in KEYS})
    output=dict(status='COMPLETE_THREE_WAY_RECONSTRUCTION_COMPARISON',signature=signature,
        videos=len(rows),frames=audit['frames'],arms=arms,equal_rate_lgvsc_comparison=False,
        independent_semantic_review='PENDING',long_video_generalization_verified=False)
    write_json(root/'RESULT.json',output);write_json(root/'status.json',dict(status=output['status'],completed=len(rows),total=len(rows)))
    print(json.dumps(output,ensure_ascii=False,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','run','status','_refine','_evaluate'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bsu-root',type=Path);p.add_argument('--lgvsc-root',type=Path)
    p.add_argument('--base',type=Path)
    p.add_argument('--model-dir',type=Path,default=REPO/'.local/checkpoints/ltx25')
    p.add_argument('--vendor',type=Path,default=REPO/'.local/vendor/LTX-2')
    p.add_argument('--ltx-python',type=Path,default=REPO/'.local/envs/bsu-ltx25/bin/python')
    args=p.parse_args()
    if args.action=='status':
        print((args.output/'status.json').read_text());return
    if args.action=='_refine': return refine(args.output)
    if args.action=='_evaluate':
        from .bsu_sequence_refinement import evaluate
        return evaluate(dict(base=str(args.base)),args.output)
    if args.bsu_root is None or args.lgvsc_root is None:
        p.error('--bsu-root and --lgvsc-root are required')
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    if args.action=='prepare': prepare(args)
    else: run(args)


if __name__=='__main__':
    main()
