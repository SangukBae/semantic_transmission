"""Local Kinetics14 comparison: UF QP30 adaptive versus frozen v3 RD search."""
import argparse
import ast
import copy
import csv
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

from .sparc_selector_data import read, save, sha, digest
from .sparc_selector_workers import REPO
from . import sparc_selector_distributed as distributed
from . import sparc_selector_train as supervision
from . import sparc_selector_train_v3 as training
from . import sparc_selector_v3_data as data
from . import sparc_webvid_evaluate as shared

MODULE = 'semantic_transmission.sparc_kinetics_evaluate'
DEFAULT = REPO/'outputs/sparc_kinetics14_qp30_v3_20261022'
TRAINED = REPO/'.local/sparc_kinetics_v3_assets_20261022'
CHECKPOINT_SHA = '1e622f956013c0e6e13744340225245473c466eec7d9a254ca509e13f442c957'
MANIFEST_SHA = '8c10e3bc1ff80a083472d4b5ea9dbeaecd7a52a973f0b111a47fce0cf83e5f0e'


def function_tree(source, name):
    return ast.dump(next(n for n in ast.parse(source).body
                        if isinstance(n, ast.FunctionDef) and n.name == name), include_attributes=False)


def source_rows(dataset, manifest, training_rows):
    forbidden = {r.get(k) for r in training_rows for k in ('sha256', 'raw_sha256')}
    rows = []
    for i, item in enumerate(csv.DictReader(manifest.open()), 1):
        raw = dataset/'raw'/item['filename']
        movie = dataset/'processed'/item['filename']
        receipt = read(dataset/'preprocess_records'/(item['filename']+'.json'))
        if item['sha256'] in forbidden or receipt['sha256'] in forbidden:
            raise ValueError('Kinetics source overlaps selector training/validation/test corpus')
        if (sha(raw) != item['sha256'] or receipt['source_sha256'] != item['sha256']
                or sha(movie) != receipt['sha256']):
            raise ValueError('Kinetics source/preprocessing hash mismatch: '+item['filename'])
        info = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=width,height,avg_frame_rate,nb_frames','-of','json',str(movie)],text=True))['streams'][0]
        frames = int(info['nb_frames'])
        if ((info['width'],info['height'],info['avg_frame_rate']) != (576,320,'24/1')
                or frames != int(receipt['nb_read_frames']) or frames < 2):
            raise ValueError('Kinetics prepared timeline mismatch')
        rows.append(dict(id=f'kinetics_{i:03d}_{item["sha256"][:8]}', parent='kinetics:'+item['sha256'],
            dataset='Kinetics-400-author-subset', split='benchmark', training_split='external_test',
            path=str(movie), filename=item['filename'], sha256=receipt['sha256'], raw_sha256=item['sha256'],
            kinetics_label_id=int(item['kinetics_label_id']), frames=frames, width=576,height=320,fps=24.,
            start_seconds=0., duration_seconds=frames/24.))
    if len(rows) != 14:
        raise ValueError('expected all 14 author Kinetics videos')
    return rows


def protocol():
    parent = read(TRAINED/'protocol.json')
    result = read(TRAINED/'RESULT.json')
    checkpoint = TRAINED/'training/best.pt'
    if (result['status'] != 'TRAINED_RD_EVALUATED' or result['checkpoint_sha256'] != CHECKPOINT_SHA
            or sha(checkpoint) != CHECKPOINT_SHA):
        raise ValueError('frozen v3 checkpoint changed')
    current_code = {}
    for name, expected in parent['supervision_protocol']['code'].items():
        file = REPO/'src/semantic_transmission'/f'{name}.py'
        if sha(file) != expected:
            raise ValueError('frozen measurement code changed: '+name)
        current_code[str(file.relative_to(REPO))] = expected
    training_file = 'src/semantic_transmission/sparc_selector_train_v3.py'
    reference = TRAINED/'frozen_training_module.py'
    if sha(reference) != parent['code'][training_file]:
        raise ValueError('original evaluation source checksum changed')
    for name, expected in parent['code'].items():
        actual = sha(REPO/name)
        if actual != expected:
            if name != training_file:
                raise ValueError('frozen selector code changed: '+name)
            # Local training resume fixes do not change either evaluation function.
            for func in ('evaluate','allocate_cohort'):
                if function_tree((REPO/name).read_text(),func) != function_tree(reference.read_text(),func):
                    raise ValueError('evaluation differs from frozen v3: '+func)
        current_code[name] = actual
    for remote, expected in parent['supervision_protocol']['assets'].items():
        local = REPO/'.local'/remote.split('/.local/',1)[1]
        if sha(local) != expected:
            raise ValueError('frozen UF/RIFE asset changed: '+str(local))
    from .sparc_uf_transport import UPSTREAM
    vendor = REPO/'.local/dcvc_uf_vendor'
    if (subprocess.check_output(['git','-C',str(vendor),'rev-parse','HEAD'],text=True).strip() != UPSTREAM
            or subprocess.check_output(['git','-C',str(vendor),'diff','HEAD'])):
        raise ValueError('UF vendor changed')
    if sha(data.backbone_weights()) != parent['features']['weights_sha256']:
        raise ValueError('feature extractor changed')
    if read(REPO/'configs/sparc.json') != parent['supervision_protocol']['semantic_policy']:
        raise ValueError('semantic selection policy changed')
    manifest = REPO/'docs/eval_manifest_kinetics.csv'
    if sha(manifest) != MANIFEST_SHA:
        raise ValueError('author Kinetics manifest changed')
    dataset = Path(read(REPO.parent/'semantic_transmission/.local/datasets.json')['kinetics400_14'])
    rows = source_rows(dataset,manifest,read(TRAINED/'sources.json'))
    runtimes = read(REPO/'outputs/sparc_selector_distributed_20261021/local/.distributed/runtimes.json')
    runtimes['semantic'] = str(REPO/'.local/qwen35_env/bin/python')
    if not all(Path(v).is_file() for v in runtimes.values()):
        raise ValueError('local Python worker runtime missing')
    for name in ('sparc_kinetics_evaluate','sparc_webvid_evaluate','official_quality'):
        path = f'src/semantic_transmission/{name}.py'
        current_code[path] = sha(REPO/path)
    p = dict(schema='sparc_kinetics14_v3_comparison_v1', rows=rows, total_frames=sum(r['frames'] for r in rows),
        author_manifest_sha256=sha(manifest), trained_root=str(TRAINED), trained_protocol=parent,
        checkpoint_sha256=CHECKPOINT_SHA, target_cbr=parent['target_cbr'], runtime_paths=runtimes,
        code=current_code, no_retraining=True, all_inputs_disjoint_from_selector_corpus=True,
        baseline='SPARC-UF QP30 with previous adaptive selection (max_gap=4, MAE=6/255)',
        evaluation_mode='frozen v3 candidate ranker plus reference-assisted measured RD search',
        source_preparation='decode existing author release-preprocessed MP4 without renormalization',
        metrics=['psnr_db','ssim','lpips_vgg','clip','dists'], channel='AWGN 10dB, seed42, LDPC 6144/9216, 16QAM')
    p['signature'] = digest(p)
    return p


def initialize(root):
    p = protocol()
    root.mkdir(parents=True,exist_ok=True)
    if (root/'protocol.json').exists() and read(root/'protocol.json') != p:
        raise ValueError('frozen comparison changed; use a new output root')
    save(root/'protocol.json',p)
    save(root/'sources.json',p['rows'])
    dest = root/'selector.pt'
    if not dest.exists():
        shutil.copyfile(TRAINED/'training/best.pt',dest.with_suffix('.tmp'))
        os.replace(dest.with_suffix('.tmp'),dest)
    if sha(dest) != CHECKPOINT_SHA:
        raise ValueError('comparison checkpoint changed')
    return p


def final_metrics(root,p,progress):
    """Evaluate both delivered MP4s with the same official five-metric evaluator."""
    import numpy as np
    from .official_quality import OfficialMetrics, PROFILE, KEYS
    summary = read(root/'all14_summary.json')
    table = list(csv.DictReader((root/'all14_per_video.csv').open()))
    by_id = {r['id']:r for r in p['rows']}
    for i,record in enumerate(table):
        row = by_id[record['id']]
        originals = None
        for role in ('baseline','selected'):
            supervision.interrupted()
            movie = Path(record[role+'_video'])
            if sha(movie) != record[role+'_video_sha256']:
                raise ValueError('published reconstruction changed')
            identity = dict(protocol=p['signature'],source_sha256=row['sha256'],movie_sha256=sha(movie),profile=PROFILE)
            dest = root/'quality'/row['id']/(role+'.json')
            if dest.exists():
                measured = read(dest)
                if measured['identity'] != identity:
                    raise ValueError('final metric evidence changed')
            else:
                if originals is None:
                    originals = shared.decode_input(row)
                raw = subprocess.check_output(['ffmpeg','-v','error','-threads','2','-i',str(movie),
                    '-vsync','0','-threads','2','-f','rawvideo','-pix_fmt','rgb24','-'])
                restored = np.frombuffer(raw,np.uint8).reshape(-1,row['height'],row['width'],3)
                def update(done,total):
                    supervision.interrupted()
                    progress('final_quality',clip=row['id'],metric_video_done=i*2+(role=='selected'),
                             metric_video_total=len(table)*2,metric_role=role,metric_done=done,metric_total=total)
                values,per_frame = OfficialMetrics('cuda').evaluate(originals,restored,update)
                measured = dict(identity=identity,summary=values,per_frame=per_frame)
                save(dest,measured)
            for key in KEYS:
                record[role+'_'+key] = measured['summary'][key]
        for key in KEYS:
            record[key+'_delta'] = record['selected_'+key]-record['baseline_'+key]
    result = dict(status='COMPLETE',videos=len(table),frames=summary['frames'],protocol=p['signature'],
        checkpoint_sha256=CHECKPOINT_SHA,budget_feasible=summary['budget_feasible'],target_cbr=p['target_cbr'],
        baseline={},selected={},evaluation_mode=p['evaluation_mode'],no_retraining=True,
        source_disjoint=True,quality_aggregation='equal-video mean of all-frame means',
        cbr_aggregation='sum(complex channel uses) / sum(3*F*H*W)')
    for role in ('baseline','selected'):
        result[role] = {k:summary[role][k] for k in ('channel_uses','aggregate_cbr')}
        result[role]['quality'] = {k:sum(float(r[role+'_'+k]) for r in table)/len(table) for k in KEYS}
    result['channel_reduction_percent'] = 100*(1-result['selected']['channel_uses']/result['baseline']['channel_uses'])
    result['quality_delta'] = {k:result['selected']['quality'][k]-result['baseline']['quality'][k] for k in KEYS}
    dest = root/'comparison_per_video.csv'
    with dest.with_suffix('.tmp').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    os.replace(dest.with_suffix('.tmp'),dest)
    save(root/'COMPARISON.json',result)
    return result


def run(root):
    import torch
    root.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        lock=stack.enter_context((root/'run.lock').open('a'));fcntl.flock(lock,fcntl.LOCK_EX)
        signal.signal(signal.SIGTERM,supervision.stop);signal.signal(signal.SIGINT,supervision.stop)
        old=read(root/'status.json') if (root/'status.json').exists() else {}
        started=old.get('started_unix',time.time());completed=[];current={};done=0
        def progress(stage,**extra):
            value=dict(status='RUNNING',stage=stage,done=done,total=14,
                pid=os.getpid(),start_ticks=distributed.proc_identity(os.getpid()),started_unix=started,
                updated_unix=time.time(),clip=current.get('id','-'),clip_frames=current.get('frames'),
                completed_frames=sum(c['pixels']//(576*320) for c in completed))
            value.update(extra);save(root/'status.json',value)
        try:
            progress('verify_protocol');p=initialize(root)
            gpu=stack.enter_context((REPO/'.local/etri_60s_check.lock').open('a'))
            while True:
                supervision.interrupted()
                try:fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:progress('waiting_for_local_gpu');time.sleep(2)
            supervision.runtimes=lambda:p['runtime_paths']
            torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            save(root/'environment.json',dict(host=socket.gethostname(),python=sys.version,
                torch=torch.__version__,gpu=torch.cuda.get_device_name(0),runtimes=p['runtime_paths']))
            eval_protocol=copy.deepcopy(p['trained_protocol'])
            eval_protocol['search']['benchmark_clips']=14;eval_protocol['signature']=p['signature']
            for i,row in enumerate(p['rows']):
                supervision.interrupted();current=row;done=i;t=time.monotonic()
                progress('source_frames');shared.prepare(root,row,p,progress);shared.features(root,row,p,progress)
                saved=root/'evaluation/benchmark'/row['id']/(row['id']+'.json')
                if saved.exists():shared.verify_clip(root,row,p,read(saved)['clip'])
                def rd_progress(stage,*args,**extra):
                    progress('candidate_reconstruction',candidate=extra.get('candidate_measurements'),candidate_cap=17)
                training.evaluate(root,root,[row],eval_protocol,root/'selector.pt','benchmark',row['id'],rd_progress)
                clip=read(saved)['clip'];shared.verify_clip(root,row,p,clip)
                completed.append(clip);done=i+1
                folder=root/'clips'/row['id'];receipt=folder/'evaluation_receipt.json'
                if not receipt.exists():
                    save(receipt,dict(protocol=p['signature'],input_receipt_sha256=sha(folder/'input_receipt.json'),
                        evaluation_sha256=sha(saved),elapsed_seconds=time.monotonic()-t,frames=row['frames']))
                elif read(receipt)['evaluation_sha256']!=sha(saved):
                    raise ValueError('completed video evaluation changed')
                progress('video_complete')
            progress('aggregate_and_export');summary=shared.export(root,p,completed,'all14')
            summary.update(independent_of_training_and_validation=True,dataset='Kinetics-400-author-subset')
            save(root/'all14_summary.json',summary)
            result=final_metrics(root,p,progress);save(root/'RESULT.json',result)
            progress('complete',metric_video_done=28,metric_video_total=28)
            value=read(root/'status.json');value.update(status='COMPLETE',exit_code=0)
            save(root/'status.json',value);return dict(status='COMPLETE',videos=14)
        except BaseException as error:
            value=read(root/'status.json') if (root/'status.json').exists() else {}
            interrupted=supervision.STOP or isinstance(error,InterruptedError)
            value.update(status='INTERRUPTED' if interrupted else 'FAILED',error=repr(error),
                         exit_code=130 if interrupted else 1,updated_unix=time.time())
            save(root/'status.json',value);raise


def launch(root):
    root.mkdir(parents=True,exist_ok=True)
    with (root/'run.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return dict(status='ALREADY_RUNNING')
        if (root/'launcher.json').exists() and distributed.active(read(root/'launcher.json')):
            return dict(status='ALREADY_RUNNING')
        initialize(root)
        if (root/'RESULT.json').exists():return dict(status='ALREADY_COMPLETE')
        env=dict(os.environ,PYTHONPATH=str(REPO/'src'),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
            PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
            CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1')
        env.pop('LD_LIBRARY_PATH',None)
        with (root/'run.log').open('ab') as log:
            proc=subprocess.Popen([sys.executable,'-m',MODULE,'run','--root',str(root)],cwd=REPO,env=env,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(root/'launcher.json',dict(pid=proc.pid,start_ticks=distributed.proc_identity(proc.pid)))
    return dict(status='LAUNCHED',pid=proc.pid,output=str(root))


def stop(root):
    record=read(root/'launcher.json')
    if not distributed.active(record):return dict(status='NOT_RUNNING')
    args=Path(f'/proc/{record["pid"]}/cmdline').read_bytes().split(b'\0')
    if MODULE.encode() not in args or b'run' not in args or str(root).encode() not in args:
        raise ValueError('process identity mismatch; no signal sent')
    os.kill(record['pid'],signal.SIGTERM)
    return dict(status='STOP_REQUESTED',boundary='current candidate finishes; completed results retained')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['check','run','launch','stop'])
    parser.add_argument('--root',type=Path,default=DEFAULT)
    args=parser.parse_args();root=args.root.resolve()
    if args.command=='check':
        p=initialize(root);result=dict(status='PREFLIGHT_PASSED',videos=len(p['rows']),frames=p['total_frames'],checkpoint=CHECKPOINT_SHA)
    elif args.command=='launch':result=launch(root)
    elif args.command=='stop':result=stop(root)
    else:result=run(root)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
