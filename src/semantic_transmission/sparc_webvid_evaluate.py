"""Resumable WebVid53 evaluation of the frozen v3 ranker with measured RD search.

The input is the already normalized LGVSC-release MP4, decoded without another
resize, frame-rate conversion or lossy encode. This is reference-assisted search,
not a single-forward-pass deployment benchmark. No optimizer is run here.
"""
import argparse
import copy
import csv
from contextlib import ExitStack
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from . import sparc_selector_distributed as distributed
from . import sparc_selector_train as supervision
from . import sparc_selector_train_v3 as training
from . import sparc_selector_v3_data as data
from . import sparc_selector_budget as budget
from .sparc_selector_data import read, save, sha, digest, source_features
from .sparc_selector_workers import REPO, Client

MODULE = 'semantic_transmission.sparc_webvid_evaluate'
DEFAULT = REPO/'outputs/sparc_webvid53_qp30_selector_v3_20261022'
TRAINED = REPO/'outputs/sparc_selector_budget_v3_20261022'
AUTHOR_MANIFEST_SHA = '6afe2ca53ac9162036a5485cd450f868a8e16182e0c034447da7f3b56ea2f579'


def protocol(trained):
    main = REPO.parent/'semantic_transmission'
    dataset = Path(read(main/'.local/datasets.json')['webvid55'])
    manifest = main/'docs/eval_manifest_webvid.csv'
    if sha(manifest) != AUTHOR_MANIFEST_SHA:
        raise ValueError('author WebVid manifest changed')
    parent = read(trained/'protocol.json')
    result = read(trained/'RESULT.json')
    checkpoint = trained/'training/best.pt'
    if result['status'] != 'TRAINED_RD_EVALUATED' or sha(checkpoint) != result['checkpoint_sha256']:
        raise ValueError('trained selector is incomplete or best checkpoint changed')
    split = {r['sha256']: r['split'] for r in read(trained/'sources.json')
             if 'webvid' in r['dataset'].lower()}
    rows, missing = [], []
    for index, item in enumerate(csv.DictReader(manifest.open()), 1):
        raw, path = dataset/'raw'/item['filename'], dataset/'processed'/item['filename']
        if not raw.exists():
            missing.append(item['filename']); continue
        receipt = read(dataset/'preprocess_records'/(item['filename']+'.json'))
        if (sha(raw) != item['sha256'] or receipt['source_sha256'] != item['sha256']
                or sha(path) != receipt['sha256']):
            raise ValueError('source/preprocessing hash mismatch: '+item['filename'])
        info = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=width,height,avg_frame_rate,nb_frames,duration',
            '-of','json',str(path)], text=True))['streams'][0]
        frames = int(info['nb_frames'])
        if ((info['width'],info['height'],info['avg_frame_rate']) != (576,320,'24/1')
                or frames != int(receipt['nb_read_frames']) or not 0 < float(info['duration']) <= 16):
            raise ValueError('unexpected prepared video geometry')
        rows.append(dict(id=f'webvid_{index:03d}_{item["sha256"][:8]}',
            parent='webvid:'+item['sha256'], dataset='WebVid-author-subset', split='benchmark',
            training_split=split[item['sha256']], path=str(path), sha256=receipt['sha256'],
            raw_sha256=item['sha256'], filename=item['filename'], frames=frames,
            width=576,height=320,fps=24.,start_seconds=0.,duration_seconds=frames/24.))
    if len(rows) != 53 or len(missing) != 2 or sum(r['frames'] for r in rows) != 17382:
        raise ValueError('expected the audited 53-video / 17382-frame cohort')
    for name, expected in parent['supervision_protocol']['code'].items():
        if sha(REPO/'src/semantic_transmission'/f'{name}.py') != expected:
            raise ValueError('frozen measurement code changed: '+name)
    for name, expected in parent['code'].items():
        if sha(REPO/name) != expected:
            raise ValueError('frozen selector code changed: '+name)
    for name, expected in parent['supervision_protocol']['assets'].items():
        if sha(name) != expected:
            raise ValueError('frozen model asset changed: '+name)
    from .sparc_uf_transport import UPSTREAM
    vendor = REPO/'.local/dcvc_uf_vendor'
    if (subprocess.check_output(['git','-C',str(vendor),'rev-parse','HEAD'],text=True).strip() != UPSTREAM
            or subprocess.check_output(['git','-C',str(vendor),'diff','HEAD'])):
        raise ValueError('UF vendor revision changed')
    if sha(data.backbone_weights()) != parent['features']['weights_sha256']:
        raise ValueError('feature extractor weights changed')
    if read(REPO/'configs/sparc.json') != parent['supervision_protocol']['semantic_policy']:
        raise ValueError('semantic policy changed')
    runtimes = supervision.runtimes()
    if not all(Path(x).is_file() for x in runtimes.values()):
        raise ValueError('missing worker Python environment')
    p = dict(schema='sparc_webvid53_evaluation_v1', rows=rows, missing_author_sources=missing,
        author_manifest_sha256=AUTHOR_MANIFEST_SHA, author_expected_videos=55,
        trained_root=str(trained), trained_protocol=parent, checkpoint_sha256=sha(checkpoint),
        target_cbr=parent['target_cbr'], runtime_paths=runtimes,
        code_sha256=sha(Path(__file__)),
        evaluation_mode='learned candidate ranker plus reference-assisted measured RD search',
        source_preparation='decode existing release-preprocessed MP4; no additional normalization',
        no_retraining=True, all53_is_independent_heldout=False,
        training_split_counts={s:sum(r['training_split']==s for r in rows) for s in ('train','val','test')})
    p['signature'] = digest(p)
    return p


def initialize(root, trained):
    p = protocol(trained)
    root.mkdir(parents=True, exist_ok=True)
    if (root/'protocol.json').exists() and read(root/'protocol.json') != p:
        raise ValueError('frozen evaluation identity changed; use another output root')
    save(root/'protocol.json',p); save(root/'sources.json',p['rows'])
    dest = root/'selector.pt'
    if not dest.exists():
        shutil.copyfile(trained/'training/best.pt',dest.with_suffix('.tmp'))
        os.replace(dest.with_suffix('.tmp'),dest)
    if sha(dest) != p['checkpoint_sha256']:
        raise ValueError('local evaluation checkpoint changed')
    return p


def decode_input(row):
    if sha(row['path']) != row['sha256']:
        raise ValueError('processed source changed')
    raw = subprocess.check_output(['ffmpeg','-v','error','-threads','2','-i',row['path'],
        '-map','0:v:0','-vsync','0','-threads','2','-f','rawvideo','-pix_fmt','rgb24','-'])
    frames = np.frombuffer(raw,np.uint8).reshape(-1,row['height'],row['width'],3).copy()
    if len(frames) != row['frames']:
        raise ValueError('decoded frame count differs from audited input')
    return frames


def array_sha(frames):
    stream = io.BytesIO(); np.save(stream,frames)
    return hashlib.sha256(stream.getbuffer()).hexdigest()


def check_hashes(folder, hashes):
    for name, expected in hashes.items():
        if sha(folder/name) != expected:
            raise ValueError('cached artifact changed: '+str(folder/name))


def prepare(root,row,p,progress):
    folder = root/'clips'/row['id']; folder.mkdir(parents=True,exist_ok=True)
    receipt = folder/'input_receipt.json'
    if receipt.exists():
        saved = read(receipt)
        if saved['protocol'] != p['signature'] or saved['input'] != row:
            raise ValueError('prepared input identity changed')
        check_hashes(folder,saved['files'])
        ready = read(folder/'prepared.json')
        check_hashes(Path(ready['source']['directory']),ready['source']['hashes'])
        from .sparc_semantics import validate_manifest
        validate_manifest(folder/'semantic.json',ready['source'],
                          p['trained_protocol']['supervision_protocol']['semantic_policy'])
        return ready
    frames = decode_input(row)
    frame_dir = folder/'frames'; frame_dir.mkdir(exist_ok=True)
    for i, frame in enumerate(frames):
        supervision.interrupted()
        path = frame_dir/f'{i}.png'
        if path.exists():
            if not np.array_equal(np.asarray(Image.open(path).convert('RGB')),frame):
                raise ValueError('partial source frame changed')
        else:
            Image.fromarray(frame).save(path)
    source = dict(directory=str(frame_dir),video=dict(frames=len(frames),width=576,height=320,fps=24.),
        hashes={f'{i}.png':sha(frame_dir/f'{i}.png') for i in range(len(frames))})
    progress('semantic_keys',semantic_candidates=len(range(4,len(frames),4))+(int((len(frames)-1)%4!=0)))
    client = Client('semantic',p['runtime_paths']['semantic'],root)
    try:
        semantic = client.call(dict(source=source,output=str(folder)))
    finally:
        client.close()
    supervision.interrupted()
    _,_,cuts = source_features(frames)
    ready = dict(input=row,source=source,keys=semantic['keys'],cuts=cuts,semantic=semantic,
                 protocol=p['signature'])
    save(folder/'prepared.json',ready)
    save(receipt,dict(protocol=p['signature'],input=row,source_array_sha256=array_sha(frames),
        files={name:sha(folder/name) for name in ('prepared.json','semantic.json')}))
    return ready


def features(root,row,p,progress):
    folder = root/'clips'/row['id']; receipt = folder/'feature_receipt.json'
    identity = dict(protocol=p['signature'],input_receipt_sha256=sha(folder/'input_receipt.json'))
    if receipt.exists():
        saved = read(receipt)
        if saved['identity'] != identity:raise ValueError('feature source identity changed')
        check_hashes(folder,saved['files']); return
    progress('features')
    frames,_ = data.load_source(folder)
    extractor = data.Extractor('cuda')
    try:
        if extractor.identity != p['trained_protocol']['features']:
            raise ValueError('feature extractor differs from training')
        data.atomic_npz(folder/'features.npz',**extractor.extract(frames))
    finally:
        del extractor,frames
        import torch
        torch.cuda.empty_cache()
    save(receipt,dict(identity=identity,files={'features.npz':sha(folder/'features.npz')}))


def verify_clip(root,row,p,clip):
    folder = root/'clips'/row['id']
    ready = read(folder/'prepared.json'); source_sha = read(folder/'input_receipt.json')['source_array_sha256']
    if clip['id'] != row['id'] or clip['parent'] != row['parent'] or clip['pixels'] != row['frames']*576*320:
        raise ValueError('evaluation source identity changed')
    for obs in clip['observations']:
        path = folder/'observations'/digest(obs['identity'])[:24]/'result.json'
        actual = data.verify_observation(path,p['trained_protocol']['supervision_protocol'],ready,source_sha)
        if actual != obs:raise ValueError('evaluation observation differs from verified artifacts')
    if clip['baseline'] not in clip['observations']:
        raise ValueError('baseline missing from measured candidates')


def output_choice(clips,target):
    cohort = training.allocate_cohort(clips,target)
    if cohort['feasible']:
        choices = [next(o for o in c['observations'] if o['identity']['indices']==s['indices'])
                   for c,s in zip(clips,cohort['selected'])]
    else:
        # Preserve actual reconstructions even when the requested budget is infeasible.
        choices = [min(c['observations'],key=lambda o:(o['channel']['complex_channel_uses'],
                    budget.quality_cost(o,c['baseline']))) for c in clips]
    return cohort,choices


def publish_video(source,dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        if sha(dest) != sha(source):raise ValueError('published video differs from frozen choice')
    else:
        os.link(source,dest)


def export(root,p,clips,scope):
    cohort,choices = output_choice(clips,p['target_cbr'])
    rows = {r['id']:r for r in p['rows']}; table = []
    for clip,obs in zip(clips,choices):
        row = rows[clip['id']]; folder = root/'clips'/row['id']
        record = dict(id=row['id'],filename=row['filename'],training_split=row['training_split'],
            frames=row['frames'],semantic_keys=len(read(folder/'prepared.json')['keys']),
            selected_frames=len(obs['identity']['indices']),candidate_count=len(clip['observations']))
        for name,value in [('baseline',clip['baseline']),('selected',obs)]:
            directory = folder/'observations'/digest(value['identity'])[:24]
            movie = root/'videos'/scope/name/(row['id']+'.mp4')
            publish_video(directory/'delivered.mp4',movie)
            record.update({name+'_cbr':budget.cbr(value,clip['pixels']),
                name+'_channel_uses':value['channel']['complex_channel_uses'],
                name+'_psnr_db':value['quality']['psnr_db'],name+'_lpips_vgg':value['quality']['lpips_vgg'],
                name+'_video':str(movie),name+'_video_sha256':sha(movie)})
        record['psnr_delta']=record['selected_psnr_db']-record['baseline_psnr_db']
        record['lpips_delta']=record['selected_lpips_vgg']-record['baseline_lpips_vgg']
        table.append(record)
    summary = dict(videos=len(table),frames=sum(r['frames'] for r in table),
        target_cbr=p['target_cbr'],budget_feasible=cohort['feasible'],allocation=cohort,
        reference_assisted_search=True,selection_scope=scope,
        independent_of_training_and_validation=(scope=='heldout7'))
    for name in ('baseline','selected'):
        summary[name]=dict(channel_uses=sum(r[name+'_channel_uses'] for r in table),
            aggregate_cbr=sum(r[name+'_channel_uses'] for r in table)/(3*576*320*summary['frames']),
            mean_video_cbr=sum(r[name+'_cbr'] for r in table)/len(table),
            mean_video_psnr_db=sum(r[name+'_psnr_db'] for r in table)/len(table),
            mean_video_lpips_vgg=sum(r[name+'_lpips_vgg'] for r in table)/len(table),
            frame_weighted_psnr_db=sum(r[name+'_psnr_db']*r['frames'] for r in table)/summary['frames'],
            frame_weighted_lpips_vgg=sum(r[name+'_lpips_vgg']*r['frames'] for r in table)/summary['frames'])
    save(root/(scope+'_summary.json'),summary)
    path=root/(scope+'_per_video.csv'); temp=path.with_suffix('.tmp')
    with temp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    os.replace(temp,path)
    return summary


def run(root,trained):
    import torch
    root.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        lock=stack.enter_context((root/'run.lock').open('a'));fcntl.flock(lock,fcntl.LOCK_EX)
        signal.signal(signal.SIGTERM,supervision.stop);signal.signal(signal.SIGINT,supervision.stop)
        began=time.time();completed=[];current={};done=0
        old=read(root/'status.json') if (root/'status.json').exists() else {}
        started=old.get('started_unix',began)
        def progress(stage,**extra):
            save(root/'status.json',dict(status='RUNNING',stage=stage,done=done,total=53,
                pid=os.getpid(),start_ticks=distributed.proc_identity(os.getpid()),
                started_unix=started,updated_unix=time.time(),clip=current.get('id','-'),
                clip_frames=current.get('frames'),completed_frames=sum(c['pixels']//(576*320) for c in completed),
                current_session_started=began,**extra))
        try:
            progress('verify_protocol');p=initialize(root,trained)
            gpu=stack.enter_context((REPO/'.local/etri_60s_check.lock').open('a'))
            while True:
                supervision.interrupted()
                try:fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:progress('waiting_for_gpu');time.sleep(3)
            torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            save(root/'environment.json',dict(host=socket.gethostname(),python=sys.version,
                torch=torch.__version__,gpu=torch.cuda.get_device_name(0),runtimes=p['runtime_paths'],
                ffmpeg=subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0]))
            eval_protocol=copy.deepcopy(p['trained_protocol'])
            eval_protocol['search']['benchmark_clips']=53
            eval_protocol['signature']=p['signature']
            supervision.runtimes=lambda:p['runtime_paths']
            for i,row in enumerate(p['rows']):
                supervision.interrupted();current=row;done=i;t=time.monotonic()
                progress('source_frames');prepare(root,row,p,progress);features(root,row,p,progress)
                folder=root/'clips'/row['id'];saved=root/'evaluation/benchmark'/row['id']/(row['id']+'.json')
                if saved.exists():verify_clip(root,row,p,read(saved)['clip'])
                def rd_progress(stage,*args,**extra):
                    progress('candidate_reconstruction',candidate=extra.get('candidate_measurements'),candidate_cap=17)
                training.evaluate(root,root,[row],eval_protocol,root/'selector.pt',
                                  'benchmark',row['id'],rd_progress)
                clip=read(saved)['clip'];verify_clip(root,row,p,clip)
                completed.append(clip);done=i+1
                receipt=folder/'evaluation_receipt.json'
                if not receipt.exists():
                    save(receipt,dict(protocol=p['signature'],input_receipt_sha256=sha(folder/'input_receipt.json'),
                        evaluation_sha256=sha(saved),elapsed_seconds=time.monotonic()-t,frames=row['frames']))
                elif read(receipt)['evaluation_sha256']!=sha(saved):
                    raise ValueError('completed video evaluation changed')
                progress('video_complete')
            progress('aggregate_and_export')
            all53=export(root,p,completed,'all53')
            heldout={r['id'] for r in p['rows'] if r['training_split']=='test'}
            heldout7=export(root,p,[c for c in completed if c['id'] in heldout],'heldout7')
            result=dict(status='COMPLETE',protocol=p['signature'],checkpoint_sha256=p['checkpoint_sha256'],
                videos=53,all53=all53,heldout7=heldout7,author_expected_videos=55,
                missing_author_sources=p['missing_author_sources'],
                training_overlap=p['training_split_counts'],no_retraining=True,
                evaluation_mode=p['evaluation_mode'])
            save(root/'RESULT.json',result);progress('complete')
            state=read(root/'status.json');state.update(status='COMPLETE',exit_code=0);save(root/'status.json',state)
            return dict(status='COMPLETE',videos=53,budget_feasible=all53['budget_feasible'])
        except BaseException as error:
            state=read(root/'status.json') if (root/'status.json').exists() else {}
            state.update(status='INTERRUPTED' if supervision.STOP or isinstance(error,InterruptedError) else 'FAILED',
                error=repr(error),exit_code=1,updated_unix=time.time())
            save(root/'status.json',state);raise


def launch(root,trained):
    root.mkdir(parents=True,exist_ok=True)
    with (root/'run.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return dict(status='ALREADY_RUNNING')
        if (root/'launcher.json').exists() and distributed.active(read(root/'launcher.json')):
            return dict(status='ALREADY_RUNNING')
        initialize(root,trained)
        if (root/'RESULT.json').exists():return dict(status='ALREADY_COMPLETE',output=str(root))
        env=dict(os.environ,PYTHONPATH=str(REPO/'src'),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
            PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
            CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1')
        env.pop('LD_LIBRARY_PATH',None)
        with (root/'run.log').open('ab') as log:
            proc=subprocess.Popen([sys.executable,'-m',MODULE,'run','--root',str(root),'--trained',str(trained)],
                cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(root/'launcher.json',dict(pid=proc.pid,start_ticks=distributed.proc_identity(proc.pid)))
    return dict(status='LAUNCHED',pid=proc.pid,output=str(root))


def stop(root):
    record=read(root/'launcher.json')
    if not distributed.active(record):return dict(status='NOT_RUNNING')
    args=Path(f'/proc/{record["pid"]}/cmdline').read_bytes().split(b'\0')
    if MODULE.encode() not in args or b'run' not in args or str(root).encode() not in args:
        raise ValueError('process identity mismatch; no signal sent')
    os.kill(record['pid'],signal.SIGTERM)
    return dict(status='STOP_REQUESTED',boundary='current semantic stage or candidate completes first')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['check','run','launch','stop'])
    parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--trained',type=Path,default=TRAINED)
    args=parser.parse_args();root=args.root.resolve();trained=args.trained.resolve()
    if args.command=='check':
        p=protocol(trained);result=dict(status='PREFLIGHT_PASSED',videos=len(p['rows']),
            frames=sum(r['frames'] for r in p['rows']),checkpoint_sha256=p['checkpoint_sha256'])
    elif args.command=='launch':result=launch(root,trained)
    elif args.command=='run':result=run(root,trained)
    else:result=stop(root)
    print(json.dumps(result))


if __name__=='__main__':main()
