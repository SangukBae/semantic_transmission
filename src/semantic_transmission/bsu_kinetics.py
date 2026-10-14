"""One-command, resumable BSU Kinetics-400/14 reconstruction and action test."""
import argparse
import copy
import csv
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

from . import bsu_default as default, bsu_webvid as batch, kinetics_assets as assets, bsu_kinetics_base as current_base
from .artifacts import sha256, write_json
from .cli import settings
from .etri_60s_check import lock
from .qwen35_pairs_video import freeze, validate_source_window
from .webvid5 import fingerprint

REPO = batch.REPO
ROOT = REPO/'outputs/bsu_kinetics_context'
POLICY = 'bsu_kinetics14_context_action_v1'
REFERENCE = REPO/'outputs/bsu_zea_16s_20261009/protocol.json'
MANIFEST = REPO/'docs/eval_manifest_kinetics.csv'
RAW = REPO/'data/kinetics400_14/raw'
CODE = ('bsu_kinetics.py','kinetics_assets.py','kinetics_timesformer.py','decoder_memory.py','bsu_kinetics_base.py')
read = batch.read
event = assets.event


def originals(raw=RAW, manifest=MANIFEST):
    if sha256(manifest) != assets.MANIFEST_SHA:
        raise ValueError('Kinetics manifest differs from the exact 14 author clips')
    with Path(manifest).open() as stream: rows=list(csv.DictReader(stream))
    if len(rows)!=14 or len({r['filename'] for r in rows})!=14:
        raise ValueError('expected 14 unique author clips')
    for index,row in enumerate(rows,1):
        if Path(row['filename']).name != row['filename'] or not 0<=int(row['kinetics_label_id'])<400:
            raise ValueError('unsafe filename or invalid K400 label')
        path=Path(raw)/row['filename']
        if sha256(path)!=row['sha256']:raise ValueError('author original bytes differ: '+str(path))
        row.update(id=f'kinetics_{index:03d}_{row["sha256"][:8]}',source_path=str(path.resolve()),
                   label=int(row['kinetics_label_id']))
    return rows


def prepare_sources(root, rows):
    spec=importlib.util.spec_from_file_location('kinetics_source_prep',REPO/'scripts/prepare_fc_dataset_sources.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    records=[]
    for index,row in enumerate(rows,1):
        event(current=f'{row["id"]} ({index}/14)',stage='원본 전체 길이 정규화·프레임 준비')
        info=module.probe(row['source_path']);stream=info['streams'][0]
        item=dict(id=row['id'],dataset='kinetics',source_id=Path(row['filename']).stem,
            source_path=row['source_path'],source_sha256=row['sha256'],
            source_duration_seconds=float(stream.get('duration',info['format']['duration'])),
            benchmark_memberships=[],label=row['label'],filename=row['filename'])
        prepared=module.prepare_one(root/'inputs',item)
        validate_source_window(prepared)
        records.append(item)
        event(stage=f'원본 준비 {index}/14 ({100*index/14:.1f}%)')
    inventory=root/'inputs/inventory.json'
    freeze(inventory,dict(dataset='Kinetics-400',videos=records,full_source=True,
        author_manifest_sha256=assets.MANIFEST_SHA,recipe=module.RECIPE))
    return inventory


def runtime_template(root, reference=REFERENCE):
    """Bind today's code in a new template, retaining pinned model/policy bytes."""
    from .qwen35_backend import SNAPSHOT, MODEL_ID, REVISION
    from .qwen35_caption_adapter import default_adapter
    old=read(reference);local=settings(REPO);target=root/'config/bsu_template.json'
    if target.exists():
        value=read(target)
        for p,d in value['code'].items():
            if sha256(REPO/p)!=d:raise ValueError('frozen BSU code changed: '+p)
        if value['reference']['sha256']!=sha256(reference):raise ValueError('reference changed')
        for p,e in value['base_model_files'].items():
            stat=Path(p).stat()
            if (stat.st_size,stat.st_mtime_ns)!=(e['size'],e['mtime_ns']) or sha256(p)!=e['sha256']:
                raise ValueError('Qwen model file changed: '+p)
        if value['caption']['caption_adapter']!=default_adapter():raise ValueError('caption adapter changed')
        return target
    caption=copy.deepcopy(old['caption'])
    if (caption['model_id'],caption['model_revision'],caption['quantization'])!=(MODEL_ID,REVISION,'nf4'):
        raise ValueError('reference is not the current pinned BSU caption model')
    caption.update(model_path=str(SNAPSHOT),caption_adapter=default_adapter())
    receiver=copy.deepcopy(old['receiver'])
    receiver.update(models=read(REPO/'.local/model_paths.json'),snr_db=10.,seed=2025,channel_seed=42,steps=30)
    for field in ('input','input_sha256'):receiver.pop(field,None)
    for path in receiver['models'].values():
        if not Path(path).is_dir():raise FileNotFoundError(path)
    interpreters=dict(qwen=str(REPO/'.local/qwen35_env/bin/python'),python=local['python'],channel_python=local['channel_python'])
    for path in interpreters.values():
        if not Path(path).is_file():raise FileNotFoundError(path)
    files={}
    for index,(old_path,entry) in enumerate(old['base_model_files'].items(),1):
        path=SNAPSHOT/Path(old_path).name
        if path.stat().st_size!=entry['size'] or sha256(path)!=entry['sha256']:
            raise ValueError('pinned Qwen weights changed: '+str(path))
        files[str(path)]=dict(entry,mtime_ns=path.stat().st_mtime_ns)
        event(stage=f'BSU 가중치 검증 {index}/{len(old["base_model_files"])} ({100*index/len(old["base_model_files"]):.1f}%)')
    code={p:sha256(REPO/p) for p in (*old['code'],'src/semantic_transmission/decoder_memory.py',
        'src/semantic_transmission/bsu_kinetics_base.py')}
    # The historical protocol is an architectural/weight reference, never an
    # assertion that old experiment code and today's receiver are identical.
    value=dict(policy='current_bsu_kinetics_template_v1',caption=caption,receiver=receiver,
        interpreters=interpreters,base_model_files=files,code=code,
        reference=dict(path=str(reference),sha256=sha256(reference)),
        current_code_changes={p:dict(before=d,now=code[p]) for p,d in old['code'].items() if d!=code[p]})
    freeze(target,value)
    return target


def cohort(inventory, manifest, ids=None, require_complete=False):
    rows=originals(Path(read(inventory)['videos'][0]['source_path']).parent,manifest)
    expected={r['id']:r for r in rows};videos=[]
    for row in read(inventory)['videos']:
        if row['id'] not in expected or row['dataset']!='kinetics':raise ValueError('unexpected source inventory')
        e=expected[row['id']]
        if row['source_sha256']!=e['sha256'] or row['label']!=e['label']:raise ValueError('source/label mismatch')
        prepared=inventory.parent/'kinetics'/row['id']/'source_prepared.json';source=read(prepared)
        validate_source_window(source)
        if source['source_sha256']!=e['sha256'] or (source['width'],source['height'],source['fps'])!=(576,320,24):
            raise ValueError('prepared Kinetics source mismatch')
        videos.append(dict(id=row['id'],prepared=str(prepared),prepared_sha256=sha256(prepared),
            source_sha256=e['sha256'],filename=e['filename'],label=e['label'],frames=source['frames'],fps=24,width=576,height=320))
    if ids is not None or len(videos)!=14 or len({v['id'] for v in videos})!=14:
        raise ValueError('the action benchmark requires all 14 unique clips')
    return videos,dict(expected=14,available=14,selected=14,missing=[],full_manifest_selected=True,exact_paper_reproduction=False)


def plan(root, inventory, template, runtime, manifest=MANIFEST):
    target=root/'benchmark'
    if (target/'protocol.json').exists():
        value=read(target/'protocol.json');batch.verify_plan(value)
        if (value['action_runtime']!=runtime or value['mode']!=default.receiver.DEFAULT_MODE
                or value['root']!=str(target) or value['policy']!=POLICY
                or value['inventory']!=str(inventory) or value['template']!=str(template)
                or value['manifest']!=str(manifest.resolve())):
            raise ValueError('frozen action runtime/mode changed')
        return value
    args=SimpleNamespace(root=target,inventory=inventory,template=template,manifest=manifest,
        ids=None,require_complete_manifest=True,snr=[10.],mode=default.receiver.DEFAULT_MODE,
        extra_rate_ratio=.05,reuse_base=None)
    old_cohort=batch.cohort
    with default.current_pipeline():
        batch.CODE=(*default.CODE,*CODE);batch.cohort=cohort
        try:value=batch.make_plan(args)
        finally:batch.cohort=old_cohort
    from .receiver_key_context import PARAMETERS
    value.update(policy=POLICY,dataset='Kinetics-400',action_runtime=runtime,
        selection_format_repair=current_base.REPAIR_POLICY,
        receiver_refinement=dict(mode=default.receiver.DEFAULT_MODE,context_enabled=True,
            context_parameters=PARAMETERS,sender_mode='combined',additional_context_transmission=0),
        paper=dict(source='https://arxiv.org/html/2606.12899v2#S3.SS2',task='action classification',
            clips=14,snr_db=10,model='TimeSformer_divST_8x32_224_K400',metrics=['top1','top5']),
        model_description='current BSU semantic selection + caption LoRA + repaired Open-Sora + combined_context',
        exact_paper_reproduction=False)
    del value['paper_sha256']
    value['code'].update({p:sha256(REPO/p) for p in ['scripts/prepare_fc_dataset_sources.py',
        '07_downstream/timesformer/TimeSformer_divST_8x32_224_TEST.yaml',
        '07_downstream/timesformer/patches/timesformer_modifications.patch']})
    value['signature']=fingerprint({k:v for k,v in value.items() if k!='signature'})
    freeze(target/'protocol.json',value)
    return value


def prepare(args):
    root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    freeze(root/'request.json',dict(policy=POLICY,root=str(root),raw=str(args.raw.resolve()),
        manifest=str(args.manifest.resolve()),runtime=str(args.runtime.resolve()),reference=str(args.reference.resolve())))
    event(overall='입력·실행 환경 준비 중',current='Kinetics-400 14편',stage='저자 원본 SHA256 검증')
    rows=originals(args.raw,args.manifest)
    runtime=assets.setup(args.runtime)
    freeze(root/'config/action_runtime.json',runtime)
    event(stage='TimeSformer 로딩 사전 검사 (진행률 미확정)')
    env=dict(os.environ,PYTHONPATH=str(REPO/'src'),PYTHONNOUSERSITE='1')
    batch.command([runtime['python'],'-m','semantic_transmission.kinetics_timesformer',
        '--check-runtime',root/'config/action_runtime.json'],root/f'logs/runtime_{time.time_ns()}.log',env)
    inventory=prepare_sources(root,rows)
    template=runtime_template(root,args.reference)
    value=plan(root,inventory,template,runtime,args.manifest)
    event(stage='실행 설정과 전체 입력 검증 완료',state='준비 완료')
    return value


def progress(reconstructed, evaluated=0):
    return f'복원 {reconstructed}/14 ({100*reconstructed/14:.1f}%) | 행동평가 {evaluated}/28 ({100*evaluated/28:.1f}%)'


def run(value):
    from .kinetics_timesformer import evaluation_job, validate_evaluation
    root=Path(value['root']);outer=root.parent
    batch.verify_plan(value)
    results=[];pending=[]
    for video in value['videos']:
        path=root/'snr_10'/video['id'];item=batch.load_result(path,value)
        if item:results.append(item)
        else:pending.append(video)
    state=dict(status='RUNNING',reconstructed=len(results),total=14,action_evaluated=0,mode=value['mode'])
    write_json(outer/'status.json',state)
    try:
        with default.current_pipeline(), current_base.current_pipeline():
            for video in pending:
                event(overall=progress(len(results)),current=video['id'],stage='BSU 선택·캡션·송수신·복원·연결')
                item=batch.run_one(value,video,10.,root/'snr_10'/video['id']);results.append(item)
                state.update(reconstructed=len(results));write_json(outer/'status.json',state)
        # Preserve manifest order even when some completed results were reused.
        lookup={r['id']:r for r in results};results=[lookup[v['id']] for v in value['videos']]
        event(overall=progress(14),current='원본·최종 BSU 출력',stage='TimeSformer 행동 분류')
        job=evaluation_job(value,results);job_path=outer/'action_job.json';freeze(job_path,job)
        env=dict(os.environ,PYTHONPATH=str(REPO/'src'),PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
        with lock(REPO/'.local/etri_60s_check.lock'):
            batch.command([value['action_runtime']['python'],'-m','semantic_transmission.kinetics_timesformer',
                '--job',job_path],outer/f'logs/action_{time.time_ns()}.log',env)
        action=validate_evaluation(job)
        result=dict(status='COMPLETE_RECONSTRUCTION_QUALITY_ACTION_EVALUATION',signature=value['signature'],
            clips=14,frames=sum(v['frames'] for v in value['videos']),snr_db=10,mode=value['mode'],
            quality=batch.aggregate(results),action=action['summary'],
            videos=[dict(id=v['id'],filename=v['filename'],label=v['label'],frames=v['frames'],
                normalized_original=read(v['prepared'])['normalized_video'],reconstruction=r['video'],
                total_channel_uses=r['total_channel_uses']) for v,r in zip(value['videos'],results)],
            training_executed=False,independent_semantic_review='PENDING',exact_paper_reproduction=False,
            scope='BSU on the exact author clips at 10dB; published TimeSformer preprocessing/config and official checkpoint. No newly reconstructed LGVSC or codec comparators.',
            action_result_sha256=sha256(Path(job['output'])/'RESULT.json'))
        freeze(outer/'RESULT.json',result)
        lines=['variant,videos,top1_correct,top1_percent,top5_correct,top5_percent']
        for arm,metrics in result['action'].items():
            lines.append(','.join(map(str,[arm,metrics['videos'],metrics['top1_correct'],metrics['top1_percent'],metrics['top5_correct'],metrics['top5_percent']])))
        (outer/'action_accuracy.csv').write_text('\n'.join(lines)+'\n')
        write_json(outer/'status.json',dict(state,status='COMPLETE',reconstructed=14,action_evaluated=28))
        event(overall=progress(14,28),stage='복원·화질·행동 평가 완료',state='완료')
        print('REPORT: '+str(outer/'RESULT.json'),flush=True)
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error,KeyboardInterrupt) else 'FAILED',error=str(error))
        write_json(outer/'status.json',state)
        raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',nargs='?',choices=('run','prepare','check','status'),default='run')
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--raw',type=Path,default=RAW)
    p.add_argument('--manifest',type=Path,default=MANIFEST);p.add_argument('--runtime',type=Path,default=assets.ROOT)
    p.add_argument('--reference',type=Path,default=REFERENCE)
    return p


def main():
    args=parser().parse_args();args.root=args.root.resolve()
    if args.action=='status':
        print(json.dumps(read(args.root/'status.json'),ensure_ascii=False,indent=2));return
    if args.action=='check':
        rows=originals(args.raw,args.manifest)
        if (args.root/'benchmark/protocol.json').exists():
            value=read(args.root/'benchmark/protocol.json');batch.verify_plan(value)
            from .kinetics_timesformer import validate_runtime
            validate_runtime(value['action_runtime'])
            state='READY_TO_RUN'
        else:state='READY_TO_PREPARE'
        print('CHECK: '+json.dumps(dict(status=state,author_clips=len(rows),mode='combined_context',snr_db=10,
            model_inference_started=False),ensure_ascii=False));return
    from .qwen35_pairs_comparison import forbid_overlap
    forbid_overlap(args.root,[args.raw,args.runtime,args.reference.parent,REPO/'.local',REPO/'data'])
    args.root.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    with lock(args.root/'execution.lock'):
        try:
            value=prepare(args)
        except BaseException as error:
            previous=read(args.root/'status.json') if (args.root/'status.json').exists() else {}
            write_json(args.root/'status.json',dict(previous,status='INTERRUPTED' if isinstance(error,KeyboardInterrupt) else 'FAILED',
                stage='preparation',error=str(error)))
            raise
        if args.action=='run':run(value)
        else:
            if not (args.root/'status.json').exists() or read(args.root/'status.json').get('stage')=='preparation':
                write_json(args.root/'status.json',dict(status='READY_TO_RUN',reconstructed=0,total=14,action_evaluated=0))
            print('CHECK: READY_TO_RUN; no BSU inference started',flush=True)


if __name__=='__main__':main()
