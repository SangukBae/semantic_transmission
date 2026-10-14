"""LGVSC-only inference with the frozen BSU TimeSformer evaluator and runtime."""
import argparse
from pathlib import Path
import time

from . import kinetics_timesformer as base
from .artifacts import sha256, write_json
from .kinetics_assets import event
from .qwen35_pairs_video import freeze

POLICY = 'matched_lgvsc_author_timesformer_14_v1'
read = base.read


def evaluation_config(original):
    return dict(original['evaluation'], reconstruction='delivered LGVSC matched MP4')


def make_job(plan, results):
    if len(results) != 14: raise ValueError('all 14 LGVSC reconstructions are required')
    old_path = Path(plan['bsu_root'])/'action_job.json'
    old = read(old_path); base.validate_evaluation(old)
    if old['runtime'] != plan['action_runtime']: raise ValueError('action runtime differs from BSU')
    originals = {r['id']:r for r in old['rows'] if r['variant']=='original'}
    rows = []
    for v, r in zip(plan['videos'], results):
        if (r['id'], r['frames'], r['signature'], r['status']) != (
                v['id'], v['frames'], plan['signature'], 'COMPLETE_RECONSTRUCTION_QUALITY'):
            raise ValueError('LGVSC action input differs from the comparison protocol')
        row = dict(originals[v['id']], variant='lgvsc', path=r['video'], sha256=sha256(r['video']))
        if row['sha256'] != r['video_sha256']: raise ValueError('LGVSC movie bytes changed')
        rows.append(row)
    return base.signed(dict(policy=POLICY, rows=rows, output=str(Path(plan['root'])/'action'),
        runtime=old['runtime'], config=old['config'], config_sha256=old['config_sha256'],
        evaluation=evaluation_config(old), benchmark_signature=plan['signature'],
        bsu_action_job=str(old_path), bsu_action_job_sha256=sha256(old_path),
        code=dict(old['code'], **{str(Path(__file__).resolve()):sha256(__file__)})))


def validate_job(job):
    base.verify_signature(job)
    if (job['policy'] != POLICY or job['config'] != str(base.CONFIG)
            or sha256(job['config']) != job['config_sha256']
            or sha256(job['bsu_action_job']) != job['bsu_action_job_sha256']):
        raise ValueError('action policy/BSU reference/config changed')
    old = read(job['bsu_action_job']); base.validate_evaluation(old)
    if job['runtime'] != old['runtime'] or job['evaluation'] != evaluation_config(old):
        raise ValueError('TimeSformer conditions differ from BSU')
    for p, digest in job['code'].items():
        if sha256(p) != digest: raise ValueError('action implementation changed')
    expected = {r['id']:(r['label'], r['frames']) for r in old['rows'] if r['variant']=='original'}
    rows = job['rows']
    if len(rows)!=14 or {r['id']:(r['label'],r['frames']) for r in rows} != expected:
        raise ValueError('expected exactly the same 14 clip identities/labels/frames')
    for row in rows:
        if row['variant'] != 'lgvsc' or sha256(row['path']) != row['sha256']:
            raise ValueError('LGVSC action input changed')
        if ',' in row['path'] or '\n' in row['path']: raise ValueError('unsupported CSV path')


def summary(records):
    if len(records)!=14 or any(r is None for r in records): raise ValueError('incomplete action evaluation')
    a=sum(r['top1_correct'] for r in records); b=sum(r['top5_correct'] for r in records)
    return dict(videos=14, top1_correct=a, top1_percent=100*a/14, top5_correct=b, top5_percent=100*b/14)


def validate_evaluation(job):
    validate_job(job)
    records=[base.load_record(job, row) for row in job['rows']]
    result=read(Path(job['output'])/'RESULT.json'); base.verify_signature(result)
    hashes={str(base.record_path(job,row).relative_to(job['output'])):sha256(base.record_path(job,row)) for row in job['rows']}
    if (result['status']!='COMPLETE_ACTION_EVALUATION' or result['job_signature']!=job['signature']
            or result['summary']!=summary(records) or result['artifacts']!=hashes):
        raise ValueError('LGVSC action result provenance/summary changed')
    return result


def evaluate(job):
    validate_job(job); base.validate_runtime(job['runtime'])
    output=Path(job['output']); output.mkdir(parents=True,exist_ok=True)
    if (output/'RESULT.json').exists(): return validate_evaluation(job)
    records=[base.load_record(job, row) for row in job['rows']]
    import random
    import numpy as np
    import torch
    csv=''.join(f'{r["path"]},{r["label"]}\n' for r in job['rows'])
    path=output/'test.csv'
    if path.exists() and path.read_text()!=csv: raise ValueError('action CSV changed')
    path.write_text(csv)
    cfg=base.configure(job['runtime'],output)
    from timesformer.datasets.kinetics import Kinetics
    random.seed(cfg.RNG_SEED); np.random.seed(cfg.RNG_SEED); torch.manual_seed(cfg.RNG_SEED)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for TimeSformer')
    event(stage='TimeSformer 로딩 (진행률 미확정)')
    model=base.load_model(job['runtime'],'cuda'); dataset=Kinetics(cfg,'test')
    if len(dataset)!=42: raise ValueError('expected 14 inputs x 3 crops')
    done=sum(r is not None for r in records); started=time.monotonic(); new=0
    for i,row in enumerate(job['rows']):
        if records[i] is not None: continue
        event(current=row['id'],stage=f'LGVSC 행동평가 {done}/14 ({100*done/14:.1f}%)')
        tick=time.monotonic(); logits,ranking=base.classify(model,dataset,i,row['label'],'cuda')
        record=base.signed(dict(job_signature=job['signature'],input=row,crops=3,summed_logits=logits,top5=ranking,
            top1_correct=row['label']==ranking[0],top5_correct=row['label'] in ranking,seconds=time.monotonic()-tick))
        freeze(base.record_path(job,row),record); records[i]=record; done+=1; new+=1
        write_json(output/'status.json',dict(status='RUNNING',complete=done,total=14))
        event(overall=f'LGVSC 복원 14/14 (100.0%) | 행동평가 {done}/14 ({100*done/14:.1f}%)',
              stage=f'TimeSformer {done}/14 ({100*done/14:.1f}%) | ETA {(time.monotonic()-started)/new*(14-done):.0f}s')
    result=base.signed(dict(status='COMPLETE_ACTION_EVALUATION',job_signature=job['signature'],summary=summary(records),
        artifacts={str(base.record_path(job,row).relative_to(output)):sha256(base.record_path(job,row)) for row in job['rows']}))
    freeze(output/'RESULT.json',result)
    write_json(output/'status.json',dict(status='COMPLETE',complete=14,total=14))
    return validate_evaluation(job)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--job',type=Path,required=True)
    args=parser.parse_args(); evaluate(read(args.job))


if __name__=='__main__': main()
