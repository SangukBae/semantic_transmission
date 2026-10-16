"""Finish only the failed action dispatch after verifying completed DCVC trials."""
import argparse
from pathlib import Path
import subprocess
import os

from .artifacts import sha256, write_json
from .webvid_ablation import Stages, snapshot
from .webvid5 import fingerprint
from .bsu_dcvc_validation import read, REPO
from .key_reduction_selection import guard


def finish(root):
    p=read(root/'protocol.json')
    if fingerprint({k:v for k,v in p.items() if k!='signature'}) != p['signature']:
        raise ValueError('protocol changed')
    for name,digest in p['code'].items():
        if sha256(REPO/name)!=digest:
            raise ValueError('tested source changed: '+name)
    decisions=[]
    for item in p['candidates']:
        directory=root/item['case']['id']
        for receipt in directory.glob('*/stages/*.json'):
            saved=read(receipt)
            if saved['status']!='PASSED' or snapshot(receipt.parent.parent,saved['required'])!=saved['artifacts']:
                raise ValueError('incomplete/changed trial: '+str(receipt))
        d=read(directory/'decision.json')
        if d['seed2025']!=guard(d['baseline'],d['candidate'],p['criteria']):
            raise ValueError('quality verdict changed')
        if d.get('seed2026'):
            a=read(directory/'baseline_2026/quality.json');b=read(directory/'dcvc_2026/quality.json')
            if d['seed2026']!=guard(a,b,p['criteria']):raise ValueError('confirmation verdict changed')
        decisions.append(d)
    code=root/'tested_code';code.mkdir(exist_ok=True)
    (code/'bsu_dcvc_validation.py').write_bytes((REPO/'src/semantic_transmission/bsu_dcvc_validation.py').read_bytes())
    def action(log):
        env=dict(os.environ,PYTHONPATH=str(REPO/'src'),PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2')
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('x') as stream:
            subprocess.run([p['runtimes']['action'],'-m','semantic_transmission.bsu_key_reduction',
                            '--worker','action','--job',str(root/'action_job.json')],cwd=REPO,env=env,
                           stdout=stream,stderr=subprocess.STDOUT,check=True)
    Stages(root,p['signature']).step('action',['action'],['action/RESULT.json'],action)
    result=read(root/'action/RESULT.json')
    for d in decisions:
        pair={r['input']['variant']:r for r in result['records'] if r['input']['id']==d['id']}
        checks=[]
        for seed in [2025,2026] if 'seed2026' in d else [2025]:
            a,b=pair[f'baseline_{seed}'],pair[f'dcvc_{seed}']
            checks.append(all(not a[k] or b[k] for k in ['top1_correct','top5_correct']))
        d['action_noninferior']=all(checks)
        d['accepted']=d['accepted'] and d['action_noninferior']
    write_json(root/'RESULT.json',dict(status='COMPLETE_DCVC_BSU_CANDIDATE_VALIDATION',decisions=decisions,
        accepted_videos=sum(d['accepted'] for d in decisions),tested_videos=len(decisions),
        criteria=p['criteria'],action=result['summary'],production_defaults_changed=False,
        limitation='seen development videos; no accepted candidate; no generalization claim'))
    write_json(root/'action_recovery.json',dict(reason='action-only job has no run/case fields; dispatch directly to original action worker',
        reconstruction_rerun=False,quality_rerun=False,protocol_sha256=sha256(root/'protocol.json'),
        recovery_code_sha256=sha256(Path(__file__)),preserved_tested_source_sha256=sha256(code/'bsu_dcvc_validation.py')))
    write_json(root/'completion_receipt.json',dict(status='PASSED',
        artifacts={str(f.relative_to(root)):sha256(f) for f in root.rglob('*')
                   if f.is_file() and 'failed_attempts' not in f.parts and f.name!='completion_receipt.json'}))
    print('COMPLETE_DCVC_BSU_CANDIDATE_VALIDATION',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    finish(parser.parse_args().root.resolve())
