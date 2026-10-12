"""One-factor correspondence-gate follow-up; reuse all fixed parent inputs."""
import argparse
import copy
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from . import bsu_trajectory_experiment as parent
from .artifacts import sha256,write_json
from .webvid5 import read_json,fingerprint

ROOT=parent.REPO/'outputs/bsu_trajectory_guard_20261012_v1'
PARENT=parent.REPO/'outputs/bsu_trajectory_cadence_20261012_v2'


def prepare(root):
    if (root/'protocol.json').exists():parent.load(root);print('RESUME guard preparation verified',flush=True);return
    p=copy.deepcopy(parent.load(PARENT));p.pop('signature')
    p.update(policy='sparse_trajectory_correspondence_guard_v1',parent=str(PARENT),
             scope='posthoc one-factor follow-up on the same seven sources after double-contour inspection; no unseen test claim',
             intervention='only reference blending gains a generated/reference compatibility gate; trajectory bytes, baseline, cadence module unchanged')
    p['limitations'].append('guard added after reviewing all seven parent outputs; group names describe the parent split, not a fresh unseen follow-up test')
    from .guarded_trajectory_receiver import GUARD
    p['reference_guard']=GUARD
    for rel in ['src/semantic_transmission/guarded_trajectory_receiver.py','src/semantic_transmission/bsu_guarded_trajectory_experiment.py',
                'scripts/experiment_bsu_trajectory_guard.sh','tests/test_guarded_trajectory_receiver.py']:
        p['code'][str(parent.REPO/rel)]=sha256(parent.REPO/rel)
    for c in p['cases']:
        c['reuse_baseline']=str(PARENT/'generation'/c['name']/'baseline')
        for arm in parent.ARMS:
            directory=PARENT/'generation'/c['name']/arm
            for rel in ['result.json','evaluation.json','per_frame.json']:
                p['inputs'][str(directory/rel)]=sha256(directory/rel)
            for rel,digest in read_json(directory/'result.json')['artifacts'].items():
                p['inputs'][str(directory/rel)]=digest
        for path in (PARENT/'received'/c['name']).iterdir():
            if path.is_file():p['inputs'][str(path)]=sha256(path)
    p['inputs'][str(PARENT/'RESULT.json')]=sha256(PARENT/'RESULT.json')
    p['signature']=fingerprint(p);write_json(root/'protocol.json',p)
    for source,digest in p['code'].items():
        target=root/'frozen_code'/Path(source).relative_to(parent.REPO);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target);assert sha256(target)==digest
    for c in p['cases']:
        shutil.copytree(PARENT/'received'/c['name'],root/'received'/c['name'],dirs_exist_ok=True)
        for arm in ['baseline','cadence']:
            origin=PARENT/'generation'/c['name']/arm;target=root/'generation'/c['name']/arm
            target.mkdir(parents=True,exist_ok=True);receipt=read_json(origin/'result.json')
            for rel in receipt['artifacts']:
                dest=target/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(origin/rel,dest)
            receipt.update(signature=p['signature'],reused_from=str(origin));write_json(target/'result.json',receipt)
            evaluation=read_json(origin/'evaluation.json')
            evaluation.update(signature=p['signature'],receipt_sha256=sha256(target/'result.json'),reused_evaluation=str(origin/'evaluation.json'))
            write_json(target/'evaluation.json',evaluation);shutil.copyfile(origin/'per_frame.json',target/'per_frame.json')
    write_json(root/'channel_reuse.json',dict(parent=str(PARENT),extra_channel_uses=0,exact_received_packets=True))
    print('PREPARE 7/7 (100%): packets, baseline and cadence arms reused exactly; guard only changes',flush=True)


def refine(root):
    # Keep the frozen parent runner and its transport/evaluation machinery intact.
    # The explicitly declared reference implementation is the sole intervention.
    from . import trajectory_receiver
    from .guarded_trajectory_receiver import refine_references
    original=trajectory_receiver.refine_references
    trajectory_receiver.refine_references=refine_references
    try:parent.refine(root)
    finally:trajectory_receiver.refine_references=original


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--stage',choices=['all','prepare','refine','evaluate','verify'],default='all')
    args=parser.parse_args();root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    if args.stage!='all':
        {'prepare':prepare,'refine':refine,'evaluate':parent.evaluate,'verify':parent.verify}[args.stage](root);return
    with (root/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.monotonic()
        for i,stage in enumerate(['prepare','refine','evaluate','verify']):
            parent.progress('OVERALL guard stages',i,4,start)
            write_json(root/'status.json',dict(status='RUNNING',stage=stage,completed=i,total=4))
            code=subprocess.run([sys.executable,'-m','semantic_transmission.bsu_guarded_trajectory_experiment','--root',str(root),'--stage',stage]).returncode
            if code:
                write_json(root/'status.json',dict(status='FAILED',stage=stage,returncode=code));raise SystemExit(code)
        parent.progress('OVERALL guard stages',4,4,start)
        write_json(root/'status.json',dict(status='COMPLETE_PILOT',seconds=time.monotonic()-start))


if __name__=='__main__':main()
