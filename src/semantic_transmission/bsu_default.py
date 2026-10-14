"""Current BSU batch entry point; frozen v1 batches keep their original policy."""
import argparse
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import signal

from . import bsu_webvid as legacy
from . import bsu_receiver as receiver
from .webvid5 import fingerprint

REPO = legacy.REPO
POLICY = 'bsu_webvid_key_context_v2'
ROOT = REPO/'outputs/bsu_webvid_context'
CODE = (*legacy.CODE, 'bsu_default.py', 'bsu_receiver.py', 'receiver_key_context.py')


@contextmanager
def current_pipeline():
    """Process-local adapter, following bsu_webvid_base's frozen-runner pattern."""
    code, policy, command = legacy.CODE, legacy.POLICY, legacy.command
    def routed(cmd, log, env):
        cmd = list(cmd)
        if len(cmd)>2 and cmd[1:3]==['-m','semantic_transmission.bsu_sequence_refinement']:
            cmd[2] = receiver.MODULE
        return command(cmd,log,env)
    legacy.CODE, legacy.POLICY, legacy.command = CODE, POLICY, routed
    try: yield
    finally: legacy.CODE, legacy.POLICY, legacy.command = code, policy, command


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('check','run','status','report'))
    p.add_argument('--root',type=Path,default=ROOT)
    # None distinguishes an omitted option from an explicit incompatible change
    # when resuming a frozen older batch.
    for name in ('inventory','template','manifest','reuse-base'):p.add_argument('--'+name,type=Path)
    p.add_argument('--ids',nargs='+');p.add_argument('--snr',nargs='+',type=float)
    p.add_argument('--mode',choices=receiver.MODES,
        help='new batches default to combined_context; existing batches retain their frozen mode')
    p.add_argument('--extra-rate-ratio',type=float)
    p.add_argument('--require-complete-manifest',action='store_true')
    p.add_argument('--keep-going',action='store_true')
    return p


def load_or_plan(args):
    root=args.root.resolve();protocol=root/'protocol.json'
    if protocol.exists():
        plan=legacy.read(protocol);legacy.verify_plan(plan)
        if Path(plan['root']).resolve()!=root:
            raise ValueError('frozen benchmark root differs; do not relocate an active batch')
        for name in ('inventory','template','manifest','reuse_base'):
            value=getattr(args,name)
            if value is not None and str(value.resolve())!=plan.get(name):
                raise ValueError(f'frozen {name} differs; use a new --root')
        for name in ('mode','snr','extra_rate_ratio'):
            value=getattr(args,name);key='snr_db' if name=='snr' else name
            if value is not None and value!=plan[key]:
                raise ValueError(f'frozen {name} differs; use a new --root')
        if args.ids is not None and (len(args.ids)!=len(set(args.ids)) or set(args.ids)!={v['id'] for v in plan['videos']}):
            raise ValueError('frozen video selection differs; use a new --root')
        if args.require_complete_manifest and not plan['coverage']['full_manifest_selected']:
            raise ValueError('frozen benchmark does not cover the full manifest')
        receiver.sender_mode(plan['mode'])
        if plan['mode']==receiver.DEFAULT_MODE and plan['policy']!=POLICY:
            raise ValueError('key-context mode requires the v2 receiver protocol')
        print(f'RESUME: frozen mode={plan["mode"]}; new default is not applied retroactively',flush=True)
        return plan
    new=copy.copy(args);new.root=root
    defaults=dict(inventory=REPO/'outputs/remote_webvid_full_20261012/inventory.json',
        template=REPO/'.local/deployment/bsu_template.json',manifest=REPO/'docs/eval_manifest_webvid.csv',
        snr=[6.,8.,10.],mode=receiver.DEFAULT_MODE,extra_rate_ratio=.05)
    for name,value in defaults.items():
        if getattr(new,name) is None:setattr(new,name,value)
    with current_pipeline(): plan=legacy.make_plan(new)
    from .receiver_key_context import PARAMETERS
    plan['receiver_refinement']=dict(mode=new.mode,context_enabled=new.mode==receiver.DEFAULT_MODE,
        context_parameters=PARAMETERS if new.mode==receiver.DEFAULT_MODE else None,
        sender_mode=receiver.sender_mode(new.mode),additional_context_transmission=0)
    plan['model_description']='Qwen semantic selection + pinned caption LoRA + repaired Open-Sora; '+(
        'unguarded trajectory synthesis, cadence, then exact received-key context' if new.mode==receiver.DEFAULT_MODE
        else f'explicit legacy receiver mode {new.mode}')
    plan['signature']=fingerprint({k:v for k,v in plan.items() if k!='signature'})
    return plan


def run(plan, keep_going=False):
    if plan['policy']==POLICY:
        with current_pipeline():return legacy.run(plan,keep_going)
    return legacy.run(plan,keep_going)


def main():
    args=parser().parse_args()
    if args.action=='status':
        print(json.dumps(legacy.read(args.root/'status.json'),ensure_ascii=False,indent=2));return
    if args.action=='report':
        plan=legacy.read(args.root/'protocol.json')
        with legacy.lock(args.root/'execution.lock'):legacy.report(plan)
        return
    plan=load_or_plan(args)
    if args.action=='check':
        print(json.dumps(dict(status='READY',coverage=plan['coverage'],snr_db=plan['snr_db'],mode=plan['mode'],
            jobs=len(plan['videos'])*len(plan['snr_db']),model_inference_started=False),ensure_ascii=False,indent=2));return
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    raise SystemExit(run(plan,args.keep_going))


if __name__=='__main__':main()
