"""Read-only fixed terminal rows for local continuation training."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from semantic_transmission.sparc_selector_distributed import active
from watch_sparc_selector import duration


def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}


def lines(root):
    v=read(root/'status.json');d=read(root/'import_receipt.json');p=read(root/'protocol.json')
    state=v.get('status','NOT_STARTED');alive=active(v)
    if state=='RUNNING' and not alive:state='STOPPED unexpectedly; inspect log and resume'
    done,total=v.get('done'),v.get('total')
    progress=f'{done}/{total} ({100*done/total:.1f}%)' if total and done is not None else 'pending / variable work'
    stage=v.get('stage','pending')
    if stage=='train_v3':
        progress=f'{v["batch_done"]}/{v["batch_total"]} ({100*v["batch_done"]/v["batch_total"]:.1f}%) of current epoch'
    eta='pending; validation search cost varies, early stopping enabled'
    measured_batches=v.get('batch_done',0)-v.get('epoch_start_batch',0)
    if stage=='train_v3' and measured_batches>=20:
        elapsed=time.time()-v['stage_started_unix']
        eta='~'+duration(elapsed/measured_batches*(v['batch_total']-v['batch_done']))+' to current epoch end; validation extra'
    try:
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader,nounits'],text=True,timeout=2).splitlines()[0].split(',')
        gpu=f'{raw[0].strip()}% | VRAM {raw[1].strip()}/{raw[2].strip()} MiB (all processes)'
    except (OSError,subprocess.SubprocessError,IndexError):gpu='unavailable'
    history=root/'training/history.jsonl';last={}
    if history.exists():
        try:last=json.loads(history.read_text().splitlines()[-1])
        except (OSError,ValueError,IndexError):pass
    baseline=read(root/'evaluation/val/initial/result.json')
    best=last.get('measured_validation',{})
    return [
        'SPARC-UF QP30 | local long-WebVid continuation from remote best.pt',
        f'Status    {state} | PID {v.get("pid","-")} | alive={alive}',
        f'Stage     {stage}',
        f'Progress  {progress}',
        f'Dataset   long {d.get("long_clips",60)} clips / 4359 states + replay {d.get("replay_clips",651)} clips',
        f'Epoch     {v.get("epoch","-")}/{p.get("train",{}).get("max_epochs",20)} maximum | optimizer step {v.get("step","-")}',
        f'Loss      {v.get("loss","-")}',
        f'Current   {v.get("clip","-")} | measured candidate {v.get("candidate_measurements","-")}',
        f'Validation initial score {baseline.get("selection_score","pending")} | last {best.get("selection_score","pending")}',
        f'Budget    CBR 0.00065 | validation every 2 epochs; 4 misses stop',
        f'Elapsed   {duration(time.time()-v.get("started_unix",time.time()))}',
        f'ETA       {eta}',
        f'GPU       {gpu}',
        f'Failure   exit={v.get("exit_code","-")} {v.get("error","-")}',
        'Resume    model / optimizer / scheduler / RNG / sampled order / budget',
        f'Run root  outputs/{root.name}',
        'Log       run.log | training/history.jsonl | status.json',
        'Weights   training/best.pt | training/last.pt',
        'Resume    bash scripts/run_sparc_selector_finetune.sh resume',
        'Stop      bash scripts/run_sparc_selector_finetune.sh stop (safe checkpoint)',
        'Ctrl-C closes this display only; training continues.',
    ]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--once',action='store_true');args=parser.parse_args()
    if args.once or not sys.stdout.isatty():print('\n'.join(lines(args.root)));return
    previous=()
    try:
        sys.stdout.write('\x1b[?1049h\x1b[?25l')
        while True:
            size=shutil.get_terminal_size((120,25))
            current=tuple(s[:max(1,size.columns-1)] for s in lines(args.root)[:max(1,size.lines-1)])
            changed=''.join(f'\x1b[{i+1};1H{s}\x1b[K' for i,s in enumerate(current) if i>=len(previous) or previous[i]!=s)
            if changed:sys.stdout.write(changed);sys.stdout.flush()
            previous=current;time.sleep(1)
    except KeyboardInterrupt:pass
    finally:sys.stdout.write('\x1b[?25h\x1b[?1049l');sys.stdout.flush()


if __name__=='__main__':main()
