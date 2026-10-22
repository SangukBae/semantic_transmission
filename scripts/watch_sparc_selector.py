"""Read-only, fixed-row selector training monitor. Ctrl-C stops this display only."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def duration(seconds):
    seconds = max(0,int(seconds))
    return f'{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}'


def rows(root, title='SPARC-UF QP30 learned auxiliary selector',
         resume_command='bash scripts/run_sparc_selector.sh resume'):
    try:
        v = json.loads((root/'status.json').read_text())
    except (OSError,ValueError):
        v = {}
    done,total = v.get('done'),v.get('total')
    progress = f'{done}/{total} ({100*done/total:.1f}%)' if total and done is not None else 'unknown total'
    now = v.get('ended_unix',time.time())
    elapsed = now-v.get('started_unix',now)
    stage_elapsed = now-v.get('stage_started_unix',now)
    eta = 'unknown (variable clip/model workload)'
    if v.get('stage') == 'train' and done and total and done >= 3:
        eta = '~'+duration(stage_elapsed/done*(total-done))+' to epoch cap; early stop may finish earlier'
    try:
        gpu = subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total',
                                      '--format=csv,noheader,nounits'],text=True,timeout=2).strip()
        fields = gpu.splitlines()[0].split(',')
        gpu = f'{fields[0].strip()}% | VRAM {fields[1].strip()}/{fields[2].strip()} MiB (all processes)'
    except (OSError,subprocess.SubprocessError):
        gpu = 'telemetry unavailable'
    pid = v.get('pid')
    alive = bool(pid and Path(f'/proc/{pid}').exists())
    state = v.get('status','NOT STARTED')
    if state == 'RUNNING' and not alive:
        state = 'STOPPED unexpectedly; inspect logs and resume'
    return [
        title,
        f'Status    {state} | PID {pid} | alive={alive}',
        f'Stage     {v.get("stage","unknown")}',
        f'Progress  {progress}',
        f'Current   {v.get("clip","-")} | pair {v.get("pair","-")}/{v.get("pairs_this_clip","-")}',
        f'Labels    {v.get("samples","pending")} | optimizer step {v.get("step","not started")}',
        f'Training  epoch {v.get("epoch","-")} | batch {v.get("batch_done","-")}/{v.get("batch_total","-")}',
        f'Loss      {v.get("loss","-")} | best validation {v.get("best_val","-")}',
        f'Elapsed   {duration(elapsed)} | stage {duration(stage_elapsed)}',
        f'ETA       {eta}',
        f'GPU       {gpu}',
        f'Failure   exit={v.get("exit_code","-")} {v.get("error","-")}',
        f'Resume    {v.get("resume","completed clips and full optimizer state retained")}',
        f'Log       {root/"run.log"}',
        f'Worker    {v.get("log",str(root/("quality.log" if v.get("stage") in ("measure_labels","prepare_v4_labels","labels_complete") else "training/history.jsonl" if v.get("stage")=="train" else "semantic.log")))}',
        f'Resume    {resume_command}',
        'Ctrl-C closes the monitor; the experiment keeps running.',
    ]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--title',default='SPARC-UF QP30 learned auxiliary selector')
    parser.add_argument('--resume-command',default='bash scripts/run_sparc_selector.sh resume')
    args=parser.parse_args()
    if args.once or not sys.stdout.isatty():
        print('\n'.join(rows(args.output,args.title,args.resume_command)))
        return
    previous=()
    try:
        sys.stdout.write('\x1b[?1049h\x1b[?25l')
        while True:
            size=shutil.get_terminal_size((100,24))
            if size.columns < 1 or size.lines < 1:
                size=os.terminal_size((100,24))
            current=tuple(x[:max(1,size.columns-1)] for x in
                          rows(args.output,args.title,args.resume_command)[:max(1,size.lines-1)])
            changes=''.join(f'\x1b[{i+1};1H{x}\x1b[K' for i,x in enumerate(current)
                            if i>=len(previous) or previous[i]!=x)
            if changes:
                sys.stdout.write(changes);sys.stdout.flush()
            previous=current
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write('\x1b[?25h\x1b[?1049l');sys.stdout.flush()


if __name__=='__main__':
    main()
