"""Fixed terminal rows for the detached local Kinetics comparison."""
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
    s=read(root/'status.json');p=read(root/'protocol.json')
    done=s.get('done',0);total=s.get('total',14);state=s.get('status','NOT_STARTED');alive=active(s)
    if state=='RUNNING' and not alive:state='STOPPED unexpectedly; inspect run.log and resume'
    stage=s.get('stage','pending')
    try:
        a=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader,nounits'],text=True,timeout=2).splitlines()[0].split(',')
        gpu=f'{a[0].strip()}% | VRAM {a[1].strip()}/{a[2].strip()} MiB (all processes)'
    except (OSError,subprocess.SubprocessError,IndexError):gpu='unavailable'
    eta='pending: need at least 3 complete video timings'
    samples=[]
    for path in (root/'clips').glob('*/evaluation_receipt.json'):
        r=read(path);seconds=r.get('elapsed_seconds',0);source=path.parent/'input_receipt.json'
        if seconds>10 and source.exists() and source.stat().st_mtime>=path.stat().st_mtime-seconds-2:
            samples.append(r)
    if len(samples)>=3:
        rate=sum(r['elapsed_seconds'] for r in samples)/sum(r['frames'] for r in samples)
        remaining=rate*max(0,p.get('total_frames',0)-s.get('completed_frames',0))
        eta='~'+duration(remaining)+' search remaining; final 5-metric evaluation additional'
    if stage=='waiting_for_local_gpu':eta='unknown: queued behind the active local GPU job'
    if stage=='final_quality':eta='pending: final five-metric evaluation in progress'
    if state=='COMPLETE':eta='finished'
    command='bash scripts/run_sparc_kinetics_v3.sh '
    return [
        'Kinetics14 | SPARC-UF QP30 | adaptive versus frozen v3 + measured RD search',
        f'Status     {state} | PID {s.get("pid","-")} | alive={alive}',
        f'Videos     {done}/{total} ({100*done/total:.1f}%) search complete',
        f'Frames     {s.get("completed_frames",0)}/{p.get("total_frames","-")}',
        f'Stage      {stage}',
        f'Current    {s.get("clip","-")} | frames {s.get("clip_frames","-")}',
        f'RD trials  {s.get("candidate","-")}/{s.get("candidate_cap","-")} maximum per video',
        f'Quality    {s.get("metric_video_done",0)}/{s.get("metric_video_total",28)} delivered videos | {s.get("metric_role","-")} | {s.get("metric_done","-")}/{s.get("metric_total","-")} frame-metric operations',
        f'Elapsed    {duration(time.time()-s.get("started_unix",time.time()))}',
        f'ETA        {eta}',
        f'GPU        {gpu}',
        'Protocol   576x320 / 24fps / AWGN10dB seed42 / target aggregate CBR 0.00065',
        'Checkpoint 1e622f956013 | frozen; no Kinetics retraining',
        f'Failure    {s.get("error","-")}',
        f'Log        {root}/run.log',
        f'Output     {root}/COMPARISON.json | comparison_per_video.csv',
        'Videos     videos/all14/baseline/ | videos/all14/selected/',
        'Resume     '+command+'resume (verifies cached artifacts)',
        'Stop       '+command+'stop (safe candidate boundary)',
        'Ctrl-C closes this display only; background reconstruction continues.',
    ]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--once',action='store_true');args=parser.parse_args()
    if args.once or not sys.stdout.isatty():print('\n'.join(lines(args.root)));return
    previous=()
    try:
        sys.stdout.write('\x1b[?1049h\x1b[?25l')
        while True:
            size=shutil.get_terminal_size((120,24))
            current=tuple(s[:max(1,size.columns-1)] for s in lines(args.root)[:max(1,size.lines-1)])
            changes=''.join(f'\x1b[{i+1};1H{s}\x1b[K' for i,s in enumerate(current)
                            if i>=len(previous) or previous[i]!=s)
            if changes:sys.stdout.write(changes);sys.stdout.flush()
            previous=current;time.sleep(2)
    except KeyboardInterrupt:pass
    finally:sys.stdout.write('\x1b[?25h\x1b[?1049l');sys.stdout.flush()


if __name__=='__main__':main()
