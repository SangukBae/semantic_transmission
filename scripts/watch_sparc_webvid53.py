"""Read-only fixed-row display; Ctrl-C leaves the detached evaluation running."""
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
    v=read(root/'status.json');done=v.get('done',0);total=v.get('total',53)
    state=v.get('status','NOT_STARTED');alive=active(v)
    recovery=read(root/'recovery_status.json')
    if active(recovery) and recovery.get('status') in ('REPAIRING','RESUMING'):
        state='RECOVERING: '+recovery['status']
    if state=='RUNNING' and not alive:state='STOPPED unexpectedly; inspect failure/log and resume'
    try:
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader,nounits'],text=True,timeout=2).splitlines()[0].split(',')
        gpu=f'{raw[0].strip()}% | VRAM {raw[1].strip()}/{raw[2].strip()} MiB (all processes)'
    except (OSError,subprocess.SubprocessError,IndexError):gpu='unavailable'
    clip=root/'clips'/v.get('clip','-')
    calls=sum(1 for _ in clip.glob('semantic_cache/calls/*/*.json')) if v.get('stage')=='semantic_keys' else None
    receipt_paths=list((root/'clips').glob('*/evaluation_receipt.json'))
    samples=[]
    for path in receipt_paths:
        r=read(path);elapsed=r.get('elapsed_seconds',0)
        source=path.parent/'input_receipt.json'
        # A resumed partial clip times only the remaining work. Exclude it from
        # throughput estimation rather than extrapolating that short duration.
        if elapsed>10 and source.exists() and source.stat().st_mtime >= path.stat().st_mtime-elapsed-2:
            samples.append(r)
    eta='planning estimate 6-8 h total; measured remaining ETA pending'
    if len(samples)>=3:
        rate=sum(r['elapsed_seconds'] for r in samples)/sum(r['frames'] for r in samples)
        eta='~'+duration(rate*max(0,17382-v.get('completed_frames',0)))+' remaining (observed complete videos)'
    if state=='COMPLETE':eta='finished'
    return [
        'WebVid53 | SPARC-UF QP30 + frozen v3 selector | measured RD search',
        f'Status    {state} | PID {v.get("pid","-")} | alive={alive}',
        f'Videos    {done}/{total} ({100*done/total:.1f}%) | completed search, before cohort allocation',
        f'Stage     {v.get("stage","pending")}',
        f'Current   {v.get("clip","-")} | source frames {v.get("clip_frames","-")}',
        f'Keys      cached calls {calls if calls is not None else "-"} | planned candidates {v.get("semantic_candidates","-")}',
        f'RD trial  {v.get("candidate","-")}/{v.get("candidate_cap","-")} maximum per-video trials',
        f'Elapsed   {duration(time.time()-v.get("started_unix",time.time()))}',
        f'ETA       {eta}',
        f'GPU       {gpu}',
        'Protocol  576x320 / 24fps / <=16s / AWGN10dB / seed42 / target CBR 0.00065',
        'Cohort    train 33 / validation 13 / held-out 7; separate heldout7 allocation',
        f'Recovery  {recovery.get("status","not enabled")} | repairs {len(list((root/"repairs").glob("*/receipt.json")))}',
        f'Failure   {recovery.get("error",v.get("error","-")) if recovery.get("status")=="FAILED" else v.get("error","-")}',
        f'Run root  outputs/{root.name}',
        'Logs      run.log | semantic.log | receiver.log | quality.log',
        'Output    videos/all53/selected/ | all53_per_video.csv | heldout7_summary.json',
        'Resume    bash scripts/run_sparc_webvid53_remote.sh resume',
        'Stop      bash scripts/run_sparc_webvid53_remote.sh stop (safe boundary)',
        'Ctrl-C closes this display only; remote reconstruction continues.',
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
