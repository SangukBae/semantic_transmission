"""Read-only fixed-row progress for the local long-WebVid teacher dataset."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from semantic_transmission.sparc_selector_distributed import active
from watch_sparc_selector import duration


def lines(root):
    try:v=json.loads((root/'status.json').read_text())
    except (OSError,ValueError):v={}
    done,total=v.get('done'),v.get('total')
    progress=f'{done}/{total} ({100*done/total:.1f}%)' if total and done is not None else 'pending'
    try:
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader,nounits'],text=True,timeout=2).splitlines()[0].split(',')
        gpu=f'{raw[0].strip()}% | VRAM {raw[1].strip()}/{raw[2].strip()} MiB (all processes)'
    except (OSError,subprocess.SubprocessError,IndexError):gpu='unavailable'
    state=v.get('status','NOT_STARTED');alive=active(v)
    if state=='RUNNING' and not alive:state='STOPPED unexpectedly; inspect log and resume'
    return [
        'SPARC long-WebVid teacher dataset | local GPU | train split only',
        f'Status     {state} | PID {v.get("pid","-")} | alive={alive}',
        f'Stage      {v.get("stage","pending")}',
        f'Windows    {progress}',
        f'Parent job {v.get("parent_status","completed / GPU lock acquired" if v.get("stage") not in (None,"verify_protocol","waiting_for_local_labels") else "pending")} | {v.get("parent_done","-")}/{v.get("parent_total","-")}',
        f'Current    {v.get("clip","-")}',
        f'Candidates {v.get("candidate","-")}/{v.get("candidate_cap","-")} new-trial cap | measured states {v.get("measured_states","-")}',
        f'Round      {v.get("round","-")}/{v.get("rounds","-")}',
        'Budget     CBR 0.0005 / 0.00065 / 0.0008 / 0.001; actual packet cost',
        f'Elapsed    {duration(time.time()-v.get("started_unix",time.time()))}',
        'ETA        unknown: candidate counts and codec/receiver cost vary',
        f'GPU        {gpu}',
        f'Failure    {v.get("error","-")}',
        f'Log        {root/"run.log"}',
        f'Receiver   {root/"receiver.log"} | metrics {root/"quality.log"}',
        f'Dataset    {root/"dataset.json"}',
        'Resume     bash scripts/run_sparc_webvid_search.sh resume',
        'Stop       bash scripts/run_sparc_webvid_search.sh stop (commits current candidate)',
        'Ctrl-C closes this display only; the queued/running job continues.',
    ]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--once',action='store_true')
    a=p.parse_args()
    if a.once or not sys.stdout.isatty():print('\n'.join(lines(a.root)));return
    previous=()
    try:
        sys.stdout.write('\x1b[?1049h\x1b[?25l')
        while True:
            size=shutil.get_terminal_size((110,24))
            current=tuple(s[:max(1,size.columns-1)] for s in lines(a.root)[:max(1,size.lines-1)])
            changes=''.join(f'\x1b[{i+1};1H{s}\x1b[K' for i,s in enumerate(current) if i>=len(previous) or previous[i]!=s)
            if changes:sys.stdout.write(changes);sys.stdout.flush()
            previous=current;time.sleep(2)
    except KeyboardInterrupt:pass
    finally:sys.stdout.write('\x1b[?25h\x1b[?1049l');sys.stdout.flush()


if __name__=='__main__':main()
