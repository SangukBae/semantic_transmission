"""Opt-in dense guide before BSU's verified schedule/context/noise wrappers."""
import json
import os
from pathlib import Path
import runpy
from semantic_transmission.dense_generation_runtime import install

if __name__=='__main__':
    run=Path(os.environ['ETRI_PRECISION_RUN']);records=install(run)
    runpy.run_path(str(Path(__file__).with_name('etri_schedule_fix_decoder.py')),run_name='__main__')
    keys=json.loads((run/'keyframes.json').read_text())['indices']
    if len(records)!=len(keys)-1 or any(len(r['calls'])!=30 for r in records):
        raise ValueError('incomplete received-video guidance trace')
