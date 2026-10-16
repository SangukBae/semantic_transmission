"""Opt-in adapter before the unchanged BSU context, collision and noise hooks."""
import os
from pathlib import Path
import runpy
from semantic_transmission.trajectory_adapter_runtime import install

if __name__=='__main__':
    run=Path(os.environ['ETRI_PRECISION_RUN'])
    records=install(run)
    runpy.run_path(str(Path(__file__).with_name('etri_schedule_fix_decoder.py')),run_name='__main__')
    from semantic_transmission.bsu_motion_validation import read
    if len(records)!=len(read(run/'keyframes.json')['indices'])-1:raise ValueError('incomplete adapter trace')
