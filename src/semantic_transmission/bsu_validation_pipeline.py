"""Run the four opt-in stages sequentially, or verify a completed experiment."""
import argparse
from pathlib import Path
import subprocess
import sys
from . import bsu_motion_validation as common


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=common.ROOT)
    root=parser.parse_args().root
    if (root/'RESULT.json').exists():
        if common.read(root/'RESULT.json')['status']!='COMPLETE_FOUR_STAGE_LOCAL_VALIDATION':raise ValueError('unexpected completion report')
        from .bsu_motion_report import verify
        verify(root);print(f'All stages already complete; verified reuse: {root}/report.html',flush=True);return
    for module,stage in [('bsu_motion_validation','flow'),('bsu_motion_validation','flow_reconstruction'),
        ('bsu_budget_validation','run'),('bsu_tracking_validation','run'),('bsu_adapter_validation','run'),
        ('bsu_adapter_native','run'),('bsu_motion_report','audit'),('bsu_motion_report','summarize')]:
        subprocess.run([sys.executable,'-m',__package__+'.'+module,stage,'--root',str(root)],check=True)


if __name__=='__main__':main()
