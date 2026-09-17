#!/usr/bin/env python3
"""Record the venv-launch correction before the first heldout observation.

This narrowly scoped migration keeps the original declaration and all completed
artifacts. It requires an exact calibration replay in the corrected environment.
It cannot amend metrics, data, thresholds, or an experiment that has scored RGB.
"""
import argparse
import copy
from pathlib import Path
import shutil
import time

from semantic_transmission.artifacts import sha256
from semantic_transmission.metric_campaign import (REPO, campaign_lock, check_files, file_inventory,
    probe_evaluation_python, read, receipt, runtime_inventory, save, verify_receipt)

BROKEN_CONTROLLER_SHA256 = '0d00c359a30f877b90d067ea9f4a76a8a19de4eadefdc7cf4fb25cc1ac65aa2f'


def repair(root, replay):
    with campaign_lock(root):
        declaration = root / 'campaign.json'
        previous = read(declaration)
        controller = str(REPO / 'src/semantic_transmission/metric_campaign.py')
        if previous['files'].get(controller) != BROKEN_CONTROLLER_SHA256:
            raise ValueError('Only the known venv-launch bug declaration can be amended')
        # All scoring, generation, data and model/profile declarations stay frozen.
        check_files({p: value for p, value in previous['files'].items() if p != controller})
        runtime = runtime_inventory(previous['config'])
        if any(runtime.get(p) != value for p, value in previous['runtime'].items()):
            raise ValueError('Previously recorded environment changed')
        added_runtime = set(runtime) - set(previous['runtime'])
        venv = REPO / '.local/metric_v2_env'
        if any(not Path(p).is_relative_to(venv) for p in added_runtime):
            raise ValueError('Unexpected environment addition')
        state = read(root / 'state.json')
        if state['execution_status'] != 'FAILED' or state['current_task'] != '01_cross/visual':
            raise ValueError('Expected the first visual worker to have failed')
        if "ModuleNotFoundError: No module named 'sam2'" not in (root / 'logs/01_cross_visual.log').read_text():
            raise ValueError('Failure does not match the known missing-SAM2 launch error')
        if (list(root.glob('*/scores/*/*.json')) or list(root.glob('visual_cache/*/*.pkl.gz'))
                or list(root.glob('object_cache/**/*.npz'))):
            raise ValueError('Heldout observations or scores already exist; cannot apply this migration')
        expected = {receipt(root, '01_cross', name) for name in ('prepare', 'calibrate')}
        if set(root.glob('*/receipts/*.json')) != expected:
            raise ValueError('Unexpected completed stages')
        for path in expected:
            verify_receipt(path)
        old_scores = root / '01_cross/development_scores'
        new_scores = replay / '01_cross/development_scores'
        old_names = {p.name for p in old_scores.glob('*.json')}
        if len(old_names) != 600 or old_names != {p.name for p in new_scores.glob('*.json')}:
            raise ValueError('Exactly 600 replayed development rows required')
        if any(read(old_scores / name) != read(new_scores / name) for name in old_names):
            raise ValueError('Calibration replay differs from the completed results')
        calibration = root / '01_cross/calibration.json'
        if read(calibration) != read(replay / '01_cross/calibration.json'):
            raise ValueError('Replayed thresholds differ')
        interpreter = probe_evaluation_python(previous['config'])
        amendments = root / 'amendments'
        amendments.mkdir(exist_ok=True)
        backup = amendments / '0001_campaign_before_python_fix.json'
        if backup.exists() and backup.read_bytes() != declaration.read_bytes():
            raise ValueError('Existing original-declaration backup differs')
        if not backup.exists():
            backup.write_bytes(declaration.read_bytes())
        archived_replay = amendments / '0001_calibration_replay'
        replay_paths = [replay / '01_cross/calibration.json', replay / '01_cross/calibration_inputs.json',
                        *new_scores.glob('*.json')]
        archived_paths = []
        for path in replay_paths:
            destination = archived_replay / path.relative_to(replay / '01_cross')
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and sha256(destination) != sha256(path):
                raise ValueError('Existing replay archive differs')
            if not destination.exists():
                shutil.copy2(path, destination)
            archived_paths.append(destination)
        updated = copy.deepcopy(previous)
        updated['files'][controller] = sha256(controller)
        updated['runtime'] = runtime
        amendment = amendments / '0001_python_venv.json'
        save(amendment, {'kind': 'execution_environment_correction', 'created_unix': time.time(),
             'reason': 'Preserve venv bin/python path; resolving its symlink had invoked the base interpreter without SAM2',
             'previous_declaration': str(backup), 'previous_declaration_sha256': sha256(backup),
             'controller_before_sha256': BROKEN_CONTROLLER_SHA256, 'controller_after_sha256': sha256(controller),
             'repair_script_sha256': sha256(Path(__file__)), 'interpreter': interpreter,
             'added_runtime_entries': {p: runtime[p] for p in sorted(added_runtime)},
             'completed_receipts_sha256': file_inventory(expected), 'calibration_sha256': sha256(calibration),
             'replay_files': file_inventory(archived_paths),
             'replayed_development_rows_exact': 600, 'heldout_observations': 0, 'heldout_scores': 0,
             'scoring_definition_changed': False, 'thresholds_changed': False})
        updated.setdefault('execution_amendments', []).append({'path': str(amendment), 'sha256': sha256(amendment)})
        save(declaration, updated)
        print('Repaired interpreter launch; preserved prepare/calibrate:', root, flush=True)
        print('Recorded amendment:', amendment, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--calibration-replay', required=True, type=Path)
    args = parser.parse_args()
    repair(args.output.resolve(), args.calibration_replay.resolve())
