"""Recover only the known action-manifest failure; never rerun generation."""
import argparse
import math
from pathlib import Path
import shutil

from . import bsu_key_reduction as experiment
from .artifacts import sha256, write_json
from .qwen35_pairs_video import freeze
from .webvid5 import fingerprint
from .webvid_ablation import Stages

FILE = 'src/semantic_transmission/bsu_key_reduction.py'
LEGACY_SHA = '3ba796a0e24d662950449193c782d55a83ca2cba6b0f5cb2a923aeae75d44a19'
FIXED_SHA = '5b53e7258aa5170f8abb71712bbaa5924e8116e58f56f49bcb93e182a48959eb'
RECOVERY = 'action_manifest_v1'
read = experiment.read


def verify_code(plan, repo):
    if fingerprint({k:v for k,v in plan.items() if k != 'signature'}) != plan['signature']:
        raise ValueError('frozen experiment signature changed')
    if plan['code'].get(FILE) != LEGACY_SHA:
        raise ValueError('recovery only supports the known failed implementation')
    for name, digest in plan['code'].items():
        if sha256(repo/name) != (FIXED_SHA if name == FILE else digest):
            raise ValueError('unrelated or unrecognized code change: '+name)


def validate_action(job, result):
    records = result['records']
    if [r['input'] for r in records] != job['rows'] or result['labels_used_for_key_selection'] is not False:
        raise ValueError('action records differ from frozen inputs')
    summaries = {}
    for row in records:
        logits, top = row['summed_logits'], row['top5']; label = row['input']['label']
        if (len(logits) != 400 or not all(math.isfinite(x) for x in logits)
                or len(top) != 5 or len(set(top)) != 5 or any(type(i) is not int or not 0 <= i < 400 for i in top)
                or [logits[i] for i in top] != sorted(logits, reverse=True)[:5]
                or row['top1_correct'] != (label == top[0]) or row['top5_correct'] != (label in top)):
            raise ValueError('invalid action prediction or accuracy')
        arm = row['input']['variant']; counts = summaries.setdefault(arm, dict(videos=0, top1_correct=0, top5_correct=0))
        counts['videos'] += 1; counts['top1_correct'] += row['top1_correct']; counts['top5_correct'] += row['top5_correct']
    if summaries != result['summary']:
        raise ValueError('action summary differs from predictions')


class CompletedOnlyStages(Stages):
    """Pre-action work must pass normal hash and dependency checks, without writes."""
    def step(self, name, owned, required, callback):
        path = self.root/'stages'/f'{name}.json'
        if not path.exists() or read(path).get('status') != 'PASSED':
            raise ValueError('recovery refuses to rerun incomplete stage: '+str(path))
        def forbidden(_):
            raise AssertionError('completed stage callback must never execute')
        return super().step(name, owned, required, forbidden)


def recover(root):
    repo = experiment.REPO; folder = root/'recovery'/RECOVERY
    protected = {}; stage_count = 0
    original_prepare, original_stages = experiment.prepare, experiment.Stages

    def remember(path, digest=None):
        protected[str(path)] = digest or sha256(path)

    def prepare(saved_root):
        if saved_root != root:
            raise ValueError('unexpected recovery root')
        p = read(root/'protocol.json'); verify_code(p, repo)
        if (p['policy'] != 'bsu_motion_ranked_reconstruction_key_removal_v1' or len(p['cases']) != 14
                or p['criteria'] != experiment.CRITERIA or p['parameters'] != experiment.PARAMETERS):
            raise ValueError('unsupported experiment cohort or decisions')
        from .lgvsc_kinetics import verify_bsu
        verify_bsu(experiment.BASE)
        for name, field in [('RESULT.json', 'source_result_sha256'), ('benchmark/protocol.json', 'source_plan_sha256')]:
            if sha256(experiment.BASE/name) != p[field]:
                raise ValueError('original BSU comparison changed')
        for case in p['cases']:
            path = experiment.BASE/'benchmark/snr_10'/case['id']/'RESULT.json'
            if sha256(path) != case['original_result_sha256']:
                raise ValueError('original case changed')
            remember(root/case['id']/'decision.json')
            for job in (root/case['id']).glob('*/job.json'):
                remember(job)
        remember(root/'protocol.json'); remember(root/'action_job.json')
        legacy = folder/'legacy_bsu_key_reduction.py'
        if not legacy.exists() or sha256(legacy) != LEGACY_SHA:
            raise ValueError('verified original source backup required')
        backup = folder/'failed_action'
        if not (folder/'receipt.json').exists():
            job = read(root/'action_job.json'); failed = read(root/'stages/action.json')
            expected = ''.join(f'{r["path"]} {r["label"]}\n' for r in job['rows'])
            if failed['status'] != 'FAILED' or (root/'action/test.csv').read_text() != expected:
                raise ValueError('failed stage is not the known space-delimited action manifest')
            for relative in ['action/test.csv', 'logs/action.log', 'stages/action.json']:
                src, dest = root/relative, backup/relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists() and sha256(dest) != sha256(src):
                    raise ValueError('failed-stage backup changed')
                if not dest.exists():
                    shutil.copyfile(src, dest)
        return p

    class RecoveryStages(CompletedOnlyStages):
        def step(self, name, owned, required, callback):
            nonlocal stage_count
            if self.root == root and name == 'action':
                from .kinetics_timesformer import CONFIG
                receipt = dict(policy=RECOVERY, original_code_sha256=LEGACY_SHA, fixed_code_sha256=FIXED_SHA,
                    recovery_code_sha256=sha256(__file__), config_sha256=sha256(CONFIG),
                    protocol_sha256=sha256(root/'protocol.json'), action_job_sha256=sha256(root/'action_job.json'),
                    completed_stages=stage_count, protected_files=protected,
                    original_failure={str(p.relative_to(folder)):sha256(p) for p in (folder/'failed_action').rglob('*') if p.is_file()},
                    reconstruction_rerun=False, thresholds_changed=False)
                freeze(folder/'receipt.json', receipt)
                experiment.event(overall='복원·화질·감축 판정 14/14 완료', stage='TimeSformer 평가 재개 0/32 (0.0%)')
                Stages.step(self, name, owned, required, callback)
                validate_action(read(root/'action_job.json'), read(root/'action/RESULT.json'))
                for path, digest in protected.items():
                    if sha256(path) != digest:
                        raise ValueError('completed artifact changed during recovery: '+path)
                freeze(folder/'verification.json', dict(status='PASSED', completed_stages=stage_count,
                    protected_files=len(protected), all_protected_hashes_unchanged=True,
                    action_result_sha256=sha256(root/'action/RESULT.json')))
                return
            super().step(name, owned, required, callback)
            stage_count += 1
            path = self.root/'stages'/f'{name}.json'; remember(path)
            for relative, digest in read(path)['artifacts'].items():
                remember(self.root/relative, digest)

    experiment.prepare, experiment.Stages = prepare, RecoveryStages
    try:
        experiment.run_experiment(root)
        result = read(root/'RESULT.json')
        result['recovery'] = dict(policy=RECOVERY, receipt=str(folder/'receipt.json'),
            receipt_sha256=sha256(folder/'receipt.json'), verification=str(folder/'verification.json'),
            verification_sha256=sha256(folder/'verification.json'), reconstructed_again=False)
        write_json(root/'RESULT.json', result)
        write_json(root/'completion_receipt.json', dict(status='PASSED', protocol_sha256=sha256(root/'protocol.json'),
            artifacts={str(p.relative_to(root)):sha256(p) for p in root.rglob('*.json')
                       if 'failed_attempts' not in p.parts and p.name != 'completion_receipt.json'}))
        print('REPORT: '+str(root/'RESULT.json'), flush=True)
    finally:
        experiment.prepare, experiment.Stages = original_prepare, original_stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=experiment.ROOT)
    args = parser.parse_args(); recover(args.root.resolve())


if __name__ == '__main__':
    main()
