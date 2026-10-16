"""Finish an already reconstructed benchmark after the config-path validation fix.

Historical protocol/job/stage receipts stay immutable. Only the exact known
action-validator replacement is accepted; all reconstruction evidence is audited.
"""
import argparse
from pathlib import Path
import signal
import time

from . import lgvsc_kinetics as run, lgvsc_kinetics_action as action
from .artifacts import sha256, write_json
from .etri_60s_check import lock
from .kinetics_assets import event
from .qwen35_pairs_video import freeze
from .webvid5 import fingerprint
from .webvid_ablation import environment, snapshot

ACTION_FILE = 'src/semantic_transmission/lgvsc_kinetics_action.py'
LEGACY_SHA = 'eb69bc456df1181695a662ffb7ab5fda633b55b4e1adf4492011c57718b6f5ff'
FIXED_SHA = '4793bffbcd63fd60eca58763377723d1893e40177bf25ad4e2583f5221b7a899'
RECOVERY = 'action_config_path_v1'
read = run.read


def verify_plan(root):
    p = read(root/'protocol.json'); action.base.verify_signature(p)
    if p['policy'] != run.POLICY or p['root'] != str(root) or len(p['videos']) != 14:
        raise ValueError('unexpected frozen comparison root/policy/cohort')
    if p['code'].get(ACTION_FILE) != LEGACY_SHA or sha256(run.REPO/ACTION_FILE) != FIXED_SHA:
        raise ValueError('recovery requires the exact legacy-to-fixed action validator')
    for path, digest in {**p['inputs'], **p['checkpoints']}.items():
        if sha256(path) != digest: raise ValueError('frozen input changed: ' + path)
    for path, digest in p['code'].items():
        if sha256(run.REPO/path) != (FIXED_SHA if path == ACTION_FILE else digest):
            raise ValueError('unrelated frozen code changed: ' + path)
    for path, entry in p['model_files'].items():
        stat = Path(path).stat()
        if (stat.st_size, stat.st_mtime_ns) != (entry['size'], entry['mtime_ns']):
            raise ValueError('model file changed: ' + path)
    return p


def completed_results(root, p):
    results, receipts = [], {}
    for n, video in enumerate(p['videos']):
        event(current=video['id'], stage=f'완료 산출물 해시 검증 {n}/14 ({100*n/14:.1f}%)')
        dest = root/'videos'/video['id']; completed = {}
        stages = [('initialize', run.initialization_files(video['frames']))]
        stages += [(name, ['run/'+x for x in products]+[f'resources/{name}.json']+
                    (['execution_protocol.json'] if name == 'prepare-text' else []))
                   for name, products in run.jobs()]
        for name, required in stages:
            path = dest/'stages'/f'{name}.json'; saved = read(path)
            if (saved['status'] != 'PASSED' or saved['identity'] != p['signature']
                    or saved['dependencies'] != {k:fingerprint(v) for k,v in completed.items()}
                    or saved['required'] != required
                    or snapshot(dest, required) != saved['artifacts']):
                raise ValueError(f'completed stage changed or incomplete: {video["id"]}/{name}')
            completed[name] = saved; receipts[str(path)] = sha256(path)
        audit = read(dest/'run/output_audit.json'); quality = read(dest/'run/quality.json')
        if (audit['status'] != 'PASSED' or quality['status'] != 'PASSED'
                or audit['video_sha256'] != quality['video_sha256']
                or sha256(audit['video']) != audit['video_sha256']
                or any(quality[b]['frames'] != video['frames'] for b in ('lossless_frames','delivered_mp4'))):
            raise ValueError('reconstruction/quality mismatch: ' + video['id'])
        result = dict(status='COMPLETE_RECONSTRUCTION_QUALITY', signature=p['signature'], id=video['id'],
            frames=video['frames'], video=audit['video'], quality=quality,
            rate=read(dest/'run/rate.json'), video_sha256=audit['video_sha256'])
        if read(dest/'RESULT.json') != result: raise ValueError('completed result changed: ' + video['id'])
        receipts[str(dest/'RESULT.json')] = sha256(dest/'RESULT.json'); results.append(result)
    return results, receipts


def prepare(root):
    p = verify_plan(root); results, receipts = completed_results(root, p)
    job = action.make_job(p, results)
    old_path = root/'action_job.json'; old = read(old_path); action.base.verify_signature(old)
    # The old job must match exactly except for the single repaired code hash.
    expected = dict(job, code=dict(job['code'], **{str(run.REPO/ACTION_FILE):LEGACY_SHA}))
    expected = action.base.signed({k:v for k,v in expected.items() if k != 'signature'})
    if old != expected: raise ValueError('historical action job differs beyond the known validator repair')
    rate = read(root/'RATE_RESULT.json')
    if rate != dict(status='COMPLETE_RATE_COMPARISON', signature=p['signature'],
                    summary=run.aggregate_rate(p, [r['rate'] for r in results]),
                    videos=[r['rate'] for r in results], scope=p['comparison']):
        raise ValueError('completed transmission comparison changed')
    folder = root/'recovery'/RECOVERY
    job = action.base.signed(dict({k:v for k,v in job.items() if k != 'signature'}, output=str(folder/'action')))
    action.validate_job(job); action.base.validate_runtime(job['runtime'])
    receipt = action.base.signed(dict(recovery=RECOVERY, protocol_sha256=sha256(root/'protocol.json'),
        original_action_job_sha256=sha256(old_path), rate_result_sha256=sha256(root/'RATE_RESULT.json'),
        original_action_code_sha256=LEGACY_SHA, fixed_action_code_sha256=FIXED_SHA,
        recovery_code_sha256=sha256(__file__), completed_receipts=receipts,
        new_action_job_signature=job['signature'], reconstructed_again=False,
        config_paths=[job['config'], str(action.base.CONFIG)], config_sha256=job['config_sha256']))
    return p, results, job, receipt, folder


def recover(root, check=False):
    # Holding both existing locks also prevents concurrent generation/recovery.
    with lock(root/'execution.lock'), lock(run.REPO/'.local/etri_60s_check.lock'):
        p, results, job, receipt, folder = prepare(root)
        if check:
            print('CHECK: READY_ACTION_RECOVERY (14 verified reconstructions; no inference)', flush=True)
            return
        freeze(folder/'receipt.json', receipt); freeze(folder/'action_job.json', job)
        before = read(root/'status.json')
        if not (folder/'status_before.json').exists(): freeze(folder/'status_before.json', before)
        state = dict(status='RUNNING', total=14, transmitted=14, reconstructed=14, action_evaluated=0,
                     mode='action-recovery', recovery=RECOVERY)
        write_json(root/'status.json', state)
        try:
            env = dict(environment(2025), PYTHONPATH=str(run.REPO/'src'))
            run.batch.command([job['runtime']['python'], '-m', 'semantic_transmission.lgvsc_kinetics_action',
                '--job', folder/'action_job.json'], root/f'logs/action_recovery_{time.time_ns()}.log', env)
            evaluated = action.validate_evaluation(job); old = read(Path(p['bsu_root'])/'RESULT.json')
            quality = run.quality_summary(results)
            result = dict(status='COMPLETE_RATE_QUALITY_ACTION_COMPARISON', signature=p['signature'],
                rate=run.aggregate_rate(p, [r['rate'] for r in results]),
                quality=dict(lgvsc=quality, bsu=old['quality']['refined']),
                quality_difference_bsu_minus_lgvsc={b:{k:old['quality']['refined'][b]['video_mean'][k]-quality[b]['video_mean'][k]
                    for k in run.batch.KEYS} for b in quality},
                action=dict(old['action'], lgvsc=evaluated['summary']), videos=results, scope=p['comparison'],
                independent_semantic_review='PENDING',
                action_recovery=dict(receipt=str(folder/'receipt.json'), receipt_sha256=sha256(folder/'receipt.json'),
                    job=str(folder/'action_job.json'), job_sha256=sha256(folder/'action_job.json'),
                    result=str(folder/'action/RESULT.json'), result_sha256=sha256(folder/'action/RESULT.json')))
            freeze(root/'RESULT.json', result)
            state.update(status='COMPLETE', action_evaluated=14)
            event(overall='LGVSC 전송·복원·화질·행동평가 14/14 (100.0%)', stage='비교 결과 저장 완료', state='완료')
            print('REPORT: ' + str(root/'RESULT.json'), flush=True)
            return result
        except BaseException as error:
            state.update(status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED', error=str(error))
            raise
        finally:
            write_json(root/'status.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=run.ROOT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    recover(args.root.resolve(), args.check)


if __name__ == '__main__': main()
