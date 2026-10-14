"""Apply the frozen key-context receiver to a snapshot of newly completed jobs.

Only completed current RGB, received keys/packets, source evaluation RGB and
receipts are copied. The original generator output and model weights stay remote.
"""
import argparse
import inspect
import json
import os
from pathlib import Path
import shutil
import statistics as st
import subprocess
import sys
import time

from . import bsu_compare_transfer as transfer
from . import bsu_receiver_compare as common
from . import bsu_key_context_experiment as parent
from .artifacts import sha256, write_json

REPO = common.REPO
ROOT = REPO/'outputs/bsu_context_extension_20261013_v1'
MODULE = 'semantic_transmission.bsu_context_extension'
ARMS = ('current', 'context_only')
CODE = (*parent.CODE, 'bsu_context_extension.py')
BOUNDARIES = ('lossless_frames', 'delivered_mp4')
inventory, digest = transfer.inventory, transfer.digest
read, event = common.read, common.event


def inventory_selected(repo, benchmark, cohort):
    """Runs read-only on the remote using the standard-library inventory source."""
    merged = None
    for selection in cohort:
        item = inventory(repo, benchmark, [selection['id']], selection['snrs'])
        remove = set()
        for source in item['sources']:
            remove.add(source['movie']); source['movie'] = None
        for case in item['cases']:
            remove.update(case['raw_frames']); remove.add(case['base_movie'])
            case['raw_frames'] = []; case['base_movie'] = None
            result = json.loads((Path(repo)/case['result_path']).read_text())
            for boundary in ('lossless_frames', 'delivered_mp4'):
                relative = str(Path(case['result_path']).parent/f'refined/quality_{boundary}.csv')
                path = Path(repo)/relative
                expected = result['artifacts'][f'refined/quality_{boundary}.csv']
                if path.is_symlink() or not path.resolve().is_relative_to(Path(repo).resolve()):
                    raise ValueError('unsafe metric path')
                if digest(path) != expected:
                    raise ValueError('metric CSV receipt mismatch')
                item['files'][relative] = dict(sha256=expected, size=path.stat().st_size)
        for name in remove:
            item['files'].pop(name)
        if merged is None:
            merged = item
        else:
            if merged['benchmark_signature'] != item['benchmark_signature']:
                raise ValueError('remote benchmark changed during inventory')
            for name, value in item['files'].items():
                if name in merged['files'] and merged['files'][name] != value:
                    raise ValueError('conflicting source receipts')
            merged['files'].update(item['files'])
            merged['cases'].extend(item['cases']); merged['sources'].extend(item['sources'])
            merged['ids'].extend(item['ids'])
    if merged is None:
        raise ValueError('empty completed cohort')
    merged['snrs'] = sorted({c['snr'] for c in merged['cases']})
    merged['selection'] = cohort
    merged['includes_raw_generation'] = False
    merged['evaluator_code'] = {name:digest(Path(repo)/'src/semantic_transmission'/name)
        for name in ('official_quality.py', 'research_quality.py', 'stutter_metrics.py')}
    return merged


def transport(control):
    if control:
        transfer.ssh_args = lambda host: ['ssh', '-S', control, '-o', 'ControlMaster=auto',
            '-o', 'ControlPersist=600', '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=yes']


def obtain_manifest(args):
    path = args.root/'transfer_manifest.json'
    if path.exists():
        m = transfer.validate_manifest(read(path))
        if m['selection'] != read(args.root/'cohort.json')['videos']:
            raise ValueError('frozen cohort changed')
        return m
    if args.offline:
        raise ValueError('offline run requires a frozen transfer manifest')
    cohort = read(args.root/'cohort.json')['videos']
    source = Path(transfer.__file__).read_text()+'\n'+inspect.getsource(inventory_selected)
    source += '\nprint(json.dumps(inventory_selected(**'+repr(dict(repo=args.remote_repo,
        benchmark=args.remote_root, cohort=cohort))+')))\n'
    event(stage='완료 산출물 목록과 원격 영수증 검사 (진행률 미확정)')
    proc = subprocess.run([*transfer.ssh_args(args.host), '-o', 'BatchMode=yes', args.host, 'python3 -'],
        input=source, text=True, capture_output=True, check=True)
    m = json.loads(proc.stdout); m['host'] = args.host
    m['signature'] = transfer.fingerprint(m)
    transfer.validate_manifest(m); write_json(path, m)
    return m


def prepare(args, m):
    from .receiver_key_context import PARAMETERS
    old_path = args.parent/'protocol.json'; old = read(old_path)
    for name, expected in old['code'].items():
        if sha256(REPO/'src/semantic_transmission'/name) != expected:
            raise ValueError('previous receiver/evaluator code changed: '+name)
    for name, expected in m['evaluator_code'].items():
        if sha256(REPO/'src/semantic_transmission'/name) != expected:
            raise ValueError('remote evaluator differs: '+name)
    old_sources = {c['source_sha256'] for c in old['cases']}
    if old_sources & {s['source_sha256'] for s in m['sources']}:
        raise ValueError('extension overlaps receiver development sources')
    if len({s['source_sha256'] for s in m['sources']}) != len(m['sources']):
        raise ValueError('duplicate source identities')
    runtime = common.runtime_identity(args.device)
    if runtime != old['runtime']:
        raise ValueError('local evaluation runtime differs from frozen experiment')
    mirror = args.root/'inputs'; cases = []
    for c in m['cases']:
        source = next(s for s in m['sources'] if s['id'] == c['id'])
        result = read(mirror/c['result_path'])
        # The remote refinement stores flat quality summaries. The evaluation
        # helper expects the comparison schema, so retain a derived donor wrapper.
        donor_quality = args.root/'donors'/common.case_name(c)/'quality.json'
        value = {b:dict(quality=result['refined_quality'][b]) for b in BOUNDARIES}
        if donor_quality.exists() and read(donor_quality) != value:
            raise ValueError('donor quality changed')
        if not donor_quality.exists(): write_json(donor_quality, value)
        received = common.receiver_inputs(mirror, c)
        received.update(current_frames=[str(mirror/p) for p in c['current_frames']],
            packet_format='separate_trajectory', guard_frames=None)
        donor = dict(frames_paths=received['current_frames'], movie=str(mirror/c['current_movie']),
            quality=str(donor_quality), layout='comparison',
            scores={b:str(mirror/Path(c['result_path']).parent/f'refined/quality_{b}.csv') for b in BOUNDARIES})
        cases.append(dict(name=common.case_name(c), id=c['id'], snr=c['snr'], frames=c['frames'], keys=c['keys'],
            source_sha256=source['source_sha256'], source_frames=[str(mirror/p) for p in source['frames_paths']],
            source_movie=None, receiver=received, donor=donor,
            channel_uses=c['base_channel_uses']+c['trajectory_channel_uses']))
    inputs = {str(mirror/n):v['sha256'] for n,v in m['files'].items()}
    for path in [old_path, args.root/'transfer_manifest.json', args.root/'cohort.json',
                 *[Path(c['donor']['quality']) for c in cases]]:
        inputs[str(path)] = sha256(path)
    p = dict(policy='frozen_key_context_completed_extension_v1', arms=list(ARMS), cases=cases,
        previous_signature=old['signature'], parameters=PARAMETERS, runtime=runtime, inputs=inputs,
        code={n:sha256(REPO/'src/semantic_transmission'/n) for n in CODE},
        primary_boundary='delivered_mp4', primary_cohort='sources with all snapshot SNRs',
        aggregation='SNR mean within source, then equal source mean; partial SNR sources separate',
        receiver_reads_source=False, generator_reused=True, channel_reused=True, additional_transmission=0,
        tuning='Frozen context_only; no path/residual/gate changes and no per-video tuning',
        scope='New to this receiver comparison by source hash, not proof of globally unseen data; completion-selected cohort',
        independent_semantic_review='PENDING', automatically_adopted=False)
    p['signature'] = transfer.fingerprint(p)
    path = args.root/'protocol.json'
    if path.exists() and read(path) != p:
        raise ValueError('frozen inputs/code/runtime changed; choose a new root')
    if not path.exists():
        write_json(path, p)
        for n in CODE:
            target = args.root/'frozen_code'/n; target.parent.mkdir(exist_ok=True)
            shutil.copyfile(REPO/'src/semantic_transmission'/n, target)
    return p


def compact_metrics(q, boundary):
    b = q[boundary]
    return dict(**{k:b['quality'][k] for k in parent.METRICS},
        **{k:b['temporal'][k] for k in ('flow_error_mean', 'cadence_error_mean', 'false_freeze_rate')},
        nonkey_psnr=b['partitions']['nonkeys']['psnr_db'],
        boundary_error=st.mean(v['maximum_change_error'] for v in b['key_boundary_changes']))


def aggregate(rows, cases):
    """Never count SNR replications as independent sources or mix partial pairs."""
    snrs = sorted({c['snr'] for c in cases}); sources = []
    for vid in sorted({c['id'] for c in cases}):
        expected = [c for c in cases if c['id'] == vid]
        selected = [r for r in rows if r['id'] == vid]
        if len(selected) != len(expected)*len(ARMS): continue
        out = dict(id=vid, snrs=sorted(c['snr'] for c in expected), boundaries={})
        out['complete_snr_set'] = out['snrs'] == snrs
        for b in BOUNDARIES:
            means = {a:{k:st.mean(r[b][k] for r in selected if r['arm'] == a)
                for k in selected[0][b]} for a in ARMS}
            out['boundaries'][b] = dict(means=means, delta={k:means['context_only'][k]-v for k,v in means['current'].items()})
        sources.append(out)
    groups = []
    for complete, label in ((True,'complete_snr_sources'),(False,'partial_snr_sources')):
        subset = [s for s in sources if s['complete_snr_set'] == complete]
        if not subset: continue
        for b in BOUNDARIES:
            means = {a:{k:st.mean(s['boundaries'][b]['means'][a][k] for s in subset)
                for k in subset[0]['boundaries'][b]['means'][a]} for a in ARMS}
            groups.append(dict(cohort=label,boundary=b,source_count=len(subset),means=means,
                delta={k:means['context_only'][k]-v for k,v in means['current'].items()},
                relative_percent={k:100*(means['context_only'][k]/v-1) if v else None for k,v in means['current'].items()}))
    return sources, groups


def report(root, p, verify=False):
    rows=[]; pending=[]
    for c in p['cases']:
        for arm in ARMS:
            folder=root/'comparison'/c['name']/arm
            ready = (common.verify_receipt(folder,'evaluation',p['signature']) if verify else (folder/'evaluation.json').exists())
            if not ready: pending.append([c['name'],arm]); continue
            if read(folder/'evaluation.json')['signature'] != p['signature']: raise ValueError('evaluation signature differs')
            q=read(folder/'quality.json')
            rows.append(dict(id=c['id'],case=c['name'],snr=c['snr'],arm=arm,channel_uses=c['channel_uses'],
                **{b:compact_metrics(q,b) for b in BOUNDARIES},
                keys_exact=sum(k['exact_received_rgb'] for k in q['lossless_frames']['key_fidelity']),key_count=len(c['keys'])))
    sources,groups=aggregate(rows,p['cases'])
    value=dict(status='COMPLETE_QUANTITATIVE_REVIEW_PENDING' if not pending else 'PARTIAL',signature=p['signature'],
        completed=len(rows),total=len(p['cases'])*len(ARMS),rows=rows,pending=pending,sources=sources,groups=groups,
        independent_semantic_review='PENDING',automatically_adopted=False)
    write_json(root/'RESULT.json',value)
    return value


def worker(stage,job,folder,p,arm,device):
    subprocess.run([sys.executable,'-m',parent.MODULE,'--worker',stage,'--job',str(job),'--output',str(folder),
        '--signature',p['signature'],'--arm',arm,'--device',device],check=True)


def run(args):
    args.root=args.root.resolve();args.parent=args.parent.resolve();transport(args.ssh_control)
    with common.lock(args.root):
        m=obtain_manifest(args)
        transfer_record=transfer.fetch(m,args.root/'inputs',common.tick('입력 전송·해시 검증'),offline=args.offline)
        if not (args.root/'transfer_result.json').exists():write_json(args.root/'transfer_result.json',transfer_record)
        p=prepare(args,m);r=report(args.root,p,verify=True)
        if args.action=='check':print(f'CHECK: {len(m["sources"])} sources, {len(p["cases"])} cases, {r["total"]} arms');return
        started=time.monotonic();done=r['completed'];total=r['total']
        try:
            for c in p['cases']:
                for arm in ARMS:
                    folder=args.root/'comparison'/c['name']/arm;folder.mkdir(parents=True,exist_ok=True)
                    event(overall=f'{done}/{total} ({100*done/total:.1f}%) 평가 완료',current=f'{c["name"]} / {arm}',state='실행 중')
                    if not common.verify_receipt(folder,'generation',p['signature']):
                        if arm=='current':
                            info=folder/'generation_info.json';write_json(info,dict(arm=arm,**c['donor'],imported=True))
                            common.receipt(folder,'generation',p['signature'],[info])
                        else:
                            job=args.root/'jobs'/c['name']/'receiver.json';write_json(job,c['receiver'])
                            worker('generate',job,folder,p,arm,args.device)
                    if common.verify_receipt(folder,'evaluation',p['signature']):continue
                    info=read(folder/'generation_info.json')
                    job=args.root/'jobs'/c['name']/(arm+'_evaluation.json')
                    write_json(job,dict(arm=arm,source_frames=c['source_frames'],source_movie=None,frames=c['frames'],
                        keys=c['keys'],key_paths=c['receiver']['key_paths'],frames_paths=info['frames_paths'],movie=info['movie'],donor=c['donor']))
                    worker('evaluate',job,folder,p,arm,args.device)
                    done+=1
                    report(args.root,p)
            result=report(args.root,p,verify=True)
            if result['pending']:raise ValueError('evaluation incomplete')
            if not (args.root/'execution.json').exists():write_json(args.root/'execution.json',dict(seconds=time.monotonic()-started,completed=done))
            event(overall=f'{done}/{total} (100.0%) 평가 완료',stage='입력·출력 검증과 집계 완료',state='정량 완료 / 의미 검수 대기')
            print('REPORT: '+str(args.root/'RESULT.json'),flush=True)
        except BaseException as error:
            write_json(args.root/'last_failure.json',dict(error=str(error),type=type(error).__name__,completed=done,total=total,resume='repeat command'))
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',nargs='?',default='run',choices=('run','check'))
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--parent',type=Path,default=parent.ROOT)
    parser.add_argument('--host',default='sangukbae@155.230.15.77')
    parser.add_argument('--remote-repo',default='/home/sangukbae/ETRI/Semantic/semantic_transmission')
    parser.add_argument('--remote-root',default='outputs/bsu_webvid_paper')
    parser.add_argument('--ssh-control');parser.add_argument('--offline',action='store_true')
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--driver',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args();transport(args.ssh_control)
    if args.driver:run(args);return
    manifest=args.root/'transfer_manifest.json'
    need_connection=not manifest.exists() or bool(transfer.inspect_files(transfer.validate_manifest(read(manifest)),args.root/'inputs'))
    if need_connection and not args.offline:transfer.authenticate(args.host)
    from .terminal_progress import run as display
    raise SystemExit(display([sys.executable,'-m',MODULE,*sys.argv[1:],'--driver'],
        REPO/'.local/terminal_logs/bsu_context_extension',os.environ.get('BSU_PROGRESS','auto')))


if __name__=='__main__':main()
