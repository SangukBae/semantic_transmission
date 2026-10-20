"""Frozen packet-size, exact restoration and actual Sionna channel validation."""
import argparse
import base64
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from . import bsu_metadata_compression as compression
from .packets import accounting
from .wire import unpack

REPO = Path(__file__).resolve().parents[2]
DEFAULT = REPO/'outputs/bsu_metadata_compression_20261019'
BUNDLE = REPO/'.local/validation/bsu_metadata_compression_20261019/remote_packets.json'
COMPLETE = 'COMPLETE_BSU_METADATA_COMPRESSION_VALIDATION'
METHODS = ('raw','pack5','zlib','pack5_zlib','auto')


def read(path):
    return json.loads(Path(path).read_text())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    return digest(Path(path).read_bytes())


def identity(value):
    return digest(json.dumps(value, sort_keys=True, separators=(',',':')).encode())


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temp.replace(path)


def freeze_bytes(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=data: raise ValueError(f'frozen file changed: {path}')
    else:
        path.write_bytes(data)


def sources():
    bundle=read(BUNDLE)
    inputs={str(BUNDLE):file_digest(BUNDLE)}
    rows=[]
    for item in bundle['records']:
        packet=base64.b64decode(item['packet_b64'],validate=True)
        trajectory=base64.b64decode(item['trajectory_b64'],validate=True)
        assert digest(packet)==item['packet_sha256']
        assert digest(trajectory)==item['trajectory_sha256']
        rows.append(dict(cohort='remote14',id=item['id'],packet=packet,trajectory=trajectory,
            visual_uses=item['visual_uses'], recorded_base_uses=item['channel']['total_complex_channel_uses'],
            recorded_trajectory_uses=item['trajectory_channel'].get('complex_channel_uses',0),
            provenance=dict(remote_root=item['source_root'],result_sha256=item['source_result_sha256'])))
    local=REPO/'outputs/bsu_key_budget_pilot_20261019/RESULT.json'
    recorded=read(REPO/'docs/validation/2026-10-19-bsu-key-budget-pilot.json')
    assert file_digest(local)==recorded['evidence_sha256'][str(local.relative_to(REPO))]
    inputs[str(local)]=file_digest(local)
    for item in read(local)['records']:
        transport=Path(item['transport']);run=transport/'run'
        packet=(run/'transmitter/metadata.bin').read_bytes()
        trajectory=(transport/'trajectory_sender/trajectory.bin').read_bytes()
        assert packet==(run/'received/metadata.bin').read_bytes()
        assert trajectory==(transport/'trajectory_received/trajectory.bin').read_bytes()
        for path in (run/'transmitter/metadata.bin',transport/'trajectory_sender/trajectory.bin'):
            inputs[str(path)]=file_digest(path)
        channels=read(run/'channel_accounting.json')
        assert digest(packet)==channels['received_files']['metadata.bin']['sha256']
        assert (run/'transmitter/visual.c64').stat().st_size==channels['visual_complex_channel_uses']*8
        rows.append(dict(cohort='local6',id=item['id']+'_'+item['arm'],packet=packet,trajectory=trajectory,
            visual_uses=channels['visual_complex_channel_uses'],recorded_base_uses=item['base_uses'],
            recorded_trajectory_uses=item['trajectory_uses'],provenance=dict(run=str(run),arm=item['arm'])))
    assert len(rows)==20
    return rows,inputs


def prepare(root):
    rows,inputs=sources()
    paths=[Path(__file__),Path(compression.__file__),REPO/'src/semantic_transmission/wire.py',
        REPO/'src/semantic_transmission/packets.py',REPO/'src/semantic_transmission/metadata_channel.py',
        REPO/'scripts/validate_bsu_metadata_compression.sh']
    code={str(p):file_digest(p) for p in paths}
    records=[];jobs=[]
    for row in rows:
        packet=row['packet'];trajectory=row['trajectory']
        _,rates=compression.validate_original(packet)
        prefix=Path('packets')/row['cohort']/row['id']
        freeze_bytes(root/prefix/'original.bin',packet)
        freeze_bytes(root/prefix/'trajectory.bin',trajectory)
        tc=accounting(len(trajectory))['complex_channel_uses'] if trajectory else 0
        assert tc==row['recorded_trajectory_uses']
        assert row['visual_uses']+accounting(len(packet))['complex_channel_uses']==row['recorded_base_uses']
        if trajectory:
            assert unpack(trajectory)[0]['base_metadata_sha256']==digest(packet)
        choices=compression.variants(packet);choices['auto']=compression.encode(packet)
        methods={}
        for method,encoded in choices.items():
            assert compression.decode(encoded)==packet
            path=prefix/(method+'.bin');freeze_bytes(root/path,encoded)
            cost=accounting(len(encoded))
            methods[method]=dict(path=str(path),sha256=digest(encoded),packet_bytes=len(encoded),
                digital_uses=cost['complex_channel_uses'],blocks=cost['blocks'],padding_bits=cost['padding_bits'],
                total_uses=row['visual_uses']+tc+cost['complex_channel_uses'],selected=compression.method_name(encoded))
        records.append({k:row[k] for k in ('cohort','id','visual_uses','provenance')} | dict(
            trajectory_uses=tc,trajectory_path=str(prefix/'trajectory.bin'),trajectory_sha256=digest(trajectory),
            original_path=str(prefix/'original.bin'),original_sha256=digest(packet),
            rate_indices=len(rates),keys=len(unpack(packet)[0]['keyframes']),methods=methods))
        for snr,seed,methods_to_run in ((10,42,('raw','pack5','zlib','pack5_zlib')),
                (10,2025,('raw','auto')),(8,42,('raw','auto')),(8,2025,('raw','auto'))):
            for method in methods_to_run:
                jobs.append(dict(cohort=row['cohort'],id=row['id'],method=method,snr_db=snr,seed=seed,
                    source_path=methods[method]['path'],source_sha256=methods[method]['sha256'],
                    original_sha256=digest(packet)))
    plan=dict(version=1,code=code,inputs=inputs,records=records,jobs=jobs,
        primary_snr_db=10,diagnostic_snr_db=8,seeds=[42,2025],
        channel=dict(k=6144,n=9216,bits_per_symbol=4),
        scope=['20 packets from 14 remote videos and six local short-clip policies; overlapping sources',
            'visual symbols and trajectory transport are retained at their existing measured costs',
            'compression restores existing packet bytes before decoder and trajectory checks',
            'no neural inference, new video quality evaluation or default adoption',
            '8dB is a small reliability diagnostic, not a packet-error-rate benchmark'])
    plan['signature']=identity(plan)
    encoded=(json.dumps(plan,ensure_ascii=False,indent=2)+'\n').encode()
    freeze_bytes(root/'protocol.json',encoded)
    write(root/'RATE_REPORT.json',aggregate(plan))
    return plan


def aggregate(plan):
    summary={}
    for cohort in ('remote14','local6'):
        records=[r for r in plan['records'] if r['cohort']==cohort]
        if not records: continue
        methods={}
        baseline=sum(r['methods']['raw']['total_uses'] for r in records)
        for method in METHODS:
            totals={k:sum(r['methods'][method][k] for r in records) for k in ('packet_bytes','digital_uses','total_uses','blocks')}
            totals['total_saving_percent']=100*(1-totals['total_uses']/baseline)
            totals['packets_with_reduced_channel_uses']=sum(r['methods'][method]['total_uses']<r['methods']['raw']['total_uses'] for r in records)
            methods[method]=totals
        summary[cohort]=dict(packets=len(records),rate_indices=sum(r['rate_indices'] for r in records),
            methods=methods,per_packet=[dict(id=r['id'],methods=r['methods']) for r in records])
    return summary


def verify(plan,root):
    if identity({k:v for k,v in plan.items() if k!='signature'})!=plan['signature']:
        raise ValueError('protocol signature changed')
    for group in ('code','inputs'):
        for path,expected in plan[group].items():
            if file_digest(path)!=expected: raise ValueError(f'frozen {group} changed: {path}')
    for record in plan['records']:
        for path_key,hash_key in (('original_path','original_sha256'),('trajectory_path','trajectory_sha256')):
            if file_digest(root/record[path_key])!=record[hash_key]: raise ValueError('frozen original/trajectory changed')
        for method in record['methods'].values():
            if file_digest(root/method['path'])!=method['sha256']: raise ValueError('frozen candidate changed')


def run(root):
    from .metadata_channel import transmit
    import fcntl
    root.mkdir(parents=True,exist_ok=True)
    with (root/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=read(root/'protocol.json') if (root/'protocol.json').exists() else prepare(root)
        verify(plan,root)
        if (root/'RESULT.json').exists():
            completed=read(root/'RESULT.json')
            if completed['status']!=COMPLETE or completed['signature']!=plan['signature']:
                raise ValueError('completed result identity changed')
            for path,expected in completed['artifacts'].items():
                if file_digest(root/path)!=expected: raise ValueError(f'completed artifact changed: {path}')
            print('REUSE complete result; frozen inputs and all artifact hashes verified',flush=True)
            return
        state=read(root/'status.json') if (root/'status.json').exists() else {}
        state=dict(status='RUNNING',stage='actual channel transmission',started_epoch=state.get('started_epoch',time.time()),
            total=len(plan['jobs']),done=0,errors=0,pid=os.getpid(),process_start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21])
        records=[]
        write(root/'status.json',state)
        try:
            for index,job in enumerate(plan['jobs']):
                receipt=root/'channel'/f'{index:04d}.json'
                target=receipt.with_suffix('.bin')
                if receipt.exists():
                    item=read(receipt)
                    if item['job']!=job or item['signature']!=plan['signature'] or file_digest(target)!=item['received_sha256']:
                        raise ValueError('completed channel receipt/input changed')
                else:
                    packet=(root/job['source_path']).read_bytes();started=time.monotonic()
                    received,cost=transmit(packet,job['snr_db'],job['seed'])
                    expected=accounting(len(packet))
                    if any(cost[k]!=v for k,v in expected.items()): raise ValueError('actual channel cost mismatch')
                    try:
                        restored=compression.decode(received)
                        accepted=True;exact=digest(restored)==job['original_sha256'];error=None
                    except (ValueError,UnicodeError) as e:
                        accepted=False;exact=False;error=str(e)
                    item=dict(job=job,signature=plan['signature'],channel=cost,
                        transmitted_exact=received==packet,restored_exact=exact,accepted=accepted,error=error,
                        seconds=time.monotonic()-started,received_sha256=digest(received))
                    freeze_bytes(target,received);write(receipt,item)
                    print(f'CHANNEL {index+1}/{len(plan["jobs"])} {job["cohort"]}/{job["id"]} {job["method"]} SNR={job["snr_db"]} seed={job["seed"]} restored={exact}',flush=True)
                records.append(item)
                if (item['accepted'] and not item['restored_exact']) or (job['snr_db']==10 and not item['restored_exact']):
                    state['errors']+=1
                state['done']=index+1;write(root/'status.json',state)
                if index%10==0: gc.collect()
            channel_summary=[]
            for cohort in ('remote14','local6'):
                for snr in (10,8):
                    for method in ('raw','pack5','zlib','pack5_zlib','auto'):
                        group=[r for r in records if r['job']['cohort']==cohort and r['job']['snr_db']==snr and r['job']['method']==method]
                        if group:
                            channel_summary.append(dict(cohort=cohort,snr_db=snr,method=method,cases=len(group),
                                restored_exact=sum(r['restored_exact'] for r in group),
                                rejected=sum(not r['accepted'] for r in group),
                                undetected_errors=sum(r['accepted'] and not r['restored_exact'] for r in group),
                                bit_errors=sum(r['channel']['bit_errors'] for r in group)))
            if state['errors']: raise ValueError(f'{state["errors"]} primary or undetected failures')
            state.update(status=COMPLETE,stage='complete',ended_epoch=time.time())
            result=dict(status=COMPLETE,signature=plan['signature'],rate=aggregate(plan),channel=channel_summary,
                channel_cases=len(records),elapsed_seconds=state['ended_epoch']-state['started_epoch'],
                original_metadata_restored_before_decoder=True,trajectory_binding_unchanged=True,
                neural_inference_executed=False,quality_remeasured=False,default_adopted=False,
                physical_link_overhead_included=False,scope=plan['scope'],
                artifacts={str(path.relative_to(root)):file_digest(path) for path in
                    [root/'protocol.json',root/'RATE_REPORT.json',*sorted((root/'channel').glob('*')),*sorted((root/'packets').rglob('*.bin'))]})
            write(root/'RESULT.json',result);write(root/'status.json',state)
        except BaseException as error:
            state.update(status='INTERRUPTED' if isinstance(error,KeyboardInterrupt) else 'FAILED',
                error=f'{type(error).__name__}: {error}',ended_epoch=time.time())
            write(root/'status.json',state);raise


def lines(root):
    state=read(root/'status.json') if (root/'status.json').exists() else {}
    status=state.get('status','PLANNED' if (root/'protocol.json').exists() else 'NOT_STARTED')
    if status=='RUNNING':
        try:
            if Path(f'/proc/{state["pid"]}/stat').read_text().split()[21]!=state['process_start_ticks']:
                status='STOPPED_WITHOUT_FINAL_RESULT'
        except OSError: status='STOPPED_WITHOUT_FINAL_RESULT'
    elapsed=max(0,state.get('ended_epoch',time.time())-state.get('started_epoch',time.time()))
    done,total=state.get('done',0),state.get('total',0)
    gpu='unavailable'
    try:
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader'],text=True,timeout=3).strip()
    except (OSError,subprocess.SubprocessError): pass
    return [f'Status      {status}',f'Stage       {state.get("stage","packet preparation")}',
        f'Channel     {done}/{total or "unknown"}'+(f' ({100*done/total:.1f}%)' if total else ''),
        f'Elapsed     {int(elapsed)//3600}:{int(elapsed)%3600//60:02d}:{int(elapsed)%60:02d} | ETA unknown',
        f'GPU         utilization / used VRAM / total VRAM: {gpu} (channel runs on CPU)',
        'Resume      same run command; protocol, packets and completed channel receipts verified',
        f'Log         {root/"run.log"}',f'Result      {root/"RESULT.json"}',
        f'Failure     {state.get("error","none recorded")}']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('plan','run','status','watch'))
    parser.add_argument('--output',type=Path,default=DEFAULT)
    args=parser.parse_args();root=args.output.resolve()
    if args.action=='plan':
        p=prepare(root);print(json.dumps(dict(status='PLANNED',packets=len(p['records']),channel_cases=len(p['jobs']))))
    elif args.action=='run':
        def stop(*_): raise KeyboardInterrupt('termination requested')
        signal.signal(signal.SIGTERM,stop);run(root)
    elif args.action=='status': print('\n'.join(lines(root)))
    else:
        first=True
        try:
            while True:
                rows=lines(root)
                if not first and os.isatty(1):print(f'\033[{len(rows)}F',end='')
                for row in rows:print(('\033[2K' if os.isatty(1) else '')+row)
                first=False
                if any(s in rows[0] for s in ('COMPLETE_','FAILED','INTERRUPTED','STOPPED_')):break
                time.sleep(1)
        except KeyboardInterrupt: pass


if __name__=='__main__': main()
