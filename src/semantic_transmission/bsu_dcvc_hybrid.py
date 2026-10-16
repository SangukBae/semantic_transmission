"""Keep original first/last received anchors, predictively code interior keys."""
import argparse
import copy
import hashlib
from pathlib import Path
import shutil

from . import bsu_dcvc_validation as full
from .artifacts import sha256, write_json
from .webvid5 import fingerprint
from .webvid_ablation import Stages
from .qwen35_pairs_video import freeze
from .key_reduction_selection import CRITERIA, guard

REPO=full.REPO
MODULE='semantic_transmission.bsu_dcvc_hybrid'
read=full.read


def worker(j,stage):
    if stage not in ('channel','decode','initialize'):
        full.worker(j,stage);return
    full.worker(j,stage)
    run,old=Path(j['run']),Path(j['case']['original_run'])
    anchors=[j['keys'][0],j['keys'][-1]]
    if stage=='channel':
        from .wire import unpack,pack
        from .metadata_channel import transmit
        h,_=unpack((run/'received/metadata.bin').read_bytes())
        records=[r for r in h['keyframes'] if r['index'] in anchors]
        tx=(old/'transmitter/visual.c64').read_bytes()
        rx=(old/'received/visual.c64').read_bytes()
        def subset(raw):
            return b''.join(raw[r['complex_offset']*8:(r['complex_offset']+r['complex_count'])*8] for r in records)
        tx,rx=subset(tx),subset(rx)
        uses=len(tx)//8
        mode=pack(dict(schema='bsu_dcvc_first_last_anchors_v1',indices=anchors,
            original_metadata_sha256=sha256(run/'received/metadata.bin'),
            dcvc_packet_sha256=sha256(run/'received/keys.bin'),visual_complex_count=uses),b'')
        recovered,report=transmit(mode,10,44,bits_per_symbol=4)
        if recovered!=mode or report['bit_errors']:raise ValueError('hybrid mode packet failed')
        (run/'received/mode.bin').write_bytes(recovered)
        (run/'received/anchor_visual.c64').write_bytes(rx)
        accounting=read(run/'channel_accounting.json')
        accounting.update(channel_uses=accounting['channel_uses']+uses+report['complex_channel_uses'],
            original_anchor_complex_uses=uses,mode_report=report,
            original_anchor_channel_reused=True,
            original_anchor_tx_sha256=hashlib.sha256(tx).hexdigest(),
            original_anchor_rx_sha256=hashlib.sha256(rx).hexdigest(),
            assumption='same historical frame-seeded AWGN realization for original anchor symbols',
            all_channel_seeds_tested_for_hybrid=False)
        write_json(run/'channel_accounting.json',accounting)
    elif stage=='decode':
        from .wire import unpack
        mode,_=unpack((run/'received/mode.bin').read_bytes())
        if (mode['indices']!=anchors or mode['dcvc_packet_sha256']!=sha256(run/'received/keys.bin')
                or mode['original_metadata_sha256']!=sha256(run/'received/metadata.bin')):
            raise ValueError('hybrid mode binding failed')
        # Exact historical AWGN samples permit exact cached NTSCC receiver decoding.
        for k in mode['indices']:
            src=old/f'receiver/frames/sample/key_frames_received/{k}.png'
            dst=run/f'receiver/frames/sample/key_frames_received/{k}.png'
            shutil.copyfile(src,dst)
        write_json(run/'receiver/anchor_reuse.json',dict(
            indices=anchors,receiver_reads_source=False,cached_original_decoding_reused=True,
            original_channel_sha256=sha256(old/'channel_accounting.json'),
            keys={str(k):sha256(run/f'receiver/frames/sample/key_frames_received/{k}.png') for k in anchors}))
    else:
        policy=read(run/'receiver_policy.json')
        policy['noise_contract']['inputs']=fingerprint({str(p.relative_to(run)):sha256(p)
            for p in (run/'received').iterdir() if p.is_file()})
        write_json(run/'receiver_policy.json',policy)


def run(args):
    from .etri_60s_check import lock
    root=args.root.resolve()
    source=args.source.resolve();old=read(source/'protocol.json')
    # Fixed boundary-anchor policy; only interior-containing screened candidates.
    candidates=[c for c in old['candidates'] if len(c['case']['original_keys'])>2]
    p=dict(criteria=CRITERIA,candidates=candidates,runtimes=old['runtimes'],
        action_runtime=old['action_runtime'],baseline_result=old['baseline_result'],worker_module=MODULE,
        previous_protocol_sha256=sha256(source/'protocol.json'),
        code={str(f.relative_to(REPO)):sha256(f) for f in (REPO/'src/semantic_transmission').glob('*.py')},
        hypothesis='unchanged first/last anchor conditions may reduce temporal regression from key codec replacement',
        fixed_seed_selection=2025,fixed_seed_confirmation=2026,
        accounting='all DCVC keys still transmitted for predictive context; original anchor symbols and paid mode packet added',
        no_quality_threshold_relaxation=True,production_defaults_changed=False)
    p['signature']=fingerprint(p)
    decisions=[];action_rows=[]
    labels={v['id']:v['label'] for v in read(REPO/'outputs/bsu_kinetics_context/benchmark/protocol.json')['videos']}
    with lock(REPO/'.local/etri_60s_check.lock'):
        freeze(root/'protocol.json',p)
        for i,c in enumerate(candidates):
            case=c['case'];full.legacy.event(done=i,total=len(candidates),current=case['id'],stage='hybrid anchors')
            baseline=read(source/case['id']/'decision.json')['baseline']
            q,r=full.trial(root,p,case,c,2025)
            a=read(Path(case['original_run'])/'receiver/generation_noise.json');b=read(r/'receiver/generation_noise.json')
            if a['records']!=b['records'] or a['runtime']!=b['runtime']:raise ValueError('noise/runtime differs')
            verdict=guard(baseline,q)
            decision=dict(id=case['id'],baseline=baseline,candidate=q,seed2025=verdict,
                          saving_percent=100*(1-q['channel_uses']/baseline['channel_uses']),
                          original_anchors=[case['original_keys'][0],case['original_keys'][-1]],
                          noise_records_exact=True,accepted=False)
            pairs=[(2025,baseline,q)]
            if verdict['passed']:
                # Original branch has unchanged receiver anchors and therefore uses the original worker.
                original=copy.deepcopy(p);original['worker_module']=full.MODULE
                qa,ra=full.trial(root,original,case,c,2026,original=True)
                qb,rb=full.trial(root,p,case,c,2026)
                a=read(ra/'receiver/generation_noise.json');b=read(rb/'receiver/generation_noise.json')
                if a['records']!=b['records'] or a['runtime']!=b['runtime']:raise ValueError('confirmation noise differs')
                decision['seed2026']=guard(qa,qb);decision['accepted']=decision['seed2026']['passed']
                pairs.append((2026,qa,qb))
            for seed,a,b in pairs:
                for variant,value in [('baseline',a),('hybrid',b)]:
                    action_rows.append(dict(id=case['id'],variant=f'{variant}_{seed}',label=labels[case['id']],
                                            path=value['movie'],sha256=value['movie_sha256']))
            decisions.append(decision);write_json(root/case['id']/'decision.json',decision)
        freeze(root/'action_job.json',dict(runtime=p['action_runtime'],rows=action_rows,output=str(root/'action')))
        Stages(root,p['signature']).step('action',['action'],['action/RESULT.json'],
            lambda log:full.execute(p,'action',root/'action_job.json',log))
        action=read(root/'action/RESULT.json')
        for d in decisions:
            pair={r['input']['variant']:r for r in action['records'] if r['input']['id']==d['id']}
            passes=[]
            for seed in [2025,2026] if 'seed2026' in d else [2025]:
                a,b=pair[f'baseline_{seed}'],pair[f'hybrid_{seed}']
                passes.append(all(not a[k] or b[k] for k in ['top1_correct','top5_correct']))
            d['action_noninferior']=all(passes);d['accepted']=d['accepted'] and d['action_noninferior']
        write_json(root/'RESULT.json',dict(status='COMPLETE_DCVC_HYBRID_VALIDATION',decisions=decisions,
            accepted_videos=sum(d['accepted'] for d in decisions),tested_videos=len(decisions),
            action=action['summary'],criteria=CRITERIA,production_defaults_changed=False))
        write_json(root/'completion_receipt.json',dict(status='PASSED',artifacts={str(f.relative_to(root)):sha256(f)
            for f in root.rglob('*') if f.is_file() and 'failed_attempts' not in f.parts and f.name!='completion_receipt.json'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=REPO/'outputs/bsu_dcvc_hybrid_20261015_v1')
    parser.add_argument('--source',type=Path,default=REPO/'outputs/bsu_dcvc_validation_20261015_v2')
    parser.add_argument('--worker');parser.add_argument('--job',type=Path)
    args=parser.parse_args()
    if args.worker:worker(read(args.job),args.worker)
    else:run(args)
