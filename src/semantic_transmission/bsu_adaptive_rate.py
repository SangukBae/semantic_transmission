"""Resumable sender search for adaptive frame spacing and DCVC quantization.

The frozen NEW BSU is the reference. Trial reconstruction happens at TX. Labels
are read only after decisions are frozen; RX jobs contain packet/model paths.
"""
import argparse
import copy
import csv
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

from .artifacts import sha256, write_json
from .webvid5 import fingerprint
from .qwen35_pairs_video import freeze
from .webvid_ablation import Stages
from .wire import pack, unpack
from .packets import accounting
from .bsu_dense_validation import REPO, read, evaluate, progress
from . import bsu_dense_neural as dense
from . import bsu_dense_temporal as temporal
from . import dense_temporal_receiver as rife
from . import adaptive_rate as policy
from .key_reduction_selection import guard, CRITERIA

MODULE='semantic_transmission.bsu_adaptive_rate'
BASE='outputs/bsu_temporal_routed_qp32_s2_min5_20261016_v1/RESULT.json'
POOL=((2,31),(2,28),(3,32))


def wrapped(inner):
    from .bsu_dense_router import POLICY
    return pack(dict(schema=POLICY,mode='neural',lengths=[len(inner)],visual_complex_count=0),inner)


def frame_rows(path):
    with Path(path).open() as stream:
        rows=[{k:int(v) if k=='frame' else float(v) for k,v in row.items()} for row in csv.DictReader(stream)]
    if [r['frame'] for r in rows] != list(range(len(rows))):
        raise ValueError('quality rows are not a complete ordered timeline')
    return rows


def worker(job,stage):
    out=Path(job['output'])
    if stage=='sender':
        import numpy as np
        from PIL import Image
        from .adaptive_rate_codec import Codec, encode
        case=job['case'];old=Path(case['original_run'])
        h,_=unpack((old/'transmitter/metadata.bin').read_bytes())
        context=dict(video=h['video'],indices=case['original_keys'],segments=h['segments'],decoder=h['decoder'])
        images=[]
        for i in range(case['source']['frames']):
            f=Path(case['source']['source_frames'])/f'{i}.png'
            if sha256(f) != case['source']['source_frame_hashes'][str(i)]:
                raise ValueError('sender source hash changed')
            images.append(np.asarray(Image.open(f).convert('RGB')))
        images=np.stack(images);cuts=[]
        for i in range(1,len(images)):
            if np.abs(images[i].astype(np.float32)-images[i-1]).mean()>40:
                cuts.extend((i-1,i))
        indices,qps=policy.schedule(len(images),job['blocks'],cuts)
        codec=Codec(REPO/'.local/vendor/DCVC-RT',REPO/'.local/checkpoints/dcvc_rt')
        video=encode(codec,images[indices],indices,qps)
        vh,bits=unpack(video)
        vh['temporal']=dict(policy=rife.POLICY,stride=4,frames=len(images),
            weights=rife.WEIGHT_SHA,network=rife.NETWORK_SHA,warp=rife.WARP_SHA)
        inner=dense.packet(context,pack(vh,bits));wire=wrapped(inner)
        dest=out/'sender';dest.mkdir(parents=True)
        (dest/'packet.bin').write_bytes(inner);(dest/'wire.bin').write_bytes(wire)
        write_json(dest/'sender.json',dict(policy=policy.POLICY,indices=indices,qps=qps,
            blocks=job['blocks'],channel_uses=accounting(len(wire))['complex_channel_uses'],
            packet_sha256=sha256(dest/'packet.bin'),wire_sha256=sha256(dest/'wire.bin'),
            cuts=sorted(set(cuts)),continuous_predictive_state=True,
            transmitted_qps_in_each_NAL=True,action_labels_read=False))
    elif stage=='channel':
        from .metadata_channel import transmit
        from .bsu_dense_router import receive
        original=Path(job['wire']).read_bytes();dest=out/'received';dest.mkdir(parents=True)
        reports=[]
        for seed in (42,1042,2042):
            data,report=transmit(original,10,seed,bits_per_symbol=4)
            if data != original or report['bit_errors']:
                raise ValueError('received packet differs; no free fallback')
            mode,payload=receive(data)
            if mode != 'neural':raise ValueError('unexpected selected route')
            dense.unpack_packet(payload['neural']);reports.append(report)
            if seed==42:
                (dest/'wire.bin').write_bytes(data);(dest/'packet.bin').write_bytes(payload['neural'])
        expected=accounting(len(original))['complex_channel_uses']
        if any(r['complex_channel_uses'] != expected for r in reports):
            raise ValueError('actual channel accounting differs')
        write_json(dest/'channel.json',dict(channel_uses=expected,reports=reports,all_bytes_exact=True))
    elif stage in ('decode','restore'):
        temporal.worker(job,stage)
    elif stage=='evaluate':
        evaluate(job['case'],out/'reconstruction.mp4',out,job['channel_uses'])
    elif stage=='action':
        from .bsu_key_reduction import action
        action(job)
    else:raise ValueError(stage)


class Experiment:
    def __init__(self,root,case_ids,extended_joint=False,reuse_trials=()):
        self.root=root;root.mkdir(parents=True,exist_ok=True)
        previous=read(REPO/'outputs/bsu_key_reduction_20261015_v2/protocol.json')
        self.base=read(REPO/BASE)
        by_id={r['id']:r for r in self.base['rows']}
        self.cases=[c for c in previous['cases'] if not case_ids or c['id'] in case_ids]
        if case_ids and set(case_ids) != {c['id'] for c in self.cases}:raise ValueError('unknown case')
        self.reference={c['id']:by_id[c['id']] for c in self.cases}
        self.pool=POOL+((3,36),(4,40)) if extended_joint else POOL
        self.reuse_trials=[Path(p).resolve() for p in reuse_trials]
        self.done=0;self.current='';self.started=time.time()
        files=['adaptive_rate','adaptive_rate_codec','bsu_adaptive_rate','bsu_dcvc_keys',
            'bsu_dense_neural','bsu_dense_temporal','dense_temporal_receiver',
            'bsu_dense_router','bsu_dense_validation','key_reduction_selection','official_quality',
            'stutter_metrics','bsu_key_reduction','metadata_channel','packets','wire']
        self.p=dict(policy=policy.POLICY,cases=self.cases,reference=BASE,
            reference_sha256=sha256(REPO/BASE),pool=list(self.pool),window_frames=24,
            reuse_trials=[dict(root=str(p),protocol_sha256=sha256(p/'protocol.json')) for p in self.reuse_trials],
            allocation_margin=.75,admission_margin=.85,maximum_rollbacks=3,
            criteria=CRITERIA,arms=policy.ARMS,channel=dict(snr_db=10,seeds=[42,1042,2042]),
            runtimes={**previous['runtimes'],'python':str(REPO/'.local/bsu_codec_eval_env/bin/python'),
                'codec':str(REPO/'.local/dcvc_rt_env/bin/python')},action_runtime=previous['action_runtime'],
            code={n:sha256(REPO/f'src/semantic_transmission/{n}.py') for n in files},
            action_labels_used_for_selection=False,new_training=False,
            scope='14 development clips; optimize neural route; preserve existing fallback clips',
            sender_uses_source_and_trial_reconstruction=True,
            search_cost_not_transmission_cost=True,physical_radio_overhead_included=False)
        self.p['signature']=fingerprint(self.p);freeze(root/'protocol.json',self.p)
        self.identity=self.p['signature']
        snapshot=root/'validated_source';snapshot.mkdir(exist_ok=True)
        for name in ('adaptive_rate','adaptive_rate_codec','bsu_adaptive_rate'):
            source=REPO/f'src/semantic_transmission/{name}.py';saved=snapshot/source.name
            if saved.exists() and sha256(saved)!=sha256(source):raise ValueError('runtime snapshot changed')
            if not saved.exists():shutil.copy2(source,saved)
        for case in self.cases:
            q=self.reference[case['id']]['quality']
            if sha256(q['movie']) != q['movie_sha256'] or q['source_movie_sha256'] != case['source']['normalized_sha256']:
                raise ValueError('NEW BSU reference provenance mismatch')

    def status(self,stage):
        progress(stage,self.done,len(self.cases),self.current)
        write_json(self.root/'status.json',dict(status='RUNNING',stage=stage,current=self.current,
            completed=self.done,total=len(self.cases),elapsed_seconds=time.time()-self.started))

    def execute(self,stage,job,log):
        key='codec' if stage in ('sender','decode') else 'channel' if stage=='channel' else 'action' if stage=='action' else 'python'
        env=dict(os.environ,PYTHONPATH=str(REPO/'src'),PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',
            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
        if stage=='channel':env['CUDA_VISIBLE_DEVICES']='-1'
        self.status(stage);log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('x') as stream:
            rc=subprocess.run([self.p['runtimes'][key],'-m',MODULE,'--worker',stage,'--job',str(job)],
                cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
        if rc:
            print(log.read_text(errors='replace')[-6000:],flush=True)
            raise RuntimeError(f'{stage} failed ({rc}): {log}')

    def step(self,stages,stage,owned,job,required=None):
        path=stages.root/f'{stage}_job.json';freeze(path,job)
        stages.step(stage,owned,required or owned,lambda log:self.execute(stage,path,log))

    def cached_trial(self,case,name,blocks):
        import ast
        from .webvid_ablation import snapshot
        def worker_text(path):
            source=Path(path).read_text()
            node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='worker')
            return ast.dump(node,include_attributes=False)
        for root in self.reuse_trials:
            out=root/case['id']/'trials'/name
            if not (out/'stages/evaluate.json').exists():continue
            old=read(root/'protocol.json')
            if next(c for c in old['cases'] if c['id']==case['id']) != case:
                raise ValueError('cached source contract differs')
            for module,digest in old['code'].items():
                if module!='bsu_adaptive_rate' and self.p['code'].get(module)!=digest:
                    raise ValueError('cached decoder/evaluator code differs')
            if worker_text(root/'validated_source/bsu_adaptive_rate.py')!=worker_text(Path(__file__)):
                raise ValueError('cached worker implementation differs')
            if read(out/'sender_job.json')['blocks']!=blocks:continue
            for stage in ('sender','decode','restore','evaluate'):
                receipt=read(out/f'stages/{stage}.json')
                if receipt['status']!='PASSED' or snapshot(out,receipt['required'])!=receipt['artifacts']:
                    raise ValueError('cached stage artifacts changed')
            q=read(out/'quality.json');sender=read(out/'sender/sender.json')
            if sha256(q['movie'])!=q['movie_sha256'] or q['source_movie_sha256']!=case['source']['normalized_sha256']:
                raise ValueError('cached movie/source differs')
            print(f'REUSE VERIFIED TRIAL {case["id"]}/{name}',flush=True)
            return dict(name=name,quality=q,frame_rows=frame_rows(out/'quality.csv'),
                packet=str(out/'sender/packet.bin'),wire=str(out/'sender/wire.bin'),blocks=blocks,
                sender=sender,reused=True,reuse_root=str(root))
        return None

    def trial(self,case,name,blocks):
        cached=self.cached_trial(case,name,blocks)
        if cached is not None:return cached
        out=self.root/case['id']/'trials'/name;out.mkdir(parents=True,exist_ok=True)
        steps=Stages(out,fingerprint(dict(protocol=self.identity,blocks=blocks)))
        self.current=f'{case["id"]}/{name}'
        self.step(steps,'sender',['sender'],dict(output=str(out),case=case,blocks=blocks))
        sender=read(out/'sender/sender.json')
        self.step(steps,'decode',['decoded','decoded.json'],dict(output=str(out),packet=str(out/'sender/packet.bin')))
        self.step(steps,'restore',['frames','reconstruction.mp4','receiver.json'],dict(output=str(out),decoded=str(out/'decoded')))
        self.step(steps,'evaluate',['quality.json','quality.csv'],dict(output=str(out),case=case,channel_uses=sender['channel_uses']))
        return dict(name=name,quality=read(out/'quality.json'),frame_rows=frame_rows(out/'quality.csv'),
            packet=str(out/'sender/packet.bin'),wire=str(out/'sender/wire.bin'),blocks=blocks,
            sender=sender,reused=False)

    def reuse_stride4(self,case):
        out=REPO/'outputs/bsu_dense_temporal_qp32_s4_20261016_v1'/case['id']
        q=read(out/'native/quality.json');inner=out/'sender/packet.bin'
        if (sha256(q['movie']) != q['movie_sha256'] or q['source_movie_sha256'] != case['source']['normalized_sha256'] or
                sha256(inner) != read(out/'decoded.json')['packet_sha256'] or
                inner.read_bytes() != (out/'received/packet.bin').read_bytes()):
            raise ValueError('reused stride4 provenance differs')
        q['channel_uses']=accounting(len(wrapped(inner.read_bytes())))['complex_channel_uses']
        blocks=[dict(start=a,end=b,stride=4,qp=32,candidate='s4_q32') for a,b in policy.windows(case['source']['frames'])]
        return dict(name='s4_q32',quality=q,frame_rows=frame_rows(out/'native/quality.csv'),
            packet=str(inner),blocks=blocks,reused=True,stride=4,qp=32)

    def admitted(self,base,record):
        q=record['quality']
        return q['channel_uses'] < base['channel_uses'] and policy.metric_checks(base,q,.85)['passed']

    def deliver(self,case,arm,record):
        out=self.root/case['id']/f'delivery_{arm}';out.mkdir(parents=True,exist_ok=True)
        inner=Path(record['packet']).read_bytes();wire=wrapped(inner)
        tx=out/'transmitter';tx.mkdir(exist_ok=True)
        if (tx/'wire.bin').exists() and (tx/'wire.bin').read_bytes() != wire:raise ValueError('selected packet changed')
        (tx/'wire.bin').write_bytes(wire)
        steps=Stages(out,fingerprint(dict(protocol=self.identity,wire=sha256(tx/'wire.bin'))))
        self.current=f'{case["id"]}/{arm}/independent_receiver'
        self.step(steps,'channel',['received'],dict(output=str(out),wire=str(tx/'wire.bin')))
        self.step(steps,'decode',['decoded','decoded.json'],dict(output=str(out),packet=str(out/'received/packet.bin')))
        self.step(steps,'restore',['frames','reconstruction.mp4','receiver.json'],dict(output=str(out),decoded=str(out/'decoded')))
        uses=read(out/'received/channel.json')['channel_uses']
        self.step(steps,'evaluate',['quality.json','quality.csv'],dict(output=str(out),case=case,channel_uses=uses))
        q=read(out/'quality.json');base=self.reference[case['id']]['quality']
        return dict(quality=q,guard=guard(base,q),decision=record['name'],
            fresh_receiver=True,channel=read(out/'received/channel.json'),
            blocks=record['blocks'],packet_sha256=sha256(out/'received/packet.bin'))

    def run_case(self,case):
        cid=case['id'];base=self.reference[cid]['quality'];dest=self.root/cid
        dest.mkdir(exist_ok=True)
        fallback=dict(quality=base,guard=dict(passed=True,failures=[],unchanged_reference=True),
            decision='unchanged_new_bsu',fresh_receiver=False,blocks=[])
        if self.reference[cid]['choice']['mode'] != 'neural':
            result={arm:copy.deepcopy(fallback) for arm in policy.ARMS}
            write_json(dest/'RESULT.json',result);return result
        baseline_rows=frame_rows(Path(base['movie']).parent/'quality.csv')
        pool={'s4_q32':self.reuse_stride4(case)}
        frames=case['source']['frames']
        for stride,qp in self.pool:
            name=f's{stride}_q{qp}'
            blocks=[dict(start=a,end=b,stride=stride,qp=qp,candidate=name) for a,b in policy.windows(frames)]
            pool[name]={**self.trial(case,name,blocks), 'stride':stride,'qp':qp}
        selected={};search={}
        for arm in policy.ARMS:
            candidates=[r for r in pool.values() if
                (arm!='temporal' or r['qp']==32) and (arm!='quantization' or r['stride']==2)]
            accepted=[r for r in candidates if self.admitted(base,r)]
            blocks=policy.allocate(base,baseline_rows,pool,frames,case['original_keys'],arm)
            attempts=[]
            for attempt in range(3):
                if all(b['stride']==2 and b['qp']==32 for b in blocks):break
                name=f'{arm}_adaptive_{attempt}'
                rec=self.trial(case,name,blocks)
                g=guard(base,rec['quality'])
                attempts.append(dict(name=name,channel_uses=rec['quality']['channel_uses'],guard=g,blocks=blocks))
                if self.admitted(base,rec):
                    accepted.append(rec);break
                blocks=policy.rollback(blocks)
            # Freeze each decision before action inference. No clip identity rules.
            best=min(accepted,key=lambda r:r['quality']['channel_uses']) if accepted else None
            selected[arm]=best
            search[arm]=dict(attempts=attempts,selected=best['name'] if best else 'unchanged_new_bsu',
                selection_uses_action_labels=False)
        # Joint search includes its two restricted subspaces.
        choices=[r for r in selected.values() if r is not None]
        if choices:selected['joint']=min(choices,key=lambda r:r['quality']['channel_uses'])
        search['joint']['selected']=selected['joint']['name'] if selected['joint'] else 'unchanged_new_bsu'
        freeze(dest/'selection.json',search)
        result={};deliveries={}
        for arm in policy.ARMS:
            rec=selected[arm]
            if rec is None:result[arm]=copy.deepcopy(fallback);continue
            digest=sha256(rec['packet'])
            if digest not in deliveries:deliveries[digest]=self.deliver(case,arm,rec)
            result[arm]=copy.deepcopy(deliveries[digest])
        write_json(dest/'RESULT.json',result);return result

    def run(self):
        rows=[]
        for case in self.cases:
            rows.append(dict(id=case['id'],variants=self.run_case(case)))
            self.done+=1;self.status('completed_video')
        # Labels become available to this process only after every choice freezes.
        labels={v['id']:v['label'] for v in read(REPO/'outputs/bsu_kinetics_context/RESULT.json')['videos']}
        action_rows=[]
        for r in rows:
            variants={'new_bsu':dict(quality=self.reference[r['id']]['quality']),**r['variants']}
            for arm,v in variants.items():
                q=v['quality'];action_rows.append(dict(id=r['id'],variant=arm,label=labels[r['id']],path=q['movie'],sha256=q['movie_sha256']))
        self.step(Stages(self.root,self.identity),'action',['action'],dict(output=str(self.root/'action'),
            runtime=self.p['action_runtime'],rows=action_rows),['action/RESULT.json'])
        action=read(self.root/'action/RESULT.json');by={(r['input']['id'],r['input']['variant']):r for r in action['records']}
        summaries={};original=sum(self.reference[r['id']]['quality']['channel_uses'] for r in rows)
        from .official_quality import KEYS
        for arm in policy.ARMS:
            group=[r['variants'][arm] for r in rows];uses=sum(r['quality']['channel_uses'] for r in group)
            regressions=[r['id'] for r in rows if any(by[r['id'],'new_bsu'][k] and not by[r['id'],arm][k]
                for k in ('top1_correct','top5_correct'))]
            summaries[arm]=dict(channel_uses=uses,saving_vs_new_bsu_percent=100*(1-uses/original),
                ratio_to_lgvsc=uses/1407617 if len(rows)==14 else None,
                mean_quality={k:statistics.mean(r['quality']['quality'][k] for r in group) for k in KEYS},
                mean_temporal={k:statistics.mean(r['quality']['temporal'][k] for r in group) for k in
                    ('flow_error_mean','cadence_error_mean','false_freeze_rate')},
                quality_motion_passes=sum(r['guard']['passed'] for r in group),
                changed_videos=sum(r['fresh_receiver'] for r in group),action=action['summary'][arm],
                action_regressions=regressions,
                performance_preserved=all(r['guard']['passed'] for r in group) and not regressions)
        value=dict(status='COMPLETE_ADAPTIVE_RATE_VALIDATION',baseline_channel_uses=original,
            baseline_action=action['summary']['new_bsu'],summary=summaries,rows=rows,
            reference_result_sha256=self.p['reference_sha256'],protocol_sha256=sha256(self.root/'protocol.json'),
            generalization_proven=False,cohort='previously used development clips',
            selection_is_sender_trial_reconstruction=True,action_labels_used_for_selection=False,
            source_available_at_receiver=False,shared_pretrained_weights=True,new_training=False,
            elapsed_seconds=time.time()-self.started)
        write_json(self.root/'RESULT.json',value)
        write_json(self.root/'status.json',dict(status=value['status'],completed=len(rows),total=len(rows)))
        print('FINAL_SUMMARY '+str(summaries),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=REPO/'outputs/bsu_adaptive_rate_20261016_v1')
    parser.add_argument('--case',action='append');parser.add_argument('--worker');parser.add_argument('--job',type=Path)
    parser.add_argument('--extended-joint',action='store_true',help='also test fewer, higher-quality anchors')
    parser.add_argument('--reuse-trials',type=Path,action='append',default=[])
    args=parser.parse_args()
    if args.worker:return worker(read(args.job),args.worker)
    from .etri_60s_check import lock
    with lock(REPO/'.local/etri_60s_check.lock'):
        try:Experiment(args.root.resolve(),args.case,args.extended_joint,args.reuse_trials).run()
        except BaseException as error:
            write_json(args.root/'status.json',dict(status='FAILED',error=str(error)))
            raise


if __name__=='__main__':main()
