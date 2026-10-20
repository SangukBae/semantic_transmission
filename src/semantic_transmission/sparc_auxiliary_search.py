"""Bounded, resumable SPARC remove/relocate/remove experiment on real packets."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from . import sparc, sparc_auxiliary_selection as select
from .artifacts import sha256, write_json
from .qwen35_pairs_video import freeze
from .webvid5 import fingerprint
from .webvid_ablation import Stages, snapshot

REPO=sparc.REPO
DEFAULT=REPO/'outputs/kinetics_001_auxiliary_search_20261020'
REFERENCE=REPO/'outputs/kinetics_001_adaptive_v2_20261019'
MOTION=REPO/'outputs/kinetics_001_sparse_motion_20261019'
CONFIG=REPO/'configs/sparc_auxiliary_search.json'
COMPLETE='COMPLETE_SPARC_AUXILIARY_SEARCH_PILOT'


def read(path): return json.loads(Path(path).read_text())


def trial_protocol(parent,folder,indices):
    value=dict(parent,output=str(folder),parent_protocol_signature=parent['signature'],trial_indices=indices)
    value.pop('signature')
    value['signature']=fingerprint(value)
    return value


def preflight(args):
    from .sparc_motion_cache import audit_reference
    root=args.output.resolve(); ref=args.reference.resolve()
    source,semantic=audit_reference(ref)
    opts=sparc.parser().parse_args(['check','--source-frames',source['directory'],'--fps',str(source['video']['fps']),
        '--semantic-replay',str(ref/'semantic.json'),'--output',str(root)])
    p=sparc.preflight(opts);p.pop('signature')
    old=read(ref/'protocol.json')
    for name in ('sparc','sparc_selection','sparc_transport','predictive_codec','dense_temporal_receiver',
                 'official_quality','metadata_channel','packets','wire'):
        key='src/semantic_transmission/'+name+'.py'
        if old['code'][key]!=p['code'][key]:raise ValueError('baseline inference code differs: '+key)
    if old['vendor_code']!=p['vendor_code']:raise ValueError('baseline codec vendor differs')
    for k in ('codec','receiver','channel'):
        if old['runtimes'][k]!=p['runtimes'][k]:raise ValueError('baseline runtime differs')
    if any(p['assets'].get(k)!=v for k,v in old['assets'].items() if 'qwen35' not in k):
        raise ValueError('baseline assets differ')
    p.update(reference=str(ref),baseline_files={str(f):sha256(f) for f in [ref/'protocol.json',ref/'RESULT.json',
        ref/'selection.json',ref/'semantic.json',ref/'reconstruction/reconstruction.mp4']},search=read(CONFIG))
    if p['search']['ranking_resolution']!=[144,80] or p['search']['max_gap']!=4:
        raise ValueError('unregistered search geometry')
    p['code']['configs/sparc_auxiliary_search.json']=sha256(CONFIG)
    p['code']['scripts/run_sparc_auxiliary_search.sh']=sha256(REPO/'scripts/run_sparc_auxiliary_search.sh')
    p['legacy_windows']=[];p['legacy_provenance']={}
    if args.motion_reference:
        motion=args.motion_reference.resolve();mp=read(motion/'protocol.json')
        if mp['source']!=source or mp['reference_semantic_sha256']!=sha256(ref/'semantic.json'):
            raise ValueError('cached tracks source or keys differ')
        for f,d in mp['trajectory_reuse']['provenance'].items():
            if sha256(f)!=d:raise ValueError('cached tracks provenance changed')
        p['legacy_windows']=mp['legacy_windows']
        p['legacy_provenance']={str(motion/'protocol.json'):sha256(motion/'protocol.json'),**mp['trajectory_reuse']['provenance']}
    p['signature']=fingerprint(p)
    if (root/'protocol.json').exists() and read(root/'protocol.json')!=p:
        raise ValueError('frozen search inputs changed; preserve output and use a new root')
    return p


class Quality:
    def __init__(self,source,signals):
        import cv2,numpy as np,torch,lpips
        self.cv2,self.np,self.torch,self.lpips=cv2,np,torch,lpips
        self.source=source;self.small=signals.frames.astype(np.float32);self.masks=signals.masks
        self.model=lpips.LPIPS(net='vgg').cuda().eval()

    def evaluate(self,movie):
        from .research_quality import read_video
        np,cv2,t=self.np,self.cv2,self.torch
        frames=read_video(movie)
        if frames.shape!=self.source.shape:raise ValueError('final MP4 timeline or dimensions differ')
        psnr=[];perceptual=[]
        with t.inference_mode():
            for a,b in zip(self.source,frames):
                psnr.append(float(cv2.PSNR(a,b)))
                perceptual.append(float(self.model(self.lpips.im2tensor(a).cuda(),self.lpips.im2tensor(b).cuda()).item()))
        small=np.stack([cv2.resize(f,(144,80),interpolation=cv2.INTER_AREA) for f in frames]).astype(np.float32)
        error=np.abs(small-self.small).mean(-1);roi=[]
        for e,mask in zip(error,self.masks):roi.append(float(e[mask].mean() if mask.any() else e.mean()))
        temporal=np.abs(np.diff(small,axis=0)-np.diff(self.small,axis=0)).mean(axis=(1,2,3))
        return dict(quality=dict(psnr_db=float(np.mean(psnr)),lpips_vgg=float(np.mean(perceptual)),
                    roi_mae=float(np.mean(roi)),temporal_mae=float(temporal.mean())),
            per_frame=dict(psnr_db=psnr,lpips_vgg=perceptual,roi_mae=roi,temporal_mae=temporal.tolist()))


class Experiment:
    def __init__(self,p):
        import numpy as np,cv2
        from PIL import Image
        cv2.setNumThreads(2)
        self.p=p;self.root=Path(p['output']);self.ref=Path(p['reference']);self.config=p['search']
        self.plan=read(self.ref/'selection.json');self.frames=p['source']['video']['frames']
        self.mandatory=select.protected(self.plan['semantic_keys'],self.plan['cuts'],self.frames)
        self.visited=set();self.measured=set();self.history=[];self.started=time.time();self.accepted=0
        self.state('source motion and visibility analysis')
        source=np.stack([np.asarray(Image.open(Path(p['source']['directory'])/f'{i}.png').convert('RGB')) for i in range(self.frames)])
        self.signals=select.Signals(source,self.config,p['legacy_windows'])
        np.savez_compressed(self.root/'source_signals.npz',motion_risk=self.signals.risk,
            cached_track_risk=self.signals.track_risk,roi_masks=self.signals.masks)
        self.quality=Quality(source,self.signals)
        write_json(self.root/'reuse.json',dict(new_qwen_calls=0,semantic_keys=len(self.plan['semantic_keys']),
            cached_trajectory_windows=len(p['legacy_windows']),new_dense_flow_for_selector=True,
            source_signals_transmitted=False,receiver_reads_source=False,baseline_movie_reused=True,
            changed_selection_requires_full_dcvc_reencode=True))

    def state(self,stage,status='RUNNING',**extra):
        value=dict(status=status,stage=stage,elapsed_seconds=time.time()-getattr(self,'started',time.time()),
            evaluated_trials=len(getattr(self,'measured',[])),maximum_trials=sum(self.config['phase_trial_limits'])+1,
            accepted_changes=getattr(self,'accepted',0),eta='unknown: bounded adaptive search',
            pid=os.getpid(),**extra)
        write_json(self.root/'status.json',value)
        print('COMPARE_PROGRESS '+json.dumps(dict(stage=stage,state=status,current='Kinetics-400 001',
            overall=f"{value['evaluated_trials']} evaluated / at most {value['maximum_trials']} trials; {value['accepted_changes']} accepted")),flush=True)

    def measure(self,folder,movie,indices,uses):
        record=folder/'measurement.json'
        identity=dict(protocol=self.p['signature'],movie_sha256=sha256(movie),indices=indices,channel_uses=uses)
        if record.exists():
            value=read(record)
            if value['identity']!=identity or value['checksum']!=fingerprint({k:v for k,v in value.items() if k!='checksum'}):
                raise ValueError('cached measured quality changed')
            return value
        value=dict(identity=identity,indices=indices,channel_uses=uses,movie=str(movie),**self.quality.evaluate(movie))
        value['checksum']=fingerprint(value);write_json(record,value);return value

    def evaluate(self,proposal,label):
        indices=proposal['indices'];select.validate_anchors(indices,self.frames,4,self.plan['cuts'])
        if not self.mandatory.issubset(indices):raise ValueError('semantic/cut key removed')
        key=fingerprint(indices)[:20];folder=self.root/'trials'/key;folder.mkdir(parents=True,exist_ok=True)
        self.visited.add(key);self.state(label+' full DCVC encode',trial=key)
        freeze(folder/'protocol.json',trial_protocol(self.p,folder,indices))
        freeze(folder/'selection.json',dict(indices=indices,cuts=self.plan['cuts']))
        stages=Stages(folder,fingerprint(dict(protocol=self.p['signature'],indices=indices)))
        jobs=[('sender',dict(source=self.p['source'],profile=self.p['profile']),['sender'],'codec'),
            ('channel',dict(packet=str(folder/'sender/packet.bin')),['received'],'channel'),
            ('decode',dict(packet=str(folder/'received/packet.bin')),['decoded','decoded.json'],'codec'),
            ('restore',dict(decoded=str(folder/'decoded')),['reconstruction'],'receiver')]
        for stage,extra,products,role in jobs:
            self.state(label+' '+stage,trial=key)
            job=folder/f'{stage}_job.json';freeze(job,dict(output=str(folder),**extra))
            stages.step(stage,products,products,lambda log,s=stage,j=job,r=role:sparc.launch(s,j,log,self.p['runtimes'][r]))
        self.state(label+' final MP4 quality',trial=key)
        uses=read(folder/'received/channel.json')['complex_channel_uses']
        result=self.measure(folder,folder/'reconstruction/reconstruction.mp4',indices,uses)
        self.measured.add(key)
        self.state(label+' measured',trial=key)
        return result

    def search(self,current,baseline,phase,limit):
        count=0;reason='candidate_exhaustion'
        while count<limit:
            if phase=='relocate':proposals=self.signals.relocations(current['indices'],self.mandatory,self.plan['cuts'])
            else:
                rank=self.signals.rank(current['indices'],self.mandatory,self.plan['cuts'])
                proposals=select.removal_proposals(current['indices'],rank,self.mandatory,self.frames,
                    self.plan['cuts'],self.config['remove_batch_sizes'])
            proposals=[v for v in proposals if fingerprint(v['indices'])[:20] not in self.visited]
            if not proposals:break
            accepted=False
            # Evaluate up to three alternatives, then choose their best actual saving per loss.
            assessed=[]
            for proposal in proposals[:min(3,limit-count)]:
                result=self.evaluate(proposal,phase);count+=1
                gate=select.quality_gate(result,baseline,self.config)
                saving=current['channel_uses']-result['channel_uses']
                loss=max(0,current['quality']['psnr_db']-result['quality']['psnr_db'])/.2
                loss+=max(0,result['quality']['lpips_vgg']-current['quality']['lpips_vgg'])/.005
                feasible=gate['passed'] and saving>0
                assessed.append((saving/(1+loss) if feasible else -1,result,proposal,gate))
                self.history.append(dict(phase=phase,proposal=proposal,measurement=result['identity'],
                    quality=result['quality'],quality_gate=gate,saved_uses=saving,accepted=False))
                write_json(self.root/'history.json',self.history)
            if assessed:
                score,result,proposal,gate=max(assessed,key=lambda v:v[0])
                if score>0:
                    current=result;self.accepted+=1;accepted=True
                    for h in reversed(self.history):
                        if h['measurement']==result['identity']:h['accepted']=True;break
                    write_json(self.root/'history.json',self.history)
                    write_json(self.root/'best.json',current)
            # Continue other unvisited alternatives even when the first group failed.
            if count>=limit:reason='predeclared_trial_limit'
        return current,dict(phase=phase,evaluated=count,stop_reason=reason)

    def finish(self,current,baseline,control,phases):
        from .official_quality import OfficialMetrics
        from .research_quality import read_video
        reports={}
        for name,value in [('baseline',baseline),('selected',current),('uniform_same_count',control)]:
            self.state('official quality '+name)
            path=self.root/(name+'_official.json')
            if path.exists():
                cached=read(path)
                if cached['movie_sha256']!=sha256(value['movie']):raise ValueError('final metric movie changed')
            else:
                quality,rows=OfficialMetrics().evaluate(self.quality.source,read_video(value['movie']))
                cached=dict(quality=quality,per_frame=rows,movie_sha256=sha256(value['movie']))
                write_json(path,cached)
            if any(abs(cached['quality'][k]-value['quality'][k])>1e-6 for k in ('psnr_db','lpips_vgg')):
                raise ValueError('search and official final quality disagree')
            reports[name]=dict(anchor_count=len(value['indices']),semantic_keys=len(self.plan['semantic_keys']),
                auxiliary_count=len(value['indices'])-len(self.plan['semantic_keys']),channel_uses=value['channel_uses'],
                quality=cached['quality'],diagnostics=value['quality'],movie=value['movie'],indices=value['indices'],
                quality_gate=select.quality_gate(value,baseline,self.config))
        result=dict(status=COMPLETE,protocol_sha256=sha256(self.root/'protocol.json'),results=reports,phases=phases,
            channel_reduction_percent=100*(1-current['channel_uses']/baseline['channel_uses']),
            all_semantic_keys_preserved=True,max_gap=4,actual_channel='LDPC(6144,9216)+16QAM+AWGN 10dB seed42',
            full_dcvc_reference_chain_reencoded=True,final_mp4_metrics=True,independent_action_validation=False,
            same_count_control=True,equal_actual_rate=current['channel_uses']==control['channel_uses'],
            limits=['one development video; no held-out generalization claim','bounded heuristic search; not a global optimum',
                'motion ROI and visibility use optical-flow heuristics, not object labels or independent semantic review',
                'quality preservation means the preregistered tolerances, not byte-identical reconstruction',
                'same anchor count does not guarantee equal actual channel uses'])
        write_json(self.root/'RESULT.json',result);self.state('complete',COMPLETE)


def run(p):
    from .etri_60s_check import lock
    root=Path(p['output']);root.mkdir(parents=True,exist_ok=True)
    def stop(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,stop)
    with lock(REPO/'.local/etri_60s_check.lock'),lock(root/'execution.lock'):
        freeze(root/'protocol.json',p)
        exp=None
        try:
            exp=Experiment(p)
            baseline=exp.measure(root,exp.ref/'reconstruction/reconstruction.mp4',exp.plan['indices'],read(exp.ref/'RESULT.json')['channel_uses'])
            current=baseline;phases=[]
            for name,limit in zip(('remove_first','relocate','remove_second'),exp.config['phase_trial_limits']):
                current,phase=exp.search(current,baseline,name,limit);phases.append(phase)
            uniform=select.uniform_semantic(exp.frames,len(current['indices']),exp.mandatory,exp.plan['cuts'])
            control=exp.evaluate(dict(indices=uniform),'uniform count control')
            exp.finish(current,baseline,control,phases)
        except BaseException as error:
            status='INTERRUPTED' if isinstance(error,KeyboardInterrupt) else 'FAILED'
            if exp:exp.state(str(error),status,error=f'{type(error).__name__}: {error}')
            else:write_json(root/'status.json',dict(status=status,error=f'{type(error).__name__}: {error}'))
            raise


def watch(root,interval,once=False):
    from .terminal_progress import fit
    import shutil
    live=sys.stdout.isatty() and not once
    previous=[]
    if live:print('\033[?1049h\033[?25l',end='',flush=True)
    try:
        while True:
            value=read(root/'status.json') if (root/'status.json').exists() else {}
            gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total',
                '--format=csv,noheader'],capture_output=True,text=True,timeout=3).stdout.strip()
            best=read(root/'best.json') if (root/'best.json').exists() else {}
            elapsed=value.get('elapsed_seconds',0)
            rows=[f"Status   {value.get('status','NOT_STARTED')}",f"Stage    {value.get('stage','pending')}",
                f"Trials   {value.get('evaluated_trials',0)} evaluated; maximum {value.get('maximum_trials',33)} (adaptive total)",
                f"Accepted {value.get('accepted_changes',0)} changes",
                f"Best     {len(best.get('indices',[])) or 'pending'} anchors | {best.get('channel_uses','pending')} channel uses",
                f"Elapsed  {elapsed/60:.1f} minutes at last stage update | ETA unknown",
                f"GPU      utilization / used VRAM / total VRAM: {gpu or 'unavailable'}",
                f"Log      {root/'run.log'}",f"Error    {value.get('error','none')}",
                'Resume   bash scripts/run_sparc_auxiliary_search.sh run (verified stages reused)']
            if not live:print('\n'.join(rows));return
            width=shutil.get_terminal_size((110,24)).columns
            rows=[fit(row,width) for row in rows]
            for i,row in enumerate(rows):
                if i>=len(previous) or previous[i]!=row:print(f'\033[{i+1};1H{row}\033[K',end='')
            sys.stdout.flush();previous=rows;time.sleep(interval)
    except KeyboardInterrupt:pass
    finally:
        if live:print('\033[?25h\033[?1049l',end='',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['check','run','status','watch'])
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--reference',type=Path,default=REFERENCE)
    parser.add_argument('--motion-reference',type=Path,default=MOTION)
    parser.add_argument('--interval',type=float,default=1)
    args=parser.parse_args()
    if args.command in ('status','watch'):
        if not 0<args.interval<=60:parser.error('interval must be in (0,60]')
        return watch(args.output.resolve(),args.interval,args.command=='status')
    p=preflight(args)
    if args.command=='check':print('READY: verified source, baseline, semantic cache, tracks, models and runtimes');return
    run(p)


if __name__=='__main__':main()
