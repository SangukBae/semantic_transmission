"""Fixed-budget source-event pair selection, with an overlapping-observation ablation.

The extractor never reads reference annotations. Original caption prompts,
target samples and caption notes remain fixed; only selection changes.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
from pathlib import Path
import time

from . import auto_extraction as ae, qwen35_feedback as fb, qwen35_state_delta as sd
from . import qwen35_state_delta_resume as recovery
from .artifacts import sha256, write_json

ROOT = fb.REPO / 'outputs/qwen35_event_pairs_20261005'
ARMS = ('pairs', 'overlap_pairs')
WEIGHTS = dict(entry_exit=3., occlusion=2.5, scene_change=3., appearance=2., pose=1., viewpoint=.75)


def source_pairs(observations):
    grouped = {}
    for row in observations:
        for number, event in enumerate(row['result']['events']):
            a,b = event['before'],event['after']
            if not a < b or a not in row['indices'] or b not in row['indices']:
                raise ValueError('pair lacks chronological source evidence')
            key = (a,b)
            weight = WEIGHTS[fb.KIND_ALIASES.get(event['kind'],event['kind'])]
            entry = grouped.setdefault(key,dict(before=a,after=b,weight=weight,origins=[]))
            entry['weight'] = max(entry['weight'],weight)
            entry['origins'].append(dict(window=row['window'],event=number,description=event['description']))
    return [grouped[k] for k in sorted(grouped)]


def paired_keys(frames, count, pairs, prior_keys, novelty, max_gap=24):
    """Lexicographic MILP: complete pairs first, then preserve existing keys.

    One-sided endpoints earn no event reward. Duplicate endpoint pairs earn
    only their maximum event weight. Close evidence pairs have no short-gap
    penalty. No reference time ranges or evaluation scores enter this solver.
    """
    import numpy as np
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import lil_matrix, vstack
    if not 2 <= count <= frames or count < math.ceil((frames-1)/max_gap)+1:
        raise ValueError('infeasible key budget')
    if len(novelty)!=frames or not np.isfinite(novelty).all():
        raise ValueError('invalid novelty vector')
    n=frames+len(pairs)
    rows=[]; lower=[]; upper=[]
    def add(items,lo,hi):
        rows.append(items);lower.append(lo);upper.append(hi)
    add({i:1 for i in range(frames)},count,count)
    add({0:1},1,1);add({frames-1:1},1,1)
    for start in range(1,frames-max_gap):
        add({i:1 for i in range(start,start+max_gap)},1,np.inf)
    for j,pair in enumerate(pairs):
        if not 0 <= pair['before'] < pair['after'] < frames or pair['weight']<=0:
            raise ValueError('invalid source pair')
        for index in (pair['before'],pair['after']):
            add({frames+j:1,index:-1},-np.inf,0)
    matrix=lil_matrix((len(rows),n))
    for j,row in enumerate(rows):
        for i,value in row.items():matrix[j,i]=value
    matrix=matrix.tocsc();lo=np.array(lower);hi=np.array(upper)
    weights=np.array([p['weight'] for p in pairs]);cost=np.r_[np.zeros(frames),-weights]
    opts=dict(time_limit=60,mip_rel_gap=0.,presolve=True)
    start=time.monotonic()
    first=milp(cost,integrality=np.ones(n),bounds=Bounds(0,1),
               constraints=LinearConstraint(matrix,lo,hi),options=opts)
    if not first.success:raise RuntimeError(f'pair optimum unproven: {first.message}')
    optimum=round(-first.fun,6)
    reward=lil_matrix((1,n));reward[0,frames:]=weights
    matrix=vstack([matrix,reward.tocsc()],format='csc')
    prior=set(prior_keys)
    # < 0.1 total secondary novelty cannot offset one retained original key.
    scale=max(float(max(novelty)),1e-9)
    tie=np.array([(i not in prior) - .001*float(novelty[i])/scale
                  + 1e-9*((i*2654435761)%104729)/104729 for i in range(frames)])
    second=milp(np.r_[tie,np.zeros(len(pairs))],integrality=np.ones(n),bounds=Bounds(0,1),
        constraints=LinearConstraint(matrix,np.r_[lo,optimum-1e-7],np.r_[hi,np.inf]),options=opts)
    if not second.success:raise RuntimeError(f'secondary optimum unproven: {second.message}')
    keys=np.flatnonzero(second.x[:frames]>.5).tolist()
    validate_keys(keys,frames,count,max_gap)
    covered=[i for i,p in enumerate(pairs) if p['before'] in keys and p['after'] in keys]
    if abs(sum(pairs[i]['weight'] for i in covered)-optimum)>1e-5:
        raise ValueError('pair objective replay mismatch')
    return keys,dict(weighted_pair_coverage=optimum,covered_pairs=covered,total_pairs=len(pairs),
        retained_prior_keys=len(prior.intersection(keys)),solver='scipy.milp/HiGHS',optimal=True,
        elapsed_seconds=time.monotonic()-start,primary_gap=first.mip_gap,secondary_gap=second.mip_gap)


def validate_keys(keys,frames,count,gap):
    if (len(keys)!=count or keys!=sorted(set(keys)) or keys[0]!=0 or keys[-1]!=frames-1
            or max(b-a for a,b in zip(keys,keys[1:]))>gap):
        raise ValueError('key count, endpoints, order or maximum gap changed')


def overlap_windows(frames):
    # Fixed schedule, not selected by uncertainty, reference errors, or outcomes.
    return [list(range(c-24,min(c+24,frames-1)+1,4)) for c in range(48,frames-24,48)]


def prepare(root,parent=sd.ROOT):
    if (root/'plan.json').exists():raise FileExistsError('plan already frozen')
    original=sd.check_plan(parent)
    paths=[parent/'plan.json',parent/'ledger.json',parent/'selection.json',parent/'budget_state.json']
    plan=dict(original,policy='source_event_pairs_and_overlap_v1',created_utc=ae.now(),arms=list(ARMS),
        parent_root=str(parent.resolve()),parent_hashes={str(p.resolve()):sha256(p) for p in paths},
        overlap_windows=overlap_windows(original['frames']),weights=WEIGHTS,
        selection_objective='lexicographic complete source-event pairs, retain prior keys, pixel novelty',
        caption_notes='original parent observation ledger only in both arms',
        comparisons=['pairs vs parent: same original observations and caption rules; changed selector',
                     'overlap_pairs vs pairs: same selector and caption rules; 29 fixed overlapping observations added only to selector'],
        reference_access='No evaluator/reference files are inputs. Source analysis is a development repair, not held-out validation.',
        max_solver_seconds_per_stage=60,code_sha256=dict(original['code_sha256']))
    for path in [Path(__file__),Path(recovery.__file__)]:plan['code_sha256'][str(path)]=sha256(path)
    plan.pop('signature',None);plan['signature']=ae.digest(plan)
    write_json(root/'plan.json',plan)
    return plan


def check_plan(root):
    plan=fb.check_plan(root)
    for path,digest in plan['parent_hashes'].items():
        if sha256(path)!=digest:raise ValueError('parent source artifact changed')
    return plan


class Session(recovery.RecoverySession):
    def __init__(self,root,plan):
        super().__init__(root,plan)
        self.parent_cache=[]
        for path in sorted((Path(plan['parent_root'])/'calls').glob('state_caption*/*.json')):
            self.parent_cache.append((path,ae.unseal(path)))

    def call(self,role,ids,prompt,validator,tokens=640):
        if role=='state_caption':
            tail='\nVALID SOURCE FRAME IDS: '+json.dumps(ids)+'. Use these exact IDs, never image positions or example numbers. Use compact JSON without indentation.'
            acceptable=[prompt+tail,sd.explicit_source_examples(prompt,ids)+tail]
            for path,record in self.parent_cache:
                identity=record['identity']
                if (identity['ids']==ids and identity['tokens']==tokens and identity['prompt'] in acceptable
                        and identity['frame_hashes']==[self.plan['source_frame_hashes'][str(i)] for i in ids]):
                    self.validate_record(record,validator)
                    self.calls.append(dict(role=identity['role'],path=str(path),sha256=sha256(path),reused=True))
                    return record['result']
        return super().call(role,ids,prompt,validator,tokens)


def extract(session):
    root,plan=session.root,session.plan;parent=Path(plan['parent_root'])
    original=ae.unseal(parent/'ledger.json')['observations']
    parent_bundle=ae.unseal(parent/'budget_state.json')
    observation_calls=[c for c in parent_bundle['calls'] if c['role'].removesuffix('_schema')=='state_observe']
    parent_selection=ae.unseal(parent/'selection.json')
    extra=[];extra_calls=[]
    for arm in ARMS:
        if arm=='overlap_pairs':
            session.calls=[]
            for w,ids in enumerate(plan['overlap_windows']):
                prompt=sd.explicit_source_examples(plan['prompts']['observe'],ids)
                result=session.call('overlap_observe',ids,prompt,lambda x:sd.validate_observation(x,ids),768)
                extra.append(dict(window=f'overlap_{w:02d}',indices=ids,result=result))
                write_json(root/'partial/overlap_observations'/f'{w:03d}.json',extra[-1])
            extra_calls=list(session.calls)
            write_json(root/'overlap_ledger.json',ae.seal(dict(signature=plan['signature'],observations=extra,calls=extra_calls)))
        observations=original+extra
        pairs=source_pairs(observations)
        keys,solver=paired_keys(plan['frames'],plan['key_budget'],pairs,parent_bundle['keys'],
                               parent_selection['details']['pixel_novelty'],plan['max_gap'])
        write_json(root/f'{arm}_selection.json',ae.seal(dict(signature=plan['signature'],keys=keys,pairs=pairs,solver=solver)))
        session.calls=list(observation_calls)+list(extra_calls)
        captions=[]
        for segment,(a,b) in enumerate(zip(keys,keys[1:])):
            ids=fb.target_samples(a,b);prompt=sd.caption_prompt(plan,original,a,b)
            value=session.call('state_caption',ids,prompt,lambda x:sd.validate_caption(x,ids),plan['tokens']['caption'])
            row=dict(segment=segment,start=a,end_exclusive=b,source_indices=ids,caption=sd.caption_text(value),
                     structured=value,uncertainties=value['uncertainties'],needs_check=value['needs_check'])
            captions.append(row);write_json(root/'partial'/arm/f'{segment:04d}.json',row)
        write_json(root/f'{arm}.json',ae.seal(dict(signature=plan['signature'],arm=arm,keys=keys,captions=captions,calls=list(session.calls))))


def verify(root,partial=False):
    plan=check_plan(root);parent=Path(plan['parent_root'])
    original=ae.unseal(parent/'ledger.json')['observations'];results={}
    for arm in ARMS:
        path=root/f'{arm}.json'
        if partial and not path.exists():continue
        bundle=ae.unseal(path);selection=ae.unseal(root/f'{arm}_selection.json')
        if bundle['signature']!=plan['signature'] or bundle['keys']!=selection['keys']:
            raise ValueError('selection provenance changed')
        validate_keys(bundle['keys'],plan['frames'],plan['key_budget'],plan['max_gap'])
        extra=[] if arm=='pairs' else ae.unseal(root/'overlap_ledger.json')['observations']
        if extra and [r['indices'] for r in extra]!=plan['overlap_windows']:raise ValueError('overlap schedule changed')
        if selection['pairs']!=source_pairs(original+extra):raise ValueError('source pair list changed')
        records={}
        for call in bundle['calls']:
            if sha256(call['path'])!=call['sha256']:raise ValueError('raw call changed')
            record=ae.unseal(call['path']);fb.Session(root,plan).validate_record(record,lambda x:x)
            identity=record['identity']
            if identity['frame_hashes']!=[plan['source_frame_hashes'][str(i)] for i in identity['ids']]:
                raise ValueError('source image hashes differ')
            records[call['path']]=record
        for obs in original+extra:
            sd.validate_observation(obs['result'],obs['indices'])
            if not any(r['identity']['ids']==obs['indices'] and r['result']==obs['result']
                       and r['identity']['role'].removesuffix('_schema') in ('state_observe','overlap_observe') for r in records.values()):
                raise ValueError('observation has no actual model response')
        if len(bundle['captions'])!=plan['key_budget']-1:raise ValueError('caption count mismatch')
        for row,(a,b) in zip(bundle['captions'],zip(bundle['keys'],bundle['keys'][1:])):
            ids=fb.target_samples(a,b);prompt=sd.caption_prompt(plan,original,a,b)
            if (row['start'],row['end_exclusive'],row['source_indices'])!=(a,b,ids):raise ValueError('caption timeline changed')
            sd.validate_caption(row['structured'],ids)
            if sd.caption_text(row['structured'])!=row['caption']:raise ValueError('caption changed from model output')
            matches=[r for r in records.values() if r['identity']['role'].removesuffix('_schema')=='state_caption'
                     and r['identity']['ids']==ids and r['result']==row['structured']
                     and any(r['identity']['prompt'].startswith(p+'\nVALID SOURCE') for p in [prompt,sd.explicit_source_examples(prompt,ids)])]
            if len(matches)!=1:raise ValueError('caption lacks exact source/prompt/output trace')
        results[arm]=dict(keyframes=len(bundle['keys']),captions=len(bundle['captions']),
            max_gap=max(b-a for a,b in zip(bundle['keys'],bundle['keys'][1:])),
            short_intervals=sum(b-a<=4 for a,b in zip(bundle['keys'],bundle['keys'][1:])),
            unique_calls=len(records),generation_seconds=sum(r['seconds'] for r in records.values()),
            peak_allocated_gib=max(r['peak_allocated_bytes'] for r in records.values())/1024**3,
            solver=selection['solver'])
    out=dict(status='PASS_EVENT_PAIR_EXTRACTION' if len(results)==2 else 'PARTIAL',arms=results,
             reference_used_in_extraction=False,reconstruction_executed=False)
    write_json(root/'EXTRACTION_RESULT.json',out);return out


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','run','check'])
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--partial',action='store_true')
    args=parser.parse_args();root=args.root.resolve();recovery.install_normalizer()
    if args.action=='prepare':result=prepare(root)
    elif args.action=='check':result=verify(root,args.partial)
    else:
        with (root/'run.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);plan=check_plan(root)
            recovery.declare(root);session=Session(root,plan);start=time.monotonic();started=ae.now()
            try:extract(session)
            finally:write_json(root/'execution.json',dict(started_utc=started,ended_utc=ae.now(),elapsed_seconds=time.monotonic()-start))
            result=verify(root)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
