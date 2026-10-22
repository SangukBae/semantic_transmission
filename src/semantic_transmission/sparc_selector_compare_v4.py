"""Matched-source, matched-measurement-cap v3/v4 comparison on prepared clips."""
import argparse
import copy
import fcntl
import json
import math
from pathlib import Path
import signal
import time

import torch
import torch.nn.functional as F

from . import sparc_selector_budget as budget
from . import sparc_selector_train as supervision
from . import sparc_selector_train_v3 as v3
from . import sparc_selector_train_v4 as v4
from . import sparc_selector_v3_data as data
from . import sparc_selector_v4 as runner
from .sparc_selector_data import read, save, sha, digest
from .sparc_selector_workers import REPO


def control(root, source_root, rows, p, checkpoint):
    """The original v3 ranking rule and allocator, with an explicit measurement cap."""
    begin = time.monotonic()
    root.mkdir(parents=True, exist_ok=True)
    state = torch.load(checkpoint,map_location='cpu',weights_only=False)
    if state['architecture'] != budget.ARCHITECTURE:
        raise ValueError('original v3 control checkpoint required')
    model = budget.BudgetSelector(**state['model_config']).to(p['train']['device']).eval()
    clips, per_video, actual_calls = [], {}, 0
    try:
        for row in rows:
            supervision.interrupted()
            measure = data.Measurer(root,source_root,row,p)
            try:
                save(root/'clips'/row['id']/'prepared.json',measure.ready)
                extractor = data.Extractor(p['train']['device'])
                if extractor.identity != p['features']:
                    raise ValueError('control feature extractor mismatch')
                features = extractor.extract(measure.frames)
                del extractor
                tensors = [torch.from_numpy(features[k]) for k in ('patches','motion')]
                def rank(states, baseline):
                    scores=[]
                    with torch.inference_mode():
                        for i in range(0,len(states),4):
                            ys,u,s=budget.predict(model,tensors,states[i:i+4],measure.ready,p['target_cbr'],
                                budget.cbr(baseline,measure.pixels),state['normalization'],p['train']['device'])
                            scores.extend((s-4*F.relu(ys[:,4]/p['target_cbr']-1)-.1*u[:,4]).cpu().tolist())
                    return scores
                cfg = p['search']
                per_round=cfg['per_round']
                if cfg['measurements_per_clip'] % per_round:
                    raise ValueError('matched v3 control requires measurements_per_clip divisible by per_round')
                decision, observations, baseline=budget.search(features['proxies'],measure.ready['keys'],measure.ready['cuts'],
                    math.floor(p['target_cbr']*3*measure.pixels),measure,rank=rank,beam=cfg['beam'],
                    rounds=cfg['measurements_per_clip']//per_round,per_round=per_round,stop=supervision.interrupted)
                clips.append(dict(id=row['id'],pixels=measure.pixels,baseline=baseline,observations=observations,decision=decision))
                per_video[row['id']]=measure.calls
                actual_calls += measure.calls
            finally:
                measure.close()
        cohort=v3.allocate_cohort(clips,p['target_cbr'])
        return dict(clips=clips,cohort=cohort,summary=v4.summarize(clips,cohort,p['quality_goal']),
            measurement_attempts=actual_calls,per_video_attempts=per_video,
            elapsed_seconds=time.monotonic()-begin,checkpoint_sha256=sha(checkpoint))
    finally:
        del model
        if torch.cuda.is_available():torch.cuda.empty_cache()


def compare(trained, source_root, output, *, runtimes=None):
    if output.resolve() in (trained.resolve(),source_root.resolve()):
        raise ValueError('comparison output must be separate from source and training evidence')
    original=read(trained/'protocol.json');p=copy.deepcopy(original)
    for name,expected in p['code'].items():
        if sha(REPO/name)!=expected:
            raise ValueError('trained v4 code changed: '+name)
    for name,expected in p['supervision_protocol']['code'].items():
        if sha(REPO/'src/semantic_transmission'/f'{name}.py')!=expected:
            raise ValueError('frozen reconstruction dependency changed: '+name)
    if runtimes is not None:p['runtimes']=read(runtimes)
    supervision.runtimes=lambda:p['runtimes']
    benchmark=read(source_root/'protocol.json')
    rows=benchmark.get('rows') or read(source_root/'sources.json')
    training_rows=read(trained/'sources.json')
    known={r['parent'] for r in training_rows}
    known_hashes={r[k] for r in training_rows for k in ('sha256','raw_sha256') if r.get(k)}
    if any(r['parent'] in known or any(r.get(k) in known_hashes for k in ('sha256','raw_sha256')) for r in rows):
        raise ValueError('external benchmark overlaps training/calibration/validation/test parents')
    p['benchmark_ids']=[r['id'] for r in rows]
    checkpoint=trained/'training/best.pt'
    v3_checkpoint=Path(p['initialization']['path'])
    if sha(v3_checkpoint)!=p['initialization']['sha256']:
        raise ValueError('v3 reference checkpoint changed')
    output.mkdir(parents=True,exist_ok=True)
    identity=dict(training_protocol=p['signature'],v4_checkpoint=sha(checkpoint),v3_checkpoint=sha(v3_checkpoint),
                  rows=digest(rows),source=str(source_root),runtimes=p['runtimes'])
    if (output/'identity.json').exists() and read(output/'identity.json')!=identity:
        raise ValueError('comparison identity changed; new output required')
    save(output/'identity.json',identity)
    callback=lambda stage,done=None,total=None,**kw:runner.progress(output,stage,done,total,**kw)
    current=v4.evaluate(output,source_root,rows,p,checkpoint,'benchmark','v4',callback)
    previous=control(output/'v3',source_root,rows,p,v3_checkpoint)
    for a,b in zip(current['clips'],previous['clips']):
        if a['id']!=b['id'] or a['baseline']!=b['baseline']:
            raise ValueError('v3/v4 sources or baseline reconstruction differ')
    # Identical source-derived seed policy and the same global upper bound.
    cap=current['measurement_cap']
    result=dict(status='MATCHED_V3_V4_COMPARISON_COMPLETE',identity=identity,
        v3=previous,v4=current,measurement_cap=cap,identical_source_and_baseline=True,
        same_measurement_cap=previous['measurement_attempts']<=cap and current['measurement_attempts']<=cap,
        runtime_note='wall time includes cache reuse; use fresh output directories for timing comparisons',
        benchmark_role='development/regression if previously used for design; not a fresh generalization test')
    save(output/'RESULT.json',result)
    runner.progress(output,'complete',len(rows),len(rows))
    value=read(output/'status.json');value.update(status='COMPLETE',exit_code=0);save(output/'status.json',value)
    return dict(status=result['status'],v3=previous['summary'],v4=current['summary'],measurement_cap=cap)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trained',type=Path,default=runner.DEFAULT)
    parser.add_argument('--source',type=Path,required=True,help='completed Kinetics/WebVid run containing prepared source frames')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--runtimes',type=Path)
    args=parser.parse_args()
    signal.signal(signal.SIGTERM,supervision.stop);signal.signal(signal.SIGINT,supervision.stop)
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'run.lock').open('a') as lock, (REPO/'.local/etri_60s_check.lock').open('a') as gpu:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps(compare(args.trained.resolve(),args.source.resolve(),args.output.resolve(),runtimes=args.runtimes)))


if __name__=='__main__':main()
