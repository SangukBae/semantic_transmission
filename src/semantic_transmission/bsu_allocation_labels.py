"""Build 600 measured NTSCC+/LTX allocation labels, audit, then fit the selector.

No synthetic quality labels. Each candidate is actually transmitted, decoded,
generated and evaluated. Dataset groups are never shuffled across source splits.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import statistics
import subprocess
import time

from .artifacts import sha256, write_json
from .webvid5 import fingerprint
from .qwen35_pairs_video import freeze
from .webvid_ablation import Stages, environment, command
from .bsu_allocation_transport import bundle, receipt, validate_candidate, wire_cost

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO/'configs/bsu_allocation_labels.json'
DEFAULT = REPO/'outputs/bsu_allocation_labels_600_20261019'
MODULE = 'semantic_transmission.bsu_allocation_labels'
ARMS = ('baseline', 'more_keys', 'key_quality', 'unequal_rates', 'different_times', 'random_valid')
COMPLETE = 'COMPLETE_BSU_ALLOCATION_600_LABELS'


def read(path):
    return json.loads(Path(path).read_text())


def maybe(path):
    return read(path) if Path(path).exists() else {}


def step(folder, name, identity, owned, callback):
    folder.mkdir(parents=True, exist_ok=True)
    Stages(folder, identity).step(name, owned, owned, callback)


def prepare_protocol(root):
    from . import ltx_receiver, ntscc_plus_codec
    from .qwen35_backend import MODEL_ID, REVISION, SNAPSHOT
    from .qwen35_caption_adapter import default_adapter
    from . import auto_extraction as ae
    if (root/'protocol.json').exists():
        return verify(root)
    corpus = read(root/'corpus/corpus.json'); policy = read(CONFIG)
    if corpus['policy'] != policy or len(corpus['records']) != 100:
        raise ValueError('expected the registered 100-source corpus')
    paths = list((REPO/'src/semantic_transmission').glob('*.py'))
    paths += [CONFIG, REPO/'configs/bsu_allocation.json', REPO/'configs/bsu_gen_ltx25.json',
              REPO/'configs/bsu_ltx_receiver_v2.json', REPO/'scripts/run_bsu_allocation_labels.sh']
    paths += list((REPO/'scripts').glob('qwen35*.py'))
    paths += list((REPO/'scripts').glob('qwen3vl*.py'))
    paths += list((REPO/'.local/vendor/NTSCC_plus').rglob('*.py'))
    code = {str(p):sha256(p) for p in paths if p.is_file()}
    settings = read(REPO/'.local/settings.json')
    p = dict(policy=policy, corpus=receipt(root/'corpus/corpus.json'), code=code,
        runtime=ltx_receiver.runtime(REPO/'.local/checkpoints/ltx25', REPO/'.local/vendor/LTX-2'),
        codec=ntscc_plus_codec.installation(REPO),
        interpreters=dict(general=settings['python'], channel=settings['channel_python'],
            codec=str(REPO/'.local/ntscc_plus_env/bin/python'),
            ltx=str(REPO/'.local/envs/bsu-ltx25/bin/python'),
            caption=str(REPO/'.local/qwen35_env/bin/python')),
        caption=dict(backend='qwen35', model_id=MODEL_ID, model_revision=REVISION,
            model_path=str(SNAPSHOT), quantization='nf4', max_pixels=262144, caption_max_words=79,
            caption_adapter=default_adapter(), prompts=dict(caption=ae.CAPTION_PROMPT, review=ae.REVIEW_PROMPT)),
        candidates=list(ARMS), normalized_source_is_evaluation_reference=True,
        annotation='actual final MP4 quality under fixed channel/generation seeds',
        baseline='uniform three temporal keys, not the production learned key selector',
        captions='one source-only Qwen draft+visual review per clip; identical facts for all candidates',
        limits=corpus['limits']+['finite six-candidate pool; no all-allocation oracle',
            'same budget cap, not necessarily identical utilization',
            'algorithmic motion diagnostics, not independent action/semantic review'])
    p['signature'] = fingerprint(p)
    freeze(root/'protocol.json', p)
    for source, digest in code.items():
        target = root/'code_snapshot'/Path(source).relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        if sha256(target) != digest: raise ValueError('snapshot copy changed')
    freeze(root/'code_snapshot/MANIFEST.json', code)
    return p


def verify(root):
    p = read(root/'protocol.json')
    if p['signature'] != fingerprint({k:v for k,v in p.items() if k != 'signature'}):
        raise ValueError('protocol changed')
    for path, digest in p['code'].items():
        if sha256(path) != digest:
            raise ValueError('frozen code changed; restore code_snapshot before resuming: '+path)
    if sha256(p['corpus']['path']) != p['corpus']['sha256']:
        raise ValueError('frozen source manifest changed')
    return p


def candidates(features, source_id, frames, spacing, seed):
    """Six source-only finite proposals. No received/generation quality input."""
    import numpy as np
    mid = (frames-1)//2
    base = [0, mid, frames-1]
    pixels = features[:, :48]
    scored = []
    for a,b in zip(base, base[1:]):
        for i in range(a+spacing, b-spacing+1):
            u = (i-a)/(b-a)
            scored.append((float(np.abs(pixels[i]-(1-u)*pixels[a]-u*pixels[b]).mean()), -i, i))
    added = max(scored)[2]
    alternative = max((float(np.abs(pixels[i]-(1-i/(frames-1))*pixels[0]
                        -i/(frames-1)*pixels[-1]).mean()), -i, i)
                      for i in range(spacing, frames-spacing) if i != mid)[2]
    rng = random.Random(f'{seed}:{source_id}')
    random_keys = sorted([0, rng.choice([i for i in range(spacing, frames-spacing)
                                      if i not in (mid,alternative)]), frames-1])
    sets = [base, sorted(base+[added]), [0,frames-1], base,
            [0,alternative,frames-1], random_keys]
    weights = [[1.]*len(k) for k in sets]
    weights[3] = [.8,1.4,.8]
    weights[5] = [round(rng.uniform(.7,1.3), 4) for _ in random_keys]
    result = [dict(id=arm, keys=k, weights=w) for arm,k,w in zip(ARMS,sets,weights)]
    for c in result:
        validate_candidate(dict(c, etas=[.2*w for w in c['weights']]), frames, spacing)
    return result


def caption_source(root, p, row, vlm):
    from PIL import Image
    from . import auto_extraction as ae, qwen35_pairs_comparison as pc
    dest = root/'labels'/row['id']; dest.mkdir(parents=True, exist_ok=True)
    indices = sorted({round(i*(row['video']['frames']-1)/7) for i in range(8)})
    paths = [root/'corpus/sources'/row['id']/'frames'/f'{i:05d}.png' for i in indices]
    identity = dict(protocol=p['signature'], source_id=row['source_id'],
        source_indices=indices, frame_hashes=[sha256(v) for v in paths])
    images = [Image.open(path).convert('RGB') for path in paths]
    labels = [f'Source frame {i}, time {i/24:.6f} seconds' for i in indices]
    try:
        cfg = p['caption']
        draft = pc.infer(vlm, dest/'caption_calls/draft.json', dict(identity,task='draft'),
            images, labels, cfg['prompts']['caption'], ae.caption_validator(cfg))
        review = pc.infer(vlm, dest/'caption_calls/review.json', dict(identity,task='visual_review',
            draft_sha256=ae.digest(draft)), images, labels, cfg['prompts']['review']+draft['caption'],
            ae.caption_validator(cfg, True))
    finally:
        for im in images: im.close()
    freeze(dest/'caption.json', dict(identity=identity, draft=draft, review=review,
        calls={str(v):sha256(v) for v in (dest/'caption_calls').glob('*.json')},
        independent_verification=False))


def encode_source(root, p, row, model):
    import numpy as np
    from .bsu_allocation_transport import encode_candidate
    dest = root/'labels'/row['id']; policy=p['policy']
    source = root/'corpus/sources'/row['id']/'frames'
    with np.load(root/'corpus/sources'/row['id']/'features.npz') as data:
        proposals = candidates(data['features'], row['source_id'], row['video']['frames'],
                               policy['minimum_key_spacing'], policy['seed'])
        flow = float(data['features'][1:,68].mean()*32)
    baseline_keys = proposals[0]['keys']
    caption = read(dest/'caption.json')['review']['caption']
    source_row = dict(row, keys=baseline_keys,
        segments=[dict(text=caption,flow=flow) for _ in baseline_keys[1:]])
    budget = None
    for proposal in proposals:
        folder = dest/proposal['id']; weights=proposal['weights']
        def work(_, proposal=proposal, folder=folder, budget=budget, weights=weights):
            trials = []
            def trial(scale):
                etas = [round(scale*w,8) for w in weights]
                candidate = dict(id=proposal['id'], keys=proposal['keys'], etas=etas)
                path = folder/'trials'/fingerprint(candidate)[:20]
                identity = fingerprint(dict(protocol=p['signature'], candidate=candidate,
                                            source=row['normalized_sha256'],caption=caption))
                def encode(_):
                    encode_candidate(model,source_row,candidate,source,path/'tx',seed=policy['encoder_seed'])
                step(path,'encode',identity,['tx'],encode)
                transport=bundle(path/'tx/visual.c64',path/'tx/metadata.bin')
                ledger=wire_cost(transport,candidate,row['video']['frames'])
                value=dict(candidate=candidate,bundle=transport,ledger=ledger,scale=scale)
                trials.append(value); return value
            if proposal['id']=='baseline':
                best=trial(policy['baseline_eta'])
            else:
                low=.005/min(weights); high=1/max(weights)
                invalid=[]
                # Released rate=1 masking can leave no channel symbols for a
                # very smooth image. Reject that rate, without inventing a label
                # or changing the source/noise seed, and probe the next rate.
                while True:
                    try:
                        lower=trial(low); break
                    except ValueError as exc:
                        if str(exc)!='invalid NTSCC+ channel input' or low>=high: raise
                        invalid.append(dict(scale=low,error=str(exc)))
                        write_json(folder/'invalid_rate_trials.json',invalid)
                        low=min(high,low*2)
                if lower['ledger']['total_uses']>budget:
                    raise ValueError('INFEASIBLE_REGISTERED_BUDGET at minimum eta')
                upper=trial(high)
                if upper['ledger']['total_uses']>budget:
                    for _ in range(policy['rate_search_steps']):
                        middle=(low+high)/2; value=trial(middle)
                        if value['ledger']['total_uses']<=budget: low=middle
                        else: high=middle
                best=max((t for t in trials if t['ledger']['total_uses']<=budget),
                         key=lambda t:(t['ledger']['total_uses'],t['scale']))
            shutil.copytree(Path(best['bundle']['visual']['path']).parent,folder/'chosen')
            write_json(folder/'allocation.json',dict(**best['candidate'],base_cost=best['ledger'],
                base_budget=budget or best['ledger']['total_uses'], trials=trials,
                quality_used_for_selection=False, proposal=proposal))
        # Trial receipts remain available if this outer stage is interrupted.
        step(folder,'allocation',p['signature'],['chosen','allocation.json'],work)
        if budget is None: budget=read(folder/'allocation.json')['base_cost']['total_uses']
        print(f'ENCODE {row["id"]} {proposal["id"]}',flush=True)


def prepare_transports(root,p,row):
    import cv2
    import numpy as np
    from PIL import Image
    from . import bsu_sequence_refinement as seq, bsu_ltx_pipeline as pipeline
    from .packets import accounting
    policy=p['policy']; cv2.setNumThreads(2); cv2.setRNGSeed(policy['encoder_seed'])
    dest=root/'labels'/row['id']; base=read(dest/'baseline/allocation.json')['base_cost']['total_uses']
    cap=math.floor(base*policy['trajectory_ratio']); baseline_trajectory=None
    frames=np.stack([cv2.resize(np.asarray(Image.open(root/'corpus/sources'/row['id']/'frames'/f'{i:05d}.png')),
                     (288,160),interpolation=cv2.INTER_AREA) for i in range(row['video']['frames'])])
    for arm in ARMS:
        folder=dest/arm; candidate=read(folder/'allocation.json')
        def work(_,folder=folder,candidate=candidate,cap=cap):
            run=folder/'transport/run'; run.mkdir(parents=True,exist_ok=True)
            shutil.copytree(folder/'chosen',run/'transmitter')
            write_json(run/'run_config.json',dict(row['video'],snr_db=policy['snr_db'],
                channel_seed=policy['channel_seed'],visual_channel='source_frame_seeded_numpy_awgn_v1',
                visual_codec='NTSCC+',generator_backend='ltx25',generation_conditioning=pipeline.MULTIKEY,
                seed=policy['generation_seed'],steps=11))
            options=[seq.sender_candidates(frames[a:b+1],a,b) for a,b in zip(candidate['keys'],candidate['keys'][1:])]
            packet,audit=seq.choose_packet(options,row['video'],sha256(run/'transmitter/metadata.bin'),cap)
            sender=folder/'transport/trajectory_sender'; sender.mkdir()
            uses=accounting(len(packet),bits_per_symbol=4)['complex_channel_uses'] if packet else 0
            if packet: (sender/'trajectory.bin').write_bytes(packet)
            write_json(sender/'sender.json',dict(packet_present=packet is not None,
                complex_channel_uses=uses,cap=cap,audit=audit))
        step(folder,'transport',p['signature'],['transport/run/transmitter',
            'transport/run/run_config.json','transport/trajectory_sender'],work)
        trajectory=folder/'transport/trajectory_sender/trajectory.bin'
        trajectory_cost=read(trajectory.parent/'sender.json')['complex_channel_uses']
        if arm=='baseline': baseline_trajectory=trajectory_cost; cap=trajectory_cost
        transport=bundle(folder/'chosen/visual.c64',folder/'chosen/metadata.bin',
                         [trajectory] if trajectory.exists() else [])
        cost=wire_cost(transport,candidate,row['video']['frames'])
        if cost['total_uses']>base+baseline_trajectory: raise ValueError('candidate exceeds full wire budget')
        freeze(folder/'job.json',dict(candidate=candidate,bundle=transport,cost=cost,
            budget=base+baseline_trajectory,source_id=row['source_id'],source=row['source'],
            source_sha256=row['normalized_sha256']))


def all_jobs(root,rows):
    return [root/'labels'/r['id']/arm for r in rows for arm in ARMS]


def verified_batch(root, rows, receipt_path):
    from .webvid_ablation import snapshot
    saved=read(receipt_path)
    if saved['ids']!=[r['id'] for r in rows]: raise ValueError('batch source order changed')
    for folder in all_jobs(root,rows):
        path=folder/'label.json'
        if saved['labels'].get(str(path))!=sha256(path): raise ValueError('completed label changed')
        # Check both outer pipeline receipts and the generator's interval receipts.
        for stage_root in (folder,folder/'raw'):
            for record in (stage_root/'stages').glob('*.json'):
                stage=read(record)
                if stage['status']!='PASSED' or snapshot(stage_root,stage['required'])!=stage['artifacts']:
                    raise ValueError('completed batch stage changed: '+str(record))
        label=read(path)
        for entry in (label['label_evidence'],label['final_movie']):
            if sha256(entry['path'])!=entry['sha256']: raise ValueError('completed evidence changed')


def worker(root,stage,ids,arms=None):
    p=verify(root); rows=[r for r in read(p['corpus']['path'])['records'] if r['id'] in ids]
    if len(rows)!=len(ids): raise ValueError('unknown or repeated source IDs')
    jobs=[f for f in all_jobs(root,rows) if f.name in (ARMS if arms is None else arms)]
    if stage=='caption':
        from .qwen35_backend import LocalQwen35
        vlm=LocalQwen35(p['caption'])
        for row in rows: caption_source(root,p,row,vlm)
    elif stage=='encode':
        from .ntscc_plus_codec import load_codec
        model,_=load_codec(REPO)
        for row in rows: encode_source(root,p,row,model)
    elif stage=='transport':
        for row in rows: prepare_transports(root,p,row)
    elif stage=='channel':
        from .bsu_budget_pilot import channel_one
        for folder in jobs:
            step(folder,'channel',p['signature'],['transport/run/received',
                'transport/run/channel_accounting.json','transport/trajectory_received'],
                lambda _,f=folder:channel_one(f/'transport',p))
    elif stage=='receive':
        from .ntscc_plus_codec import load_codec
        from .bsu_allocation_transport import decode_received
        model,_=load_codec(REPO)
        for folder in jobs:
            def decode(_,folder=folder):
                audit=decode_received(folder/'transport/run/received',folder/'received_keys',model=model)
                write_json(folder/'receiver_audit.json',audit)
            step(folder,'receive',p['signature'],['received_keys','receiver_audit.json'],decode)
    elif stage=='conditions':
        from . import ltx_conditioning as cond
        for folder in jobs:
            def conditions(_,folder=folder):
                job=read(folder/'job.json'); actual=read(folder/'transport/run/channel_accounting.json')['total_complex_channel_uses']
                actual+=read(folder/'transport/trajectory_received/channel.json')['complex_channel_uses']
                if actual!=job['cost']['total_uses'] or actual>job['budget']: raise ValueError('channel ledger mismatch')
                trajectory=folder/'transport/trajectory_received/trajectory.bin'
                value=cond.prepare(folder/'transport/run/received/metadata.bin',folder/'received_keys',
                    folder/'conditions',variant='context',trajectory_packet=trajectory if trajectory.exists() else None,
                    seed=p['policy']['generation_seed'])
                if value.get('source_pixels_available') is not False or any(
                    not Path(path).resolve().is_relative_to(folder.resolve()) for path in value['inputs']):
                    raise ValueError('generator condition escaped received-only candidate directory')
                write_json(folder/'request.json',value)
            step(folder,'conditions',p['signature'],['conditions','request.json'],conditions)
    elif stage=='generate':
        from . import ltx_receiver as ltx, ltx_conditioning as cond
        if ltx.runtime(REPO/'.local/checkpoints/ltx25',REPO/'.local/vendor/LTX-2')!=p['runtime']:
            raise ValueError('LTX runtime changed since protocol freeze')
        assets=dict(p['runtime'],execution_offload=p['policy']['execution_offload']); generator=None
        def cached(current):
            nonlocal generator
            if generator is None: generator=ltx.Generator(current)
            return generator
        for folder in jobs:
            value=read(folder/'request.json'); cond.validate(value)
            ltx.reconstruct(value,assets,folder/'raw',factory=cached)
            freeze(folder/'generation.json',dict(request_sha256=sha256(folder/'request.json'),
                movie=receipt(folder/'raw/reconstruction.mp4')))
            print('GENERATED '+str(folder),flush=True)
    elif stage=='evaluate':
        from PIL import Image
        import numpy as np
        from . import bsu_ltx_improvements as rx
        from .ltx_improvement_validation import evaluate
        for folder in jobs:
            job=read(folder/'job.json'); value=read(folder/'request.json'); rx.verified_raw(folder/'raw',value)
            trajectory=folder/'transport/trajectory_received/trajectory.bin'
            step(folder,'refine',p['signature'],['refined'],lambda _,f=folder,v=value,t=trajectory:
                rx.refinement(v,f/'raw',f/'refined',f/'transport/run/received/metadata.bin',
                              f/'received_keys',t if t.exists() else None))
            for key in job['candidate']['keys']:
                if not np.array_equal(np.asarray(Image.open(folder/'received_keys'/f'{key}.png')),
                    np.asarray(Image.open(folder/'refined/frames'/f'{key:05d}.png'))):
                    raise ValueError('refinement changed received key')
            step(folder,'evaluate',p['signature'],['evaluation'],lambda _,f=folder,j=job:
                evaluate(j['source'],f/'refined/reconstruction.mp4',f/'evaluation',j['candidate']['keys']))
            q=read(folder/'evaluation/quality.json')
            metrics=dict(psnr_db=q['quality']['psnr_db'],lpips_vgg=q['quality']['lpips_vgg'],
                         flow_error_mean=q['temporal']['flow_error_mean'])
            if not all(math.isfinite(v) for v in metrics.values()): raise ValueError('nonfinite ground truth')
            freeze(folder/'label.json',dict(id=job['candidate']['id'],keys=job['candidate']['keys'],
                etas=job['candidate']['etas'],bundle=job['bundle'],metrics=metrics,
                label_evidence=receipt(folder/'evaluation/quality.json'),
                final_movie=receipt(folder/'refined/reconstruction.mp4'),actual_channel_uses=job['cost']['total_uses']))
            print('LABEL '+str(folder),flush=True)
    else: raise ValueError('unknown worker stage')


def export(root,p):
    from .bsu_allocation_data import SCHEMA, FEATURE_SCHEMA, load_dataset
    rows=read(p['corpus']['path'])['records']; records=[]
    for row in rows:
        dest=root/'labels'/row['id']; labels=[]
        for arm in ARMS:
            folder=dest/arm; label=read(folder/'label.json'); job=read(folder/'job.json')
            q=read(label['label_evidence']['path'])
            if (sha256(label['final_movie']['path'])!=label['final_movie']['sha256']
                or q['video_sha256']!=label['final_movie']['sha256']
                or q['source_sha256']!=row['normalized_sha256'] or q['frames']!=row['video']['frames']):
                raise ValueError('quality label provenance mismatch')
            expected=dict(psnr_db=q['quality']['psnr_db'],lpips_vgg=q['quality']['lpips_vgg'],
                          flow_error_mean=q['temporal']['flow_error_mean'])
            if label['metrics']!=expected: raise ValueError('label differs from actual evaluator output')
            if wire_cost(label['bundle'],label,row['video']['frames'])['total_uses']>job['budget']:
                raise ValueError('over-budget training candidate')
            labels.append(label)
        budget=read(dest/'baseline/job.json')['budget']
        records.append(dict(id=row['id'],source_id=row['source_id'],source_sha256=row['source_sha256'],
            split=row['split'],frames=row['video']['frames'],budget=budget,snr_db=p['policy']['snr_db'],
            baseline_id='baseline',feature=row['feature'],minimum_key_spacing=p['policy']['minimum_key_spacing'],
            candidates=labels))
    document=dict(schema=SCHEMA,feature_schema=FEATURE_SCHEMA,feature_dim=71,pipeline_id=p['signature'],records=records)
    freeze(root/'dataset.json',document)
    _,loaded=load_dataset(root/'dataset.json')
    counts={s:sum(r['split']==s for r in records) for s in ('train','validation')}
    if counts!={'train':80,'validation':20} or sum(len(r['candidates']) for r in records)!=600:
        raise ValueError('600-label/80+20 source requirement not met')
    # Compare all six fixed family choices on the validation set. These are
    # controls, not post-hoc rules deployed on the same validation set.
    controls={}
    for arm in ARMS:
        deltas=[]
        for r in records:
            if r['split']!='validation': continue
            base=next(c for c in r['candidates'] if c['id']=='baseline')
            c=next(c for c in r['candidates'] if c['id']==arm)
            deltas.append({k:c['metrics'][k]-base['metrics'][k] for k in base['metrics']})
        controls[arm]={k:statistics.mean(d[k] for d in deltas) for k in deltas[0]}
    result=dict(status='VERIFIED_600_MEASURED_LABELS',sources=len(loaded),candidates=600,
        split_sources=counts,all_candidates_within_budget=True,validation_fixed_family_deltas=controls,
        dataset=receipt(root/'dataset.json'),generalization_proven=False,
        independent_semantic_review='PENDING',limits=p['limits'])
    write_json(root/'DATASET_AUDIT.json',result)
    return result


def validate_predictor(root):
    import numpy as np
    from .bsu_allocation_data import load_dataset
    from .bsu_allocation_training import load_predictor, evaluate_records
    document,records=load_dataset(root/'dataset.json')
    model,metadata=load_predictor(root/'predictor/best.json',document,device='cpu')
    validation=[r for r in records if r['row']['split']=='validation']
    result=evaluate_records(model,validation,metadata['objective'],'cpu')
    by_id={r['row']['id']:r['row'] for r in validation}; differences=[]
    for selected in result['rows']:
        row=by_id[selected['id']]
        baseline=next(c for c in row['candidates'] if c['id']=='baseline')
        differences.append({k:selected['selected_metrics'][k]-baseline['metrics'][k]
                            for k in baseline['metrics']})
    result['mean_metric_delta_vs_baseline']={k:statistics.mean(v[k] for v in differences) for k in differences[0]}
    rng=np.random.default_rng(20261019)
    gains=np.array([r['utility_gain_over_baseline'] for r in result['rows']])
    boot=gains[rng.integers(0,len(gains),(2000,len(gains)))].mean(1)
    result.update(utility_gain_bootstrap_95_percent=np.quantile(boot,[.025,.975]).tolist(),
        checkpoint=receipt(root/'predictor/best.json'),split='validation',
        checkpoint_selected_using_this_validation_set=True,untouched_test=False,
        performance_improvement_claim='requires inspection of individual metrics and independent test sources')
    write_json(root/'PREDICTOR_VALIDATION.json',result)
    return result


def run(args):
    from .etri_60s_check import lock
    from .ntscc_plus_codec import runtime_environment
    root=args.output.resolve(); root.mkdir(parents=True,exist_ok=True)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    with lock(root/'execution.lock'):
        p=prepare_protocol(root); rows=read(p['corpus']['path'])['records']
        previous=maybe(root/'status.json')
        state=dict(status='RUNNING',stage='starting',started_epoch=previous.get('started_epoch',time.time()),
            pid=os.getpid(),process_start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21])
        def phase(name): state.update(stage=name); write_json(root/'status.json',state)
        try:
            # First source is a real six-arm end-to-end gate; then 4-source batches.
            batches=[rows[:1]]+[rows[i:i+p['policy']['batch_sources']]
                for i in range(1,len(rows),p['policy']['batch_sources'])]
            for batch_index,batch in enumerate(batches):
                ids=[r['id'] for r in batch]
                completed=root/'batches'/f'{batch_index:03d}.json'
                if completed.exists():
                    phase(f'verify completed batch {batch_index+1}/{len(batches)}')
                    verified_batch(root,batch,completed)
                    print(f'REUSE completed batch {batch_index+1}',flush=True)
                    continue
                sequence=[(s,None) for s in ('caption','encode','transport','channel','receive','conditions')]
                if batch_index==0:
                    sequence += [('generate',['baseline']),('evaluate',['baseline'])]
                sequence += [('generate',None),('evaluate',None)]
                for stage,arms in sequence:
                    phase(f'batch {batch_index+1}/{len(batches)} {stage}'+(' baseline gate' if arms else ''))
                    runtime={'caption':'caption','encode':'codec','channel':'channel',
                             'receive':'codec','generate':'ltx'}.get(stage,'general')
                    python=p['interpreters'][runtime]; env=environment(p['policy']['seed'])
                    env.update(PYTHONPATH=str(REPO/'src'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
                    if runtime=='codec': env=runtime_environment(python,env)
                    if runtime=='channel': env['CUDA_VISIBLE_DEVICES']='-1'
                    log=root/'workers'/f'{batch_index:03d}_{stage}_{time.time_ns()}.log'
                    invocation=[python,'-m',MODULE,'worker','--output',root,'--stage',stage,'--ids',*ids]
                    if arms: invocation += ['--arms',*arms]
                    command(REPO,invocation,log,env)
                freeze(root/'batches'/f'{batch_index:03d}.json',dict(ids=ids,
                    labels={str(f/'label.json'):sha256(f/'label.json') for f in all_jobs(root,batch)}))
                if args.stop_after_sources and sum((root/'labels'/r['id']/'random_valid/label.json').exists()
                    for r in rows)>=args.stop_after_sources:
                    state.update(status='PARTIAL_MEASURED_LABELS',stage='requested bounded validation complete')
                    write_json(root/'status.json',state); return
            phase('verify all 600 labels'); audit=export(root,p)
            training=None
            if p['policy']['train_after_labels']:
                phase('fit predictor and validate on 20 sources')
                from .bsu_allocation_training import train
                training=train(root/'dataset.json',root/'predictor',read(REPO/'configs/bsu_allocation.json'),device='cpu')
                if training['status']!='COMPLETE_ALLOCATION_TRAINING':
                    state.update(status='INTERRUPTED',stage='predictor training interrupted; resumable',ended_epoch=time.time())
                    write_json(root/'status.json',state); return
                phase('validate best predictor checkpoint')
                validate_predictor(root)
            write_json(root/'RESULT.json',dict(status=COMPLETE,dataset_audit=audit,
                predictor_result=str(root/'predictor/RESULT.json') if training else None,
                default_adopted=False,generalization_proven=False,signature=p['signature']))
            state.update(status=COMPLETE,stage='complete',ended_epoch=time.time()); write_json(root/'status.json',state)
        except BaseException as exc:
            state.update(status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED',
                         error=f'{type(exc).__name__}: {exc}',ended_epoch=time.time())
            write_json(root/'status.json',state); raise


def status_lines(root):
    state=maybe(root/'status.json'); corpus=maybe(root/'corpus/acquisition_status.json')
    total=600; folders=list((root/'labels').glob('*/*')) if (root/'labels').exists() else []
    counts={name:sum((f/path).exists() for f in folders) for name,path in
        [('Encoding','allocation.json'),('Channel','transport/trajectory_received/channel.json'),
         ('Decoding','receiver_audit.json'),('Generation','generation.json'),('Labels','label.json')]}
    elapsed=max(0,state.get('ended_epoch',time.time())-state.get('started_epoch',time.time()))
    try:
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader'],text=True,timeout=3).strip()
    except (OSError,subprocess.SubprocessError): gpu='unavailable'
    running=state.get('status','ACQUIRING_SOURCES')
    if running=='RUNNING':
        proc=Path(f'/proc/{state["pid"]}/stat')
        if not proc.exists() or proc.read_text().split()[21]!=state['process_start_ticks']:
            running='STOPPED_WITHOUT_COMPLETION'
    lines=[f'Status      {running}',f'Stage       {state.get("stage","source acquisition")}',
           f'Sources     {corpus.get("selected",0)}/100',
           f'Captions    {len(list((root/"labels").glob("*/caption.json")))}/100']
    lines += [f'{name:<12}{value}/{total} ({100*value/total:.1f}%)' for name,value in counts.items()]
    requests=[read(f/'request.json') for f in folders if (f/'request.json').exists()]
    known=sum(len(r['windows']) for r in requests)
    done=sum(read(v).get('status')=='PASSED' for f in folders
             for v in (f/'raw/stages').glob('interval_*.json'))
    lines += [f'Intervals   {done}/{known} prepared (total unknown until all conditions are ready)',
        f'Elapsed     {elapsed/3600:.2f} hours | ETA unknown',
        f'GPU         utilization / used VRAM / total VRAM: {gpu}',
        'Resume      same run command; completed stages and LTX intervals are hash-checked',
        f'Log         {root/"run.log"}',f'Result      {root/"RESULT.json"}',
        f'Failure     {state.get("error") or "none recorded"}']
    return lines


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['run','worker','status','watch','audit'])
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--stage'); parser.add_argument('--ids',nargs='+',default=[])
    parser.add_argument('--arms',nargs='+',choices=ARMS)
    parser.add_argument('--stop-after-sources',type=int)
    args=parser.parse_args(); args.output=args.output.resolve()
    if args.command=='run': run(args)
    elif args.command=='worker': worker(args.output,args.stage,args.ids,args.arms)
    elif args.command=='audit': print(json.dumps(export(args.output,verify(args.output)),indent=2))
    else:
        try:
            while True:
                if args.command=='watch': print('\033[2J\033[H',end='')
                print('\n'.join(status_lines(args.output)),flush=True)
                if args.command!='watch': break
                time.sleep(3)
        except KeyboardInterrupt: pass


if __name__=='__main__': main()
