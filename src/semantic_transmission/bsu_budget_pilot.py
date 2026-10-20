"""Short, actually transmitted key-count/rate pilot with frozen LTX weights.

The added key is selected with a sender interpolation-error proxy, not a trained
estimate of LTX contribution. Captions are remapped from frozen factual text.
All arms use the same budget cap; exact measured utilization is reported.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time

from .artifacts import sha256, write_json
from .webvid5 import fingerprint
from .wire import pack, unpack
from .packets import accounting
from .qwen35_pairs_video import freeze
from .webvid_ablation import Stages

REPO = Path(__file__).resolve().parents[2]
MODULE = 'semantic_transmission.bsu_budget_pilot'
CONFIG = REPO / 'configs/bsu_budget_pilot.json'
SCHEMA = 'bsu_ntscc_plus_budget_inputs_v1'
COMPLETE = 'COMPLETE_BSU_KEY_BUDGET_PILOT'
ARMS = ('baseline', 'more_keys', 'key_quality')


def read(path):
    return json.loads(Path(path).read_text())


def maybe_read(path):
    try:
        return read(path)
    except (OSError, ValueError):
        return {}


def code_identity():
    from .bsu_gen import code_identity as base
    from .bsu_ltx_improvements import code_identity as receiver
    files = {str(REPO / p): h for p, h in base().items()}
    files.update(receiver())
    names = ('bsu_budget_pilot', 'ltx_improvement_validation', 'metadata_channel', 'packets',
             'transmission_accounting', 'official_quality', 'research_quality', 'stutter_metrics')
    files.update({str(Path(__file__).with_name(n+'.py')): sha256(Path(__file__).with_name(n+'.py')) for n in names})
    for p in (CONFIG, REPO/'scripts/run_bsu_budget_pilot.sh'):
        files[str(p)] = sha256(p)
    return files


def verify_contract(root):
    p = read(root/'protocol.json')
    if p['signature'] != fingerprint({k:v for k,v in p.items() if k!='signature'}):
        raise ValueError('frozen pilot protocol signature changed')
    # Codec/channel environments deliberately omit video/metric libraries.
    # Check every recorded byte without importing those implementations.
    for path,digest in p['code'].items():
        if sha256(path)!=digest:
            raise ValueError('frozen pilot implementation changed; restore snapshot before resuming')
    for path, digest in p['inputs'].items():
        if sha256(path) != digest:
            raise ValueError(f'frozen pilot input changed: {path}')
    return p


def plan(args):
    from .cli import settings
    from .ltx_improvement_validation import cases
    from . import ltx_receiver as ltx, ntscc_plus_codec as codec, bsu_gen
    from .qwen35_pairs_comparison import forbid_overlap
    from .video_io import probe
    root = args.output.resolve(); manifest = args.manifest.resolve(strict=True)
    policy = read(CONFIG)
    rows, inputs = [], {str(manifest): sha256(manifest)}
    for row in cases(manifest):
        header, _ = unpack(Path(row['packet']).read_bytes())
        keys = [k['index'] for k in header['keyframes']][:3]
        if len(keys) != 3 or keys[0] != 0 or keys[-1]+1 > policy['max_frames']:
            raise ValueError('pilot needs first three received semantic keys within the frame limit')
        info = probe(row['source'])
        if any(info[k] != header['video'][k] for k in ('width', 'height', 'fps')):
            raise ValueError('source and frozen semantic packet geometry differ')
        if (info['width'], info['height'], info['fps']) != (576, 320, 24):
            raise ValueError('registered pilot requires 576x320 at 24 FPS')
        rows.append(dict(id=row['id'], source=row['source'], semantic_packet=row['packet'],
            video=dict(header['video'], frames=keys[-1]+1), keys=keys, segments=header['segments'][:2]))
        inputs.update({row[k]: sha256(row[k]) for k in ('source', 'packet')})
    if len(rows) != 2 or len({r['id'] for r in rows}) != 2:
        raise ValueError('this predeclared short pilot requires two distinct cases')
    forbid_overlap(root, [REPO/'src', REPO/'configs', REPO/'.local', *map(Path, inputs)])
    local = settings(REPO)
    interpreters = dict(general=local['python'], channel=local['channel_python'],
        codec=str(REPO/'.local/ntscc_plus_env/bin/python'), ltx=str(REPO/'.local/envs/bsu-ltx25/bin/python'))
    for path in interpreters.values():
        if not Path(path).is_file(): raise FileNotFoundError(path)
    p = dict(policy=policy, cases=rows, inputs=inputs, code=code_identity(),
        current_bsu_profile=bsu_gen.profile(), interpreters=interpreters,
        codec=codec.installation(REPO), runtime=ltx.runtime(REPO/'.local/checkpoints/ltx25', REPO/'.local/vendor/LTX-2'),
        seeds=args.seeds or policy['generation_seeds'],
        selection='frozen first three semantic keys; add one highest sender interpolation residual or retain endpoints',
        captions='frozen caption facts remapped by temporal overlap; no new Qwen inference',
        limits=['known development prefixes, not unseen or full-video validation',
            'key changes also change generation windows and caption segmentation',
            'sender residual is a contribution proxy, not measured LTX contribution',
            'same budget cap; actual utilization and rate mismatch reported',
            'eta changes use frozen NTSCC+ weights; no online adaptation or new training'],
        independent_semantic_review='PENDING')
    if not p['seeds'] or len(p['seeds']) != len(set(p['seeds'])):
        raise ValueError('generation seeds must be nonempty and unique')
    p['signature'] = fingerprint(p)
    if root.exists() and not (root/'protocol.json').exists() and any(
            f.name not in ('run.log', 'execution.lock') for f in root.iterdir()):
        raise ValueError('unrecognized output; choose a new directory')
    if (root/'protocol.json').exists() and read(root/'protocol.json') != p:
        raise ValueError('frozen pilot configuration or inputs changed')
    return p


def select_added_key(frames, keys, params):
    """Sender-only proxy; no generated images, task labels, or evaluator scores."""
    import cv2
    import numpy as np
    small = [cv2.resize(f, tuple(params['selection_proxy_size']), interpolation=cv2.INTER_AREA).astype(float)
             for f in frames]
    gap = params['minimum_key_spacing']; scores = []
    for a, b in zip(keys, keys[1:]):
        for i in range(a+gap, b-gap+1):
            u = (i-a)/(b-a)
            prediction = (1-u)*small[a] + u*small[b]
            scores.append(dict(index=i, score=float(np.abs(small[i]-prediction).mean()), interval=[a,b]))
    if not scores: raise ValueError('no interior candidate satisfies the common key-spacing rule')
    best = max(scores, key=lambda v: (v['score'], -v['index']))
    return sorted([*keys, best['index']]), dict(selected=best, candidates=scores)


def remap_segments(old_keys, segments, new_keys):
    rows = []
    for a,b in zip(new_keys, new_keys[1:]):
        overlaps = [(s, max(0, min(b,y)-max(a,x))) for x,y,s in zip(old_keys,old_keys[1:],segments)]
        overlaps = [(s,n) for s,n in overlaps if n]
        if sum(n for _,n in overlaps) != b-a: raise ValueError('caption timeline not covered')
        texts = list(dict.fromkeys(s['text'] for s,_ in overlaps))
        rows.append(dict(path=f'sample/{a}_{b}.mp4', text=' '.join(texts),
                         flow=sum(float(s['flow'])*n for s,n in overlaps)/(b-a)))
    return rows


def prepare_source(row, dest, params):
    import numpy as np
    from PIL import Image
    dest.mkdir(parents=True)
    subprocess.run(['ffmpeg','-v','error','-nostdin','-threads','2','-i',row['source'],
        '-frames:v',str(row['video']['frames']),'-threads','2','-start_number','0',str(dest/'%05d.png')],check=True)
    frames = [np.asarray(Image.open(dest/f'{i:05d}.png').convert('RGB')) for i in range(row['video']['frames'])]
    added, audit = select_added_key(frames, row['keys'], params)
    selections = dict(baseline=row['keys'], more_keys=added, key_quality=[0,row['keys'][-1]])
    write_json(dest/'selection.json', dict(arms=selections, proxy=audit,
        source_sha256=sha256(row['source']), source_frame_hashes={str(i):sha256(dest/f'{i:05d}.png') for i in range(len(frames))}))


def rate_search(encode, budget, low, high, steps):
    """Use measured wire cost only. Never select a rate using decoded quality."""
    trials=[]
    def trial(eta):
        item = encode(round(eta,8)); trials.append(item); return item
    upper=trial(high)
    if upper['total_base_uses'] <= budget: return upper, trials
    lower=trial(low)
    if lower['total_base_uses'] > budget:
        raise ValueError('INFEASIBLE_REGISTERED_BUDGET: minimum rate exceeds cap')
    for _ in range(steps):
        middle=(low+high)/2; item=trial(middle)
        if item['total_base_uses'] <= budget: low=middle
        else: high=middle
    feasible=[t for t in trials if t['total_base_uses'] <= budget]
    return max(feasible,key=lambda t:(t['total_base_uses'],t['eta'])), trials


def encode_trial(model, row, keys, source, eta, output, params):
    import torch
    from PIL import Image
    from torchvision.transforms.functional import pil_to_tensor
    from . import ntscc_plus_codec as codec, bsu_ltx_pipeline as pipeline
    output.mkdir(parents=True); model.eta=eta
    torch.manual_seed(params['encoder_seed'])
    decoder=dict(pipeline.decoder_fields(dict(generation_conditioning=pipeline.MULTIKEY)),
        seed=2025,steps=11,policy='endpoint_exact',concatenation_policy='endpoint_exact')
    header=dict(schema=SCHEMA, video=row['video'], decoder=decoder, keyframes=[],
        checkpoint_sha256=codec.CHECKPOINT_SHA256,visual_codec=dict(codec.SPEC,eta=eta),
        segments=remap_segments(row['keys'],row['segments'],keys),complex_dtype='little_endian_complex64')
    payload=bytearray(); offset=0
    with (output/'visual.c64').open('xb') as stream, torch.inference_mode():
        for i in keys:
            x=pil_to_tensor(Image.open(source/f'{i:05d}.png').convert('RGB')).float().unsqueeze(0).cuda()/255
            values,rates,power=codec.encode_symbols(model,x)
            packed=codec.pack_rates(rates.cpu().tolist())
            header['keyframes'].append(dict(index=i,complex_offset=offset,complex_count=values.numel(),
                average_power=power,rate_offset=len(payload),rate_bytes=len(packed),rate_count=rates.numel()))
            payload.extend(packed); offset+=values.numel()
            stream.write(values.cpu().numpy().astype('<c8').tobytes())
    packet=pack(header,payload); (output/'metadata.bin').write_bytes(packet)
    digital=accounting(len(packet),bits_per_symbol=4)['complex_channel_uses']
    return dict(eta=eta,visual_uses=offset,digital_uses=digital,total_base_uses=offset+digital,
                metadata_bytes=len(packet),directory=str(output))


def encode_all(root,p):
    from .ntscc_plus_codec import load_codec
    model,_=load_codec(REPO); params=p['policy']
    for row in p['cases']:
        case=root/'cases'/row['id']; source=case/'source'; selections=read(source/'selection.json')['arms']
        encoding=case/'encoding'; encoding.mkdir(exist_ok=True)
        stages=Stages(encoding,p['signature']); budget=None
        for arm in ARMS:
            def work(_, arm=arm, budget=budget):
                dest=encoding/arm
                def trial(eta):
                    return encode_trial(model,row,selections[arm],source,eta,dest/'trials'/f'{eta:.8f}',params)
                if arm=='baseline':
                    best=trial(params['baseline_eta']); trials=[best]
                else:
                    low,high=(params['eta_min'],params['baseline_eta']) if arm=='more_keys' else (params['baseline_eta'],params['eta_max'])
                    best,trials=rate_search(trial,budget,low,high,params['rate_search_steps'])
                shutil.copytree(best['directory'],dest/'chosen')
                write_json(dest/'rate.json',dict(best=best,trials=trials,keys=selections[arm],
                    base_budget=best['total_base_uses'] if budget is None else budget, quality_used_for_selection=False))
            stages.step(arm,[arm],[arm+'/chosen',arm+'/rate.json'],work)
            if arm=='baseline': budget=read(encoding/arm/'rate.json')['best']['total_base_uses']
            print(f'ENCODE {row["id"]} {arm}: {read(encoding/arm/"rate.json")["best"]["total_base_uses"]}/{budget}',flush=True)


def prepare_transport(root,p):
    import cv2
    import numpy as np
    from PIL import Image
    from . import bsu_sequence_refinement as seq, bsu_ltx_pipeline as pipeline
    cv2.setNumThreads(2); cv2.setRNGSeed(2025)
    jobs=[]
    for row in p['cases']:
        case=root/'cases'/row['id']; baseline=read(case/'encoding/baseline/rate.json')['best']['total_base_uses']
        trajectory_cap=math.floor(baseline*p['policy']['trajectory_ratio'])
        for arm in ARMS:
            dest=root/'transport'/row['id']/arm; run=dest/'run'; run.mkdir(parents=True)
            rate=read(case/'encoding'/arm/'rate.json')
            shutil.copytree(case/'encoding'/arm/'chosen',run/'transmitter')
            cfg=dict(row['video'],visual_codec='NTSCC+',generator_backend='ltx25',generation_conditioning=pipeline.MULTIKEY,
                seed=2025,steps=11,snr_db=p['policy']['snr_db'],channel_seed=p['policy']['channel_seed'],
                visual_channel='source_frame_seeded_numpy_awgn_v1')
            write_json(run/'run_config.json',cfg)
            candidates=[]
            for a,b in zip(rate['keys'],rate['keys'][1:]):
                frames=np.stack([cv2.resize(np.asarray(Image.open(case/'source'/f'{i:05d}.png').convert('RGB')),
                     (288,160),interpolation=cv2.INTER_AREA) for i in range(a,b+1)])
                candidates.append(seq.sender_candidates(frames,a,b))
            sender=dest/'trajectory_sender'; sender.mkdir()
            packet,audit=seq.choose_packet(candidates,row['video'],sha256(run/'transmitter/metadata.bin'),trajectory_cap)
            uses=accounting(len(packet),bits_per_symbol=4)['complex_channel_uses'] if packet else 0
            if packet: (sender/'trajectory.bin').write_bytes(packet)
            write_json(sender/'sender.json',dict(packet_present=packet is not None,complex_channel_uses=uses,
                cap=trajectory_cap,audit=audit))
            if arm=='baseline': trajectory_cap=uses
            jobs.append(dict(id=row['id'],arm=arm,transport=str(dest),source=row['source'],frames=row['video']['frames'],
                keys=rate['keys'],eta=rate['best']['eta'],base_uses=rate['best']['total_base_uses'],
                trajectory_uses=uses,total_uses=rate['best']['total_base_uses']+uses,budget=baseline+trajectory_cap))
    write_json(root/'transport_plan.json',dict(jobs=jobs))


def channel_one(dest,p):
    from .qwen35_pairs_comparison import paired_channel
    from .metadata_channel import transmit
    from .bsu_sequence_refinement import decode_packet
    paired_channel(dest/'run')
    receiver=dest/'trajectory_received'; receiver.mkdir()
    source=dest/'trajectory_sender'; info=read(source/'sender.json')
    report=dict(status='PASSED',packet_present=False,complex_channel_uses=0)
    if info['packet_present']:
        packet=(source/'trajectory.bin').read_bytes()
        restored,report=transmit(packet,p['policy']['snr_db'],p['policy']['trajectory_channel_seed'],bits_per_symbol=4)
        if restored!=packet or report['bit_errors']: raise ValueError('trajectory channel failed')
        decode_packet(restored,(dest/'run/received/metadata.bin').read_bytes())
        (receiver/'trajectory.bin').write_bytes(restored)
        report.update(status='PASSED',packet_present=True,exact_match=True)
    if report['complex_channel_uses']!=info['complex_channel_uses']: raise ValueError('trajectory cost mismatch')
    write_json(receiver/'channel.json',report)


def receive_only(run,model=None):
    """Decode solely from the two received files; no run_config or source read."""
    import numpy as np
    import torch
    from torchvision.utils import save_image
    from . import ntscc_plus_codec as codec
    packet=(run/'received/metadata.bin').read_bytes(); header,payload=unpack(packet)
    spec=header.get('visual_codec',{}); eta=spec.get('eta')
    if (header.get('schema')!=SCHEMA or header.get('checkpoint_sha256')!=codec.CHECKPOINT_SHA256
            or not isinstance(eta,(int,float)) or not math.isfinite(eta) or not .005<=eta<=1.
            or spec!=dict(codec.SPEC,eta=eta)):
        raise ValueError('unknown received budget-codec profile')
    from .bsu_sequence_refinement import validate_keys
    validate_keys([i['index'] for i in header['keyframes']],header['video']['frames'])
    model=model if model is not None else codec.load_codec(REPO)[0]
    values=np.fromfile(run/'received/visual.c64',dtype='<c8')
    output=run/'receiver/frames/sample/key_frames_received'; output.mkdir(parents=True)
    cursor=rate_cursor=0
    with torch.inference_mode():
        for item in header['keyframes']:
            a,n=item['complex_offset'],item['complex_count']; r,m=item['rate_offset'],item['rate_bytes']
            if a!=cursor or r!=rate_cursor or n<1: raise ValueError('noncontiguous received streams')
            indices=codec.unpack_rates(payload[r:r+m],item['rate_count'])
            image=codec.decode_symbols(model,torch.from_numpy(values[a:a+n].copy()).cuda(),
                torch.tensor(indices,dtype=torch.long,device='cuda'),item['average_power'],
                header['video']['height'],header['video']['width'])
            save_image(image,output/f'{item["index"]}.png'); cursor+=n; rate_cursor+=m
    if cursor!=len(values) or rate_cursor!=len(payload): raise ValueError('unaccounted received symbols or metadata')
    write_json(run/'receiver_accounting.json',dict(status='PASSED',source_pixels_read=False,
        received_packet_sha256=sha256(run/'received/metadata.bin'),eta=eta,keys=len(header['keyframes'])))


def transport_worker(root,p,stage):
    model=None
    if stage=='receive':
        from .ntscc_plus_codec import load_codec
        model,_=load_codec(REPO)
    for job in read(root/'transport_plan.json')['jobs']:
        dest=Path(job['transport']); stages=Stages(dest,p['signature'])
        # Each worker has its own dependency chain. Root stages order workers.
        if stage=='channel':
            stages.step('channel',['run/received','run/channel_accounting.json','trajectory_received'],
                ['run/received','run/channel_accounting.json','trajectory_received'],lambda _:channel_one(dest,p))
        else:
            stages.step('receive',['run/receiver','run/receiver_accounting.json'],
                ['run/receiver','run/receiver_accounting.json'],lambda _:receive_only(dest/'run',model))


def prepare_requests(root,p):
    from . import ltx_conditioning as cond
    jobs=[]
    for job in read(root/'transport_plan.json')['jobs']:
        transport=Path(job['transport']); run=transport/'run'
        channel=read(run/'channel_accounting.json'); extras=read(transport/'trajectory_received/channel.json')
        actual=channel['total_complex_channel_uses']+extras['complex_channel_uses']
        if actual!=job['total_uses'] or actual>job['budget']: raise ValueError('actual channel budget violated')
        for seed in p['seeds']:
            dest=root/'jobs'/job['id']/job['arm']/f'seed_{seed}'
            trajectory=transport/'trajectory_received/trajectory.bin'
            request=cond.prepare(run/'received/metadata.bin',run/'receiver/frames/sample/key_frames_received',
                dest/'conditions',variant='context',trajectory_packet=trajectory if trajectory.exists() else None,seed=seed)
            freeze(dest/'request.json',request)
            jobs.append(dict(job,seed=seed,output=str(dest),request_sha256=sha256(dest/'request.json'),
                             generation_windows=len(request['windows'])))
    freeze(root/'generation_plan.json',dict(jobs=jobs))


def generate_all(root,p,factory=None):
    from . import ltx_receiver as ltx, ltx_conditioning as cond
    assets=dict(p['runtime'],execution_offload=p['policy']['execution_offload'])
    generator=None
    def cached(current):
        nonlocal generator
        if generator is None: generator=(factory or ltx.Generator)(current)
        return generator
    for job in read(root/'generation_plan.json')['jobs']:
        dest=Path(job['output'])
        if sha256(dest/'request.json')!=job['request_sha256']: raise ValueError('generation request changed')
        value=read(dest/'request.json'); cond.validate(value)
        ltx.reconstruct(value,assets,dest/'raw',factory=cached)
        print(f'GENERATED {job["id"]} {job["arm"]} seed {job["seed"]}',flush=True)


def finish(root,p):
    from . import bsu_ltx_improvements as rx
    from .ltx_improvement_validation import evaluate
    from PIL import Image
    import numpy as np
    jobs=read(root/'generation_plan.json')['jobs']; records=[]
    for job in jobs:
        dest=Path(job['output']); transport=Path(job['transport']); run=transport/'run'
        request=read(dest/'request.json'); rx.verified_raw(dest/'raw',request)
        keys=run/'receiver/frames/sample/key_frames_received'; trajectory=transport/'trajectory_received/trajectory.bin'
        stages=Stages(dest,p['signature'])
        stages.step('refine',['refined'],['refined'],lambda _:rx.refinement(request,dest/'raw',dest/'refined',
            run/'received/metadata.bin',keys,trajectory if trajectory.exists() else None))
        for i in job['keys']:
            if not np.array_equal(np.asarray(Image.open(keys/f'{i}.png')),
                                  np.asarray(Image.open(dest/'refined/frames'/f'{i:05d}.png'))):
                raise ValueError('final lossless received key changed')
        for kind in ('raw','refined'):
            stages.step('evaluate_'+kind,['evaluation/'+kind],['evaluation/'+kind],
                lambda _,kind=kind:evaluate(job['source'],dest/kind/'reconstruction.mp4',dest/'evaluation'/kind,
                    job['keys'],frame_count=job['frames']))
        union={i for j in jobs if j['id']==job['id'] for i in j['keys']}
        common={}
        for kind in ('raw','refined'):
            with (dest/'evaluation'/kind/'frames.csv').open() as stream:
                rows=[r for r in csv.DictReader(stream) if int(r['frame']) not in union]
            common[kind]={k:statistics.mean(float(r[k]) for r in rows) for k in
                          ('psnr_db','ssim','lpips_vgg','dists','clip')} if rows else {}
        key_scores=[]
        for i in job['keys']:
            source=np.asarray(Image.open(root/'cases'/job['id']/'source'/f'{i:05d}.png'),dtype=float)
            received=np.asarray(Image.open(keys/f'{i}.png'),dtype=float)
            key_scores.append(dict(index=i,psnr_db=float(10*np.log10(255**2/max(np.mean((source-received)**2),1e-12)))))
        records.append(dict(job,raw=read(dest/'evaluation/raw/quality.json'),
            refined=read(dest/'evaluation/refined/quality.json'),common_nonkey=common,
            received_key_quality=key_scores,mean_received_key_psnr_db=statistics.mean(v['psnr_db'] for v in key_scores),
            budget_utilization=job['total_uses']/job['budget'],received_keys_exact=True))
    comparisons=[]
    for r in records:
        b=next(v for v in records if v['id']==r['id'] and v['seed']==r['seed'] and v['arm']=='baseline')
        comparisons.append(dict(id=r['id'],seed=r['seed'],arm=r['arm'],total_channel_uses=r['total_uses'],
            budget=r['budget'],rate_difference_fraction=r['total_uses']/b['total_uses']-1,
            rate_matched_within_one_percent=abs(r['total_uses']/b['total_uses']-1)<=p['policy']['rate_match_tolerance'],
            refined_delta={k:r['refined']['quality'][k]-b['refined']['quality'][k] for k in
                           ('psnr_db','ssim','lpips_vgg','dists','clip')},
            common_nonkey_delta={k:r['common_nonkey']['refined'][k]-b['common_nonkey']['refined'][k]
                                 for k in r['common_nonkey']['refined']}))
    write_json(root/'RESULT.json',dict(status=COMPLETE,signature=p['signature'],records=records,
        comparisons=comparisons,training=False,default_adopted=False,generalization_proven=False,
        independent_semantic_review='PENDING',scope=p['limits']))


def launch_worker(root,p,stage,log):
    from .webvid_ablation import command, environment
    from .ntscc_plus_codec import runtime_environment
    runtime={'encode':'codec','receive':'codec','channel':'channel','generate':'ltx'}[stage]
    python=p['interpreters'][runtime]
    env=environment(2025); env.update(PYTHONPATH=str(REPO/'src'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    if runtime=='codec': env=runtime_environment(python,env)
    if runtime=='channel': env['CUDA_VISIBLE_DEVICES']='-1'
    command(REPO,[python,'-m',MODULE,'worker','--stage',stage,'--output',root],log,env)


def run(args):
    from .etri_60s_check import lock
    root=args.output.resolve(); p=plan(args); root.mkdir(parents=True,exist_ok=True)
    with lock(root/'execution.lock'):
        freeze(root/'protocol.json',p)
        # Archive code before the first GPU operation, including local vendor dependencies.
        for name,digest in p['code'].items():
            dest=root/'code_snapshot'/Path(name).relative_to(REPO)
            if not dest.exists(): dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(name,dest)
            if sha256(dest)!=digest: raise ValueError('code snapshot changed')
        freeze(root/'code_snapshot/MANIFEST.json',p['code'])
        prior=maybe_read(root/'status.json'); started=prior.get('started_epoch',time.time())
        state=dict(status='RUNNING',stage='source',started_epoch=started,pid=os.getpid(),
            process_start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
            jobs=len(p['cases'])*len(ARMS)*len(p['seeds']),evaluations=len(p['cases'])*len(ARMS)*len(p['seeds'])*2,
            log=str(root/'run.log'))
        def phase(name):
            state.update(stage=name); write_json(root/'status.json',state)
        stages=Stages(root,p['signature'])
        try:
            phase('source selection')
            for row in p['cases']:
                folder='cases/'+row['id']+'/source'
                stages.step('source_'+row['id'],[folder],[folder],lambda _,r=row,f=folder:prepare_source(r,root/f,p['policy']))
            phase('NTSCC+ rate search')
            encoded=[f'cases/{r["id"]}/encoding/{a}/{f}' for r in p['cases'] for a in ARMS for f in ('chosen','rate.json')]
            stages.step('encode',[],encoded,lambda log:launch_worker(root,p,'encode',log))
            phase('trajectory preparation')
            setup=['transport_plan.json']+[f'transport/{r["id"]}/{a}/{f}' for r in p['cases'] for a in ARMS
                for f in ('run/transmitter','run/run_config.json','trajectory_sender')]
            stages.step('transport_setup',['transport','transport_plan.json'],setup,lambda _:prepare_transport(root,p))
            phase('AWGN + LDPC/16QAM transmission')
            channel=[f'transport/{r["id"]}/{a}/{f}' for r in p['cases'] for a in ARMS
                for f in ('run/received','run/channel_accounting.json','trajectory_received')]
            stages.step('channel',[],channel,lambda log:launch_worker(root,p,'channel',log))
            phase('NTSCC+ received-only decoding')
            decoded=[f'transport/{r["id"]}/{a}/{f}' for r in p['cases'] for a in ARMS
                for f in ('run/receiver','run/receiver_accounting.json')]
            stages.step('receive',[],decoded,lambda log:launch_worker(root,p,'receive',log))
            phase('received multikey conditions')
            requests=['generation_plan.json']+[f'jobs/{r["id"]}/{a}/seed_{seed}/{f}' for r in p['cases'] for a in ARMS
                for seed in p['seeds'] for f in ('conditions','request.json')]
            stages.step('conditions',['jobs','generation_plan.json'],requests,lambda _:prepare_requests(root,p))
            phase('LTX generation')
            raw=[str(Path(j['output']).relative_to(root)/'raw') for j in read(root/'generation_plan.json')['jobs']]
            stages.step('generate',[],raw,lambda log:launch_worker(root,p,'generate',log))
            phase('refinement and evaluation')
            stages.step('finish',[],['RESULT.json']+[str(Path(j['output']).relative_to(root)/f)
                for j in read(root/'generation_plan.json')['jobs'] for f in ('refined','evaluation')],lambda _:finish(root,p))
            state.update(status=COMPLETE,stage='complete',ended_epoch=time.time()); write_json(root/'status.json',state)
        except BaseException as error:
            state.update(status='INTERRUPTED' if isinstance(error,KeyboardInterrupt) else 'FAILED',
                error=f'{type(error).__name__}: {error}',ended_epoch=time.time())
            write_json(root/'status.json',state); raise


def passed(paths):
    return sum(maybe_read(p).get('status')=='PASSED' for p in paths)


def status_lines(root,telemetry=True):
    state=maybe_read(root/'status.json'); p=maybe_read(root/'protocol.json'); result=maybe_read(root/'RESULT.json')
    status=state.get('status','NOT_STARTED')
    if status=='RUNNING':
        try:
            pid=state['pid']; current=Path(f'/proc/{pid}/stat').read_text().split()[21]
            if current!=state['process_start_ticks']: status='STOPPED_WITHOUT_FINAL_RESULT'
        except (OSError,KeyError): status='STOPPED_WITHOUT_FINAL_RESULT'
    if result.get('status')==COMPLETE and status==COMPLETE: status=COMPLETE
    count=len(p.get('cases',[]))*len(ARMS)
    def row(name,done,total):
        return f'{name:<12}{done}/{total} ({100*done/total:.1f}%)' if total else f'{name:<12}{done}/unknown'
    generation=maybe_read(root/'generation_plan.json').get('jobs',[])
    intervals=sum(j['generation_windows'] for j in generation)
    elapsed=max(0,state.get('ended_epoch',time.time())-state.get('started_epoch',time.time()))
    gpu='unavailable'
    if telemetry:
        try:
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total',
                '--format=csv,noheader'],text=True,timeout=3).strip()
        except (OSError,subprocess.SubprocessError): pass
    return [f'Status      {status}',f'Stage       {state.get("stage","not started")}',
        row('Encoding',passed(root.glob('cases/*/encoding/stages/*.json')),count),
        row('Channel',passed(root.glob('transport/*/*/stages/channel.json')),count),
        row('Decoding',passed(root.glob('transport/*/*/stages/receive.json')),count),
        row('Generation',sum(maybe_read(f).get('status')=='COMPLETE_LTX_GENERATION' for f in root.glob('jobs/*/*/seed_*/raw/generation.json')),state.get('jobs',0)),
        row('Intervals',passed(root.glob('jobs/*/*/seed_*/raw/stages/interval_*.json')),intervals),
        row('Evaluation',passed(root.glob('jobs/*/*/seed_*/stages/evaluate_*.json')),state.get('evaluations',0)),
        f'Elapsed     {int(elapsed)//3600}:{int(elapsed)%3600//60:02d}:{int(elapsed)%60:02d} | ETA unknown',
        f'GPU         utilization / used VRAM / total VRAM: {gpu}',
        'Resume      same run command; completed stage and interval hashes verified',
        f'Log         {root / "run.log"}',f'Result      {root / "RESULT.json"}',
        f'Failure     {state.get("error","none recorded")}']


def watch(root):
    first=True
    try:
        while True:
            lines=status_lines(root)
            if not first and sys.stdout.isatty(): print(f'\033[{len(lines)}F',end='')
            for line in lines: print(('\033[2K' if sys.stdout.isatty() else '')+line)
            sys.stdout.flush(); first=False
            state=maybe_read(root/'status.json').get('status')
            if state==COMPLETE: return 0
            if state in ('FAILED','INTERRUPTED') or 'STOPPED_' in lines[0]: return 1
            time.sleep(1)
    except KeyboardInterrupt:
        return 0  # Detaching the monitor never terminates the experiment.


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check','run','status','watch','worker'))
    parser.add_argument('--output',type=Path,default=REPO/'outputs/bsu_key_budget_pilot_20261019')
    parser.add_argument('--manifest',type=Path,default=REPO/'.local/validation/bsu_ltx_improvements_20261019/received_replay/PILOT_MANIFEST.json')
    parser.add_argument('--seeds',type=int,nargs='+')
    parser.add_argument('--stage',choices=('encode','channel','receive','generate'))
    args=parser.parse_args(argv); root=args.output.resolve()
    if args.action=='status': print('\n'.join(status_lines(root)))
    elif args.action=='watch': return watch(root)
    elif args.action=='check':
        p=plan(args); print(json.dumps(dict(status='READY_FILES_VERIFIED',cases=[r['id'] for r in p['cases']],
            seeds=p['seeds'],jobs=len(p['cases'])*len(ARMS)*len(p['seeds']),neural_inference_executed=False),indent=2))
    elif args.action=='worker':
        p=verify_contract(root)
        if args.stage=='encode': encode_all(root,p)
        elif args.stage in ('channel','receive'): transport_worker(root,p,args.stage)
        elif args.stage=='generate': generate_all(root,p)
        else: parser.error('worker requires --stage')
    else:
        def stop(*_): raise KeyboardInterrupt('termination requested')
        signal.signal(signal.SIGTERM,stop)
        run(args)


if __name__=='__main__':
    sys.exit(main())
