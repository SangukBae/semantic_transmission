"""Native-resolution training targets and strictly received/generated conditions."""
import csv
import json
from pathlib import Path
import numpy as np
from . import bsu_motion_validation as common
from .artifacts import sha256, write_json
from .adapter_training_state import atomic_torch_save, validate_config

POLICY = 'bsu_native_trajectory_adapter_training_v1'
DEFAULT_ROOT = common.REPO/'outputs/bsu_adapter_training_v1'
DEFAULT_CONFIG = common.REPO/'configs/bsu_adapter_training.json'
CODE = ('adapter_training_data.py','adapter_training_state.py','adapter_training_memory.py',
        'bsu_adapter_training.py','adapter_training_evaluation.py','trajectory_adapter.py',
        'trajectory_adapter_runtime.py','condition_collision.py','short_schedule.py',
        'bsu_adapter_native.py','bsu_adapter_validation.py','precision_text_cache.py',
        'generation_noise.py','tail_reference.py','decoder_memory.py','bsu_receiver.py',
        'trajectory_receiver.py','receiver_key_context.py','sparse_trajectory.py')


def fingerprint(value):
    import hashlib
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def all_cases():
    from .wire import unpack
    result = [dict(c, dataset='webvid') for c in common.cases() if c['snr']==10]
    base = common.REPO/'outputs/bsu_kinetics_context/benchmark'
    for v in common.read(base/'protocol.json')['videos']:
        folder=base/'snr_10'/v['id']; run=folder/'base/reconstruction/event_pairs/run'
        packet=run/'received/metadata.bin'; h,_=unpack(packet.read_bytes())
        keys=[k['index'] for k in h['keyframes']]; prep=common.read(v['prepared'])
        trajectory=folder/'received/trajectory.bin'
        result.append(dict(id=v['id'],dataset='kinetics',snr=10,keys=keys,frames=v['frames'],
            source_sha256=v['source_sha256'],source_movie=prep['normalized_video'],
            receiver=dict(base_packet=str(packet),trajectory=str(trajectory) if trajectory.exists() else None,
                key_paths=[str(run/f'receiver/frames/sample/key_frames_received/{k}.png') for k in keys],
                raw_frames=[str(run/f'receiver/reconstruction/sample_0000_frames/{i:05d}.png') for i in range(v['frames'])])))
    return result


def source_roles(cases):
    old=common.read(common.ROOT/'adapter/plan.json')['examples']
    dev={e['source_sha256'] for e in old if e['role']!='train'}
    trained={e['source_sha256'] for e in old if e['role']=='train'}
    trained |= {c['source_sha256'] for c in cases if c['dataset']=='webvid' and int(c['id'].split('_')[1]) in common.PILOT_NUMBERS}
    if dev & trained:raise ValueError('source roles overlap')
    # Prior evaluation sources are development/validation now, never called a
    # new blind test. Remaining sources are reserved for the final checkpoint.
    roles={c['source_sha256']:('validation' if c['source_sha256'] in dev else
        'train' if c['source_sha256'] in trained else 'test') for c in cases}
    if set(roles.values())!={'train','validation','test'}:raise ValueError('three source-disjoint splits required')
    return roles


def build_protocol(config_path=DEFAULT_CONFIG):
    from .wire import unpack
    from .bsu_sequence_refinement import decode_packet
    from .condition_collision import preview, upstream_functions
    config=validate_config(common.read(config_path)); cases=all_cases(); roles=source_roles(cases)
    functions=upstream_functions(common.REPO); examples=[]; excluded=[]; evaluation=[]; inputs={}
    def pin(path):
        path=str(path)
        if path not in inputs:inputs[path]=sha256(path)
    pin(common.ROOT/'adapter/plan.json');pin(common.CASES)
    for c in cases:
        role=roles[c['source_sha256']]; c['role']=role; run=Path(c['receiver']['base_packet']).parent.parent
        packet=Path(c['receiver']['base_packet']).read_bytes(); header,_=unpack(packet)
        if (header['video']['width'],header['video']['height'],header['video']['fps'])!=(576,320,24):raise ValueError('expected native 576x320 at 24fps')
        tracks=decode_packet(Path(c['receiver']['trajectory']).read_bytes(),packet) if c['receiver']['trajectory'] else {}
        pin(c['receiver']['base_packet']); pin(run/'run_config.json'); pin(run/'receiver/decoder_inputs.json'); pin(run/'keyframes.json')
        pin(run/'receiver/metadata.csv')
        with (run/'receiver/metadata.csv').open() as f:
            actual=[dict(row,flow=float(row['flow'])) for row in csv.DictReader(f)]
        if actual!=header['segments']:raise ValueError('decoder CSV differs from received metadata')
        if c['receiver']['trajectory']:pin(c['receiver']['trajectory'])
        for p in c['receiver']['key_paths']:pin(p)
        if role in ('validation','test'):
            if tracks:
                evaluation.append(c)
                pin(c['source_movie'])
                for t in range(c['frames']):pin(common.source_directory(c)/f'{t}.png')
            else:excluded.append(dict(id=c['id'],role=role,reason='no paid trajectory; adapter would be identity'))
        candidates=[]
        for i,mask in enumerate(preview(c['keys'],functions)):
            a,b=c['keys'][i:i+2]; frames=b-a+1+(17 if i else 0)
            fixed={t for o in mask['operations'] for t in range(o['target'],o['target']+o['length'])}
            if (a,b) not in tracks or len(fixed)==mask['latent_frames']:
                excluded.append(dict(id=c['id'],segment=i,role=role,reason='no paid track or no free latent'));continue
            if role=='test':continue
            # Actual generated history from the frozen receiver, not source RGB.
            history_indices=[max(0,a-16+j) for j in range(17)] if i else []
            history=[c['receiver']['raw_frames'][t] for t in history_indices]
            e=dict(name=f'{c["id"]}_{i:03d}',id=c['id'],dataset=c['dataset'],role=role,segment=i,
                start=a,end=b,frames=frames,latent_frames=mask['latent_frames'],mask_plan=mask,
                source_sha256=c['source_sha256'],source_directory=str(common.source_directory(c)),
                key_paths=c['receiver']['key_paths'][i:i+2],history=history,history_indices=history_indices,
                packet=c['receiver']['base_packet'],trajectory=c['receiver']['trajectory'],
                text=header['segments'][i]['text'],flow=header['segments'][i]['flow'])
            candidates.append(e)
        if role=='validation' and candidates:
            ordered=sorted(candidates,key=lambda e:(e['frames'],e['segment']))
            candidates=list({e['name']:e for e in (ordered[0],ordered[-1])}.values())
        for e in candidates:
            for t in range(e['start'],e['end']+1):pin(Path(e['source_directory'])/f'{t}.png')
            for p in e['history']:pin(p)
        examples.extend(candidates)
    code={str(common.REPO/'src/semantic_transmission'/n):sha256(common.REPO/'src/semantic_transmission'/n) for n in CODE}
    for name in ('scripts/train_bsu_adapter.sh','scripts/etri_trajectory_adapter_decoder.py','configs/official_opensora.py'):
        code[str(common.REPO/name)]=sha256(common.REPO/name)
    # Pin actual model files, including frozen backbone/encoder weights.
    cfg=common.read(Path(cases[0]['receiver']['base_packet']).parent.parent/'run_config.json')
    models={}
    for folder in {str(Path(p).resolve()) for p in cfg['models'].values() if isinstance(p,str) and Path(p).is_dir()}:
        for path in Path(folder).rglob('*'):
            if path.is_file() and path.suffix in ('.json','.bin','.safetensors','.model'):models[str(path)]=sha256(path)
    value=dict(policy=POLICY,config=config,cases=cases,evaluation=evaluation,examples=examples,excluded=excluded,
        inputs=inputs,models=models,code=code,roles=roles,
        training_scope='full existing semantic intervals, native 576x320 and 24fps; no temporal subsampling or frame cap',
        history_policy='frozen received generator RGB tail17, exact collision-repaired mask; never source RGB history',
        remaining_distribution_gap='training histories come from the frozen baseline, not online updated-adapter rollouts',
        validation_policy='fixed source-disjoint development loss plus full closed-loop generation; test never selects weights',
        test_scope='sources unused by the previous adapter training/evaluation; some have prior non-adapter diagnostics, not a blind population test',
        convergence_scope='validation plateau is an operational stopping rule, not proof of optimality or sufficient data diversity')
    import importlib.metadata
    value['packages']={n:importlib.metadata.version(n) for n in ('torch','transformers','numpy','flash-attn','apex')}
    value['signature']=fingerprint(value)
    return value


def protocol(root, config_path=DEFAULT_CONFIG, create=False):
    path=Path(root)/'protocol.json'
    if path.exists():
        p=common.read(path); signature=p['signature']
        if fingerprint({k:v for k,v in p.items() if k!='signature'})!=signature:raise ValueError('protocol tampered')
        if p['config']!=common.read(config_path):raise ValueError('configuration differs; use a new --root')
        import importlib.metadata
        if any(importlib.metadata.version(n)!=v for n,v in p['packages'].items()):raise ValueError('runtime packages changed')
        for group in ('code','inputs','models'):
            for file,digest in p[group].items():
                if sha256(file)!=digest:raise ValueError(f'changed {group}: {file}; use a new --root')
        return p
    p=build_protocol(config_path)
    if create:write_json(path,p)
    return p


def summary(p):
    return dict(policy=p['policy'],signature=p['signature'],
        sources={r:len({c['source_sha256'] for c in p['cases'] if c['role']==r}) for r in ('train','validation','test')},
        training_sources_with_tracks=len({e['id'] for e in p['examples'] if e['role']=='train'}),
        intervals={r:sum(e['role']==r for e in p['examples']) for r in ('train','validation')},
        largest_native_frames=max(e['frames'] for e in p['examples']),config=p['config'],
        sufficient_training_verified=False,scope=p['training_scope'])


def prepare_text(root, examples):
    from . import precision_text_cache as cache
    from .bsu_adapter_validation import model_config
    batch=Path(root)/'text';(batch/'receiver').mkdir(parents=True,exist_ok=True)
    with (batch/'receiver/metadata.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['path','text','flow']);writer.writeheader()
        writer.writerows(dict(path=f'clips/sample/{i:05d}.mp4',text=e['text'],flow=e['flow']) for i,e in enumerate(examples))
    cache.prepare(batch,common.REPO,model_config().pretty_text,'fp32')


def prepare_latents(root, examples, signature, event):
    import torch
    from PIL import Image
    from .bsu_adapter_validation import model_config, imports
    from .trajectory_adapter_runtime import native_condition
    from .bsu_sequence_refinement import decode_packet
    from .precision_text_cache import load
    root=Path(root); report=common.read(root/'text/receiver/text_embeddings.json')
    entries={e['prompt']:e for e in report['entries']}; modes,build=imports()
    vae=build(model_config().vae,modes).cuda().to(torch.bfloat16).eval(); receipts=[]
    for i,e in enumerate(examples):
        dest=root/'data'/f'{e["name"]}.pt'; receipt=dest.with_suffix('.json')
        if receipt.exists():
            old=common.read(receipt)
            if old['signature']!=signature or old['example']!=e or old['sha256']!=sha256(dest):raise ValueError('latent cache changed')
            receipts.append(old);continue
        def rgb(p):
            with Image.open(p) as im:
                if im.size!=(576,320):raise ValueError('native image dimensions differ')
                return np.asarray(im.convert('RGB'))
        source=np.stack([rgb(Path(e['source_directory'])/f'{t}.png') for t in range(e['start'],e['end']+1)])
        keys=np.stack([rgb(p) for p in e['key_paths']]); history=np.stack([rgb(p) for p in e['history']]) if e['history'] else None
        target_pixels=np.concatenate([history,source]) if history is not None else source
        tracks=decode_packet(Path(e['trajectory']).read_bytes(),Path(e['packet']).read_bytes())[e['start'],e['end']]
        condition=native_condition(tracks,keys,len(source),40,72,history is not None)
        def tensor(x):return torch.from_numpy(x.copy()).permute(3,0,1,2)[None].cuda().to(torch.bfloat16)/127.5-1
        with torch.inference_mode():
            torch.manual_seed(2025+i);target=vae.encode(tensor(target_pixels)).cpu()
            key_latents=[vae.encode(tensor(k[None])).cpu() for k in keys]
            history_latent=vae.encode(tensor(history)).cpu() if history is not None else None
        if target.shape[2]!=e['latent_frames'] or condition.shape[1]!=e['latent_frames']:raise ValueError('native temporal alignment mismatch')
        fixed,mask=fixed_latents(e,key_latents,history_latent,target)
        text=load(entries[report['prompts'][i]],report['contract'],report['prompts'][i])
        received=dict(fixed=fixed,mask=mask,condition=torch.from_numpy(condition)[None],y=text['y'],text_mask=text['mask'],frames=e['frames'],fps=24)
        atomic_torch_save(dict(target=target,received=received,example=e,signature=signature),dest)
        item=dict(path=str(dest),sha256=sha256(dest),signature=signature,example=e)
        write_json(receipt,item);receipts.append(item)
        event('native latent preparation',i+1,len(examples))
    write_json(root/'data.json',dict(signature=signature,receipts=receipts))


def fixed_latents(example,keys,history,target):
    import torch
    fixed=torch.zeros_like(target);mask=torch.ones((1,example['latent_frames']),dtype=torch.bool)
    for operation in example['mask_plan']['operations']:
        t,n=operation['target'],operation['length']
        reference=keys[0] if example['segment']==0 and operation['reference']==0 else (
            keys[1] if operation['reference']==example['segment']+1 else history)
        if reference is None or reference.shape[2]!=n:raise ValueError('invalid received/history condition')
        fixed[:,:,t:t+n]=reference;mask[:,t:t+n]=False
    return fixed,mask
