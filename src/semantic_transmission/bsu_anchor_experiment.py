"""Rate-matched, training-free BSU ideas 1/2 pilot on the frozen harbor input.

New mixed-resolution packets and received structure video stay in this explicit
experimental path. Existing runs/default policies are never mutated.
"""
import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .artifacts import sha256, write_json
from .webvid5 import read_json, fingerprint
from .wire import pack, unpack, pack_indices, unpack_indices
from .packets import accounting
from . import structure_anchors as anchors

REPO = Path(__file__).resolve().parents[2]
BASE = REPO/'outputs/bsu_zea_16s_20261009/reconstruction/event_pairs/run'
ROOT = REPO/'outputs/bsu_structure_anchor_20261011_v2'
BUDGET = 92475


def progress(label, done, total, start, eta=False):
    elapsed = time.monotonic()-start
    remaining = f' remaining~{elapsed/done*(total-done):.0f}s' if eta and done else ' ETA unknown'
    print(f'{label}: {done}/{total} ({100*done/total:.1f}%) elapsed={elapsed:.0f}s{remaining}', flush=True)


def load(root):
    p = read_json(root/'protocol.json')
    q = dict(p); signature = q.pop('signature')
    if fingerprint(q) != signature:
        raise ValueError('protocol changed')
    for group in ('code', 'inputs'):
        for path, digest in p[group].items():
            if sha256(path) != digest:
                raise ValueError(f'frozen {group} changed; use a new output root: {path}')
    return p


def prepare(root):
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from mmengine import Config
    from .codec_transport import load_codec, encode_symbols
    from .bsu_motion_experiment import source_window
    from .qwen35_pairs_comparison import frame_noise
    if (root/'protocol.json').exists():
        load(root); print('RESUME frozen preparation verified', flush=True); return
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2); cv2.setNumThreads(2); torch.manual_seed(2025)
    original = (BASE/'transmitter/metadata.bin').read_bytes()
    old, payload = unpack(original)
    if [k['index'] for k in old['keyframes']] != [0,383]:
        raise ValueError('expected frozen harbor endpoints')
    source = source_window(BASE/'data/normalized.mp4', 0, 383)
    sent = np.fromfile(BASE/'transmitter/visual.c64', dtype='<c8')
    received = np.fromfile(BASE/'received/visual.c64', dtype='<c8')
    configs = {}
    start = time.monotonic()
    base_dir = root/'transport/baseline'; base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir/'tx.bin').write_bytes(original)
    sent.tofile(base_dir/'tx.c64'); received.tofile(base_dir/'visual.c64')
    configs['baseline'] = dict(visual_uses=len(sent), packet_bytes=len(original),
        total_uses=len(sent)+accounting(len(original))['complex_channel_uses'])
    print('PREPARE loading NTSCC; encode half-resolution end key with padding to 64 pixels', flush=True)
    model, checkpoint = load_codec(REPO)
    if sha256(checkpoint) != old['checkpoint_sha256']:
        raise ValueError('NTSCC checkpoint changed')
    half = cv2.resize(source[-1], (288,160), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(half.copy()).permute(2,0,1).unsqueeze(0).float().cuda()/255
    tensor = F.pad(tensor, (0,32,0,32), mode='replicate')
    with torch.inference_mode():
        symbols, indices, power = encode_symbols(model,tensor)
    symbols = symbols.cpu().numpy().astype('<c8')
    raw_rates = pack_indices(indices.cpu().tolist())
    del model, tensor; torch.cuda.empty_cache()
    header = copy.deepcopy(old); header['schema'] = 'bsu_mixed_resolution_structure_v1'
    first = header['keyframes'][0]
    first_rates = payload[first['rate_offset']:first['rate_offset']+first['rate_bytes']]
    header['keyframes'][1] = dict(index=383, complex_offset=first['complex_count'], complex_count=len(symbols),
        average_power=power, rate_offset=len(first_rates), rate_bytes=len(raw_rates), rate_count=indices.numel(),
        coded_height=192, coded_width=320, content_height=160, content_width=288,
        pad='replicate_right_bottom', receiver_resize='bilinear_align_corners_false')
    reduced = pack(header, first_rates+raw_rates)
    new_sent = np.concatenate([sent[:first['complex_count']], symbols])
    new_rx = np.concatenate([received[:first['complex_count']], symbols+frame_noise(383,len(symbols),42,10)])
    reduced_dir = root/'transport/endhalf'; reduced_dir.mkdir(parents=True, exist_ok=True)
    (reduced_dir/'tx.bin').write_bytes(reduced)
    new_sent.astype('<c8').tofile(reduced_dir/'tx.c64'); new_rx.astype('<c8').tofile(reduced_dir/'visual.c64')
    configs['endhalf'] = dict(visual_uses=len(new_sent), packet_bytes=len(reduced),
        total_uses=len(new_sent)+accounting(len(reduced))['complex_channel_uses'])
    progress('PREPARE key configurations', 2,2,start)
    times = list(range(0,384,8)); curve = []; selected = []
    for width,height in ((72,40),(144,80)):
        low = np.stack([cv2.resize(source[i],(width,height),interpolation=cv2.INTER_AREA) for i in times])
        trials = root/'compression'/f'{width}x{height}'; trials.mkdir(parents=True,exist_ok=True)
        candidates=[]
        for crf in (26,32,38,44,50):
            bits = anchors.compress(low,trials/f'crf{crf}.h264',3,crf)
            packet = anchors.add_video(reduced,bits,width,height,times,384)
            ledger = accounting(len(packet)); total=len(new_sent)+ledger['complex_channel_uses']
            row=dict(width=width,height=height,fps=3,crf=crf,bitstream_bytes=len(bits),packet_bytes=len(packet),
                visual_uses=len(new_sent),digital_uses=ledger['complex_channel_uses'],total_uses=total,
                within_budget=total<=BUDGET)
            curve.append(row)
            print(f'RATE {width}x{height} CRF={crf}: {len(bits)} video bytes, {total}/{BUDGET} channel uses',flush=True)
            if total<=BUDGET:candidates.append((row,packet))
        if not candidates:
            continue
        # Predeclared: least lossy CRF fitting the cap, independent of generated quality.
        row, packet = candidates[0]
        name=f'endhalf_{width}x{height}'; directory=root/'transport'/name;directory.mkdir(parents=True,exist_ok=True)
        (directory/'tx.bin').write_bytes(packet)
        shutil.copyfile(reduced_dir/'tx.c64',directory/'tx.c64')
        shutil.copyfile(reduced_dir/'visual.c64',directory/'visual.c64')
        configs[name]=dict(row); selected.append(name)
    if not selected:
        raise ValueError('no structure candidate fits budget; inspect compression curve')
    write_json(root/'rate_curve.json',dict(configurations=configs,compression=curve,budget=BUDGET))
    cfg=Config.fromfile(str(BASE/'receiver/decoder_config.py'))
    arms=[dict(name='baseline',transport='baseline',ratio=None),dict(name='endhalf',transport='endhalf',ratio=None)]
    arms.extend(dict(name=f'{name}_r{str(ratio).replace(".","")}',transport=name,ratio=ratio)
                for name in selected for ratio in (.35,.6))
    input_paths=[BASE/'data/normalized.mp4',BASE/'receiver/decoder_config.py',BASE/'receiver/generation_noise.json',
        BASE/'receiver/text_embeddings.json',BASE/'transmitter/metadata.bin',BASE/'received/visual.c64',checkpoint]
    text=read_json(BASE/'receiver/text_embeddings.json')
    input_paths.extend(Path(e['path']) for e in text['entries'])
    input_paths.extend((BASE/'receiver/frames/sample/key_frames_received').glob('*.png'))
    input_paths.extend((BASE/'receiver/reconstruction/sample_0000_frames').glob('*.png'))
    input_paths.extend(path for name in configs for path in (root/'transport'/name).iterdir() if path.is_file())
    code_names=['bsu_anchor_experiment.py','structure_anchors.py','codec_transport.py','generation_noise.py',
        'receiver_temporal_alignment.py','condition_collision.py','official_quality.py','precision_text_cache.py',
        'receiver_alignment_evaluation.py','wire.py','metadata_channel.py']
    code={str(REPO/'src/semantic_transmission'/n):sha256(REPO/'src/semantic_transmission'/n) for n in code_names}
    vendor=REPO/'.local/vendor/Open-Sora/opensora'
    for sub in ('models/vae','models/stdit','models/layers','schedulers'):
        code.update({str(p):sha256(p) for p in (vendor/sub).rglob('*.py')})
    code[str(vendor/'utils/inference_utils.py')]=sha256(vendor/'utils/inference_utils.py')
    p=dict(policy='bsu_structure_anchors_rate_pilot_v1',budget=BUDGET,frames=384,fps=24,width=576,height=320,
        generator={k:cfg[k] for k in ('model','vae','image_size','fps','seed','scheduler','multi_resolution')},
        arms=arms,transports=configs,source=str(BASE/'data/normalized.mp4'),base=str(BASE),
        inputs={str(path):sha256(path) for path in input_paths},code=code,
        source_sha256=sha256(BASE/'data/normalized.mp4'),caption_changed=False,training=False,
        protocol_prior='fixed receiver ratio per arm; installed weights/code excluded as historical comparator',
        candidate_selection='first fitting CRF in [26,32,38,44,50], no generator quality used',
        hypothesis='end-key rate reallocation plus received low-resolution intermediate structure',
        test_scope='one known harbor 16s development video; one seed/SNR; not heldout generalization',
        metrics_domain='RGB PNG, common381 and all384; independent semantic review pending',
        target_png=dict(psnr_db=20.969,lpips_vgg=.27832),snr_db=10,channel_seed=42)
    p['signature']=fingerprint(p);write_json(root/'protocol.json',p)
    print(f'PREPARE complete: {len(arms)} generation arms; {len(configs)} real packets; no training',flush=True)


def channel(root):
    from .metadata_channel import transmit
    p=load(root);start=time.monotonic();reports=[]
    for i,(name,config) in enumerate(p['transports'].items(),1):
        directory=root/'transport'/name;packet=(directory/'tx.bin').read_bytes()
        if (directory/'channel.json').exists():
            report=read_json(directory/'channel.json')
            if report['signature']!=p['signature'] or sha256(directory/'rx.bin')!=report['received_sha256']:
                raise ValueError('stale channel result')
        else:
            received,report=transmit(packet,p['snr_db'],p['channel_seed'])
            unpack(received)
            if received!=packet:raise ValueError('digital packet mismatch')
            (directory/'rx.bin').write_bytes(received)
            report.update(signature=p['signature'],exact_match=True,received_sha256=sha256(directory/'rx.bin'),
                visual_complex_channel_uses=config['visual_uses'],total_complex_channel_uses=config['total_uses'])
            write_json(directory/'channel.json',report)
        reports.append(dict(name=name,**report));progress('CHANNEL packets',i,len(p['transports']),start,True)
    write_json(root/'channel.json',dict(status='PASSED',signature=p['signature'],reports=reports))


def receive(root):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torchvision.utils import save_image
    from .codec_transport import load_codec,decode_symbols
    p=load(root);torch.set_num_threads(2);model,checkpoint=load_codec(REPO);start=time.monotonic()
    for i,name in enumerate(p['transports'],1):
        directory=root/'transport'/name;out=directory/'receiver';out.mkdir(exist_ok=True)
        header,payload=unpack((directory/'rx.bin').read_bytes())
        if sha256(checkpoint)!=header['checkpoint_sha256']:raise ValueError('wrong decoder checkpoint')
        values=np.fromfile(directory/'visual.c64',dtype='<c8')
        if len(values)!=sum(k['complex_count'] for k in header['keyframes']):raise ValueError('visual length mismatch')
        offset=rate_offset=0
        with torch.inference_mode():
            for item in header['keyframes']:
                if item['complex_offset']!=offset or item['rate_offset']!=rate_offset:
                    raise ValueError('invalid stream offsets')
                count=item['complex_count'];raw=payload[rate_offset:rate_offset+item['rate_bytes']]
                indices=unpack_indices(raw,item['rate_count'])
                restored=decode_symbols(model,torch.from_numpy(values[offset:offset+count].copy()).cuda(),
                    torch.tensor(indices,device='cuda'),item['average_power'],
                    item.get('coded_height',header['video']['height']),item.get('coded_width',header['video']['width']))
                if 'coded_height' in item:
                    restored=restored[:,:,:item['content_height'],:item['content_width']]
                    restored=F.interpolate(restored,size=(header['video']['height'],header['video']['width']),
                        mode='bilinear',align_corners=False).clamp(0,1)
                save_image(restored,out/f'{item["index"]}.png')
                offset+=count;rate_offset+=item['rate_bytes']
        if anchors.FIELD in header:
            item,bits=anchors.video_payload((directory/'rx.bin').read_bytes())
            low=anchors.decode_video(bits,item);np.save(out/'structure.npy',low)
            write_json(out/'structure.json',item)
        elif rate_offset!=len(payload):raise ValueError('unconsumed payload')
        if header['segments']!=unpack((BASE/'received/metadata.bin').read_bytes())[0]['segments']:
            raise ValueError('received caption changed')
        paths=[path for path in out.iterdir() if path.name!='receipt.json']
        write_json(out/'receipt.json',dict(status='DECODED',signature=p['signature'],source_pixels_read=False,
            artifacts={path.name:sha256(path) for path in paths}))
        progress('RECEIVE complete transports',i,len(p['transports']),start)


def generate(root):
    import numpy as np
    import torch
    from PIL import Image
    from mmengine import Config
    from opensora.registry import MODELS,SCHEDULERS,build_module
    from opensora.datasets.utils import get_transforms_image
    from opensora.utils import inference_utils as inference
    from .bsu_motion_experiment import setup_torch
    from .condition_collision import apply_guard
    from .generation_noise import NoiseControl,tensor_hash
    from .precision_text_cache import Encoder
    from .receiver_alignment_experiment import complete_arm
    from .receiver_joint_experiment import save_pixels,seed_for
    p=load(root);setup_torch(p);cfg=Config(p['generator']);signature=p['signature']
    if all(complete_arm(root/'generation'/a['name'],signature) for a in p['arms']):
        print('RESUME all generation artifacts verified',flush=True);return
    for name in p['transports']:
        directory=root/'transport'/name/'receiver';receipt=read_json(directory/'receipt.json')
        if receipt['signature']!=signature or any(sha256(directory/k)!=v for k,v in receipt['artifacts'].items()):
            raise ValueError('receiver artifacts changed')
    print('GENERATE loading frozen VAE; structure VAE cache uses received bytes only',flush=True)
    vae=build_module(cfg.vae,MODELS).to('cuda',torch.bfloat16).eval().requires_grad_(False)
    structure={}
    with torch.inference_mode():
        for name in p['transports']:
            directory=root/'transport'/name/'receiver'
            if not (directory/'structure.npy').exists():continue
            cache=directory/'structure_latent.pt';receipt=directory/'structure_latent.json'
            if receipt.exists():
                record=read_json(receipt)
                if record['signature']!=signature or record['sha256']!=sha256(cache):raise ValueError('stale structure latent')
                latent=torch.load(cache,map_location='cpu',weights_only=True)
            else:
                item=read_json(directory/'structure.json');low=np.load(directory/'structure.npy')
                expanded=anchors.expand_video(low,item,p['height'],p['width'])
                x=torch.from_numpy(expanded).permute(3,0,1,2).unsqueeze(0).to('cuda',torch.bfloat16)/127.5-1
                # Extra reference encoding never consumes endpoint/generation RNG.
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.manual_seed(seed_for('structure_vae',0,cfg.seed));latent=vae.encode(x).cpu()
                torch.save(latent,cache)
                write_json(receipt,dict(signature=signature,sha256=sha256(cache),shape=list(latent.shape),
                    seed=seed_for('structure_vae',0,cfg.seed),tensor_sha256=tensor_hash(latent),source_pixels_read=False,
                    input_sha256=sha256(directory/'structure.npy'),temporally_aligned_full_sequence=True))
                del x,expanded;torch.cuda.empty_cache()
            structure[name]=latent
            print(f'CONDITION cached {name}: {list(latent.shape)}',flush=True)
    print('GENERATE loading frozen STDiT; ETA unknown',flush=True)
    model=build_module(cfg.model,MODELS,input_size=vae.get_latent_size((p['frames'],*cfg.image_size)),
        in_channels=vae.out_channels,caption_channels=4096,model_max_length=300,
        enable_sequence_parallelism=False).to('cuda',torch.bfloat16).eval().requires_grad_(False)
    runtime=dict(packages={n:importlib.metadata.version(n) for n in ('torch','diffusers','transformers')},
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name())
    expected=read_json(BASE/'receiver/generation_noise.json')
    transform=get_transforms_image(image_size=tuple(cfg.image_size),name='resize_crop')
    start=time.monotonic();finished=0
    with torch.inference_mode():
        for arm in p['arms']:
            directory=root/'generation'/arm['name']
            if complete_arm(directory,signature):finished+=1;continue
            directory.mkdir(parents=True,exist_ok=True);arm_start=time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            control=NoiseControl(directory/'generation_noise.json',expected['contract'],runtime,expected)
            received=root/'transport'/arm['transport']/'receiver';refs=[[]]
            for j,key in enumerate((0,383)):
                with Image.open(received/f'{key}.png') as im:
                    x=transform(im.convert('RGB')).unsqueeze(0).unsqueeze(2).to('cuda',torch.bfloat16)
                with control.scope('vae',j,x):refs[0].append(vae.encode(x).squeeze(0))
            latent_size=vae.get_latent_size((p['frames'],*cfg.image_size));model.config.input_size=latent_size
            z=torch.empty(1,vae.out_channels,*latent_size,device='cuda',dtype=torch.bfloat16)
            with control.scope('initial',0,z):z.copy_(torch.randn_like(z))
            mask,conditions=apply_guard(z,refs,['0;0,1,0,-1,1'],0,5,2,inference)
            protected=torch.where(mask[0]==0)[0].tolist();plan=None
            if arm['ratio'] is not None:
                item=read_json(received/'structure.json')
                plan=anchors.anchor_slots(p['frames'],item['timestamps'],protected)
                prior=structure[arm['transport']].to('cuda')
                if prior.shape!=z.shape:raise ValueError('structure timeline mismatch')
                z[:,:,plan['slots']]=prior[:,:,plan['slots']]
                mask[:,plan['slots']]=arm['ratio'];del prior
            initial=z.clone()
            report=copy.deepcopy(read_json(BASE/'receiver/text_embeddings.json'))
            encoder=Encoder(report,directory/'text_embedding_trace.json');encode=encoder.encode
            encoder.encode=lambda texts:{k:v.to('cuda',dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
                                         for k,v in encode(texts).items()}
            encoder.y_embedder=model.y_embedder;prompt=report['prompts'][0]
            calls=[0]
            def tick(module,args):
                calls[0]+=1
                progress(f'GENERATE {arm["name"]} steps',calls[0],cfg.scheduler.num_sampling_steps,arm_start,True)
            hook=model.register_forward_pre_hook(tick)
            extra=inference.prepare_multi_resolution_info(cfg.multi_resolution,1,tuple(cfg.image_size),
                p['frames'],cfg.fps,'cuda',torch.bfloat16)
            scheduler=build_module(cfg.scheduler,SCHEDULERS)
            try:
                with control.scope('sampling',0,z):
                    sampled=scheduler.sample(model,encoder,z=z,prompts=[prompt],device='cuda',
                        additional_args=extra,progress=False,mask=mask)
            finally:hook.remove()
            control.finish(1,cfg.scheduler.num_sampling_steps)
            if not torch.equal(initial[:,:,protected],sampled[:,:,protected]):raise ValueError('hard keys changed')
            if not torch.isfinite(sampled).all():raise ValueError('nonfinite generated latent')
            decoded=vae.decode(sampled.to(torch.bfloat16),num_frames=p['frames'])
            pixels=(decoded[0].clamp(-1,1).add(1).div(2).mul(255).add(.5).clamp(0,255)
                    .permute(1,2,3,0).to('cpu',torch.uint8).numpy())
            if arm['name']=='baseline':
                from .research_quality import read_frames
                historic=read_frames(BASE/'receiver/reconstruction/sample_0000_frames')
                parity=dict(exact_pixels=bool(np.array_equal(pixels,historic)),
                    max_absolute_error=int(np.abs(pixels.astype(int)-historic.astype(int)).max()))
                write_json(root/'baseline_parity.json',parity)
                if not parity['exact_pixels']:raise ValueError('baseline replay differs; stop before candidate claims')
            save_pixels(directory,pixels,cfg.fps)
            write_json(directory/'conditions.json',dict(conditions=conditions,structure=plan,ratio=arm['ratio'],
                prompt=prompt,hard_keys_preserved=True,source_pixels_read=False,
                input_keys={str(k):sha256(received/f'{k}.png') for k in (0,383)},
                actual_generation_noise_matches_historical=True,
                initial_conditioned_latent_sha256=tensor_hash(initial),
                structure_latent_sha256=tensor_hash(structure[arm['transport']]) if plan else None))
            paths=[*sorted((directory/'frames').glob('*.png')),directory/'reconstruction.mp4',
                directory/'generation_noise.json',directory/'conditions.json']
            write_json(directory/'result.json',dict(status='GENERATED',signature=signature,arm=arm,frames=p['frames'],
                seconds=time.monotonic()-arm_start,max_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                artifacts={str(path.relative_to(directory)):sha256(path) for path in paths}))
            finished+=1;progress('GENERATE completed arms',finished,len(p['arms']),start)
            del refs,x,z,initial,sampled,decoded,pixels,encoder;torch.cuda.empty_cache()


def evaluate(root):
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from .bsu_motion_experiment import source_window
    from .research_quality import read_frames
    from .official_quality import OfficialMetrics,summarize
    from .receiver_alignment_evaluation import temporal_scores,comparison_video
    from .receiver_alignment_experiment import complete_arm
    p=load(root);torch.set_num_threads(2);cv2.setNumThreads(2)
    source=source_window(p['source'],0,p['frames']-1);start=time.monotonic();results=[]
    for number,arm in enumerate(p['arms'],1):
        directory=root/'generation'/arm['name']
        if not complete_arm(directory,p['signature']):raise ValueError('generation incomplete')
        pixels=read_frames(directory/'frames')
        if (directory/'metrics.json').exists():
            row=read_json(directory/'metrics.json')
            if row['signature']!=p['signature']:raise ValueError('stale metrics')
        else:
            def tick(done,total):
                if done==1 or done%192==0 or done==total:progress(f'QUALITY {arm["name"]}',done,total,start)
            all_frames,per=OfficialMetrics().evaluate(source,pixels,progress=tick)
            temporal=temporal_scores(source,pixels,1)
            temporal.update(mad_p95=float(np.percentile(temporal['per_transition']['generated_mad'],95)),
                seam_interpretation='single segment; seam fields unused')
            row=dict(arm=arm,signature=p['signature'],all384=all_frames,common381=summarize(per[:381],(381,320,576,3)),
                nonkeys382=summarize(per[1:-1],(382,320,576,3)),temporal=temporal,
                channel=p['transports'][arm['transport']],generation=read_json(directory/'result.json'))
            write_json(directory/'per_frame.json',per);write_json(directory/'metrics.json',row)
        results.append(row);progress('EVALUATE completed arms',number,len(p['arms']),start)
    base=results[0];reduced=results[1]
    for row in results:
        row['delta_vs_baseline']={k:row['common381'][k]-base['common381'][k] for k in ('psnr_db','lpips_vgg','ssim')}
        row['delta_vs_same_reduced_keys']={k:row['common381'][k]-reduced['common381'][k] for k in ('psnr_db','lpips_vgg','ssim')}
        row['meets_fixed22_point_targets']=dict(psnr=row['common381']['psnr_db']>=p['target_png']['psnr_db'],
            lpips=row['common381']['lpips_vgg']<=p['target_png']['lpips_vgg'],rate=row['channel']['total_uses']<=BUDGET)
    diagnostics=[]
    for name in p['transports']:
        directory=root/'transport'/name/'receiver'
        keys=[]
        for k in (0,383):
            pixels=np.asarray(Image.open(directory/f'{k}.png').convert('RGB'))
            original=np.asarray(Image.open(BASE/f'receiver/frames/sample/key_frames_received/{k}.png').convert('RGB'))
            keys.append(dict(index=k,psnr_to_source=float(cv2.PSNR(source[k],pixels)),matches_original_received=bool(np.array_equal(pixels,original))))
        diagnostics.append(dict(transport=name,keys=keys))
    direct=[]
    for name in p['transports']:
        directory=root/'transport'/name/'receiver'
        if not (directory/'structure.npy').exists():continue
        pixels=anchors.expand_video(np.load(directory/'structure.npy'),read_json(directory/'structure.json'),320,576)
        # Uses already transmitted structure alone; excludes unused keys in this descriptive rendering.
        summary,per=OfficialMetrics().evaluate(source,pixels)
        direct.append(dict(name=name,all384=summary,common381=summarize(per[:381],(381,320,576,3)),
            note='received structure upsample only, no generator; same packet, not independently rate-optimized codec'))
    write_json(root/'RESULT.json',dict(status='COMPLETE_PILOT_INDEPENDENT_SEMANTIC_REVIEW_PENDING',
        signature=p['signature'],training=False,adopted=False,baseline_parity=read_json(root/'baseline_parity.json'),
        results=results,received_key_diagnostics=diagnostics,direct_structure=direct,rate_curve=read_json(root/'rate_curve.json'),
        limitations=['one known harbor development video, one channel/generation seed, 10dB',
            'two structure resolutions and two edit ratios; other injection/training paths untested',
            'RGB PNG evaluation; MP4 metrics not mixed',
            'independent Added/Missing/Distorted review and generalization pending']))
    review=root/'review';review.mkdir(exist_ok=True)
    basepix=read_frames(root/'generation/baseline/frames');lowpix=read_frames(root/'generation/endhalf/frames')
    body=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>BSU 구조 앵커 실험</title>',
        '<style>body{font:16px system-ui;max-width:1400px;margin:24px auto}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}video,img{width:100%}</style>',
        '<h1>BSU 아이디어 1·2: 실제 예산과 무학습 구조 앵커</h1>',
        '<p>항구 16초·PNG 공통 381장·단일 seed/SNR. 학습 없음. 독립 의미 검수 대기.</p>',
        '<table><tr><th>조건</th><th>채널 사용량</th><th>PSNR</th><th>LPIPS</th><th>광류 오차</th></tr>']
    for row in results:
        q=row['common381'];name=row['arm']['name']
        body.append(f'<tr><td>{name}</td><td>{row["channel"]["total_uses"]}</td><td>{q["psnr_db"]:.3f}</td><td>{q["lpips_vgg"]:.5f}</td><td>{row["temporal"]["flow_error_mean"]:.5f}</td></tr>')
    body.append('</table>')
    for row in results[2:]:
        name=row['arm']['name'];pixels=read_frames(root/'generation'/name/'frames')
        comparison_video(review/f'{name}.mp4',[source,basepix,lowpix,pixels],['Source','Baseline 2 keys','Half end key',name],24)
        body.append(f'<h2>{name}</h2><video controls src="review/{name}.mp4"></video>')
        for index in (96,192,288,380):
            path=review/f'{name}_{index:04d}.png'
            Image.fromarray(np.concatenate([source[index],basepix[index],lowpix[index],pixels[index]],axis=1)).save(path)
    (root/'report.html').write_text('\n'.join(body+['</html>']))
    print('COMPLETE: RESULT.json and report.html; no default model change',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--stage',choices=['all','prepare','channel','receive','generate','evaluate'],default='all')
    args=parser.parse_args();root=args.root.resolve()
    if args.stage=='all':
        import fcntl
        root.mkdir(parents=True,exist_ok=True);start=time.monotonic()
        with (root/'execution.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            stages=['prepare','channel','receive','generate','evaluate']
            for i,stage in enumerate(stages):
                progress('OVERALL stages',i,len(stages),start)
                env=os.environ.copy();python=sys.executable
                if stage=='channel':
                    python='/home/sangukbae/anaconda3/envs/lgvsc-channel/bin/python';env['CUDA_VISIBLE_DEVICES']='-1'
                write_json(root/'status.json',dict(status='RUNNING',stage=stage,completed=i,total=len(stages)))
                code=subprocess.call([python,'-m','semantic_transmission.bsu_anchor_experiment','--root',str(root),'--stage',stage],env=env)
                if code:
                    write_json(root/'status.json',dict(status='FAILED',stage=stage,returncode=code))
                    raise SystemExit(code)
            write_json(root/'status.json',dict(status='COMPLETE_PILOT',seconds=time.monotonic()-start))
            progress('OVERALL stages',len(stages),len(stages),start)
    else:
        sys.path.insert(0,str(REPO/'.local/vendor/Open-Sora'))
        globals()[args.stage](root)


if __name__=='__main__':main()
