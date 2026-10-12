"""Follow-up: coarse received structure guidance without full latent replacement.

Five predeclared arms reuse one previously verified channel realization. This
module never modifies the preceding experiment or the default BSU receiver.
"""
import argparse
import copy
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .artifacts import sha256, write_json
from .webvid5 import read_json, fingerprint
from .bsu_anchor_experiment import REPO, BASE, ROOT as PARENT, load, progress
from .receiver_alignment_experiment import complete_arm

ROOT = REPO / 'outputs/bsu_frequency_anchor_20261011_v1'
TRANSPORT = 'endhalf_144x80'
ARMS = [dict(name='disabled_replay', strength=0, fraction=1),
        dict(name='full_s035', strength=.35, fraction=1),
        dict(name='low025_s035', strength=.35, fraction=.25),
        dict(name='low025_s070', strength=.7, fraction=.25),
        dict(name='low050_s070', strength=.7, fraction=.5)]


def prepare(root):
    if (root / 'protocol.json').exists():
        load(root)
        print('RESUME preparation verified', flush=True)
        return
    old = load(PARENT)
    parent_result = read_json(PARENT / 'RESULT.json')
    if not parent_result['baseline_parity']['exact_pixels']:
        raise ValueError('preceding baseline parity failed')
    transport = PARENT / 'transport' / TRANSPORT
    channel = read_json(transport / 'channel.json')
    if not channel['exact_match'] or (transport / 'tx.bin').read_bytes() != (transport / 'rx.bin').read_bytes():
        raise ValueError('unverified received packet')
    inputs = dict(old['inputs'])
    paths = [PARENT / 'protocol.json', PARENT / 'RESULT.json', PARENT / 'verification.json',
             PARENT / 'channel.json']
    paths.extend(p for p in transport.rglob('*') if p.is_file())
    for name in ('baseline', 'endhalf', 'endhalf_144x80_r035', 'endhalf_144x80_r06'):
        directory = PARENT / 'generation' / name
        if not complete_arm(directory, old['signature']):
            raise ValueError('parent generation incomplete')
        result = read_json(directory / 'result.json')
        paths.extend(directory / rel for rel in result['artifacts'])
        paths.extend(directory / rel for rel in ('result.json', 'metrics.json', 'per_frame.json'))
    for key in (0, 383):
        if sha256(transport / 'receiver' / f'{key}.png') != sha256(PARENT / 'transport/endhalf/receiver' / f'{key}.png'):
            raise ValueError('key-only replay inputs differ')
    inputs.update({str(p): sha256(p) for p in paths})
    code = dict(old['code'])
    names = ['frequency_anchors.py', 'bsu_frequency_experiment.py', 'bsu_motion_experiment.py',
             'receiver_alignment_experiment.py', 'receiver_joint_experiment.py', 'research_quality.py',
             'artifacts.py', 'webvid5.py']
    code.update({str(REPO / 'src/semantic_transmission' / n): sha256(REPO / 'src/semantic_transmission' / n) for n in names})
    for rel in ('scripts/experiment_bsu_frequency_anchors.sh', 'tests/test_frequency_anchors.py'):
        code[str(REPO / rel)] = sha256(REPO / rel)
    root.mkdir(parents=True, exist_ok=True)
    p = dict(policy='bsu_spatial_frequency_guidance_v1', parent=str(PARENT),
             parent_signature=old['signature'], source=old['source'], base=str(BASE),
             generator=old['generator'], frames=384, fps=24, height=320, width=576,
             received=str(transport / 'receiver'), channel=old['transports'][TRANSPORT], budget=old['budget'],
             arms=ARMS, code=code, inputs=inputs, training=False, default_receiver_changed=False,
             source_pixels_read_by_receiver=False, channel_reused=True, extra_transmitted_bits=0,
             noise_reference=str(BASE / 'receiver/generation_noise.json'),
             treatment='post-CFG Euler correction lambda*dt/tau*P(prior-(z+tau*v)); no latent replacement',
             spatial_filter='float32 rectangular FFT projection; periodic H/W boundary, no time filter',
             temporal_policy='same 46 received anchor positions; preserve legacy hard keys 0/110',
             strengths=[.35, .7], spatial_fractions=[.25, .5, 1],
             choice_before_generation=True, preview_frames=[96, 192, 288, 380],
             primary_gate=dict(psnr_min=20.12781681817267 - .25, lpips_max=0.35002480928353435,
                               rule='PSNR within 0.25dB of old hard-anchor arm AND LPIPS below original BSU'),
             fixed22_target=dict(psnr_db=20.969, lpips_vgg=.27832),
             limitations=['known harbor development video, one seed/SNR; no heldout generalization',
                          'low latent frequency correction does not guarantee accurate RGB high-frequency detail',
                          'same packet charged even for disabled guidance, which discards structure',
                          'legacy key temporal placement retained for paired comparison',
                          'independent Added/Missing/Distorted review pending'])
    p['signature'] = fingerprint(p)
    write_json(root / 'protocol.json', p)
    for source, digest in code.items():
        target = root / 'frozen_code' / Path(source).relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha256(target) != digest:
            raise ValueError('source snapshot changed')
    write_json(root / 'channel_reuse.json', dict(status='VERIFIED_REUSE', source=str(transport),
               channel=channel, signature=p['signature'], extra_transmitted_bits=0,
               tx_rx_exact=True, total_uses=p['channel']['total_uses']))
    print(f'PREPARE 1/1 (100%): five arms, same {p["channel"]["total_uses"]} uses, received bytes verified', flush=True)


def generate(root):
    import numpy as np
    import torch
    from PIL import Image
    from mmengine import Config
    from opensora.registry import MODELS, SCHEDULERS, build_module
    from opensora.datasets.utils import get_transforms_image
    from opensora.utils import inference_utils as inference
    from .bsu_motion_experiment import setup_torch
    from .condition_collision import apply_guard
    from .generation_noise import NoiseControl, tensor_hash
    from .precision_text_cache import Encoder
    from .receiver_joint_experiment import save_pixels
    from .research_quality import read_frames
    from .structure_anchors import anchor_slots
    from .frequency_anchors import sample
    p = load(root)
    signature = p['signature']
    if all(complete_arm(root / 'generation' / a['name'], signature) for a in p['arms']):
        print('RESUME all five generation arms verified; no inference', flush=True)
        return
    setup_torch(p)
    cfg = Config(p['generator'])
    received = Path(p['received'])
    print('GENERATE loading frozen VAE/STDiT; loading ETA unknown', flush=True)
    vae = build_module(cfg.vae, MODELS).to('cuda', torch.bfloat16).eval().requires_grad_(False)
    latent_size = vae.get_latent_size((p['frames'], *cfg.image_size))
    model = build_module(cfg.model, MODELS, input_size=latent_size, in_channels=vae.out_channels,
                         caption_channels=4096, model_max_length=300,
                         enable_sequence_parallelism=False).to('cuda', torch.bfloat16).eval().requires_grad_(False)
    prior = torch.load(received / 'structure_latent.pt', map_location='cpu', weights_only=True).to('cuda')
    expected = read_json(p['noise_reference'])
    runtime = dict(packages={n: importlib.metadata.version(n) for n in ('torch', 'diffusers', 'transformers')},
                   cuda=torch.version.cuda, gpu=torch.cuda.get_device_name())
    transform = get_transforms_image(image_size=tuple(cfg.image_size), name='resize_crop')
    start = time.monotonic()
    with torch.inference_mode():
        for number, arm in enumerate(p['arms'], 1):
            directory = root / 'generation' / arm['name']
            if complete_arm(directory, signature):
                progress('GENERATE verified arms', number, len(p['arms']), start)
                continue
            directory.mkdir(parents=True, exist_ok=True)
            arm_start = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            control = NoiseControl(directory / 'generation_noise.json', expected['contract'], runtime, expected)
            refs = [[]]
            for j, key in enumerate((0, 383)):
                with Image.open(received / f'{key}.png') as im:
                    x = transform(im.convert('RGB')).unsqueeze(0).unsqueeze(2).to('cuda', torch.bfloat16)
                with control.scope('vae', j, x):
                    refs[0].append(vae.encode(x).squeeze(0))
            model.config.input_size = latent_size
            z = torch.empty(1, vae.out_channels, *latent_size, device='cuda', dtype=torch.bfloat16)
            with control.scope('initial', 0, z):
                z.copy_(torch.randn_like(z))
            mask, conditions = apply_guard(z, refs, ['0;0,1,0,-1,1'], 0, 5, 2, inference)
            protected = torch.where(mask[0] == 0)[0].tolist()
            plan = anchor_slots(p['frames'], read_json(received / 'structure.json')['timestamps'], protected)
            if prior.shape != z.shape or len(plan['slots']) != 46 or protected != [0, 110]:
                raise ValueError('unexpected received condition alignment')
            initial = z.clone()
            report = copy.deepcopy(read_json(BASE / 'receiver/text_embeddings.json'))
            encoder = Encoder(report, directory / 'text_embedding_trace.json')
            encode = encoder.encode
            encoder.encode = lambda texts: {k: v.to('cuda', dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
                                            for k, v in encode(texts).items()}
            encoder.y_embedder = model.y_embedder
            extra = inference.prepare_multi_resolution_info(cfg.multi_resolution, 1, tuple(cfg.image_size),
                                                            p['frames'], cfg.fps, 'cuda', torch.bfloat16)
            scheduler = build_module(cfg.scheduler, SCHEDULERS)
            sampling_start = time.monotonic()
            def tick(done, total):
                progress(f'GENERATE {arm["name"]} completed steps', done, total, sampling_start, True)
            with control.scope('sampling', 0, z):
                sampled, trace = sample(scheduler, model, encoder, z, [report['prompts'][0]], 'cuda', extra,
                                        mask, prior, plan['slots'], arm['strength'], arm['fraction'], tick)
            control.finish(1, cfg.scheduler.num_sampling_steps)
            if not torch.equal(initial[:, :, protected], sampled[:, :, protected]):
                raise ValueError('hard keys changed')
            if not torch.isfinite(sampled).all():
                raise ValueError('nonfinite generated latent')
            print(f'EXPORT {arm["name"]}: VAE decode and 384 PNG; export ETA unknown', flush=True)
            decoded = vae.decode(sampled.to(torch.bfloat16), num_frames=p['frames'])
            pixels = (decoded[0].clamp(-1, 1).add(1).div(2).mul(255).add(.5).clamp(0, 255)
                      .permute(1, 2, 3, 0).to('cpu', torch.uint8).numpy())
            if arm['name'] == 'disabled_replay':
                historical = read_frames(PARENT / 'generation/endhalf/frames')
                parity = dict(exact_pixels=bool(np.array_equal(pixels, historical)),
                              max_absolute_error=int(np.abs(pixels.astype(int) - historical.astype(int)).max()))
                write_json(root / 'disabled_parity.json', parity)
                if not parity['exact_pixels']:
                    raise ValueError('disabled sampler differs from historical reduced-key output')
                del historical
            save_pixels(directory, pixels, cfg.fps)
            write_json(directory / 'guidance_trace.json', trace)
            write_json(directory / 'conditions.json', dict(conditions=conditions, structure=plan, arm=arm,
                       prompt=report['prompts'][0], hard_keys_preserved=True, source_pixels_read=False,
                       input_keys={str(k): sha256(received / f'{k}.png') for k in (0, 383)},
                       actual_generation_noise_matches_historical=True, full_band_latent_replacement=False,
                       initial_conditioned_latent_sha256=tensor_hash(initial),
                       structure_latent_sha256=tensor_hash(prior)))
            paths = [*sorted((directory / 'frames').glob('*.png')),
                     *[directory / n for n in ('reconstruction.mp4', 'generation_noise.json', 'conditions.json',
                                              'guidance_trace.json', 'text_embedding_trace.json')]]
            write_json(directory / 'result.json', dict(status='GENERATED', signature=signature, arm=arm,
                       frames=p['frames'], seconds=time.monotonic() - arm_start,
                       max_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                       artifacts={str(path.relative_to(directory)): sha256(path) for path in paths}))
            progress('GENERATE completed arms', number, len(p['arms']), start, True)
            del refs, x, z, initial, sampled, decoded, pixels, encoder
            torch.cuda.empty_cache()


def evaluate(root):
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from .bsu_motion_experiment import source_window
    from .research_quality import read_frames
    from .official_quality import OfficialMetrics, summarize
    from .receiver_alignment_evaluation import temporal_scores, comparison_video
    p = load(root)
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    source = source_window(p['source'], 0, 383)
    start = time.monotonic()
    results = []
    old = read_json(PARENT / 'RESULT.json')
    references = {r['arm']['name']: dict(common381=r['common381'], channel=r['channel'],
                  temporal=r['temporal']) for r in old['results'] if r['arm']['name'] in
                  ('baseline', 'endhalf', 'endhalf_144x80_r035', 'endhalf_144x80_r06')}
    for number, arm in enumerate(p['arms'], 1):
        directory = root / 'generation' / arm['name']
        if not complete_arm(directory, p['signature']):
            raise ValueError('generation incomplete')
        if (directory / 'metrics.json').exists():
            row = read_json(directory / 'metrics.json')
            if row['signature'] != p['signature']:
                raise ValueError('stale metrics')
        else:
            pixels = read_frames(directory / 'frames')
            metric_start = time.monotonic()
            def tick(done, total):
                if done == 1 or done % 384 == 0:
                    progress(f'QUALITY {arm["name"]}', done, total, metric_start, done >= 384)
            summary, per = OfficialMetrics().evaluate(source, pixels, progress=tick)
            temporal = temporal_scores(source, pixels, 1)
            temporal.update(mad_p95=float(np.percentile(temporal['per_transition']['generated_mad'], 95)),
                            seam_interpretation='single segment; seam fields unused')
            row = dict(signature=p['signature'], arm=arm, all384=summary,
                       common381=summarize(per[:381], (381, 320, 576, 3)),
                       nonkeys382=summarize(per[1:-1], (382, 320, 576, 3)), temporal=temporal,
                       generation=read_json(directory / 'result.json'), channel=p['channel'])
            write_json(directory / 'per_frame.json', per)
            write_json(directory / 'metrics.json', row)
            del pixels
        for name, ref in references.items():
            row[f'delta_vs_{name}'] = {k: row['common381'][k] - ref['common381'][k]
                                     for k in ('psnr_db', 'lpips_vgg', 'ssim')}
        q = row['common381']
        row['passes_primary_gate'] = (q['psnr_db'] >= p['primary_gate']['psnr_min']
                                     and q['lpips_vgg'] < p['primary_gate']['lpips_max'])
        row['reaches_fixed22_targets'] = (q['psnr_db'] >= p['fixed22_target']['psnr_db']
                                        and q['lpips_vgg'] <= p['fixed22_target']['lpips_vgg'])
        results.append(row)
        print(f'QUALITY RESULT {arm["name"]}: PSNR={q["psnr_db"]:.4f} LPIPS={q["lpips_vgg"]:.5f}', flush=True)
        progress('EVALUATE completed arms', number, len(p['arms']), start)
    full = results[1]
    for row in results:
        row['delta_vs_full_s035'] = {k: row['common381'][k] - full['common381'][k]
                                    for k in ('psnr_db', 'lpips_vgg', 'ssim')}
    write_json(root / 'RESULT.json', dict(status='COMPLETE_PILOT_INDEPENDENT_SEMANTIC_REVIEW_PENDING',
               signature=p['signature'], training=False, adopted=False, results=results,
               references=references, direct_structure=old['direct_structure'][1],
               disabled_parity=read_json(root / 'disabled_parity.json'), channel_reused=True,
               extra_transmitted_bits=0, primary_gate=p['primary_gate'], limitations=p['limitations']))
    print('EXPORT synchronized review videos and fixed sample frames; ETA unknown', flush=True)
    review = root / 'review'
    review.mkdir(exist_ok=True)
    hard = read_frames(PARENT / 'generation/endhalf_144x80_r035/frames')
    full_pixels = read_frames(root / 'generation/full_s035/frames')
    body = ['<!doctype html><html lang="ko"><meta charset="utf-8"><title>BSU 저주파 구조 주입</title>',
            '<style>body{font:16px system-ui;max-width:1400px;margin:24px auto}td,th{padding:8px;border:1px solid #aaa}video{width:100%}</style>',
            '<h1>BSU 저주파 구조 주입: 같은 수신 패킷·키·캡션·실제 생성 난수</h1>',
            '<p>384프레임·한 개발 영상·한 seed/SNR. 화질은 공통 381 PNG. 독립 의미 검수 대기.</p>',
            '<table><tr><th>조건</th><th>전송량</th><th>PSNR</th><th>LPIPS</th><th>사전 기준 통과</th></tr>']
    for row in results:
        q = row['common381']
        body.append(f'<tr><td>{row["arm"]["name"]}</td><td>{row["channel"]["total_uses"]}</td>'
                    f'<td>{q["psnr_db"]:.3f}</td><td>{q["lpips_vgg"]:.5f}</td><td>{row["passes_primary_gate"]}</td></tr>')
    body.append('</table>')
    for number, arm in enumerate(p['arms'][2:], 1):
        pixels = read_frames(root / 'generation' / arm['name'] / 'frames')
        arrays = [source, hard, full_pixels, pixels]
        labels = ['Source (eval only)', 'Old full latent r0.35', 'Full-band pull 0.35', arm['name']]
        comparison_video(review / f'{arm["name"]}.mp4', arrays, labels, 24)
        body.append(f'<h2>{arm["name"]}</h2><video controls src="review/{arm["name"]}.mp4"></video>')
        for index in p['preview_frames']:
            panels = []
            for array, label in zip(arrays, labels):
                panel = np.zeros((344, 576, 3), dtype=np.uint8)
                panel[24:] = array[index]
                cv2.putText(panel, f'{label} / frame {index}', (8, 17), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
                panels.append(panel)
            Image.fromarray(np.concatenate([np.concatenate(panels[:2], axis=1),
                                            np.concatenate(panels[2:], axis=1)], axis=0)).save(
                                            review / f'{arm["name"]}_{index:04d}.png')
        progress('EXPORT comparison videos', number, 3, start)
    (root / 'report.html').write_text('\n'.join(body + ['</html>']))
    print('COMPLETE: RESULT.json, report.html; independent semantic review pending', flush=True)


def verify(root):
    from .generation_noise import validate_trace
    p = load(root)
    expected = read_json(p['noise_reference'])
    rows = []
    initial = None
    for arm in p['arms']:
        directory = root / 'generation' / arm['name']
        if not complete_arm(directory, p['signature']):
            raise ValueError('missing generation')
        noise = read_json(directory / 'generation_noise.json')
        validate_trace(noise, expected['contract'], 1, 30)
        if noise['records'] != expected['records'] or not noise['matched_reference']:
            raise ValueError('actual noise mismatch')
        conditions = read_json(directory / 'conditions.json')
        digest = conditions['initial_conditioned_latent_sha256']
        if initial is None:
            initial = digest
        if digest != initial or not conditions['hard_keys_preserved']:
            raise ValueError('initial/key condition mismatch')
        trace = read_json(directory / 'guidance_trace.json')
        if len(trace) != 30:
            raise ValueError('incomplete sampler trace')
        max_leak = max(r.get('correction_highpass_rms', 0) for r in trace)
        if max_leak > 1e-6:
            raise ValueError('unexpected frequency leakage in correction')
        count = len(list((directory / 'frames').glob('*.png')))
        if count != 384:
            raise ValueError('missing output frames')
        rows.append(dict(arm=arm['name'], frames=count, artifacts_verified=True, noise_matched=True,
                         max_correction_highpass_rms=max_leak, hard_keys_preserved=True))
    for source, digest in p['code'].items():
        if sha256(root / 'frozen_code' / Path(source).relative_to(REPO)) != digest:
            raise ValueError('frozen source snapshot mismatch')
    if not read_json(root / 'disabled_parity.json')['exact_pixels']:
        raise ValueError('disabled parity failed')
    write_json(root / 'verification.json', dict(status='PASSED', signature=p['signature'], arms=rows,
               total_generated_frames=1920, all_actual_noise_equal=True, same_initial_condition=True,
               reused_channel_exact=True, all_within_budget=p['channel']['total_uses'] <= p['budget'],
               frozen_source_verified=True, independent_semantic_review='PENDING'))
    print('VERIFY 5/5 (100%): 1,920 PNG/artifacts, actual random draws, protected keys and spectral support passed', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--stage', choices=['all', 'prepare', 'generate', 'evaluate', 'verify'], default='all')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.stage != 'all':
        sys.path.insert(0, str(REPO / '.local/vendor/Open-Sora'))
        globals()[args.stage](root)
        return
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (root / 'execution.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stages = ['prepare', 'generate', 'evaluate', 'verify']
        for i, stage in enumerate(stages):
            progress('OVERALL completed stages', i, len(stages), start)
            write_json(root / 'status.json', dict(status='RUNNING', stage=stage, completed=i, total=len(stages)))
            code = subprocess.call([sys.executable, '-m', 'semantic_transmission.bsu_frequency_experiment',
                                    '--root', str(root), '--stage', stage], env=os.environ.copy())
            if code:
                write_json(root / 'status.json', dict(status='FAILED', stage=stage, returncode=code))
                raise SystemExit(code)
        write_json(root / 'status.json', dict(status='COMPLETE_PILOT', seconds=time.monotonic() - start))
        progress('OVERALL completed stages', len(stages), len(stages), start)


if __name__ == '__main__':
    main()
