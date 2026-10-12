"""Paired temporal-support/strength diagnostic of the known frame-35 flash.

The spatial correction and sampler are imported unchanged from the frozen
frequency experiment. Only selected temporal positions and the declared
strength control vary. No training, retransmission, or source-guided receiver.
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
from .bsu_anchor_experiment import REPO, BASE, load, progress
from .bsu_frequency_experiment import ROOT as PARENT
from .receiver_alignment_experiment import complete_arm
from .temporal_anchor_support import experiment_arms, brightness_diagnostic

ROOT = REPO / 'outputs/bsu_temporal_support_20261012_v1'
PARENT_ARM = 'low025_s070'


def prepare(root):
    if (root / 'protocol.json').exists():
        load(root)
        print('RESUME preparation verified', flush=True)
        return
    parent = load(PARENT)
    for arm in parent['arms']:
        if not complete_arm(PARENT / 'generation' / arm['name'], parent['signature']):
            raise ValueError('parent experiment incomplete')
    condition = read_json(PARENT / 'generation' / PARENT_ARM / 'conditions.json')
    arms = experiment_arms(condition['structure'])
    if [a['positions'] for a in arms] != [46, 111, 46, 111]:
        raise ValueError('unexpected latent timeline')
    inputs = dict(parent['inputs'])
    paths = [PARENT / n for n in ('protocol.json', 'RESULT.json', 'verification.json',
                                 'channel_reuse.json', 'temporal_diagnostic.json')]
    for arm in parent['arms']:
        directory = PARENT / 'generation' / arm['name']
        paths.extend(directory / rel for rel in read_json(directory / 'result.json')['artifacts'])
        paths.extend(directory / n for n in ('result.json', 'metrics.json', 'per_frame.json'))
    inputs.update({str(path): sha256(path) for path in paths})
    code = dict(parent['code'])
    for rel in ('src/semantic_transmission/temporal_anchor_support.py',
                'src/semantic_transmission/bsu_temporal_support_experiment.py',
                'scripts/experiment_bsu_temporal_support.sh', 'tests/test_temporal_anchor_support.py'):
        code[str(REPO / rel)] = sha256(REPO / rel)
    p = dict(policy='bsu_temporal_support_factorial_v1', parent=str(PARENT), parent_signature=parent['signature'],
             parent_arm=PARENT_ARM, generator=parent['generator'], source=parent['source'], base=str(BASE),
             frames=384, fps=24, height=320, width=576, received=parent['received'],
             channel=parent['channel'], budget=parent['budget'], noise_reference=parent['noise_reference'],
             arms=arms, anchor_plan=condition['structure'], code=code, inputs=inputs,
             training=False, default_receiver_changed=False, channel_reused=True, extra_transmitted_bits=0,
             source_pixels_read_by_receiver=False, spatial_sampler_changed=False,
             intervention='46 sparse vs all 111 non-key latent positions; same prior, FFT band 0.25, sampler and noise',
             strength_control='2x2: 0.7 and 0.7*46/111; sparse-high and continuous-low have equal coefficient sum',
             normalization_caveat='Equal coefficient sums do not equal realized residual/update energy.',
             continuity_caveat='All non-key positions are active; hard-key holes 0/110 and residual differences remain.',
             target_frame=35, target_window=[32, 38], preview_frames=[34, 35, 36, 96, 192, 288, 380],
             primary_comparison=['sparse_high_replay', 'continuous_high'],
             primary_flash_gate='At unchanged strength 0.7: frame35 luma impulse and window32:38 max MAD both <=50% of sparse replay.',
             primary_gate=parent['primary_gate'], fixed22_target=parent['fixed22_target'],
             limitations=['known failing development video and seed; causal diagnostic, not heldout generalization',
                          'support changes density as well as discontinuity; no proof that boundary discontinuity alone caused the flash',
                          'coefficient-sum and lower-strength controls distinguish simple weakening from support effects',
                          'mean-luma flash diagnostic misses some local/color/texture flicker',
                          'legacy key temporal positions and FFT periodic boundaries remain fixed',
                          'independent Added/Missing/Distorted review pending'])
    p['signature'] = fingerprint(p)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / 'protocol.json', p)
    for source, digest in code.items():
        target = root / 'frozen_code' / Path(source).relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha256(target) != digest:
            raise ValueError('frozen source mismatch')
    transport = Path(parent['received']).parent
    if (transport / 'tx.bin').read_bytes() != (transport / 'rx.bin').read_bytes():
        raise ValueError('channel packet no longer exact')
    write_json(root / 'channel_reuse.json', dict(status='VERIFIED_REUSE', source=str(transport),
               signature=p['signature'], received_sha256=sha256(transport / 'rx.bin'),
               total_uses=p['channel']['total_uses'], extra_transmitted_bits=0, tx_rx_exact=True))
    print('PREPARE 1/1 (100%): four arms; sparse=46, continuous=111; strengths=0.7/0.29009009', flush=True)


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
    from .frequency_anchors import sample
    p = load(root)
    signature = p['signature']
    if all(complete_arm(root / 'generation' / a['name'], signature) for a in p['arms']):
        print('RESUME all four generation arms verified; no inference', flush=True)
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
            if prior.shape != z.shape or protected != p['anchor_plan']['protected']:
                raise ValueError('received condition timeline changed')
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
                                        mask, prior, arm['slots'], arm['strength'], arm['fraction'], tick)
            control.finish(1, cfg.scheduler.num_sampling_steps)
            if not torch.equal(initial[:, :, protected], sampled[:, :, protected]):
                raise ValueError('hard key changed')
            if not torch.isfinite(sampled).all():
                raise ValueError('nonfinite generated latent')
            print(f'EXPORT {arm["name"]}: VAE decode and 384 PNG; export ETA unknown', flush=True)
            decoded = vae.decode(sampled.to(torch.bfloat16), num_frames=p['frames'])
            pixels = (decoded[0].clamp(-1, 1).add(1).div(2).mul(255).add(.5).clamp(0, 255)
                      .permute(1, 2, 3, 0).to('cpu', torch.uint8).numpy())
            if arm['name'] == 'sparse_high_replay':
                historical = read_frames(PARENT / 'generation' / PARENT_ARM / 'frames')
                parity = dict(exact_pixels=bool(np.array_equal(pixels, historical)), frames=384,
                              max_absolute_error=int(np.abs(pixels.astype(int) - historical.astype(int)).max()))
                write_json(root / 'sparse_parity.json', parity)
                if not parity['exact_pixels']:
                    raise ValueError('known failing sparse replay differs')
                del historical
            save_pixels(directory, pixels, cfg.fps)
            write_json(directory / 'guidance_trace.json', trace)
            write_json(directory / 'conditions.json', dict(conditions=conditions, arm=arm,
                       hard_keys_preserved=True, protected=protected, source_pixels_read=False,
                       prompt=report['prompts'][0],
                       input_keys={str(k): sha256(received / f'{k}.png') for k in (0, 383)},
                       actual_generation_noise_matches_historical=True,
                       initial_conditioned_latent_sha256=tensor_hash(initial), structure_latent_sha256=tensor_hash(prior)))
            paths = [*sorted((directory / 'frames').glob('*.png')),
                     *[directory / n for n in ('reconstruction.mp4', 'generation_noise.json', 'conditions.json',
                                              'guidance_trace.json', 'text_embedding_trace.json')]]
            write_json(directory / 'result.json', dict(status='GENERATED', signature=signature, arm=arm, frames=384,
                       seconds=time.monotonic() - arm_start, max_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                       artifacts={str(path.relative_to(directory)): sha256(path) for path in paths}))
            progress('GENERATE completed arms', number, len(p['arms']), start, True)
            del refs, x, z, initial, sampled, decoded, pixels, encoder
            torch.cuda.empty_cache()


def contrast(before, after):
    keys = ('psnr_db', 'lpips_vgg', 'ssim')
    diagnostic_keys = ('target_luma_impulse', 'window_max_mad', 'max_mad', 'transitions_above_mad10',
                       'max_source_adjusted_luma_step', 'max_luma_impulse')
    return dict(before=before['arm']['name'], after=after['arm']['name'],
                quality_delta={k: after['common381'][k] - before['common381'][k] for k in keys},
                flow_delta=after['temporal']['flow_error_mean'] - before['temporal']['flow_error_mean'],
                diagnostic={k: dict(before=before['flash'][k], after=after['flash'][k]) for k in diagnostic_keys})


def evaluate(root):
    import cv2
    import numpy as np
    import torch
    from .bsu_motion_experiment import source_window
    from .research_quality import read_frames
    from .official_quality import OfficialMetrics, summarize
    from .receiver_alignment_evaluation import temporal_scores
    p = load(root)
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    source = source_window(p['source'], 0, 383)
    start = time.monotonic()
    rows = []
    for number, arm in enumerate(p['arms'], 1):
        directory = root / 'generation' / arm['name']
        if not complete_arm(directory, p['signature']):
            raise ValueError('generation incomplete')
        if (directory / 'metrics.json').exists():
            row = read_json(directory / 'metrics.json')
            if row['signature'] != p['signature']:
                raise ValueError('stale evaluation')
        else:
            pixels = read_frames(directory / 'frames')
            metric_start = time.monotonic()
            def tick(done, total):
                if done == 1 or done % 384 == 0:
                    progress(f'QUALITY {arm["name"]}', done, total, metric_start)
            summary, per = OfficialMetrics().evaluate(source, pixels, progress=tick)
            temporal = temporal_scores(source, pixels, 1)
            temporal.update(mad_p95=float(np.percentile(temporal['per_transition']['generated_mad'], 95)),
                            seam_interpretation='single segment; seam fields unused')
            diagnostic = brightness_diagnostic(source, pixels, temporal, p['target_frame'], p['target_window'])
            row = dict(signature=p['signature'], arm=arm, all384=summary,
                       common381=summarize(per[:381], (381, 320, 576, 3)),
                       nonkeys382=summarize(per[1:-1], (382, 320, 576, 3)), temporal=temporal, flash=diagnostic,
                       generation=read_json(directory / 'result.json'), channel=p['channel'])
            write_json(directory / 'per_frame.json', per)
            write_json(directory / 'metrics.json', row)
            del pixels
        q = row['common381']
        row['passes_structure_detail_gate'] = (q['psnr_db'] >= p['primary_gate']['psnr_min']
                                               and q['lpips_vgg'] < p['primary_gate']['lpips_max'])
        row['reaches_fixed22_targets'] = (q['psnr_db'] >= p['fixed22_target']['psnr_db']
                                        and q['lpips_vgg'] <= p['fixed22_target']['lpips_vgg'])
        rows.append(row)
        print(f'QUALITY RESULT {arm["name"]}: PSNR={q["psnr_db"]:.4f} LPIPS={q["lpips_vgg"]:.5f} '
              f'flash35={row["flash"]["target_luma_impulse"]:.3f} maxMAD={row["flash"]["max_mad"]:.3f}', flush=True)
        progress('EVALUATE completed arms', number, len(p['arms']), start)
    parent = read_json(PARENT / 'RESULT.json')
    references = dict(parent['references'])
    disabled = parent['results'][0]
    references['same_packet_disabled'] = dict(common381=disabled['common381'], temporal=disabled['temporal'], channel=disabled['channel'])
    paired = dict(same_strength_high=contrast(rows[0], rows[1]), same_strength_low=contrast(rows[2], rows[3]),
                  same_coefficient_sum=contrast(rows[0], rows[3]), weakening_sparse_only=contrast(rows[0], rows[2]))
    gate = all(rows[1]['flash'][k] <= .5 * rows[0]['flash'][k] for k in ('target_luma_impulse', 'window_max_mad'))
    write_json(root / 'RESULT.json', dict(status='COMPLETE_PILOT_INDEPENDENT_SEMANTIC_REVIEW_PENDING',
               signature=p['signature'], training=False, adopted=False, source_pixels_used_by_receiver=False,
               channel_reused=True, extra_transmitted_bits=0, results=rows, contrasts=paired,
               primary_flash_gate_passed=gate, primary_flash_gate=p['primary_flash_gate'],
               references=references, sparse_parity=read_json(root / 'sparse_parity.json'), limitations=p['limitations']))
    render_review(root, p, source, rows)


def render_review(root, p, source, rows):
    import cv2
    import numpy as np
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .research_quality import read_frames
    from .receiver_alignment_evaluation import comparison_video
    review = root / 'review'
    review.mkdir(exist_ok=True)
    sparse = read_frames(root / 'generation/sparse_high_replay/frames')
    frames = sorted(set(p['preview_frames'] + [max(0, r['flash']['max_luma_impulse_frame'] + j)
                    for r in rows[1:] for j in (-1, 0, 1)]))
    frames = [i for i in frames if i < 384]
    write_json(review / 'sample_plan.json', dict(predeclared=p['preview_frames'],
               posthoc='largest source-adjusted mean-luma impulse and neighbors per candidate', actual=frames))
    start = time.monotonic()
    for number, row in enumerate(rows[1:], 1):
        name = row['arm']['name']
        pixels = read_frames(root / 'generation' / name / 'frames')
        comparison_video(review / f'{name}.mp4', [source, sparse, pixels], ['Source', 'Sparse high replay', name], 24)
        for index in frames:
            panels = []
            for array, label in zip([source, sparse, pixels], ['Source', 'Sparse high', name]):
                panel = np.zeros((344, 576, 3), dtype=np.uint8)
                panel[24:] = array[index]
                cv2.putText(panel, f'{label} / frame {index}', (8, 17), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
                panels.append(panel)
            Image.fromarray(np.concatenate(panels, axis=0)).save(review / f'{name}_{index:04d}.png')
        progress('EXPORT comparison videos', number, 3, start)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    for row in rows:
        name = row['arm']['name']
        axes[0].plot(range(1, 384), row['temporal']['per_transition']['generated_mad'], label=name, linewidth=.85)
        axes[1].plot(range(30, 41), row['flash']['per_frame']['luma_error'][30:41], label=name, marker='.')
    axes[0].plot(range(1, 384), rows[0]['temporal']['per_transition']['source_mad'], color='black', linewidth=.8, label='Source')
    axes[0].set(xlabel='Destination frame', ylabel='Frame MAD (0-255)', title='All 383 frame transitions')
    axes[1].set(xlabel='Frame', ylabel='Mean luma error vs source', title='Known frame-35 failure window')
    for axis in axes:
        axis.legend(fontsize=8)
        axis.grid(alpha=.25)
    fig.savefig(review / 'temporal_diagnostic.png', dpi=150)
    fig.savefig(review / 'temporal_diagnostic.svg')
    plt.close(fig)
    body = ['<!doctype html><html lang="ko"><meta charset="utf-8"><title>BSU 시간축 보정 대조</title>',
            '<style>body{font:16px system-ui;max-width:1400px;margin:24px auto}td,th{padding:8px;border:1px solid #aaa}video,img{max-width:100%}</style>',
            '<h1>희소/연속 시간 보정 × 강도 대조</h1><p>동일 수신 패킷·키·캡션·실제 생성 난수. '
            '4조건, 공통 381 PNG 화질·전체 384장 시간 지표. 독립 의미 검수 대기.</p>',
            '<table><tr><th>조건</th><th>PSNR</th><th>LPIPS</th><th>35번 밝기 충격</th><th>최대 MAD</th><th>광류 오차</th></tr>']
    for row in rows:
        q, d = row['common381'], row['flash']
        body.append(f'<tr><td>{row["arm"]["name"]}</td><td>{q["psnr_db"]:.3f}</td>'
                    f'<td>{q["lpips_vgg"]:.5f}</td><td>{d["target_luma_impulse"]:.3f}</td>'
                    f'<td>{d["max_mad"]:.3f}</td><td>{row["temporal"]["flow_error_mean"]:.5f}</td></tr>')
    body += ['</table><p>같은 보정 계수 총합은 실제 보정 에너지 일치를 뜻하지 않는다. '
             '연속 조건에도 고정 키 위치 0/110은 제외한다.</p><img src="review/temporal_diagnostic.png">']
    for row in rows[1:]:
        name = row['arm']['name']
        body.append(f'<h2>{name}</h2><video controls src="review/{name}.mp4"></video>')
    (root / 'report.html').write_text('\n'.join(body + ['</html>']))
    print('COMPLETE: RESULT.json, report.html and temporal plots', flush=True)


def verify(root):
    from .generation_noise import validate_trace
    p = load(root)
    expected = read_json(p['noise_reference'])
    historical = read_json(PARENT / 'generation' / PARENT_ARM / 'conditions.json')
    records = []
    for arm in p['arms']:
        directory = root / 'generation' / arm['name']
        if not complete_arm(directory, p['signature']):
            raise ValueError('incomplete generation')
        noise = read_json(directory / 'generation_noise.json')
        validate_trace(noise, expected['contract'], 1, 30)
        if noise['records'] != expected['records'] or not noise['matched_reference']:
            raise ValueError('actual noise mismatch')
        condition = read_json(directory / 'conditions.json')
        for field in ('initial_conditioned_latent_sha256', 'structure_latent_sha256', 'input_keys', 'prompt'):
            if condition[field] != historical[field]:
                raise ValueError(f'conditioning differs: {field}')
        if condition['arm'] != arm or not condition['hard_keys_preserved']:
            raise ValueError('wrong temporal intervention or hard keys')
        trace = read_json(directory / 'guidance_trace.json')
        if len(trace) != 30 or max(t['correction_highpass_rms'] for t in trace) > 1e-6:
            raise ValueError('spatial correction changed')
        count = len(list((directory / 'frames').glob('*.png')))
        if count != 384:
            raise ValueError('incomplete output frames')
        records.append(dict(arm=arm['name'], frames=count, positions=arm['positions'], noise_matched=True,
                            artifacts_verified=True, hard_keys_preserved=True))
    for source, digest in p['code'].items():
        if sha256(root / 'frozen_code' / Path(source).relative_to(REPO)) != digest:
            raise ValueError('source snapshot mismatch')
    if not read_json(root / 'sparse_parity.json')['exact_pixels']:
        raise ValueError('sparse replay mismatch')
    write_json(root / 'verification.json', dict(status='PASSED', signature=p['signature'], arms=records,
               total_generated_frames=1536, actual_noise_and_received_conditions_identical=True,
               spatial_sampler_unchanged=True, frozen_code_verified=True, channel_uses=p['channel']['total_uses'],
               all_within_budget=p['channel']['total_uses'] <= p['budget']))
    print('VERIFY 4/4 (100%): 1,536 PNG, actual noise, received conditions, frozen sampler and keys passed', flush=True)


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
            code = subprocess.call([sys.executable, '-m', 'semantic_transmission.bsu_temporal_support_experiment',
                                    '--root', str(root), '--stage', stage], env=os.environ.copy())
            if code:
                write_json(root / 'status.json', dict(status='FAILED', stage=stage, returncode=code))
                raise SystemExit(code)
        write_json(root / 'status.json', dict(status='COMPLETE_PILOT', seconds=time.monotonic() - start))
        progress('OVERALL completed stages', len(stages), len(stages), start)


if __name__ == '__main__':
    main()
