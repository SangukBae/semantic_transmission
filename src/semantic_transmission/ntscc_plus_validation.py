"""Local key-codec integration check; no selector or video generation benchmark."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .artifacts import sha256, write_json
from . import ntscc_plus_codec as codec

REPO = Path(__file__).resolve().parents[2]
MODULE = 'semantic_transmission.ntscc_plus_validation'


def read(path):
    return json.loads(Path(path).read_text())


def reference(run):
    import torch
    from PIL import Image
    from torch.nn.functional import pad
    from torchvision.transforms.functional import pil_to_tensor
    model, _ = codec.load_codec(REPO)
    model.channel.chan_type = 'noiseless'
    cfg = read(run / 'run_config.json')
    rows = []
    with torch.inference_mode():
        for index in read(run / 'keyframes.json')['indices']:
            x = pil_to_tensor(Image.open(run / f'data/frames/sample/{index}.png').convert('RGB'))
            x = x.float().unsqueeze(0).cuda() / 255
            torch.manual_seed(cfg['seed'])
            official = model(pad(x, codec.padding(cfg['height'], cfg['width'])))
            torch.manual_seed(cfg['seed'])
            values, rates, power = codec.encode_symbols(model, x)
            restored = codec.decode_symbols(model, values, rates, power, cfg['height'], cfg['width'])
            left, _, top, _ = codec.padding(cfg['height'], cfg['width'])
            expected = official['x_hat'][:, :, top:top + cfg['height'], left:left + cfg['width']].clamp(0, 1)
            error = float((expected - restored).abs().max())
            uses = int(official['k'])
            if error > 1e-5 or values.numel() != uses:
                raise ValueError(f'official codec mismatch: max_error={error}, uses={uses}/{values.numel()}')
            rows.append(dict(index=index, max_abs_error=error, complex_uses=uses,
                             psnr_db=float(-10 * torch.log10((x - restored).square().mean()))))
    write_json(run / 'official_reference.json', dict(status='PASSED', frames=rows,
        peak_cuda_allocated_mib=torch.cuda.max_memory_allocated() / 2**20))


def worker(stage, run):
    if stage == 'reference':
        reference(run)
    elif stage == 'channel':
        from .qwen35_pairs_comparison import paired_channel
        paired_channel(run)
    else:
        import torch
        from . import codec_transport
        started = time.monotonic()
        getattr(codec_transport, stage)(read(run / 'run_config.json'), REPO, run)
        torch.cuda.synchronize()
        write_json(run / f'{stage}_resources.json', dict(elapsed_seconds=time.monotonic() - started,
            peak_cuda_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            gpu=torch.cuda.get_device_name(), torch=torch.__version__))


def validate(prepared, output):
    import numpy as np
    from PIL import Image
    from .cli import settings
    from .qwen35_pairs_video import stage_interpreter
    if output.exists():
        raise ValueError('validation output exists; choose a new directory')
    source = read(prepared)
    indices = sorted({0, source['frames'] // 2, source['frames'] - 1})
    if len(indices) < 2:
        raise ValueError('need at least two distinct source frames')
    output.mkdir(parents=True)
    local = settings(REPO)
    plus_python = str(REPO / '.local/ntscc_plus_env/bin/python')
    env = dict(os.environ, PYTHONPATH=str(REPO / 'src'), PYTHONNOUSERSITE='1',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    env.pop('LD_LIBRARY_PATH', None)
    env.pop('PYTORCH_CUDA_ALLOC_CONF', None)
    started = time.monotonic()
    commands, all_rows = [], []
    source_hashes = {str(i): sha256(Path(source['source_frames']) / f'{i}.png') for i in indices}
    if any(source_hashes[str(i)] != source['source_frame_hashes'][str(i)] for i in indices):
        raise ValueError('prepared source key hash changed')
    for name in ('NTSCC+', 'NTSCC'):
        run = output / ('plus' if name == 'NTSCC+' else 'original')
        frames = run / 'data/frames/sample'; frames.mkdir(parents=True)
        for index in indices:
            shutil.copyfile(Path(source['source_frames']) / f'{index}.png', frames / f'{index}.png')
        cfg = dict(width=source['width'], height=source['height'], frames=source['frames'], fps=source['fps'],
            seed=2025, channel_seed=42, steps=30, snr_db=10, visual_codec=name,
            visual_channel='source_frame_seeded_numpy_awgn_v1', decoder_policy='endpoint_exact')
        write_json(run / 'run_config.json', cfg)
        write_json(run / 'keyframes.json', dict(indices=indices))
        write_json(run / 'metadata_tx.json', [dict(path=f'sample/{a}_{b}.mp4',
            text='Codec validation placeholder; semantic extraction is not run.', flow=0.)
            for a, b in zip(indices, indices[1:])])
        protocol = dict(receiver=cfg, interpreters=dict(local, codec_python=plus_python))
        stages = ['reference', 'send', 'channel', 'receive'] if name == 'NTSCC+' else ['send', 'channel', 'receive']
        for stage in stages:
            target = run
            if stage == 'receive':
                # This directory contains only channel outputs and an empty CLI
                # config; receiver selection is exclusively packet-driven.
                target = run / 'receiver_only'; target.mkdir()
                shutil.copytree(run / 'received', target / 'received')
                write_json(target / 'run_config.json', {})
            python = plus_python if stage == 'reference' else stage_interpreter(protocol, stage)
            worker_env = codec.runtime_environment(python, env) if python == plus_python else dict(env)
            if stage == 'channel':
                worker_env['CUDA_VISIBLE_DEVICES'] = '-1'
            cmd = [python, '-m', MODULE, '--worker', stage, '--output', str(target)]
            commands.append(cmd)
            print(f'{name}: {stage}', flush=True)
            with (run / f'{stage}.log').open('x') as log:
                subprocess.run(cmd, env=worker_env, stdout=log, stderr=subprocess.STDOUT, check=True)
        decoded = run / 'receiver_only/receiver/frames/sample/key_frames_received'
        rows = []
        for index in indices:
            a = np.asarray(Image.open(frames / f'{index}.png').convert('RGB'), dtype=np.float64)
            b = np.asarray(Image.open(decoded / f'{index}.png').convert('RGB'), dtype=np.float64)
            if a.shape != b.shape or not np.isfinite(b).all():
                raise ValueError('invalid decoded image')
            rows.append(dict(index=index, psnr_db=float(10 * np.log10(255**2 / np.mean((a-b)**2)))))
        receiver = read(run / 'receiver_only/receiver_accounting.json')
        if receiver['visual_codec'] != name or receiver['source_pixels_read'] or receiver['received_keyframes'] != len(indices):
            raise ValueError('receiver audit mismatch')
        all_rows.append(dict(codec=name, keys=rows, mean_key_psnr_db=float(np.mean([r['psnr_db'] for r in rows])),
            channel=read(run / 'channel_accounting.json'),
            send_resources=read(run / 'send_resources.json'),
            receive_resources=read(run / 'receiver_only/receive_resources.json')))
    write_json(output / 'RESULT.json', dict(status='PASSED_KEY_CODEC_INTEGRATION',
        source_prepared=str(prepared), source_prepared_sha256=sha256(prepared), source_key_hashes=source_hashes,
        official_reference=read(output / 'plus/official_reference.json'), codecs=all_rows,
        commands=commands, elapsed_seconds=time.monotonic()-started,
        checkpoint_sha256=codec.CHECKPOINT_SHA256, codec_spec=codec.SPEC,
        code={str(p.relative_to(REPO)): sha256(p) for p in [Path(__file__),
            REPO / 'src/semantic_transmission/ntscc_plus_codec.py', REPO / 'src/semantic_transmission/codec_transport.py']},
        full_bsu_video_generation_tested=False, equal_rate_comparison=False,
        limits=['three fixed keys from one video, no Qwen selection/caption or generator inference',
                'different checkpoints/rates; key PSNR is not a rate-matched superiority claim',
                'times include loading weights; VRAM is PyTorch allocated peak, not total device use']))
    print(str(output / 'RESULT.json'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', choices=('reference', 'send', 'channel', 'receive'))
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.output.resolve())
    elif args.prepared:
        validate(args.prepared.resolve(), args.output.resolve())
    else:
        parser.error('--prepared is required')


if __name__ == '__main__':
    main()
