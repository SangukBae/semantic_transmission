"""LTX-2.5 receiver replacement using only actually received BSU conditions.

The registered experiment replays the same paid packet, key RGBs and trajectory
as the Open-Sora arm. The selected decoder is shared configuration, not a new
unaccounted source-dependent message. Original pixels are never inputs here.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

from .artifacts import sha256, write_json
from .wire import unpack
from .webvid5 import fingerprint
from .webvid_ablation import Stages

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / 'configs/bsu_gen_ltx25.json'


def read(path):
    return json.loads(Path(path).read_text())


def profile():
    return read(CONFIG)


def padded_frames(count):
    if type(count) is not int or count < 2:
        raise ValueError('interval must contain at least two frames')
    return ((count - 1 + 7) // 8) * 8 + 1


def request(packet, key_directory):
    """Strict received-only boundary; no source path, labels, or original keys."""
    from PIL import Image
    from .bsu_sequence_refinement import validate_keys
    packet, key_directory = Path(packet).resolve(), Path(key_directory).resolve()
    header, _ = unpack(packet.read_bytes())
    video = header['video']; keys = [k['index'] for k in header['keyframes']]
    validate_keys(keys, video['frames'])
    if (video['width'], video['height'], video['fps']) != (576, 320, 24):
        raise ValueError('registered comparison requires 576x320 at 24 FPS')
    if len(header['segments']) != len(keys) - 1:
        raise ValueError('caption count differs from received key intervals')
    files = {str(packet): sha256(packet)}
    images = []
    for index in keys:
        p = key_directory / f'{index}.png'
        with Image.open(p) as image:
            if image.size != (video['width'], video['height']):
                raise ValueError('received image dimensions differ from packet')
        files[str(p)] = sha256(p); images.append(str(p))
    windows = []
    for i, (a, b) in enumerate(zip(keys, keys[1:])):
        segment = header['segments'][i]
        caption = segment['text']
        if 'motion score:' not in caption:
            caption += f" motion score: {float(segment['flow']):.1f}."
        windows.append(dict(start=a, end=b, frames=b-a+1,
            generated_frames=padded_frames(b-a+1), prompt=caption,
            images=[dict(path=images[i], frame_idx=0), dict(path=images[i+1], frame_idx=b-a)]))
    value = dict(profile=profile(), video=video, windows=windows, inputs=files,
        source_pixels_available=False, packet_decoder_overridden_by_registered_profile=True)
    value['signature'] = fingerprint(value)
    return value


def verify_request(value):
    if value['profile'] != profile() or fingerprint({k:v for k,v in value.items() if k != 'signature'}) != value['signature']:
        raise ValueError('LTX request or profile changed')
    for path, digest in value['inputs'].items():
        if sha256(path) != digest:
            raise ValueError(f'received condition changed: {path}')


def runtime(model_dir, vendor):
    value = profile(); model_dir, vendor = Path(model_dir).resolve(), Path(vendor).resolve()
    revision = subprocess.check_output(['git', '-C', str(vendor), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != value['upstream_revision']:
        raise ValueError('LTX source revision changed')
    files = {str(p): sha256(p) for p in sorted((vendor/'packages').rglob('*.py'))}
    manifest = read(model_dir/'manifest.json')
    if (manifest['repo_id'], manifest['revision']) != ('Lightricks/LTX-2.5', value['model_revision']):
        raise ValueError('LTX weight manifest differs from registered checkpoint')
    paths = {}
    for role, relative in value['weights'].items():
        p = model_dir / relative; expected = manifest['files'][relative]
        stat = p.stat()
        if stat.st_size != expected['size'] or stat.st_mtime_ns != expected['mtime_ns']:
            raise ValueError(f'LTX weights changed: {relative}; rerun setup verification')
        paths[role] = str(p)
    return dict(paths=paths, manifest_sha256=sha256(model_dir/'manifest.json'),
                vendor_revision=revision, vendor_files=files, weights=manifest['files'])


class Generator:
    def __init__(self, assets):
        import torch
        from ltx_pipelines.utils.quantization_factory import QuantizationKind
        from ltx_pipelines.distilled import DistilledPipeline
        from ltx_pipelines.utils.model_paths import ModelPaths
        from ltx_pipelines.utils.types import OffloadMode
        paths = assets['paths']
        self.torch = torch
        self.pipeline = DistilledPipeline(
            model_paths=ModelPaths.from_split(transformer_path=paths['transformer'],
                text_encoder_path=paths['text_encoder'], video_vae_path=paths['video_vae'],
                audio_vae_path=paths['audio_vae']),
            spatial_upsampler_path=paths['spatial_upsampler'], loras=[],
            quantization=QuantizationKind(profile()['quantization']).to_policy(paths['transformer']),
            offload_mode=OffloadMode.CPU)

    def __call__(self, window, value):
        from ltx_pipelines.utils.types import ImageConditioningInput
        p, v = value['profile'], value['video']
        # The upstream CLI enters inference_mode; direct pipeline calls do not.
        # Include lazy decoding so autograd cannot retain offloaded GPU weights.
        with self.torch.inference_mode():
            output = self.pipeline(prompt=window['prompt'], seed=p['generation_seed'],
                height=v['height'], width=v['width'], frame_rate=v['fps'],
                images=[ImageConditioningInput(**item, strength=p['image_strength'], crf=p['image_crf'])
                        for item in window['images']], num_frames=window['generated_frames'],
                enhance_prompt=False,
                stage_1_sigmas=self.torch.tensor(p['stage_1_sigmas']),
                stage_2_sigmas=self.torch.tensor(p['stage_2_sigmas']))
            if output.num_frames != window['generated_frames']:
                raise ValueError('LTX silently changed the requested temporal grid')
            for chunk in output.video:
                if chunk.ndim != 4 or tuple(chunk.shape[1:]) != (v['height'], v['width'], 3):
                    raise ValueError('unexpected upstream video tensor layout')
                if not self.torch.isfinite(chunk).all():
                    raise ValueError('LTX produced nonfinite pixels')
                pixels = (chunk.detach().float().clamp(0, 1)*255).round().to(self.torch.uint8).cpu().numpy()
                yield from pixels


def reconstruct(value, assets, output, factory=Generator):
    """Resume per interval; tail trim only, no time rescaling or source filling."""
    import os
    import numpy as np
    from PIL import Image
    from .qwen35_pairs_video import freeze
    from .bsu_receiver import encode_frames
    verify_request(value); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    contract = dict(request=value, runtime=assets, implementation_sha256=sha256(__file__))
    freeze(output/'protocol.json', contract)
    stages = Stages(output, fingerprint(contract)); generator = None
    started = time.monotonic()
    for i, window in enumerate(value['windows']):
        folder = f'intervals/{i:05d}'
        def generate(_, w=window, name=folder):
            nonlocal generator
            if generator is None:
                generator = factory(assets)
            dest = output/name; dest.mkdir(parents=True)
            count = 0
            for index, pixels in enumerate(generator(w, value)):
                if pixels.dtype != np.uint8 or pixels.shape != (320,576,3):
                    raise ValueError('generator output must be RGB uint8')
                if index < w['frames']:
                    Image.fromarray(pixels).save(dest/f'{index:05d}.png')
                count += 1
            if count != w['generated_frames']:
                raise ValueError('incomplete generated interval')
            write_json(dest/'receipt.json', dict(start=w['start'], end=w['end'],
                generated_frames=count, delivered_frames=w['frames'], tail_trimmed=count-w['frames']))
        stages.step(f'interval_{i:05d}', [folder], [folder], generate)
        print(f'LTX intervals {i+1}/{len(value["windows"])} ({100*(i+1)/len(value["windows"]):.1f}%) '
              f'elapsed={time.monotonic()-started:.1f}s ETA=unknown', flush=True)
    def assemble(_):
        frames = output/'frames'; frames.mkdir()
        for i, w in enumerate(value['windows']):
            # Drop each preceding segment's shared last frame; no duplication.
            count = w['frames'] if i == len(value['windows'])-1 else w['frames']-1
            for offset in range(count):
                os.link(output/f'intervals/{i:05d}/{offset:05d}.png', frames/f'{w["start"]+offset:05d}.png')
        v = value['video']
        timeline = encode_frames(frames, output/'reconstruction.mp4', v['frames'], v['fps'])
        write_json(output/'generation.json', dict(status='COMPLETE_LTX_GENERATION',
            source_frames_read=False, timeline=timeline, profile=value['profile'],
            received_input_signature=value['signature'], video_sha256=sha256(output/'reconstruction.mp4')))
    products = ['frames', 'reconstruction.mp4', 'generation.json']
    stages.step('assemble', products, products, assemble)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--vendor', type=Path, required=True)
    args = p.parse_args()
    reconstruct(read(args.request), runtime(args.model_dir, args.vendor), args.output)


if __name__ == '__main__':
    main()
