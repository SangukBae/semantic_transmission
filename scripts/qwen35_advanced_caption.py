"""Offline higher-capacity Qwen captioning; preserves the frozen NF4 baseline.

Memory limits include a reserve for the display and allocator workspaces.
Existing GPU jobs are never stopped by this program.
"""
import argparse
import os
from pathlib import Path
import subprocess
import threading
import time

for key, value in {
    'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
    'HF_HUB_DISABLE_TELEMETRY': '1', 'TOKENIZERS_PARALLELISM': 'false',
    'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
    'PYTORCH_ALLOC_CONF': 'expandable_segments:True',
}.items():
    os.environ.setdefault(key, value)

import qwen35_caption as base


class Monitor:
    """Sample device-wide usage, separately from PyTorch allocation peaks."""
    def __init__(self):
        self.samples = []
        self.done = threading.Event()

    def sample(self):
        rows = base.gpu_status().get('gpus', [])
        if rows:
            self.samples.append(dict(time_utc=base.now(), **rows[0]))

    def loop(self):
        while not self.done.is_set():
            self.sample()
            self.done.wait(0.5)

    def __enter__(self):
        self.sample()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.done.set()
        self.thread.join(timeout=3)
        self.sample()

    def summary(self):
        return dict(sample_count=len(self.samples), sample_interval_seconds=0.5,
                    sampled_device_peak_used_mib=max((x['used_mib'] for x in self.samples), default=None),
                    sampled_device_min_free_mib=min((x['free_mib'] for x in self.samples), default=None))


def load(quantization, reserve_mib=1024):
    import torch
    from transformers import BitsAndBytesConfig, Qwen3_5ForConditionalGeneration
    if reserve_mib < 1024:
        raise ValueError('Keep at least 1024 MiB headroom on the shared display GPU')
    rows = base.gpu_status().get('gpus', [])
    if not rows or rows[0]['free_mib'] < 12000:
        raise RuntimeError('At least 12000 MiB free GPU memory is required before model loading')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    state = rows[0]
    budget_mib = state['free_mib'] - reserve_mib
    torch.cuda.set_per_process_memory_fraction(budget_mib / state['total_mib'], 0)
    torch.set_num_threads(2)
    torch.manual_seed(2025)
    if quantization == 'int8':
        config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0,
                                   llm_int8_skip_modules=['visual', 'lm_head'])
    elif quantization == 'nf4':
        config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                   bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16,
                                   llm_int8_skip_modules=['visual', 'lm_head'])
    else:
        raise ValueError(quantization)
    start = time.monotonic()
    with Monitor() as monitor:
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            base.SNAPSHOT, local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, device_map={'': 'cuda:0'},
            quantization_config=config, attn_implementation='sdpa').eval()
        torch.cuda.synchronize()
    record = dict(model_id=base.MODEL_ID, revision=base.REVISION, quantization=quantization,
                  dtype='bfloat16', attention='sdpa', enable_thinking=False,
                  do_sample=False, seed=2025, reserve_mib=reserve_mib,
                  allocator_budget_mib=budget_mib, gpu_before=state,
                  model_load_seconds=time.monotonic() - start,
                  resident_allocated_bytes=torch.cuda.memory_allocated(),
                  load_device_memory=monitor.summary(), versions=base.packages())
    return model, base.load_processor(), record


def generate(model, processor, images, labels, prompt, max_pixels=1048576, max_new_tokens=256):
    import torch
    if not 1 <= len(images) <= 16:
        raise ValueError('Use 1..16 chronological images')
    if not 4096 <= max_pixels <= 1048576 or not 1 <= max_new_tokens <= 512:
        raise ValueError('Use 4096..1048576 max pixels and 1..512 output tokens')
    inputs = base.preprocess(processor, images, labels, prompt, max_pixels)
    if inputs.input_ids.shape[1] > 8192:
        raise ValueError('Input exceeds 8192 tokens; reduce frames or resolution')
    description = dict(input_tokens=inputs.input_ids.shape[1],
                       image_grid_thw=inputs.image_grid_thw.tolist(),
                       image_native_sizes=[list(im.size) for im in images],
                       max_pixels=max_pixels, image_count=len(images), max_new_tokens=max_new_tokens)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    with Monitor() as monitor:
        inputs = inputs.to('cuda:0')
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                    do_sample=False, use_cache=True)
        torch.cuda.synchronize()
        tokens = output[0, inputs.input_ids.shape[1]:]
        caption = processor.decode(tokens, skip_special_tokens=True).strip()
        description.update(caption=caption, generated_tokens=len(tokens),
                           truncated=len(tokens) >= max_new_tokens,
                           generation_seconds=time.monotonic() - start,
                           peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                           peak_reserved_bytes=torch.cuda.max_memory_reserved())
    description['device_memory'] = monitor.summary()
    if not caption or description['truncated']:
        description['status'] = 'EMPTY_OR_TRUNCATED'
    else:
        description['status'] = 'CAPTION_COMPLETE'
    return description


def video_frames(path, start, end, frames):
    import cv2
    from PIL import Image
    if not 1 <= frames <= 16 or start < 0 or (end is not None and end <= start):
        raise ValueError('Invalid interval or frame count (1..16)')
    path = Path(path).resolve()
    cap = cv2.VideoCapture(str(path))
    try:
        fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or fps <= 0:
            raise ValueError(f'Cannot decode {path}')
        a, b = int(start * fps), min(count, int(end * fps)) if end is not None else count
        if not 0 <= a < b <= count:
            raise ValueError('Invalid interval')
        n = min(frames, b - a)
        indices = sorted({min(b-1, a+int((i+.5)*(b-a)/n)) for i in range(n)})
        images = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f'Cannot decode source frame {i}')
            images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        return images, [f'Source frame {i}, timestamp {i/fps:.3f} seconds' for i in indices], dict(
            path=str(path), sha256=base.sha256(path), fps=fps, interval_frames=[a,b], sampled_indices=indices)
    finally:
        cap.release()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--start', type=float, default=0)
    p.add_argument('--end', type=float)
    p.add_argument('--frames', type=int, default=8)
    p.add_argument('--quantization', choices=['nf4','int8'], default='int8')
    p.add_argument('--max-pixels', type=int, default=1048576)
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--prompt', default=base.PROMPT)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    images, labels, source = video_frames(args.video, args.start, args.end, args.frames)
    model, processor, runtime = load(args.quantization)
    result = generate(model, processor, images, labels, args.prompt, args.max_pixels, args.max_new_tokens)
    base.write_json(args.output, dict(created_utc=base.now(), inference_provider='LOCAL_ONLY',
                                     runtime=runtime, source=source, prompt=args.prompt,
                                     quality_verified=False, **result))
    print(result['caption'])
    if result['status'] != 'CAPTION_COMPLETE':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
