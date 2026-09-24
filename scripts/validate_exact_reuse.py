"""Fast A/B at changed boundaries, with real STDiT3/VAE weights on CUDA.

Avoid re-running unchanged 8B captioning, T5, and channel transmission. Fixed
synthetic text embeddings isolate the decoder; these are not a quality dataset.
Both arms use identical inputs and 30 RFLOW steps, including endpoint masks.
"""
import argparse
import ast
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".local/vendor/Open-Sora")]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from semantic_transmission.exact_reuse import DiffusionModelReuse, FrameTensorCache


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def load_preprocessor(source):
    names = {"build_transform", "find_closest_aspect_ratio", "dynamic_preprocess", "load_image"}
    nodes = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = dict(torch=torch, T=T, Image=Image, InterpolationMode=InterpolationMode,
                 IMAGENET_MEAN=(0.485, 0.456, 0.406), IMAGENET_STD=(0.229, 0.224, 0.225))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    return scope["load_image"]


def frame_probe(output):
    relative = Path("02_semantic_encoder/skem/MLM-keyframe-internvl.py")
    before = load_preprocessor(output / "before" / relative.name)
    after = load_preprocessor(ROOT / relative)
    images = sorted((ROOT / "outputs/wsl_smoke_20260917_2055").rglob("*.png"))[:9]
    if len(images) < 5:
        raise RuntimeError("Need existing real frames for preprocessing verification")
    pairs = [(0, 1), (0, 2), (0, 3), (3, 4), (3, 0), (0, 1)]
    def run(enabled):
        load = after if enabled else before
        cache = FrameTensorCache(lambda p: load(p, max_num=12).to(torch.bfloat16).cuda(), enabled)
        values = []
        start = sync_time()
        for a, b in pairs:
            values.append(torch.cat([cache.get(str(images[a])), cache.get(str(images[b]))]))
        elapsed = sync_time() - start
        return elapsed, values, cache.loads, cache.hits
    run(False)  # warm image/transform/CUDA paths before either timed arm
    timings = {False: [], True: []}
    comparisons = []
    for repetition in range(3):
        arms = {}
        for enabled in ((False, True) if repetition % 2 == 0 else (True, False)):
            arms[enabled] = run(enabled)
            timings[enabled].append(arms[enabled][0])
        comparisons.extend(torch.equal(a, b) for a, b in zip(arms[False][1], arms[True][1]))
    old, new = (statistics.median(timings[key]) for key in (False, True))
    result = dict(all_pair_tensors_bitwise_equal=all(comparisons), pairs_per_repeat=len(pairs),
                  repeats=3, before_seconds=old, after_seconds=new, speedup=old/new,
                  before_loads=arms[False][2], after_loads=arms[True][2], hits=arms[True][3],
                  image_paths=[str(p) for p in images], timings_seconds={str(k): v for k,v in timings.items()})
    assert result["all_pair_tensors_bitwise_equal"]
    return result


@torch.inference_mode()
def decoder_probe(output):
    import opensora.models
    import opensora.schedulers
    from opensora.registry import MODELS, SCHEDULERS, build_module
    from opensora.utils.inference_utils import prepare_multi_resolution_info, apply_mask_strategy

    paths = json.loads((ROOT / ".local/model_paths.json").read_text())
    dtype, device = torch.bfloat16, "cuda"
    config = dict(type="STDiT3-XL/2", from_pretrained=paths["stdit"], force_huggingface=True,
                  qk_norm=True, enable_flash_attn=True, enable_layernorm_kernel=True)
    vae = build_module(dict(type="OpenSoraVAE_V1_2", from_pretrained=paths["vae"],
                           vae_2d_path=paths["vae2d"], force_huggingface=True,
                           micro_frame_size=17, micro_batch_size=1), MODELS).to(device, dtype).eval()
    size = (128, 128)
    lengths = (9, 17, 9)  # vary temporal size and return to the first size
    torch.manual_seed(42)
    source = sorted((ROOT / "outputs/wsl_smoke_20260917_2055").rglob("*.png"))[0]
    pixels = np.array(Image.open(source).convert("RGB").resize(size))
    pixels = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).unsqueeze(2).to(device, dtype) / 127.5 - 1
    reference = vae.encode(pixels)
    # Unchanged text-encoder boundary: deterministic embeddings, not real T5.
    embedding = torch.randn(1, 1, 8, 4096, device=device, dtype=dtype)
    class FixedText:
        def encode(self, prompts):
            return {"y": embedding.clone(), "mask": torch.ones(1, 8, device=device, dtype=torch.long)}
        def null(self, n):
            return self.y_embedder.y_embedding[:8].unsqueeze(0).unsqueeze(0).repeat(n, 1, 1, 1)
    text = FixedText()
    records = {}
    originals = []
    rng_initial = torch.cuda.get_rng_state()
    for enabled in (False, True):
        name = "after" if enabled else "before"
        cache = DiffusionModelReuse(config, device, enabled)
        torch.cuda.set_rng_state(rng_initial)
        segments = []
        previous = None
        for i, length in enumerate(lengths):
            latent_size = vae.get_latent_size((length, *size))
            rng_before_build = torch.cuda.get_rng_state()
            start = sync_time()
            model = cache.get(lambda: build_module(config, MODELS, input_size=latent_size,
                              in_channels=4, caption_channels=4096, model_max_length=300,
                              enable_sequence_parallelism=False).to(device, dtype).eval(), latent_size)
            build_seconds = sync_time() - start
            assert torch.equal(rng_before_build, torch.cuda.get_rng_state()), "Model build changed CUDA RNG"
            text.y_embedder = model.y_embedder
            scheduler = build_module(dict(type="rflow", num_sampling_steps=30, cfg_scale=7.0,
                                          use_timestep_transform=True), SCHEDULERS)
            args = prepare_multi_resolution_info("STDiT2", 1, size, length, 24, device, dtype)
            refs = [[reference[0]]]
            strategy = "0,0,0,0,1;0,0,0,-1,1"
            if previous is not None:
                overlap = vae.encode(previous[:, :, -1:])
                refs[0].append(overlap[0])
                strategy = "0,1,0,0,1;0,0,0,-1,1"
            z = torch.randn(1, 4, *latent_size, device=device, dtype=dtype)
            masks = apply_mask_strategy(z, refs, [strategy], 0, align=None)
            start = sync_time()
            samples = scheduler.sample(model, text, z=z, prompts=["fixed conditioning"], device=device,
                                       additional_args=args, progress=False, mask=masks)
            latent_seconds = sync_time() - start
            start = sync_time()
            decoded = vae.decode(samples.to(dtype), num_frames=length)
            decode_seconds = sync_time() - start
            previous = decoded
            pixels_u8 = decoded.clamp(-1, 1).add(1).div(2).mul(255).add(0.5).clamp(0, 255).to(torch.uint8)
            item = dict(frames=length, latent_size=latent_size, build_seconds=build_seconds,
                        diffusion_seconds=latent_seconds, vae_decode_seconds=decode_seconds,
                        latent_sha256=digest(samples), pixels_sha256=digest(pixels_u8))
            if not enabled:
                originals.append((samples.cpu(), decoded.cpu(), pixels_u8.cpu()))
            else:
                old_latent, old_decoded, old_pixels = originals[i]
                item.update(latent_bitwise_equal=torch.equal(samples.cpu(), old_latent),
                            decoded_bitwise_equal=torch.equal(decoded.cpu(), old_decoded),
                            pixels_bitwise_equal=torch.equal(pixels_u8.cpu(), old_pixels),
                            max_abs_pixel_float_error=(decoded.cpu().float()-old_decoded.float()).abs().max().item())
                assert item["latent_bitwise_equal"] and item["decoded_bitwise_equal"] and item["pixels_bitwise_equal"]
            segment_dir = output / name / f"segment_{i}"
            segment_dir.mkdir(parents=True, exist_ok=True)
            for j, frame in enumerate(pixels_u8[0].permute(1, 2, 3, 0).cpu().numpy()):
                Image.fromarray(frame).save(segment_dir / f"{j:03d}.png")
            segments.append(item)
            print(name, i, json.dumps(item), flush=True)
            del model
            text.y_embedder = None
            if not enabled:
                gc.collect()
        records[name] = dict(builds=cache.builds, hits=cache.hits, segments=segments)
        del cache
        gc.collect()
        torch.cuda.empty_cache()
    for entry in records.values():
        entry["build_seconds"] = sum(s["build_seconds"] for s in entry["segments"])
        entry["measured_component_seconds"] = sum(s[k] for s in entry["segments"]
            for k in ("build_seconds", "diffusion_seconds", "vae_decode_seconds"))
        # Exclude each arm's first segment: baseline pays cold checkpoint/kernel
        # startup, whereas after runs second. Report repeated-use timing instead.
        entry["subsequent_segments_seconds"] = sum(s[k] for s in entry["segments"][1:]
            for k in ("build_seconds", "diffusion_seconds", "vae_decode_seconds"))
    return dict(arms=records, frame_lengths=lengths, image_size=size, steps=30,
                text_condition="fixed synthetic embeddings; no T5 inference",
                checkpoint_paths=paths,
                speedup=records["before"]["measured_component_seconds"]/records["after"]["measured_component_seconds"],
                subsequent_segments_speedup=records["before"]["subsequent_segments_seconds"]/records["after"]["subsequent_segments_seconds"],
                limitation="Small component A/B; not full-pipeline timing or semantic-quality evaluation. Baseline runs first.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    report = dict(status="RUNNING", torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
                      ROOT/"src/semantic_transmission/exact_reuse.py",
                      ROOT/"scripts/validate_exact_reuse.py",
                      ROOT/"02_semantic_encoder/skem/MLM-keyframe-internvl.py",
                      ROOT/"04_semantic_decoder/scripts/mydemo_new_align_sh.py"]})
    def save():
        (args.output / "comparison.json").write_text(json.dumps(report, indent=2)+"\n")
    try:
        report["preprocessing"] = frame_probe(args.output)
        save()
        print("preprocessing", report["preprocessing"], flush=True)
        report["decoder"] = decoder_probe(args.output)
        report["status"] = "PASSED"
    except BaseException as error:
        report.update(status="FAILED", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
