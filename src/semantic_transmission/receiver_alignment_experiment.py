"""Paired receiver-only diagnostic windows; no source pixels enter generation.

This deliberately does not change the default decoder. Three arms isolate end
position from VAE temporal context, using identical received data and noise.
"""
import argparse
import copy
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import time

from .artifacts import sha256, write_json
from .webvid5 import fingerprint, read_json
from .receiver_temporal_alignment import capture_single_image, phase_endpoint, temporal_layout
from .temporal import segment_lengths, conditioning_indices, concatenate_segments

ARMS = ("legacy_aligned", "exact_last", "phase_last")
POLICY = "receiver_temporal_alignment_paired_windows_v1"
CODE = ("receiver_alignment_experiment.py", "receiver_temporal_alignment.py",
        "condition_collision.py", "generation_noise.py", "precision_text_cache.py", "temporal.py")


def cases(repo):
    short = repo / "outputs/etri_latest_short3_20261001"
    result = [dict(name="zea", run=str(repo / "outputs/qwen35_pairs_zea_20261007/reconstruction/event_pairs/run"),
                   keys=[144, 168, 192], scene="aerial camera movement")]
    for name, keys, scene in (("person_walk", [8, 32, 56], "person walking"),
                              ("single_subject", [0, 24, 48], "person turning, scarf and hair"),
                              ("candle_flowers", [0, 24, 48], "low-motion lampshade closeup")):
        result.append(dict(name=name, run=str(short / name / "reconstruction_fp32_fc_lgvsc_default_v1/run"),
                           keys=keys, scene=scene))
    return result


def prepare_case(case):
    run = Path(case["run"])
    keydir = run / "receiver/frames/sample/key_frames_received"
    all_keys = sorted(int(p.stem) for p in keydir.glob("*.png"))
    positions = [all_keys.index(k) for k in case["keys"]]
    if positions != list(range(positions[0], positions[0]+len(positions))):
        raise ValueError("experiment must retain adjacent received keys")
    report_path = run / "receiver/text_embeddings.json"
    report = read_json(report_path)
    if report["metadata_sha256"] != sha256(run / "receiver/metadata.csv"):
        raise ValueError("caption metadata differs from prepared T5 conditions")
    prompts = report["prompts"][positions[0]:positions[-1]]
    entries = {e["prompt"]: e for e in report["entries"]}
    inputs = {str(keydir/f"{k}.png"): sha256(keydir/f"{k}.png") for k in case["keys"]}
    inputs[str(report_path)] = sha256(report_path)
    for prompt in prompts:
        entry = entries[prompt]
        actual = sha256(Path(entry["path"]))
        if actual != entry["sha256"]:
            raise ValueError("cached text embedding changed")
        inputs[entry["path"]] = actual
    return dict(case, prompts=prompts, inputs=inputs,
                lengths=segment_lengths([k-case["keys"][0] for k in case["keys"]]))


def make_protocol(repo, selected):
    from mmengine import Config
    config_path = Path(selected[0]["run"]) / "receiver/decoder_config.py"
    cfg = Config.fromfile(str(config_path))
    # All cases must use the same generator settings; text itself stays case-specific.
    shared = ("image_size", "fps", "seed", "model", "vae", "scheduler", "multi_resolution")
    for case in selected:
        other = Config.fromfile(str(Path(case["run"])/"receiver/decoder_config.py"))
        if any(other[k] != cfg[k] for k in shared):
            raise ValueError(f"generator configuration differs: {case['name']}")
    checkpoints = {}
    roots = [Path(cfg.model.from_pretrained), Path(cfg.vae.from_pretrained),
             Path(cfg.vae.vae_2d_path)/"vae"]
    for root in roots:
        for path in sorted(root.iterdir()):
            if path.is_file() and path.suffix in {".json", ".bin", ".safetensors"}:
                checkpoints[str(path)] = sha256(path)
    vendor = repo / ".local/vendor/Open-Sora/opensora"
    code = {str(repo/"src/semantic_transmission"/name): sha256(repo/"src/semantic_transmission"/name)
            for name in CODE}
    for subdir in ("models/vae", "models/stdit", "models/layers", "schedulers"):
        code.update({str(p): sha256(p) for p in sorted((vendor/subdir).rglob("*.py"))})
    code[str(vendor/"utils/inference_utils.py")] = sha256(vendor/"utils/inference_utils.py")
    return cfg, dict(policy=POLICY, cases=selected, arms=list(ARMS),
        generator={k: cfg[k] for k in shared}, checkpoints=checkpoints, code=code,
        source_pixels_used_for_generation=False, output_pixel_interpolation=False,
        training_executed=False, window_context="restart at first selected received key",
        scope="four short diagnostic windows, not a generalization benchmark")


def mask_for_arm(z, refs, strategy, loop, mode, functions, endpoint):
    from .condition_collision import plan
    record = plan(strategy, [r.shape[1] for r in refs[0]], z.shape[2], loop, 3, 5, functions)
    effective_refs = [list(refs[0])]
    end = next(o for o in record["operations"] if o["reference"] == loop+1)
    if mode != "legacy_aligned":
        end["target"] = end["requested_target"]
    if mode == "phase_last":
        effective_refs[0][loop+1] = endpoint.squeeze(0)
    record["effective_strategy"] = ";".join(
        f"{loop},{o['reference']},{o['source']},{o['target']},{o['length']},0"
        for o in record["operations"])
    mask = functions.apply_mask_strategy(z, effective_refs, [record["effective_strategy"]], loop, align=None)
    return mask, record


def complete_arm(directory, signature):
    receipt = directory / "result.json"
    if not receipt.exists():
        return False
    result = read_json(receipt)
    if result["signature"] != signature or result["status"] != "GENERATED":
        raise ValueError(f"incompatible existing result: {directory}")
    if any(sha256(directory/name) != digest for name, digest in result["artifacts"].items()):
        raise ValueError(f"completed artifact changed: {directory}")
    return True


def generate(repo, output, cfg, protocol, signature):
    import torch
    from PIL import Image
    from opensora.registry import MODELS, SCHEDULERS, build_module
    from opensora.datasets.utils import get_transforms_image
    from opensora.utils import inference_utils as inference
    from .generation_noise import NoiseControl
    from .precision_text_cache import Encoder

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    device, dtype = "cuda", torch.bfloat16
    runtime = dict(packages={n: importlib.metadata.version(n) for n in ("torch", "diffusers", "transformers")},
                   cuda=torch.version.cuda, gpu=torch.cuda.get_device_name())
    total_arms = len(protocol["cases"])*len(ARMS)
    finished = sum(complete_arm(output/c["name"]/a, signature) for c in protocol["cases"] for a in ARMS)
    if finished == total_arms:
        print(f"RESUME generation {finished}/{total_arms} (100%): verified completed arms", flush=True)
        return
    print(f"LOAD frozen VAE / STDiT; completed arms {finished}/{total_arms}; loading ETA unknown", flush=True)
    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()
    model = build_module(cfg.model, MODELS, input_size=vae.get_latent_size((25, *cfg.image_size)),
        in_channels=vae.out_channels, caption_channels=4096, model_max_length=300,
        enable_sequence_parallelism=False).to(device, dtype).eval()
    transform = get_transforms_image(image_size=tuple(cfg.image_size), name="resize_crop")
    steps = cfg.scheduler.num_sampling_steps
    start, newly_done = time.monotonic(), 0
    with torch.inference_mode():
        for case in protocol["cases"]:
            run = Path(case["run"])
            for arm in ARMS:
                directory = output/case["name"]/arm
                if complete_arm(directory, signature):
                    print(f"RESUME {case['name']}/{arm}: verified", flush=True)
                    continue
                directory.mkdir(parents=True, exist_ok=True)
                contract = dict(seed=cfg.seed, segments=2, steps=steps, inputs=fingerprint(case),
                                code=signature, reference_policy="last_17_frames_before_vae")
                baseline = output/case["name"]/ARMS[0]/"generation_noise.json"
                expected = read_json(baseline) if arm != ARMS[0] else None
                control = NoiseControl(directory/"generation_noise.json", contract, runtime, expected)
                write_json(directory/"status.json", dict(status="GENERATING", signature=signature))
                report = copy.deepcopy(read_json(run/"receiver/text_embeddings.json"))
                report["prompts"], report["prompt_count"] = case["prompts"], len(case["prompts"])
                report["entries"] = [e for e in report["entries"] if e["prompt"] in case["prompts"]]
                encoder = Encoder(report, directory/"text_embedding_trace.json")
                original_encode = encoder.encode
                encoder.encode = lambda texts: {k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
                                                for k, v in original_encode(texts).items()}
                encoder.y_embedder = model.y_embedder
                refs, records = [[]], []
                for i, key in enumerate(case["keys"]):
                    path = run/f"receiver/frames/sample/key_frames_received/{key}.png"
                    with Image.open(path) as im:
                        x = transform(im.convert("RGB")).unsqueeze(0).unsqueeze(2).to(device, dtype)
                    with control.scope("vae", i, x), capture_single_image(vae) as record:
                        refs[0].append(vae.encode(x).squeeze(0))
                    records.append(record)
                strategy, clips, condition_trace = "0;0,1,0,-1,1;1,2,0,-1,1", [], []
                for loop, frames in enumerate(case["lengths"]):
                    if loop:
                        previous = clips[-1][:, :, conditioning_indices(clips[-1].shape[2])].to(device, dtype)
                        with control.scope("vae", 3, previous):
                            refs[0].append(vae.encode(previous).squeeze(0))
                        strategy += ";1,3,-5,0,5,0"
                    latent_size = vae.get_latent_size((frames, *cfg.image_size))
                    model.config.input_size = latent_size
                    z = torch.empty(1, vae.out_channels, *latent_size, device=device, dtype=dtype)
                    with control.scope("initial", loop, z):
                        z.copy_(torch.randn_like(z))
                    block_frames = temporal_layout(frames)[-1]["frames"]
                    endpoint = phase_endpoint(vae, records[loop+1], block_frames) if arm == "phase_last" else None
                    mask, condition = mask_for_arm(z, refs, strategy, loop, arm, inference, endpoint)
                    condition.update(frames=frames, final_block_frames=block_frames, arm=arm)
                    condition_trace.append(condition)
                    write_json(directory/"conditions.json", condition_trace)
                    initial = z.clone()
                    scheduler = build_module(cfg.scheduler, SCHEDULERS)
                    extra = inference.prepare_multi_resolution_info(cfg.multi_resolution, 1, tuple(cfg.image_size),
                                                                      frames, cfg.fps, device, dtype)
                    step = [0]
                    def progress(module, inputs):
                        step[0] += 1
                        if step[0] == 1 or step[0] % 5 == 0:
                            done = finished*2*steps + loop*steps + step[0]-1
                            total = total_arms*2*steps
                            print(f"GENERATE {done}/{total} steps ({100*done/total:.1f}%) | {case['name']}/{arm} "
                                  f"segment {loop+1}/2 step {step[0]}/{steps} | elapsed {time.monotonic()-start:.0f}s", flush=True)
                    handle = model.register_forward_pre_hook(progress)
                    try:
                        with control.scope("sampling", loop, z):
                            sampled = scheduler.sample(model, encoder, z=z, prompts=[case["prompts"][loop]],
                                device=device, additional_args=extra, progress=False, mask=mask)
                    finally:
                        handle.remove()
                    if step[0] != steps or not torch.isfinite(sampled).all():
                        raise ValueError("invalid model sampling result")
                    if not torch.equal(sampled[:, :, mask[0] == 0], initial[:, :, mask[0] == 0]):
                        raise ValueError("sampling changed fixed receiver conditions")
                    clips.append(vae.decode(sampled.to(dtype), num_frames=frames).cpu())
                control.finish(2, steps)
                video = concatenate_segments([v[0] for v in clips])
                pixels = (video.clamp(-1, 1).add(1).div(2).mul(255).add(.5).clamp(0, 255)
                          .permute(1, 2, 3, 0).to(torch.uint8).numpy())
                if len(pixels) != case["keys"][-1]-case["keys"][0]+1:
                    raise ValueError("output frame count differs from input key times")
                framedir = directory/"frames"
                framedir.mkdir(exist_ok=True)
                if any(int(p.stem) >= len(pixels) for p in framedir.glob("*.png")):
                    raise ValueError("stale frames in unfinished output")
                for i, frame in enumerate(pixels):
                    Image.fromarray(frame).save(framedir/f"{i:05d}.png")
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(cfg.fps), "-i",
                    str(framedir/"%05d.png"), "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                    str(directory/"reconstruction.mp4")], check=True)
                artifact_paths = [*sorted(framedir.glob("*.png")), directory/"generation_noise.json",
                                  directory/"conditions.json", directory/"text_embedding_trace.json",
                                  directory/"reconstruction.mp4"]
                write_json(directory/"result.json", dict(status="GENERATED", signature=signature,
                    case=case["name"], arm=arm, frames=len(pixels), fps=cfg.fps,
                    artifacts={str(p.relative_to(directory)): sha256(p) for p in artifact_paths}))
                write_json(directory/"status.json", dict(status="GENERATED", signature=signature))
                finished += 1
                newly_done += 1
                eta = (time.monotonic()-start)/newly_done*(total_arms-finished)
                print(f"DONE {case['name']}/{arm}: {finished}/{total_arms} arms ({100*finished/total_arms:.1f}%) "
                      f"elapsed={time.monotonic()-start:.0f}s estimated remaining={eta:.0f}s", flush=True)
                del clips, video, pixels, refs, records, encoder, sampled, initial
    del model, vae
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="default: a directory named by the frozen protocol signature")
    parser.add_argument("--check", action="store_true", help="verify received inputs and print protocol, no GPU sampling")
    parser.add_argument("--stage", choices=("all", "generate", "evaluate"), default="all")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo/".local/vendor/Open-Sora"))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    print("CHECK received inputs, text caches and frozen checkpoint hashes (ETA unknown)", flush=True)
    selected = [prepare_case(c) for c in cases(repo)]
    cfg, protocol = make_protocol(repo, selected)
    signature = fingerprint(protocol)
    output = (args.output or repo/"outputs/receiver_temporal_alignment_20261007"/f"paired_{signature[:12]}").resolve()
    if args.check:
        print(f"CHECK PASSED: {len(selected)} scenes x {len(ARMS)} arms x 2 segments; signature={signature}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    path = output/"protocol.json"
    # JSON round-trips tuples as lists; compare the canonical serialized contract.
    if path.exists() and fingerprint(read_json(path)) != signature:
        raise ValueError("output belongs to another protocol; choose a new --output")
    write_json(path, protocol)
    archive = output/"generation_code"
    archive.mkdir(exist_ok=True)
    for name in CODE:
        (archive/name).write_bytes((repo/"src/semantic_transmission"/name).read_bytes())
    try:
        if args.stage in {"all", "generate"}:
            write_json(output/"status.json", dict(status="GENERATING", signature=signature))
            generate(repo, output, cfg, protocol, signature)
        if args.stage in {"all", "evaluate"}:
            if not all(complete_arm(output/c["name"]/a, signature) for c in selected for a in ARMS):
                raise ValueError("evaluation requires all generation arms")
            write_json(output/"status.json", dict(status="EVALUATING", signature=signature))
            from .receiver_alignment_evaluation import evaluate
            evaluate(output, protocol, signature)
        else:
            write_json(output/"status.json", dict(status="GENERATED_EVALUATION_PENDING", signature=signature))
    except BaseException as error:
        write_json(output/"status.json", dict(status="FAILED", signature=signature, error=str(error)))
        raise


if __name__ == "__main__":
    main()
