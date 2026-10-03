"""Bounded, read-only first-segment replay and VAE counterfactual diagnosis.

Oracle intermediates are available only for diagnosis, never a receiver method.
Original decoder files and completed artifacts are not changed.
"""
import argparse
import datetime
import faulthandler
import gc
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import time

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / ".local/vendor/Open-Sora"))
from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.webvid5 import read_json


class FirstSegmentComplete(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=REPO / "outputs/etri_latest_short3_20261001/person_walk/reconstruction_fp32_condition_fix")
    parser.add_argument("--output", type=Path, default=REPO / "outputs/etri_person_walk_vae_diagnosis_20261001")
    args = parser.parse_args()
    source, dest = args.source.resolve(), args.output.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    import torch
    from opensora.models.vae.vae import VideoAutoencoderPipeline
    from opensora.schedulers.rf import RFLOW
    from opensora.datasets.utils import get_transforms_image

    original_encode, original_decode = VideoAutoencoderPipeline.encode, VideoAutoencoderPipeline.decode
    original_sample = RFLOW.sample
    run = dest / "replay_inputs"
    (run / "receiver").mkdir(parents=True)
    for name in ("receiver_policy.json", "keyframes.json", "receiver/text_embeddings.json", "receiver/decoder_config.py"):
        shutil.copyfile(source / "run" / name, run / name)
    (run / "receiver/frames").symlink_to(source / "run/receiver/frames", target_is_directory=True)
    policy = read_json(run / "receiver_policy.json")
    policy["noise_reference"] = str(source / "run/receiver/generation_noise.json")
    write_json(run / "receiver_policy.json", policy)
    os.environ.update(ETRI_PRECISION_RUN=str(run), ETRI_REFERENCE_TRACE=str(run / "receiver/reference_trace.json"),
                      ETRI_TAIL_REFERENCE_TRACE=str(run / "receiver/tail_reference_trace.json"))
    state = {"key_latents": [], "steps": []}

    def encoded(vae, x):
        z = original_encode(vae, x)
        state["key_latents"].append(z.detach().cpu().clone())
        return z

    def sampled(scheduler, model, text_encoder, z, *rest, **kwargs):
        state["before_sampling"] = z.detach().cpu().clone()
        state["mask"] = kwargs["mask"].detach().cpu().clone()
        def prehook(module, inputs, keywords):
            state["steps"].append(dict(t=inputs[1].float().cpu().tolist(),
                x_mask=keywords["x_mask"].cpu().tolist()))
        handle = model.register_forward_pre_hook(prehook, with_kwargs=True)
        try:
            result = original_sample(scheduler, model, text_encoder, z, *rest, **kwargs)
        finally:
            handle.remove()
        state["after_sampling"] = result.detach().cpu().clone()
        return result

    def decoded(vae, z, num_frames=None):
        assert num_frames == 9 and z.shape[2] == 3, (num_frames, z.shape)
        state["vae"] = vae
        state["decoded"] = original_decode(vae, z, num_frames=num_frames).detach().cpu()
        raise FirstSegmentComplete()

    VideoAutoencoderPipeline.encode = encoded
    VideoAutoencoderPipeline.decode = decoded
    RFLOW.sample = sampled
    sys.argv = [str(REPO / "scripts/etri_condition_fix_decoder.py"), "--save_dir", str(dest / "unused_full_output"),
        "--csv_path", str(source / "run/receiver/reconstruction/inputs/00000.csv"), "--method", "key_frames_received",
        "--root_path", str(run / "receiver"), str(run / "receiver/decoder_config.py")]
    try:
        runpy.run_path(str(REPO / "scripts/etri_condition_fix_decoder.py"), run_name="__main__")
        raise RuntimeError("Expected a deliberate stop after the first segment")
    except FirstSegmentComplete:
        pass
    finally:
        faulthandler.cancel_dump_traceback_later()
        VideoAutoencoderPipeline.encode = original_encode
        VideoAutoencoderPipeline.decode = original_decode
        RFLOW.sample = original_sample
    replay_seconds = time.monotonic() - start
    torch.save({k: v for k, v in state.items() if k != "vae"}, dest / "replay_capture.pt")
    gc.collect()
    torch.cuda.empty_cache()
    vae = state.pop("vae")
    transform = get_transforms_image(image_size=(320, 576), name="resize_crop")
    def tensor(paths):
        return torch.stack([transform(Image.open(p).convert("RGB")) for p in paths], dim=1)[None].to(vae.device, vae.dtype)
    def pixels(x):
        # Identical conversion boundary to the completed decoder's PNG output.
        return (x[0].clamp(-1, 1).add(1).div(2).mul(255).add(0.5).clamp(0, 255)
                .permute(1, 2, 3, 0).to("cpu", torch.uint8).numpy())
    def save_frames(name, x):
        p = dest / "frames" / name
        p.mkdir(parents=True)
        arr = pixels(x)
        for i, frame in enumerate(arr):
            Image.fromarray(frame).save(p / f"{i:02d}.png")
        return arr
    source_input_run = Path(policy["paired_source"]) / "run"
    source_paths = [source_input_run / f"data/frames/sample/{i}.png" for i in range(17)]
    received_paths = [source / f"run/receiver/frames/sample/key_frames_received/{i}.png" for i in (0, 8)]
    target = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in source_paths])
    replay = save_frames("replay", state["decoded"])
    expected = np.stack([np.asarray(Image.open(source / f"run/receiver/reconstruction/sample_0000_frames/{i:05d}.png")) for i in range(9)])
    equal = [bool(np.array_equal(a, b)) for a, b in zip(replay, expected)]
    trace = read_json(run / "receiver/generation_noise.json")
    expected_trace = read_json(source / "run/receiver/generation_noise.json")
    matched = trace["records"] == expected_trace["records"][:len(trace["records"])]
    assert len(trace["records"]) == 18 and matched, "First-segment noise prefix mismatch"
    assert all(equal), "Replay pixels differ from the completed output; do not draw paired conclusions"
    # Prefix only: deliberately do not mark an interrupted 15-segment trace PASSED.
    write_json(dest / "replay_verification.json", dict(status="PASSED_FIRST_SEGMENT_ONLY", frames=9,
        pixel_equal=equal, noise_prefix_records=18, noise_draws=sum(len(r["draws"]) for r in trace["records"]),
        noise_prefix_matches=True, full_video_reconstruction=False, replay_seconds=replay_seconds))
    print("DIAGNOSIS: first 9 output frames and noise prefix match the completed reconstruction", flush=True)

    cases = {}
    latent_archive = {k: v for k, v in state.items() if k != "steps"}
    def metrics(arr, truth):
        mse = np.mean((arr.astype(np.float64) - truth.astype(np.float64)) ** 2, axis=(1, 2, 3))
        psnr = 10 * np.log10(255 ** 2 / np.maximum(mse, 1e-12))
        return dict(psnr_per_frame=psnr.tolist(), psnr_mean=float(psnr.mean()),
                    psnr_frames_1_5=float(psnr[1:6].mean()) if len(psnr) >= 6 else None)
    def record(name, output, truth, diagnostic_only):
        arr = save_frames(name, output)
        cases[name] = dict(**metrics(arr, truth), shape=list(output.shape), diagnostic_only=diagnostic_only,
                           finite=bool(torch.isfinite(output).all()))
    def encode(x, seed=20251001):
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(seed)
            return original_encode(vae, x)
    def decode_case(name, z, frames=9, truth=None, diagnostic_only=True):
        z = z.to(device=vae.device, dtype=vae.dtype)
        latent_archive[name] = z.detach().cpu()
        result = original_decode(vae, z, num_frames=frames)
        record(name, result, target[:frames] if truth is None else truth, diagnostic_only)
        return result
    with torch.inference_mode():
        cases["replay"] = dict(**metrics(replay, target[:9]), shape=list(state["decoded"].shape), diagnostic_only=False, finite=True)
        real9 = tensor(source_paths[:9])
        real17 = tensor(source_paths)
        mixed9 = real9.clone()
        mixed9[:, :, :1] = tensor(received_paths[:1])
        mixed9[:, :, -1:] = tensor(received_paths[1:])
        oracle = encode(mixed9)
        decode_case("source9_roundtrip", encode(real9))
        decode_case("received_ends_source_middle_roundtrip", oracle)
        decode_case("source17_roundtrip", encode(real17), frames=17)
        keys = [z.to(vae.device) for z in state["key_latents"][:2]]
        for i, key in enumerate(keys):
            truth = np.asarray(Image.open(received_paths[i]).convert("RGB"))[None]
            decode_case(f"received_key_{i}_roundtrip", key, frames=1, truth=truth)
        for name, slots in (("oracle_with_single_start", (0,)), ("oracle_with_single_end", (2,)),
                            ("oracle_with_single_both", (0, 2))):
            value = oracle.clone()
            for slot in slots:
                value[:, :, slot:slot+1] = keys[0 if slot == 0 else 1]
            decode_case(name, value)
        generated = state["after_sampling"].to(device=vae.device, dtype=vae.dtype)
        value = generated.clone()
        value[:, :, 1:2] = oracle[:, :, 1:2]
        decode_case("generated_with_oracle_middle", value)
        value = generated.clone()
        value[:, :, 0:1], value[:, :, 2:3] = oracle[:, :, 0:1], oracle[:, :, 2:3]
        decode_case("generated_with_oracle_ends", value)
        repeated = real9[:, :, :1].repeat(1, 1, 9, 1, 1)
        repeat_truth = target[:1].repeat(9, axis=0)
        decode_case("still9_roundtrip", encode(repeated), truth=repeat_truth)
        decode_case("single_key_latent_repeated", keys[0].repeat(1, 1, 3, 1, 1), truth=np.asarray(Image.open(received_paths[0]))[None].repeat(9, axis=0))
        spatial = vae.spatial_vae.decode(vae.spatial_vae.encode(real9))
        record("spatial_only_roundtrip", spatial, target[:9], True)
    torch.cuda.synchronize()
    torch.save(latent_archive, dest / "latents.pt")
    stats = {}
    for name, z in latent_archive.items():
        if isinstance(z, torch.Tensor) and z.ndim == 5:
            stats[name] = [dict(mean=float(q.float().mean()), std=float(q.float().std()),
                               min=float(q.min()), max=float(q.max())) for q in z.unbind(2)]
    delta = state["after_sampling"].float() - state["before_sampling"].float()
    result = dict(status="NUMERIC_COMPLETE_VISUAL_REVIEW_PENDING", created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source=str(source), output=str(dest), first_segment_frames=9, latent_slots=3,
        replay_pixel_equal=equal, noise_prefix_matches=matched, mask=state["mask"].tolist(),
        sampling_steps=[dict(t=[v if np.isfinite(v) else None for v in step["t"]],
                             x_mask=step["x_mask"]) for step in state["steps"]], latent_stats=stats,
        sampling_slot_rms_change=[float(q.square().mean().sqrt()) for q in delta.unbind(2)],
        cases=cases, replay_seconds=replay_seconds, total_seconds=time.monotonic()-start,
        peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
        diagnostic_seed=20251001, vae_dtype=str(vae.dtype), source_frames_sha256={str(p):sha256(p) for p in source_paths+received_paths},
        code_sha256={str(p.relative_to(REPO)):sha256(p) for p in (Path(__file__),
            REPO/".local/vendor/Open-Sora/opensora/models/vae/vae.py", REPO/".local/vendor/Open-Sora/opensora/models/vae/vae_temporal.py",
            REPO/".local/vendor/Open-Sora/opensora/schedulers/rf/__init__.py")},
        caveats=["Oracle middle frames require the source video and cannot be used as a receiver mitigation.",
            "One video's first segment and one generation seed; no whole-video mitigation claim.",
            "All PSNR values use pre-MP4 RGB pixels, not delivered MP4 metrics.",
            "The replay noise file is intentionally partial, not a completed reconstruction trace."])
    write_json(dest / "RESULT.json", result)
    print(json.dumps(dict(status=result["status"], seconds=result["total_seconds"], cases={k:round(v['psnr_mean'],3) for k,v in cases.items()})), flush=True)


if __name__ == "__main__":
    main()
