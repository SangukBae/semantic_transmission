"""Diagnostic-only paired schedule correction for the captured nine-frame segment.

Does not modify the production decoder or use source frames for generation.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / ".local/vendor/Open-Sora"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnosis", type=Path, default=REPO / "outputs/etri_person_walk_vae_diagnosis_20261001_v3")
    args = parser.parse_args()
    root = args.diagnosis.resolve()
    dest = root / "schedule_probe"
    dest.mkdir(exist_ok=False)
    start = time.monotonic()
    import numpy as np
    import torch
    from PIL import Image
    from mmengine import Config
    import opensora.models
    from opensora.registry import build_module, MODELS
    from opensora.schedulers import rf
    from opensora.utils.inference_utils import prepare_multi_resolution_info
    from semantic_transmission.artifacts import sha256, write_json
    from semantic_transmission.webvid5 import read_json
    from semantic_transmission.generation_noise import NoiseControl
    from semantic_transmission.precision_text_cache import Encoder

    result = read_json(root / "RESULT.json")
    assert all(result["replay_pixel_equal"]) and result["noise_prefix_matches"]
    source = Path(result["source"])
    cfg = Config.fromfile(source / "run/receiver/decoder_config.py")
    state = torch.load(root / "replay_capture.pt", map_location="cpu", weights_only=True)
    trace = read_json(source / "run/receiver/generation_noise.json")
    sampling_record = next(r for r in trace["records"] if r["kind"] == "sampling" and r["index"] == 0)
    expected = {**trace, "records": [sampling_record]}
    torch.set_grad_enabled(False)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    z = state["before_sampling"].to("cuda")
    mask = state["mask"].to("cuda")
    model = build_module(cfg.model, MODELS, input_size=tuple(z.shape[2:]), in_channels=4,
        caption_channels=4096, model_max_length=300, enable_sequence_parallelism=False).to("cuda", torch.bfloat16).eval()
    report = read_json(source / "run/receiver/text_embeddings.json")
    encoder = Encoder(report, dest / "text_trace.json")
    encoder.y_embedder = model.y_embedder
    original_encode = encoder.encode
    def encode(prompts):
        return {k: v.to(device="cuda", dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
                for k, v in original_encode(prompts).items()}
    encoder.encode = encode
    model_args = prepare_multi_resolution_info(cfg.multi_resolution, 1, cfg.image_size, 9, cfg.fps, "cuda", torch.bfloat16)
    original_transform = rf.timestep_transform
    def corrected(t, model_kwargs, base_resolution=512*512, base_num_frames=1, scale=1.0, num_timesteps=1):
        frames = model_kwargs["num_frames"]
        assert frames.numel() == 1
        if not 1 < frames.item() < 17:
            return original_transform(t, model_kwargs, base_resolution, base_num_frames, scale, num_timesteps)
        # The exact three-slot count used by this VAE for nine frames.
        time_ratio = (((frames + 3) // 4) / base_num_frames).sqrt()
        space_ratio = ((model_kwargs["height"] * model_kwargs["width"]) / base_resolution).sqrt()
        ratio = space_ratio * time_ratio * scale
        unit = t / num_timesteps
        return ratio * unit / (1 + (ratio - 1) * unit) * num_timesteps
    rf.timestep_transform = corrected
    steps = []
    def prehook(module, inputs, keywords):
        steps.append(dict(t=inputs[1].float().cpu().tolist(), x_mask=keywords["x_mask"].cpu().tolist()))
    handle = model.register_forward_pre_hook(prehook, with_kwargs=True)
    control = NoiseControl(dest / "noise.json", trace["contract"], trace["runtime"], expected)
    scheduler = rf.RFLOW(**{k: v for k, v in cfg.scheduler.items() if k != "type"})
    with control.scope("sampling", 0, z):
        after = scheduler.sample(model, encoder, z, [report["prompts"][0]], "cuda",
            additional_args=model_args, mask=mask, progress=False)
    handle.remove()
    assert control.records == [sampling_record]
    assert all(np.isfinite(step["t"]).all() and step["t"][0] > 0 for step in steps)
    assert all(torch.equal(after[:, :, i].float().cpu(), state["before_sampling"][:, :, i].float()) for i in (0, 2))
    del model, encoder
    torch.cuda.empty_cache()
    vae = build_module(cfg.vae, MODELS).to("cuda", torch.bfloat16).eval()
    output = vae.decode(after.to(torch.bfloat16), num_frames=9).cpu()
    arr = (output[0].clamp(-1,1).add(1).div(2).mul(255).add(0.5).clamp(0,255)
           .permute(1,2,3,0).to(torch.uint8).numpy())
    (dest / "frames").mkdir()
    for i, frame in enumerate(arr):
        Image.fromarray(frame).save(dest / "frames" / f"{i:02d}.png")
    policy = read_json(source / "run/receiver_policy.json")
    paths = [Path(policy["paired_source"]) / f"run/data/frames/sample/{i}.png" for i in range(9)]
    target = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in paths])
    psnr = 10*np.log10(255**2/np.maximum(((arr.astype(float)-target.astype(float))**2).mean((1,2,3)),1e-12))
    torch.save(dict(latents=after.cpu(), output=output), dest / "capture.pt")
    details = dict(status="PAIRED_SHORT_SCHEDULE_PROBE_COMPLETE", scope="first nine frames only",
        source=str(source), source_capture_sha256=sha256(root / "replay_capture.pt"),
        input_latents_mask_captions_unchanged=True, sampling_noise_draws_matched=30,
        endpoint_latents_unchanged=True, oracle_source_frames_used_in_generation=False,
        sampling_slot_rms_change=((after.cpu().float()-state["before_sampling"].float())**2).mean((0,1,3,4)).sqrt().tolist(),
        steps=steps, psnr_per_frame=psnr.tolist(), psnr_mean=float(psnr.mean()), psnr_frames_1_5=float(psnr[1:6].mean()),
        total_seconds=time.monotonic()-start, default_method_changed=False, whole_video_verified=False,
        code_sha256=sha256(Path(__file__)), caveats=["One short segment; downstream quality and hallucination mitigation remain unverified.",
            "The schedule uses ceil(frames/4) only for this short-segment probe; full implementation is not adopted."])
    write_json(dest / "RESULT.json", details)
    print(json.dumps({k:details[k] for k in ("status","psnr_mean","psnr_frames_1_5","total_seconds")}), flush=True)


if __name__ == "__main__":
    main()
