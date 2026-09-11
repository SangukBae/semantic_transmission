"""Isolated real-model stages for the local LGVSC research profile.

Each invocation exits after one stage so large models never share VRAM.
"""

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from .artifacts import sha256, write_json


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def csv_write(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare(cfg, repo, run):
    import cv2
    import shutil
    data = run / "data"
    frames = data / "frames/sample"
    frames.mkdir(parents=True)
    normalized = data / "normalized.mp4"
    width, height = cfg["width"], cfg["height"]
    if cfg.get("official_preprocessing", False):
        from moviepy.editor import VideoFileClip
        first_pass = data / "moviepy_24fps.mp4"
        with VideoFileClip(cfg["input"]) as original:
            clip = original.subclip(0, 16) if original.duration > 16 else original
            if clip.fps > 24:
                clip = clip.set_fps(24)
            clip.write_videofile(str(first_pass), codec="libx264", fps=24)
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(first_pass),
                        "-vf", "crop=iw:ih:((iw-576)/2):((ih-320)/2),scale=576:320",
                        "-c:v", "libx264", "-crf", "18", "-preset", "medium", str(normalized)], check=True)
    elif cfg.get("preserve_input", False):
        from .video_io import probe
        info = probe(cfg["input"])
        if (info["width"], info["height"], info["frames"], info["fps"]) != (width, height, cfg["frames"], cfg["fps"]):
            raise ValueError(f"input differs from the frozen profile: {info}")
        shutil.copyfile(cfg["input"], normalized)
    else:
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", cfg["input"], "-vf",
                        f"scale={width}:{height},fps={cfg['fps']}", "-frames:v", str(cfg["frames"]),
                        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(normalized)], check=True)
    cap = cv2.VideoCapture(str(normalized))
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if not cv2.imwrite(str(frames / f"{index}.png"), frame):
            raise RuntimeError("failed to write a source frame")
        index += 1
    cap.release()
    if index != cfg["frames"]:
        raise ValueError(f"input supplied {index} frames, expected {cfg['frames']}")
    selected = sorted(set(range(0, index, cfg["selection_stride"])) | {index - 1})
    csv_write(frames / "frames.csv", [{"frame_path": str(frames / f"{i}.png")} for i in selected], ["frame_path"])
    csv_write(data / "16x24/videos.csv", [{"path": str(normalized), "frame_save_dir": str(frames)}],
              ["path", "frame_save_dir"])
    write_json(run / "prepare.json", {"frames": index, "width": width, "height": height,
                                      "fps": cfg["fps"], "source_sha256": sha256(cfg["input"]),
                                      "normalized_sha256": sha256(normalized)})


def select(cfg, repo, run):
    import shutil
    directory = run / "data/frames/sample"
    target = directory / cfg["method"]
    if cfg["selector"] == "skem":
        selector_python = sys.executable
        if cfg.get("selector_environment") == "internvl_release":
            from .cli import settings
            selector_python = settings(repo)["internvl_python"]
        command = [selector_python, str(repo / "02_semantic_encoder/skem/MLM-keyframe-internvl.py"),
                   "--csv-path", str(run / "data/16x24/videos.csv"),
                   "--method", cfg["method"].removeprefix("key_frames"),
                   "--model_path", cfg["models"]["internvl"], "--threshold", str(cfg["threshold"]),
                   "--max-new-tokens", str(cfg["max_new_tokens"]), "--max-tiles", str(cfg["max_tiles"])]
        if cfg["internvl_8bit"]:
            command.append("--load-in-8bit")
        elif cfg.get("internvl_gpu_head", False):
            command.append("--gpu-static-head")
        elif cfg.get("internvl_static_cpu_head", False):
            command.append("--cpu-static-head")
        elif cfg.get("internvl_cpu_offload", False):
            command.append("--cpu-offload")
        if cfg.get("flash_attn", False):
            command.append("--flash-attn")
        if cfg.get("internvl_offload_layers"):
            command += ["--cpu-layers", str(cfg["internvl_offload_layers"])]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(repo / "src"), str(repo / ".local/vendor/InternVL")])
        if cfg.get("internvl_allocator"):
            env["PYTORCH_CUDA_ALLOC_CONF"] = cfg["internvl_allocator"]
        write_json(run / "selector_runtime.json", {"command": command, "pythonpath": env["PYTHONPATH"],
                    "allocator": env.get("PYTORCH_CUDA_ALLOC_CONF")})
        subprocess.run(command, cwd=run, env=env, check=True)
    else:
        target.mkdir()
        indices = sorted({round(i * (cfg["frames"] - 1) / (cfg["skim_keyframes"] - 1))
                          for i in range(cfg["skim_keyframes"])})
        for index in indices:
            shutil.copyfile(directory / f"{index}.png", target / f"{index}.png")
    indices = sorted(int(path.stem) for path in target.glob("*.png"))
    if len(indices) < 2 or indices[0] != 0 or indices[-1] != cfg["frames"] - 1:
        raise ValueError("keyframes do not cover both video endpoints")
    write_json(run / "keyframes.json", {"indices": indices, "selector": cfg["selector"],
                                        "precision": ("int8" if cfg["internvl_8bit"] else "bf16")
                                        if cfg["selector"] == "skem" else None})


def source_images(run, indices):
    from PIL import Image
    return [Image.open(run / f"data/frames/sample/{i}.png").convert("RGB") for i in indices]


def semantic_clips(run, indices, fps):
    """Use the release's MoviePy stream-copy subclips, including its timestamp behavior."""
    from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
    directory = run / "data/clips/sample"
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for segment, (start, end) in enumerate(zip(indices, indices[1:])):
        path = directory / f"{segment:05d}.mp4"
        if not path.exists():
            ffmpeg_extract_subclip(str(run / "data/normalized.mp4"), start / fps, end / fps,
                                  targetname=str(path))
        paths.append(path)
    return paths


def caption(cfg, repo, run):
    import numpy as np
    import torch
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from peft import get_peft_model, LoraConfig, TaskType
    sys.path.insert(0, str(repo / ".local/vendor/PLLaVA"))
    from models.pllava import PllavaConfig, PllavaForConditionalGeneration, PllavaProcessor

    torch.manual_seed(cfg["seed"])
    model_dir = cfg["models"]["pllava"]
    config = PllavaConfig.from_pretrained(model_dir, num_frames=4, pooling_shape=(4, 12, 12))
    config.lgvsc_attention_backend = cfg.get("caption_attention", "sdpa")
    with init_empty_weights():
        model = PllavaForConditionalGeneration(config)
        model.language_model = get_peft_model(model.language_model, LoraConfig(
            task_type=TaskType.CAUSAL_LM, target_modules=["q_proj", "v_proj"], r=128,
            lora_alpha=4, lora_dropout=0, inference_mode=True))
    index = json.loads((Path(model_dir) / "model.safetensors.index.json").read_text())
    expected, supplied = set(model.state_dict()), set(index["weight_map"])
    if expected != supplied:
        raise ValueError(f"PLLaVA checkpoint key mismatch: missing={expected-supplied}, extra={supplied-expected}")
    model = load_checkpoint_and_dispatch(model, model_dir, device_map="auto", dtype=torch.bfloat16,
                                        max_memory={0: "12GiB", "cpu": "40GiB"},
                                        no_split_module_classes=["LlamaDecoderLayer", "CLIPEncoderLayer",
                                                                 "PllavaMultiModalProjector"])
    model.eval()
    processor = PllavaProcessor.from_pretrained(model_dir)
    indices = json.loads((run / "keyframes.json").read_text())["indices"]
    clips = semantic_clips(run, indices, cfg["fps"]) if cfg.get("official_semantic_clips") else None
    rows = []
    sampling = []
    for segment, (start, end) in enumerate(zip(indices, indices[1:])):
        sampled = np.linspace(start, end, 4).round().astype(int).tolist()
        if cfg.get("paper_caption", False):
            # Upstream get_index: centered samples from the half-open semantic segment.
            size = float(max(1, end - start) - 1) / 4
            sampled = [start + int(size / 2) + int(np.round(size * i)) for i in range(4)]
        images = source_images(run, sampled)
        if clips is not None:
            sys.path.insert(0, str(repo / ".local/vendor/Open-Sora"))
            from opensora.datasets.read_video import read_video_av
            from torchvision.transforms import Resize
            from PIL import Image
            decoded, _, _ = read_video_av(str(clips[segment]), pts_unit="sec", output_format="THWC")
            size = float(len(decoded) - 1) / 4
            sampled = [int(size / 2) + int(np.round(size * i)) for i in range(4)]
            images = [Resize(672)(Image.fromarray(decoded[i].numpy())) for i in sampled]
            sampling.append({"clip": str(clips[segment]), "decoded_frames": len(decoded),
                             "indices": sampled, "resize_short_edge": 672})
        prompt = ("USER: <image>\nDescribe this video. Pay attention to the objects, their actions, "
                  "and the background. Use no more than three sentences. ASSISTANT:")
        if cfg.get("paper_caption", False):
            import ast
            tree = ast.parse((repo / ".local/vendor/Open-Sora/tools/caption/pllava_dir/caption_pllava.py").read_text())
            call = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "conv_template" for t in node.targets))
            system = ast.literal_eval(next(k.value for k in call.keywords if k.arg == "system"))
            # Exact single-turn MM_INTERLEAF Conversation.get_prompt output.
            # Avoid importing the optional video-dataset evaluation stack.
            prompt = system + " USER:<image> Describe the video in details. ASSISTANT:"
        inputs = processor(text=prompt, images=images, return_tensors="pt").to("cuda", torch.bfloat16)
        with torch.inference_mode():
            generation = {}
            if clips is not None:
                generation = dict(num_beams=1, min_length=1, top_p=0.9,
                                  repetition_penalty=1.0, length_penalty=1, temperature=1.0)
            output = model.generate(**inputs, media_type="video", do_sample=False,
                                    max_new_tokens=cfg.get("caption_max_new_tokens", cfg["max_new_tokens"]),
                                    **generation)
        text = processor.batch_decode(output, skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0].split("ASSISTANT:")[-1].strip()
        if clips is not None:
            text = text.removesuffix("</s>").strip().replace("\n", " ")
        if not text:
            raise RuntimeError("PLLaVA returned an empty caption")
        rows.append({"path": f"clips/sample/{segment:05d}.mp4", "text": text, "flow": 0.0})
    write_json(run / "captions.json", rows)
    if clips is not None:
        write_json(run / "caption_sampling.json", sampling)


def flow(cfg, repo, run):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torchvision.transforms.functional import pil_to_tensor
    sys.path.insert(0, str(repo / ".local/vendor/Open-Sora"))
    from tools.scoring.optical_flow.unimatch import UniMatch
    torch.manual_seed(cfg["seed"])
    model = UniMatch(feature_channels=128, num_scales=2, upsample_factor=4, num_head=1,
                     ffn_dim_expansion=4, num_transformer_layers=6, reg_refine=True, task="flow")
    model.load_state_dict(torch.load(repo / ".local/checkpoints/unimatch.pth", map_location="cpu")["model"])
    model = model.cuda().eval()
    indices = json.loads((run / "keyframes.json").read_text())["indices"]
    rows = json.loads((run / "captions.json").read_text())
    clips = semantic_clips(run, indices, cfg["fps"]) if cfg.get("official_semantic_clips") else None
    sampling = []
    for segment, (row, (start, end)) in enumerate(zip(rows, zip(indices, indices[1:]))):
        sampled = np.linspace(start, end, 4).round().astype(int).tolist()
        if cfg.get("paper_flow", False):
            sampled = [min(start + i, max(start, end - 1)) for i in (0, 10, 20, 30)]
        images = torch.stack([pil_to_tensor(image) for image in source_images(run, sampled)]).float().cuda()
        if clips is not None:
            from tools.datasets.utils import extract_frames
            extracted = extract_frames(str(clips[segment]), frame_inds=[0, 10, 20, 30], backend="opencv")
            images = torch.stack([pil_to_tensor(image) for image in extracted]).float().cuda()
            sampling.append({"clip": str(clips[segment]), "requested_indices": [0, 10, 20, 30]})
        images = F.interpolate(images, size=(320, 576), mode="bilinear", align_corners=True)
        # The official profile evaluates the original three-pair batch.
        scores = []
        with torch.inference_mode():
            for i in ([None] if clips is not None else range(3)):
                a, b = (images[:-1], images[1:]) if i is None else (images[i:i+1], images[i+1:i+2])
                output = model(a, b, attn_type="swin", attn_splits_list=[2, 8],
                               corr_radius_list=[-1, 4], prop_radius_list=[-1, 1], num_reg_refine=6,
                               task="flow", pred_bidir_flow=False)
                scores.append(output["flow_preds"][-1].abs().mean().item())
        row["flow"] = float(np.mean(scores))
    write_json(run / "metadata_tx.json", rows)
    if clips is not None:
        write_json(run / "flow_sampling.json", sampling)


def ntscc(cfg, repo, run):
    import logging
    import math
    import torch
    from torchvision.transforms.functional import pil_to_tensor
    from torchvision.utils import save_image
    sys.path.insert(0, str(repo / ".local/vendor/NTSCC_JSAC22"))
    from config import config
    from net.NTSCC_Hyperior import NTSCC_Hyperprior
    torch.manual_seed(cfg["seed"])
    config.device, config.logger = torch.device("cuda"), logging.getLogger("ntscc")
    config.channel["chan_param"] = cfg["snr_db"]
    config.train_lambda, config.eta = 64, 0.2
    model = NTSCC_Hyperprior(config).cuda().eval()
    checkpoint = torch.load(repo / ".local/checkpoints/ntscc_hyperprior_quality_4_psnr.pth", map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    # Upstream regenerates resolution-dependent masks; all learned weights must match.
    def generated_mask(name):
        return "attn_mask" in name or "rate_adaption.mask" in name
    state = {name: value for name, value in state.items() if not generated_mask(name)}
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys or any(not generated_mask(name) for name in incompatible.missing_keys):
        raise ValueError(f"NTSCC learned-weight mismatch: {incompatible}")
    indices = json.loads((run / "keyframes.json").read_text())["indices"]
    output_dir = run / "data/frames/sample" / (cfg["method"] + "_10")
    output_dir.mkdir()
    uses = []
    original_forward = model.channel.forward
    def measured_channel(*args, **kwargs):
        # NTSCC explicitly calls channel.forward, bypassing nn.Module forward hooks.
        output = original_forward(*args, **kwargs)
        uses.append(int(output[1]))
        return output
    model.channel.forward = measured_channel
    rows = []
    with torch.inference_mode():
        for index, image in zip(indices, source_images(run, indices)):
            tensor = pil_to_tensor(image).float().unsqueeze(0).cuda() / 255.0
            result = model(tensor)
            reconstruction = result[-1].clamp(0, 1)
            mse = float((tensor - reconstruction).square().mean())
            save_image(reconstruction, output_dir / f"{index}.png")
            rows.append({"index": index, "mse": mse, "psnr_db": -10 * math.log10(max(mse, 1e-12)),
                         "complex_channel_uses": uses[-1]})
    model.channel.forward = original_forward
    write_json(run / "ntscc.json", {"keyframes": rows, "snr_db": cfg["snr_db"],
                                    "visual_complex_channel_uses": sum(uses),
                                    "rate_index_transport": "upstream_shared_metadata_not_serialized",
                                    "complete_wire_accounting": False})


def decode_video(cfg, repo, run):
    from .decoder_runner import run as run_decoder
    config = run / "decoder_config.py"
    template = (repo / "configs/rtx4080_opensora.py").read_text()
    template += f"\nimage_size = ({cfg['height']}, {cfg['width']})\n"
    template += f"seed = {cfg['seed']}\nfps = {cfg['fps']}\nsave_fps = {cfg['fps']}\n"
    template += f"scheduler['num_sampling_steps'] = {cfg['steps']}\n"
    for key, field in (("stdit", "model"), ("vae", "vae"), ("t5", "text_encoder")):
        template += f"{field}['from_pretrained'] = {cfg['models'][key]!r}\n"
    template += f"vae['vae_2d_path'] = {cfg['models']['vae2d']!r}\n"
    config.write_text(template)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo / ".local/vendor/Open-Sora") + os.pathsep + environment.get("PYTHONPATH", "")
    run_decoder(run / "metadata_rx.csv", run / "reconstruction", cfg["method"] + "_10", run / "data",
                decoder=repo / "04_semantic_decoder/scripts/mydemo_new_align_sh.py",
                config=config, environment=environment)


def evaluate(cfg, repo, run):
    import cv2
    import math
    import numpy as np
    from skimage.metrics import structural_similarity
    files = list((run / "reconstruction").glob("*.mp4"))
    if len(files) != 1:
        raise ValueError("expected exactly one reconstructed video")
    def read(path):
        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return np.stack(frames)
    source, generated = read(run / "data/normalized.mp4"), read(files[0])
    if source.shape != generated.shape:
        raise ValueError(f"source/generated video shapes differ: {source.shape} vs {generated.shape}")
    mse = float(np.square(source.astype(np.float64) - generated.astype(np.float64)).mean())
    ssim = float(np.mean([structural_similarity(a, b, channel_axis=-1, data_range=255)
                         for a, b in zip(source, generated)]))
    write_json(run / "quality.json", {"frames": len(generated), "shape": list(generated.shape),
                                      "psnr_db": 10 * math.log10(255**2 / max(mse, 1e-12)),
                                      "ssim": ssim, "video_sha256": sha256(files[0]),
                                      "evidence_scope": "single_clip_local_execution_not_paper_reproduction"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "select", "caption", "flow", "ntscc", "decode", "evaluate"])
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    repo, run = Path(__file__).resolve().parents[2], args.run_dir.resolve()
    cfg = json.loads((run / "run_config.json").read_text())
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    functions = {"prepare": prepare, "select": select, "caption": caption, "flow": flow,
                 "ntscc": ntscc, "decode": decode_video, "evaluate": evaluate}
    functions[args.stage](cfg, repo, run)


if __name__ == "__main__":
    main()
