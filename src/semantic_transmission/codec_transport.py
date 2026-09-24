"""Explicit sender/channel/receiver boundaries for pretrained NTSCC.

Continuous complex channel symbols are stored as complex64 for replay. Their file
bytes are never described as RF bits. Every sample-dependent digital decoder input
is carried by the framed metadata packet through the simulated digital channel.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys

from .artifacts import sha256, write_json
from .wire import pack, unpack, pack_indices, unpack_indices
from .temporal import resolve_concatenation_policy
from .transmission_accounting import packet_breakdown, channel_breakdown
from .validation_progress import emit as validation_progress


def load_codec(repo):
    import logging
    import torch
    sys.path.insert(0, str(repo / ".local/vendor/NTSCC_JSAC22"))
    from config import config
    from net.NTSCC_Hyperior import NTSCC_Hyperprior
    config.device, config.logger = torch.device("cuda"), logging.getLogger("ntscc")
    if config.use_side_info:
        raise ValueError("hyperprior refinement requires an additional transport schema")
    config.train_lambda, config.eta = 64, 0.2
    model = NTSCC_Hyperprior(config).cuda().eval()
    path = repo / ".local/checkpoints/ntscc_hyperprior_quality_4_psnr.pth"
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    generated = lambda name: "attn_mask" in name or "rate_adaption.mask" in name
    result = model.load_state_dict({k: v for k, v in state.items() if not generated(k)}, strict=False)
    if result.unexpected_keys or any(not generated(k) for k in result.missing_keys):
        raise ValueError(f"NTSCC learned-weight mismatch: {result}")
    return model, path


def encode_symbols(model, tensor):
    import torch
    model.update_resolution(*tensor.shape[-2:])
    result = model.forward_NTC(tensor, require_probs=True)
    y, scales, means = result[4], result[6], result[7]
    probabilities = model.feature_probs_based_Gaussian(y, means, scales)
    symbols, mask, indices = model.fe(y, probabilities.detach(), eta=model.eta)
    values = torch.masked_select(symbols, mask.bool())
    if values.numel() % 2:
        raise ValueError("NTSCC produced an odd number of real channel values")
    power = values.square().mean()
    if not torch.isfinite(values).all() or not math.isfinite(float(power)) or float(power) <= 0:
        raise ValueError("invalid NTSCC channel input")
    normalized = values / (power * 2).sqrt()
    complex_values = torch.complex(normalized[::2], normalized[1::2])
    return complex_values, indices, float(power)


def decode_symbols(model, complex_values, indices, power, height, width):
    import torch
    model.update_resolution(height, width)
    h, w = height // 16, width // 16
    if indices.numel() != h * w or bool(((indices < 0) | (indices >= 16)).any()):
        raise ValueError("invalid received NTSCC rate indices")
    rates = model.fd.rate_choice_tensor[indices].reshape(h, w)
    channels = int(model.fd.rate_choice_tensor.max())
    mask = (torch.arange(channels, device="cuda")[:, None, None] < rates[None]).unsqueeze(0)
    if int(mask.sum()) != complex_values.numel() * 2:
        raise ValueError("received visual length disagrees with decoded rate indices")
    values = torch.stack([complex_values.real, complex_values.imag], -1).flatten()
    values = values * torch.tensor(power * 2, dtype=torch.float32, device="cuda").sqrt()
    received = torch.zeros((1, channels, h, w), device="cuda")
    received[mask] = values
    return model.gs(model.fd(received, indices)).clamp(0, 1)


def send(cfg, repo, run):
    import numpy as np
    import torch
    from torchvision.transforms.functional import pil_to_tensor
    from .workers import source_images
    torch.manual_seed(cfg["seed"])
    model, checkpoint = load_codec(repo)
    indices = json.loads((run / "keyframes.json").read_text())["indices"]
    segments = json.loads((run / "metadata_tx.json").read_text())
    if len(segments) != len(indices) - 1 or indices[0] != 0 or indices[-1] != cfg["frames"] - 1:
        raise ValueError("semantic segment coverage mismatch")
    header = {"schema": "lgvsc_model_inputs_v1", "video": {k: cfg[k] for k in ("width", "height", "frames", "fps")},
              "decoder": {k: cfg[k] for k in ("seed", "steps")},
              "checkpoint_sha256": sha256(checkpoint), "segments": segments,
              "complex_dtype": "little_endian_complex64", "keyframes": []}
    header["decoder"]["policy"] = cfg.get("decoder_policy", "endpoint_exact")
    if "concatenation_policy" in cfg:
        header["decoder"]["concatenation_policy"] = resolve_concatenation_policy(
            header["decoder"]["policy"], cfg["concatenation_policy"])
    output = run / "transmitter"
    output.mkdir()
    payload = bytearray()
    offset = 0
    with (output / "visual.c64").open("xb") as stream, torch.inference_mode():
        for index, image in zip(indices, source_images(run, indices)):
            tensor = pil_to_tensor(image).float().unsqueeze(0).cuda() / 255
            values, rates, power = encode_symbols(model, tensor)
            raw_rates = pack_indices(rates.cpu().tolist())
            header["keyframes"].append({"index": index, "complex_offset": offset,
                "complex_count": values.numel(), "average_power": power,
                "rate_offset": len(payload), "rate_bytes": len(raw_rates), "rate_count": rates.numel()})
            payload.extend(raw_rates)
            stream.write(values.cpu().numpy().astype("<c8").tobytes())
            offset += values.numel()
            validation_progress(len(header["keyframes"]), len(indices))
    packet = pack(header, payload)
    (output / "metadata.bin").write_bytes(packet)
    write_json(run / "sender_accounting.json", {"visual_complex_channel_uses": offset,
        "visual_real_values": offset * 2, "visual_fp32_iq_serialization_bytes": offset * 8,
        "metadata_packet_bytes": len(packet), "packed_rate_index_bytes": len(payload),
        "header_json_and_framing_bytes": len(packet) - len(payload),
        "caption_utf8_bytes": sum(len(s["text"].encode("utf-8")) for s in segments),
        "serialized_model_input_bytes": offset * 8 + len(packet),
        "metadata_breakdown": packet_breakdown(packet),
        "visual_rf_bits": None, "visual_rf_bits_reason": "continuous_amplitude_JSCC_symbols",
        "transmitter_files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in output.iterdir()}})


def channel(cfg, repo, run):
    import numpy as np
    import subprocess
    from .metadata_channel import transmit
    packet = (run / "transmitter/metadata.bin").read_bytes()
    restored, report = transmit(packet, cfg["snr_db"], cfg.get("channel_seed", cfg["seed"]))
    try:
        header, payload = unpack(restored)
    except (ValueError, UnicodeError) as error:
        report.update(status="FAILED", error=str(error))
        write_json(run / "channel_accounting.json", report)
        raise
    output = run / "received"
    output.mkdir()
    (output / "metadata.bin").write_bytes(restored)
    sent = np.fromfile(run / "transmitter/visual.c64", dtype="<c8")
    expected = sum(k["complex_count"] for k in header["keyframes"])
    if len(sent) != expected or not np.isfinite(sent).all():
        raise ValueError("visual transport stream length/nonfinite error")
    # Same unit-power complex AWGN law as the official NTSCC channel.
    rng_name = "numpy_PCG64"
    if cfg.get("visual_channel") == "official_torch_cuda":
        from .cli import settings
        env = os.environ.copy()
        env.pop("CUDA_VISIBLE_DEVICES", None)
        subprocess.run([settings(repo)["python"], "-m", "semantic_transmission.codec_transport",
                        "visual_channel", str(run)], env=env, check=True)
        rng_name = "official_NTSCC_Channel_gaussian_noise_layer_torch_cuda"
    else:
        rng = np.random.default_rng(cfg["seed"])
        sigma = math.sqrt(1 / (2 * 10 ** (cfg["snr_db"] / 10)))
        noise = (rng.normal(0, sigma, len(sent)) + 1j * rng.normal(0, sigma, len(sent))).astype("<c8")
        (sent + noise).astype("<c8").tofile(output / "visual.c64")
    digital_uses = report["complex_channel_uses"]
    pixels = 3 * cfg["width"] * cfg["height"] * cfg["frames"]
    report.update(status="PASSED", metadata_exact_match=restored == packet,
        visual_complex_channel_uses=len(sent), digital_complex_channel_uses=digital_uses,
        total_complex_channel_uses=len(sent) + digital_uses,
        cbr_complex_uses_per_source_scalar=(len(sent) + digital_uses) / pixels,
        complete_sample_dependent_model_input_accounting=True,
        physical_link_overhead_included=False,
        shared_prior="installed model checkpoints, code and fixed architecture/configuration",
        visual_awgn_rng=rng_name, channel_seed=cfg.get("channel_seed", cfg["seed"]),
        visual_tx_mean_power=float(np.mean(np.abs(sent) ** 2)),
        transmission_breakdown=channel_breakdown(packet, len(sent), header["video"], report),
        received_files={p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in output.iterdir()})
    write_json(run / "channel_accounting.json", report)


def visual_channel(cfg, repo, run):
    """Apply the released GPU AWGN implementation to each transmitted keyframe."""
    import numpy as np
    import torch
    from types import SimpleNamespace
    sys.path.insert(0, str(repo / ".local/vendor/NTSCC_JSAC22"))
    from channel.channel import Channel
    torch.manual_seed(cfg.get("channel_seed", cfg["seed"]))
    header, _ = unpack((run / "received/metadata.bin").read_bytes())
    sent = np.fromfile(run / "transmitter/visual.c64", dtype="<c8")
    channel_model = Channel(SimpleNamespace(channel={"type": "awgn", "chan_param": cfg["snr_db"]},
                                            device=torch.device("cuda"), logger=None))
    with (run / "received/visual.c64").open("xb") as stream, torch.inference_mode():
        for frame_number, item in enumerate(header["keyframes"]):
            offset, count = item["complex_offset"], item["complex_count"]
            symbols = torch.from_numpy(sent[offset:offset + count].copy()).cuda()
            channel_model.channel_forward(symbols).cpu().numpy().astype("<c8").tofile(stream)
            validation_progress(frame_number + 1, len(header["keyframes"]))


def receive(cfg, repo, run):
    import numpy as np
    import torch
    from torchvision.utils import save_image
    from .workers import csv_write
    # No source frames, captions.json, keyframes.json or encoder tensors are read here.
    header, payload = unpack((run / "received/metadata.bin").read_bytes())
    values = np.fromfile(run / "received/visual.c64", dtype="<c8")
    if header["schema"] != "lgvsc_model_inputs_v1" or not np.isfinite(values).all():
        raise ValueError("invalid receiver input")
    model, checkpoint = load_codec(repo)
    if sha256(checkpoint) != header["checkpoint_sha256"]:
        raise ValueError("receiver checkpoint does not match negotiated codec")
    output = run / "receiver"
    frames = output / "frames/sample/key_frames_received"
    frames.mkdir(parents=True)
    offset = rate_offset = 0
    with torch.inference_mode():
        for item in header["keyframes"]:
            if item["complex_offset"] != offset or item["rate_offset"] != rate_offset:
                raise ValueError("noncontiguous receiver payload")
            count = item["complex_count"]
            rates = unpack_indices(payload[rate_offset:rate_offset + item["rate_bytes"]], item["rate_count"])
            restored = decode_symbols(model, torch.from_numpy(values[offset:offset + count].copy()).cuda(),
                                      torch.tensor(rates, device="cuda"), item["average_power"],
                                      header["video"]["height"], header["video"]["width"])
            save_image(restored, frames / f"{item['index']}.png")
            offset += count
            rate_offset += item["rate_bytes"]
    if offset != len(values) or rate_offset != len(payload):
        raise ValueError("unconsumed receiver payload")
    csv_write(output / "metadata.csv", header["segments"], ["path", "text", "flow"])
    write_json(output / "decoder_inputs.json", {"video": header["video"], "decoder": header["decoder"],
                                               "indices": [k["index"] for k in header["keyframes"]]})
    write_json(run / "receiver_accounting.json", {"status": "PASSED", "source_pixels_read": False,
        "received_keyframes": len(header["keyframes"]),
        "keyframe_png_bytes": sum(p.stat().st_size for p in frames.glob("*.png")),
        "caption_flow_csv_bytes": (output / "metadata.csv").stat().st_size,
        "note": "decoded conditioning files are receiver products, not transmitter traffic"})


def decoder_config_text(cfg, repo, inputs):
    """Select the model recipe independently of final concatenation."""
    video, decoder = inputs["video"], inputs["decoder"]
    policy = decoder.get("policy", "endpoint_exact")
    templates = {"official_release": "official_opensora.py", "endpoint_exact": "rtx4080_opensora.py"}
    if policy not in templates:
        raise ValueError(f"unknown received decoder policy: {policy}")
    template = (repo / "configs" / templates[policy]).read_text()
    template += f"\nimage_size=({video['height']},{video['width']})\nfps={video['fps']}\nsave_fps={video['fps']}\n"
    template += f"seed={decoder['seed']}\nscheduler['num_sampling_steps']={decoder['steps']}\nsave_frames=True\nverbose=2\n"
    template += f"model['enable_flash_attn']={cfg.get('flash_attn', False)!r}\n"
    concatenation = resolve_concatenation_policy(policy, decoder.get("concatenation_policy"))
    template += f"concatenation_policy={concatenation!r}\n"
    for key, field in (("stdit", "model"), ("vae", "vae"), ("t5", "text_encoder")):
        template += f"{field}['from_pretrained']={cfg['models'][key]!r}\n"
    template += f"vae['vae_2d_path']={cfg['models']['vae2d']!r}\n"
    return template


def reconstruct(cfg, repo, run):
    from .decoder_runner import run as run_decoder
    inputs = json.loads((run / "receiver/decoder_inputs.json").read_text())
    template = decoder_config_text(cfg, repo, inputs)
    path = run / "receiver/decoder_config.py"
    path.write_text(template)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo / ".local/vendor/Open-Sora")
    run_decoder(run / "receiver/metadata.csv", run / "receiver/reconstruction", "key_frames_received",
                run / "receiver", decoder=repo / "04_semantic_decoder/scripts/mydemo_new_align_sh.py",
                config=path, environment=env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["send", "channel", "visual_channel", "receive", "reconstruct"])
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    cfg = json.loads((run / "run_config.json").read_text())
    globals()[args.stage](cfg, Path(__file__).resolve().parents[2], run)


if __name__ == "__main__":
    main()
