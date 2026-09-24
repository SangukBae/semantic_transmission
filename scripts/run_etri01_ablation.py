"""Preserved-input ETRI01 decoder ablations; explicit diagnostic/oracle provenance."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.codec_transport import decoder_config_text
from semantic_transmission.decoder_runner import run as decode

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "outputs/etri01_official_20260911_v2/01_person_walk"
ROOT = REPO / "outputs/diagnostics/etri01_ablation_20260917"
CORE = Path.home() / "anaconda3/envs/lgvsc/bin/python"
CHANNEL = Path.home() / "anaconda3/envs/lgvsc-channel/bin/python"
CASES = ["replay", "aligned", "clean_keys", "action_caption", "dense_2s", "dense_1s", "combined"]


def read(path):
    return json.loads(path.read_text())


def action_caption(a, b):
    # Manually authored after inspecting source frames; NOT an automatic captioner.
    setting = "A fixed camera shows one woman in a dark blue coat and dark trousers on a sidewalk in front of a beige brick wall and green bushes. "
    if b < 168:
        action = "She walks steadily from the left side of the image toward the right, facing right and taking alternating steps."
    elif a < 179:
        action = "She walks from left to right. Near the end she slows down on the right side of the image and turns around, briefly facing the camera."
    elif b <= 199:
        action = "On the right side of the image, she finishes turning to face left, pauses briefly, then begins walking toward the left."
    else:
        action = "She faces left and walks steadily from the right side of the image toward the left, returning along the sidewalk with alternating steps."
    if a == 179 and b == 239:
        action = "Starting on the right side of the image, she finishes turning to face left, briefly pauses, then walks back toward the left along the sidewalk."
    return setting + action + " The camera and background remain stationary."


def dense_indices(gap):
    result = [0]
    for a, b in [(0, 179), (179, 239)]:
        n = math.ceil((b - a) / gap)
        result.extend(a + round((b - a) * j / n) for j in range(1, n + 1))
    return result


def run_command(command, log, env):
    with log.open("x") as stream:
        subprocess.run(list(map(str, command)), stdout=stream, stderr=subprocess.STDOUT,
                       check=True, env=env, cwd=REPO)


def preserve_original_transport(run, *, received=False):
    """Add keyframes without perturbing the three original JSCC conditions."""
    import numpy as np
    from semantic_transmission.wire import pack, unpack
    from semantic_transmission.transmission_accounting import packet_breakdown
    header, payload = unpack((run / "transmitter/metadata.bin").read_bytes())
    old, old_payload = unpack((BASE / "transmitter/metadata.bin").read_bytes())
    old_items = {item["index"]: item for item in old["keyframes"]}
    if received:
        original_values = np.fromfile(BASE / "received/visual.c64", dtype="<c8")
        values = np.fromfile(run / "transmitter/visual.c64", dtype="<c8")
        sigma = math.sqrt(1 / (2 * 10 ** (10 / 10)))
        for item in header["keyframes"]:
            start, count = item["complex_offset"], item["complex_count"]
            if item["index"] in old_items:
                previous = old_items[item["index"]]
                s = previous["complex_offset"]
                values[start:start+count] = original_values[s:s+count]
            else:
                rng = np.random.default_rng(1024 + 100000 + item["index"])
                noise = (rng.normal(0, sigma, count) + 1j * rng.normal(0, sigma, count)).astype("<c8")
                values[start:start+count] += noise
        values.tofile(run / "received/visual.c64")
        report = read(run / "channel_accounting.json")
        report["received_files"]["visual.c64"]["sha256"] = sha256(run / "received/visual.c64")
        report["visual_awgn_rng"] = "replay original official CUDA noise; added frames independent numpy_PCG64 seed=101024+index"
        write_json(run / "channel_accounting.json", report)
        return
    values = np.fromfile(run / "transmitter/visual.c64", dtype="<c8")
    original_values = np.fromfile(BASE / "transmitter/visual.c64", dtype="<c8")
    new_payload, chunks, items = bytearray(), [], []
    offset = 0
    for current in header["keyframes"]:
        original = current["index"] in old_items
        item = dict(old_items[current["index"]] if original else current)
        stream, rates = (original_values, old_payload) if original else (values, payload)
        s, n, r, rn = (item[k] for k in ("complex_offset", "complex_count", "rate_offset", "rate_bytes"))
        chunks.append(stream[s:s+n])
        item.update(complex_offset=offset, rate_offset=len(new_payload))
        new_payload.extend(rates[r:r+rn])
        items.append(item)
        offset += n
    header["keyframes"] = items
    packet = pack(header, new_payload)
    (run / "transmitter/metadata.bin").write_bytes(packet)
    np.concatenate(chunks).tofile(run / "transmitter/visual.c64")
    report = read(run / "sender_accounting.json")
    report.update(visual_complex_channel_uses=offset, visual_real_values=offset*2,
                  visual_fp32_iq_serialization_bytes=offset*8, metadata_packet_bytes=len(packet),
                  packed_rate_index_bytes=len(new_payload), header_json_and_framing_bytes=len(packet)-len(new_payload),
                  serialized_model_input_bytes=offset*8+len(packet), metadata_breakdown=packet_breakdown(packet),
                  original_keyframe_transport_replayed=True)
    report["transmitter_files"] = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                                   for p in (run / "transmitter").iterdir()}
    write_json(run / "sender_accounting.json", report)


def prepare(name, env):
    run = ROOT / name
    run.mkdir(exist_ok=False)
    (run / "data").symlink_to(BASE / "data", target_is_directory=True)
    cfg = read(BASE / "run_config.json")
    cfg["input"] = str(REPO / "data/etri_video_eval/processed/01_person_walk.mp4")
    cfg["concatenation_policy"] = "official_release" if name == "replay" else "endpoint_exact"
    write_json(run / "run_config.json", cfg)
    dense = name in {"dense_2s", "dense_1s", "combined"}
    indices = dense_indices(48 if name == "dense_2s" else 24) if dense else [0, 179, 239]
    write_json(run / "keyframes.json", {"indices": indices, "selector": "diagnostic_max_gap_preserve_original" if dense else "reused_SKEM"})
    original = read(BASE / "metadata_tx.json")
    rows = []
    for i, (a, b) in enumerate(zip(indices, indices[1:])):
        row = dict(original[0 if a < 179 else 1])
        row["path"] = f"clips/sample/{i:05d}.mp4"
        if name in {"action_caption", "combined"}:
            row["text"] = action_caption(a, b)
        rows.append(row)
    write_json(run / "metadata_tx.json", rows)
    provenance = {"name": name, "baseline": str(BASE), "indices": indices,
        "alignment": "official_release" if name == "replay" else "endpoint_exact",
        "manual_caption_oracle": name in {"action_caption", "combined"},
        "clean_keyframe_oracle": name == "clean_keys", "training": False,
        "seed": cfg["seed"], "sampling_steps": cfg["steps"],
        "flow": "original received scalar repeated for subdivisions; no flow re-estimation",
        "baseline_video_sha256": sha256(BASE / "receiver/reconstruction/sample_0000.mp4"),
        "source_sha256": sha256(BASE / "data/normalized.mp4")}
    if dense or name == "action_caption":
        if dense:
            run_command([CORE, "-m", "semantic_transmission.codec_transport", "send", run], run / "send.log", env)
            preserve_original_transport(run)
        else:
            from semantic_transmission.wire import pack, unpack
            (run / "transmitter").mkdir()
            header, payload = unpack((BASE / "transmitter/metadata.bin").read_bytes())
            header["segments"] = rows
            header["decoder"]["concatenation_policy"] = "endpoint_exact"
            (run / "transmitter/metadata.bin").write_bytes(pack(header, payload))
            (run / "transmitter/visual.c64").symlink_to(BASE / "transmitter/visual.c64")
        # Channel includes the full framed metadata, rate indices, LDPC and AWGN.
        run_command([CHANNEL, "-m", "semantic_transmission.codec_transport", "channel", run], run / "channel.log", env)
        if dense:
            preserve_original_transport(run, received=True)
        if not dense:
            # Preserve the original received visual stream exactly for a caption-only ablation.
            shutil.copyfile(BASE / "received/visual.c64", run / "received/visual.c64")
            account = read(run / "channel_accounting.json")
            account["received_files"]["visual.c64"]["sha256"] = sha256(run / "received/visual.c64")
            account["visual_replay"] = str(BASE / "received/visual.c64")
            write_json(run / "channel_accounting.json", account)
        run_command([CORE, "-m", "semantic_transmission.codec_transport", "receive", run], run / "receive.log", env)
        # Avoid visual-codec numerical drift across computers on the three controls.
        for index in [0, 179, 239]:
            shutil.copyfile(BASE / f"receiver/frames/sample/key_frames_received/{index}.png",
                            run / f"receiver/frames/sample/key_frames_received/{index}.png")
        provenance["visual_channel_note"] = ("original three transmitted and received symbol blocks and PNGs replayed; added frames independently transmitted" if dense
            else "original visual stream and PNGs replayed exactly; caption metadata retransmitted")
    else:
        shutil.copytree(BASE / "receiver/frames", run / "receiver/frames")
        shutil.copyfile(BASE / "receiver/metadata.csv", run / "receiver/metadata.csv")
        inputs = read(BASE / "receiver/decoder_inputs.json")
        inputs["decoder"]["concatenation_policy"] = cfg["concatenation_policy"]
        write_json(run / "receiver/decoder_inputs.json", inputs)
        shutil.copyfile(BASE / "channel_accounting.json", run / "channel_accounting.json")
        if name == "clean_keys":
            for index in indices:
                shutil.copyfile(BASE / f"data/frames/sample/{index}.png",
                                run / f"receiver/frames/sample/key_frames_received/{index}.png")
            provenance["transmission_claim"] = "none: clean keyframes bypass codec/channel; copied ledger is baseline only"
    inputs = read(run / "receiver/decoder_inputs.json")
    config = decoder_config_text(cfg, REPO, inputs)
    config += f"\nconditioning_alignment={'official_release' if name == 'replay' else 'endpoint_exact'!r}\n"
    if dense:
        config += "cache_text_embeddings=True\n"
    (run / "receiver/decoder_config.py").write_text(config)
    provenance["inputs"] = {str(p.relative_to(run)): sha256(p) for p in sorted((run / "receiver").rglob("*")) if p.is_file()}
    provenance["decoder_sha256"] = sha256(REPO / "04_semantic_decoder/scripts/mydemo_new_align_sh.py")
    write_json(run / "experiment.json", provenance)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    env = os.environ.copy()
    env.update(PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="8", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env.pop("LD_LIBRARY_PATH", None)
    env.pop("PYTHONPATH", None)
    for name in args.cases:
        run = ROOT / name
        if args.resume and (run / "complete.json").exists():
            print(f"SKIP completed {name}", flush=True)
            continue
        print(f"START {name}", flush=True)
        start = time.time()
        run = prepare(name, env)
        decode_env = dict(env, PYTHONPATH=str(REPO / ".local/vendor/Open-Sora"))
        decode(run / "receiver/metadata.csv", run / "receiver/reconstruction", "key_frames_received",
               run / "receiver", decoder=REPO / "04_semantic_decoder/scripts/mydemo_new_align_sh.py",
               config=run / "receiver/decoder_config.py", python=CORE, environment=decode_env)
        write_json(run / "complete.json", {"status": "PASSED", "seconds": time.time() - start})
        print(f"DONE {name} {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
