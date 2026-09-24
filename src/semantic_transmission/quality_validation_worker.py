"""Isolated real workers for the three-stage restoration validation campaign."""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from .artifacts import sha256, write_json
from .webvid5 import read_json
from .validation_progress import emit as progress


def load_view(run):
    from .research_quality import read_frames
    from .temporal import unique_output_positions, output_source_indices
    inputs = read_json(run / "receiver/decoder_inputs.json")
    policy = inputs["decoder"].get("concatenation_policy", inputs["decoder"].get("policy", "endpoint_exact"))
    output = read_frames(run / "receiver/reconstruction/sample_0000_frames")
    if len(output) != len(output_source_indices(inputs["indices"], policy)):
        raise ValueError("reconstruction does not match declared time axis")
    return output[unique_output_positions(inputs["indices"], policy)], inputs


def keys_from_receiver(run, indices):
    import numpy as np
    from PIL import Image
    folder = run / "receiver/frames/sample/key_frames_received"
    return {i: np.array(Image.open(folder / f"{i}.png").convert("RGB")) for i in indices}


def save_video(frames, folder, fps):
    from PIL import Image
    folder.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(folder / f"{i:05d}.png")
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-framerate", str(fps),
                    "-i", str(folder / "%05d.png"), "-an", "-c:v", "libx264", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-threads", "2", str(folder.parent / "video.mp4")], check=True)


def evaluate(source, output, indices, destination, pixel_only=False, callback=progress):
    import numpy as np
    from .quality_methods import temporal_error
    from .official_quality import OfficialMetrics, pixel_scores
    if source.shape != output.shape:
        raise ValueError(f"unaligned metric inputs: {source.shape} != {output.shape}")
    if pixel_only:
        rows = [dict(frame=i, **pixel_scores(a, b)) for i, (a,b) in enumerate(zip(source, output))]
    else:
        _, rows = OfficialMetrics().evaluate(source, output, progress=callback)
    fields = list(rows[0])
    def mean(selected):
        return {k: float(np.mean([r[k] for r in selected])) for k in fields if k != "frame"} if selected else None
    keyset = set(indices)
    interior = [r for r in rows if r["frame"] not in keyset]
    result = {"all": mean(rows), "anchors": mean([r for r in rows if r["frame"] in keyset]),
              "interior": mean(interior), "interior_frames": len(interior),
              "temporal_error": temporal_error(source, output), "frames": len(rows),
              "metric_profile": "PIXEL_ONLY_TEST" if pixel_only else "lgvsc_official_metrics_v1",
              "hallucination_review": "PENDING"}
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "metrics.json", result)
    with (destination / "per_frame.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    return result


def correction(spec, repo):
    from .research_quality import read_video
    from .quality_methods import anchor_correction, choose_strength
    run, dest = Path(spec["base"]), Path(spec["dest"])
    values, inputs = load_view(run)
    source = read_video(run / "data/normalized.mp4")  # evaluation only
    keys = keys_from_receiver(run, inputs["indices"])
    strengths = spec.get("strengths")
    if strengths is None:
        strengths = sorted({0.0, read_json(Path(spec["frozen"]))["anchor_strength"]})
    results = {}
    for index, strength in enumerate(strengths):
        def cb(done, total):
            progress(index + .4 * done/total, len(strengths))
        corrected = anchor_correction(values, keys, strength, cb)
        case = dest / str(float(strength))
        save_video(corrected, case / "frames", inputs["video"]["fps"])
        results[str(float(strength))] = evaluate(source, corrected, inputs["indices"], case,
            spec.get("pixel_only", False), lambda d,t: progress(index+.4+.6*d/t, len(strengths)))
        progress(index+1, len(strengths))
    write_json(dest / "results.json", results)
    if spec.get("freeze"):
        selected = choose_strength(results)
        write_json(dest / "frozen.json", dict(anchor_strength=selected,
            lowres_strength=spec["lowres_strength"], tuned_only_on=str(run),
            selection="minimum interior LPIPS with PSNR/temporal guards; includes no-op",
            correction_finding="NO_IMPROVEMENT_SELECTED" if selected == 0 else "DEVELOPMENT_GAIN_ONLY",
            independent_confirmation_required=True))


def make_bank(spec, repo):
    from .quality_methods import rank_insertions
    from .research_quality import read_video
    from .ablation_transport import subdivide_metadata
    base, run = Path(spec["base"]), Path(spec["run"])
    run.mkdir(parents=True)
    cfg = read_json(base / "run_config.json")
    cfg["models"] = read_json(repo / ".local/model_paths.json")
    indices = read_json(base / "keyframes.json")["indices"]
    source = read_video(base / "data/normalized.mp4")
    count = min(len(source)-len(indices), max(12, len(indices)*3))
    adaptive = rank_insertions(source, indices, count, True)
    progress(1, 2)
    uniform = rank_insertions(source, indices, count, False)
    union = sorted(set(indices + adaptive + uniform))
    (run / "data").symlink_to((base / "data").resolve(), target_is_directory=True)
    write_json(run / "run_config.json", cfg)
    write_json(run / "keyframes.json", {"indices": union})
    write_json(run / "metadata_tx.json", subdivide_metadata(indices, union, read_json(base / "metadata_tx.json")))
    write_json(run / "rankings.json", {"original": indices, "adaptive": adaptive, "uniform": uniform})
    progress(2, 2)


def encode_side(spec, repo):
    from .wire import pack
    base, dest = Path(spec["base"]), Path(spec["dest"])
    dest.mkdir(parents=True)
    cfg = read_json(base / "run_config.json")
    size = spec["lowres_size"]
    mp4 = dest / "sender.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(base / "data/normalized.mp4"),
                    "-vf", f"scale={size[0]}:{size[1]}:flags=area", "-an", "-c:v", "libx264",
                    "-crf", str(spec["lowres_crf"]), "-pix_fmt", "yuv420p", "-threads", "2", str(mp4)], check=True)
    header = dict(schema="lgvsc_lowres_v1", frames=cfg["frames"], fps=cfg["fps"],
                  width=size[0], height=size[1], codec="h264_mp4")
    (dest / "packet.bin").write_bytes(pack(header, mp4.read_bytes()))
    write_json(dest / "encoding.json", {"header": header, "payload_bytes": mp4.stat().st_size,
                "packet_sha256": sha256(dest / "packet.bin")})


def side_channel(spec, repo):
    from .metadata_channel import transmit
    from .wire import unpack
    src, dest = Path(spec["encoded"]), Path(spec["dest"])
    dest.mkdir(parents=True)
    packet = (src / "packet.bin").read_bytes()
    received, ledger = transmit(packet, spec["snr_db"], spec["channel_seed"])
    if received != packet or ledger["bit_errors"]:
        raise ValueError("low-resolution payload failed channel integrity")
    header, payload = unpack(received)
    (dest / "received.mp4").write_bytes(payload)
    from .video_io import probe
    info = probe(dest / "received.mp4")
    if any(info[k] != header[k] for k in ("frames", "fps", "width", "height")):
        raise ValueError("low-resolution receiver time axis mismatch")
    write_json(dest / "channel.json", dict(ledger, header=header, status="PASSED",
        received_sha256=sha256(dest / "received.mp4"), physical_link_headers_included=False))


def subset_wire(header, payload, symbols, indices, original, rows):
    import numpy as np
    from .ablation_transport import subdivide_metadata
    from .wire import pack
    by_index = {k["index"]: k for k in header["keyframes"]}
    result = copy.deepcopy(header)
    result["segments"] = subdivide_metadata(original, sorted(indices), rows)
    result["keyframes"] = []
    packed, chunks, offset = bytearray(), [], 0
    for i in sorted(indices):
        item = dict(by_index[i])
        start, count = item["complex_offset"], item["complex_count"]
        r, n = item["rate_offset"], item["rate_bytes"]
        chunks.append(symbols[start:start+count])
        item.update(complex_offset=offset, rate_offset=len(packed))
        packed.extend(payload[r:r+n]); offset += count
        result["keyframes"].append(item)
    return pack(result, packed), np.concatenate(chunks)


def choose_budget(original, ranking, build, budget, side_uses):
    from .packets import accounting
    selected = sorted(original)
    def cost(keys):
        packet, symbols = build(keys)
        return len(symbols) + accounting(len(packet))["complex_channel_uses"] + side_uses
    if cost(selected) > budget:
        raise ValueError("fixed budget cannot carry original keys plus side information; use a new protocol with a larger budget")
    for candidate in ranking:
        proposed = sorted(set(selected + [candidate]))
        if cost(proposed) <= budget:
            selected = proposed
    return selected, cost(selected)


def plan_wires(spec, repo):
    import numpy as np
    from .wire import unpack
    from .packets import accounting
    base, bank, dest = (Path(spec[k]) for k in ("base", "bank", "dest"))
    dest.mkdir(parents=True)
    cfg = read_json(base / "run_config.json")
    cfg["models"] = read_json(repo / ".local/model_paths.json")
    header, payload = unpack((bank / "transmitter/metadata.bin").read_bytes())
    symbols = np.fromfile(bank / "transmitter/visual.c64", dtype="<c8")
    ranking = read_json(bank / "rankings.json")
    rows = read_json(base / "metadata_tx.json")
    build = lambda indices: subset_wire(header, payload, symbols, indices, ranking["original"], rows)
    packet, baseline_symbols = build(ranking["original"])
    baseline_uses = len(baseline_symbols) + accounting(len(packet))["complex_channel_uses"]
    budget = int(baseline_uses * spec["budget_multiplier"])
    side = read_json(Path(spec["side"]) / "channel.json")["complex_channel_uses"]
    report = {"baseline_uses": baseline_uses, "budget": budget, "variants": {},
              "comparison": "same upper symbol budget; actual uses and slack explicitly reported"}
    for name, order, extra in (("uniform", ranking["uniform"], 0),
                               ("adaptive", ranking["adaptive"], 0),
                               ("adaptive_lowres", ranking["adaptive"], side)):
        indices, total = choose_budget(ranking["original"], order, build, budget, extra)
        path = dest / name
        (path / "transmitter").mkdir(parents=True)
        (path / "data").symlink_to((base / "data").resolve(), target_is_directory=True)
        write_json(path / "run_config.json", cfg)
        write_json(path / "keyframes.json", {"indices": indices})
        packet, selected_symbols = build(indices)
        (path / "transmitter/metadata.bin").write_bytes(packet)
        selected_symbols.astype("<c8").tofile(path / "transmitter/visual.c64")
        report["variants"][name] = dict(indices=indices, total_complex_channel_uses=total,
            lowres_complex_channel_uses=extra, unused_budget=budget-total,
            same_budget_cap=True, equal_actual_rate=False)
    write_json(dest / "budget.json", report)


def seed_view(spec, repo):
    base, run = Path(spec["base"]), Path(spec["run"])
    run.mkdir(parents=True)
    cfg = read_json(base / "run_config.json")
    cfg["models"] = read_json(repo / ".local/model_paths.json")
    cfg["seed"] = spec["seed"]
    write_json(run / "run_config.json", cfg)
    (run / "data").symlink_to((base / "data").resolve(), target_is_directory=True)
    receiver = run / "receiver"
    receiver.mkdir()
    (receiver / "frames").symlink_to((base / "receiver/frames").resolve(), target_is_directory=True)
    (receiver / "metadata.csv").symlink_to((base / "receiver/metadata.csv").resolve())
    inputs = read_json(base / "receiver/decoder_inputs.json")
    inputs["decoder"]["seed"] = spec["seed"]
    write_json(receiver / "decoder_inputs.json", inputs)
    write_json(run / "seed_protocol.json", {"shared_evaluation_seed": spec["seed"],
        "channel_reused": True, "source": str(base), "no_sample_specific_seed_search": True})


def assess(spec, repo):
    from .research_quality import read_video
    from .quality_methods import anchor_correction, low_resolution_constraint
    run, dest = Path(spec["run"]), Path(spec["dest"])
    output, inputs = load_view(run)
    source = read_video(run / "data/normalized.mp4")
    if spec.get("correction"):
        alpha = read_json(Path(spec["frozen"]))["anchor_strength"]
        output = anchor_correction(output, keys_from_receiver(run, inputs["indices"]), alpha,
                                  lambda d,t: progress(.3*d/t, 1))
    if spec.get("side"):
        low = read_video(Path(spec["side"]) / "received.mp4")
        alpha = read_json(Path(spec["frozen"]))["lowres_strength"]
        output = low_resolution_constraint(output, low, alpha, lambda d,t: progress(.3*d/t, 1))
    save_video(output, dest / "frames", inputs["video"]["fps"])
    evaluate(source, output, inputs["indices"], dest, spec.get("pixel_only", False),
             lambda d,t: progress(.3+.7*d/t, 1))


def pipeline(spec, repo):
    from . import workers, codec_transport
    run = Path(spec["run"])
    if spec["stage"] == "prepare":
        from .video_io import probe
        cfg = read_json(repo / "configs/webvid5.json")
        cfg.update(spec["config"])
        cfg.update(probe(cfg["input"]))
        cfg["models"] = read_json(repo / ".local/model_paths.json")
        run.mkdir(parents=True, exist_ok=True)
        write_json(run / "run_config.json", cfg)
    else:
        cfg = read_json(run / "run_config.json")
    target = codec_transport if spec["stage"] in {"send", "channel", "receive", "reconstruct"} else workers
    if spec["stage"] == "channel":
        # Common random numbers per keyframe, independent of subset/order.
        # Same complex AWGN law, with its RNG explicitly declared in the ledger.
        import numpy as np
        from .wire import unpack
        codec_transport.channel(dict(cfg, visual_channel="numpy_PCG64"), repo, run)
        header, _ = unpack((run / "received/metadata.bin").read_bytes())
        sent = np.fromfile(run / "transmitter/visual.c64", dtype="<c8")
        received = sent.copy()
        sigma = (1 / (2 * 10 ** (cfg["snr_db"] / 10))) ** .5
        for i, item in enumerate(header["keyframes"]):
            start, count = item["complex_offset"], item["complex_count"]
            rng = np.random.default_rng(cfg.get("channel_seed", cfg["seed"]) + 100000 + item["index"])
            received[start:start+count] += (rng.normal(0, sigma, count)+1j*rng.normal(0, sigma, count)).astype("<c8")
            progress(i+1, len(header["keyframes"]))
        received.tofile(run / "received/visual.c64")
        report = read_json(run / "channel_accounting.json")
        report["visual_awgn_rng"] = "paired numpy_PCG64: channel_seed+100000+keyframe_index"
        report["received_files"]["visual.c64"]["sha256"] = sha256(run / "received/visual.c64")
        write_json(run / "channel_accounting.json", report)
    else:
        getattr(target, spec["stage"])(cfg, repo, run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    args = parser.parse_args()
    spec = read_json(args.spec)
    repo = Path(__file__).resolve().parents[2]
    start = time.monotonic()
    globals()[spec["action"]](spec, repo)
    progress(1, 1)
    print(json.dumps({"action": spec["action"], "seconds": time.monotonic()-start}))


if __name__ == "__main__":
    main()
