"""Paired short-window conditioning experiment using an existing received AWGN packet.

Changes only align=5 -> None. No retransmission, keyframe insertion, caption edit,
model replacement or training. Window contexts are restarted identically; these
are paired diagnostics, not a replay of the full-video random/context trajectory.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.cli import settings
from semantic_transmission.etri_60s_check import lock, run_command
from semantic_transmission.video_io import probe
from semantic_transmission.webvid5 import execution_identity, fingerprint, read_json
from semantic_transmission.webvid_ablation import Stages, environment

REPO = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = REPO / "outputs/etri_60s_tv_low_08_42057b2ee8ed"
WINDOWS = {"people_10s": [217, 228, 234, 260, 266],
           "car_55s": [1303, 1305, 1311, 1331, 1334],
           "door_59s": [1400, 1404, 1409, 1427, 1439]}
CASES = {"baseline": 5, "no_rounding": None}


def initialize(root, baseline):
    base = baseline / "baseline"
    config = read_json(base / "run_config.json")
    keys = read_json(base / "keyframes.json")["indices"]
    inputs = read_json(base / "receiver/decoder_inputs.json")
    with (base / "receiver/metadata.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    pairs = []
    for name, selected in WINDOWS.items():
        start, end = selected[0], selected[-1]
        positions = [keys.index(k) for k in selected]
        if positions != list(range(positions[0], positions[-1] + 1)):
            raise ValueError("diagnostic windows must use consecutive original keyframes")
        shared = root / "inputs" / name
        shared.mkdir(parents=True)
        source = shared / "source.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(base / "data/normalized.mp4"),
            "-vf", f"trim=start_frame={start}:end_frame={end+1},setpts=PTS-STARTPTS", "-an",
            "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv420p", str(source)], check=True)
        if probe(source)["frames"] != end - start + 1:
            raise ValueError("window frame count mismatch")
        for case, align in CASES.items():
            run = root / name / case
            local_keys = [k - start for k in selected]
            cfg = dict(config, profile="etri_conditioning_diagnostic_v1", input=str(source),
                       input_sha256=sha256(source), frames=end-start+1, max_frames=end-start+1,
                       diagnostic_align=align, conditioning_alignment="official_release")
            for key in ("selector_checkpoint", "caption_checkpoint"):
                cfg.pop(key, None)
            write_json(run / "run_config.json", cfg)
            (run / "data").mkdir()
            shutil.copyfile(source, run / "data/normalized.mp4")
            target = run / "receiver/frames/sample/key_frames_received"
            target.mkdir(parents=True)
            for original, local in zip(selected, local_keys):
                shutil.copyfile(base / f"receiver/frames/sample/key_frames_received/{original}.png", target / f"{local}.png")
            with (run / "receiver/metadata.csv").open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=["path", "text", "flow"])
                writer.writeheader()
                for local, original in enumerate(positions[:-1]):
                    writer.writerow(dict(rows[original], path=f"clips/sample/{local:05d}.mp4"))
            write_json(run / "keyframes.json", {"indices": local_keys})
            write_json(run / "receiver/decoder_inputs.json", dict(inputs,
                indices=local_keys, video=dict(inputs["video"], frames=end-start+1)))
            pairs.append({"window": name, "case": case, "original_frames_inclusive": [start, end],
                "original_keyframes": selected, "original_segments": positions[:-1], "align": align,
                "source_sha256": sha256(source),
                "receiver_metadata_sha256": sha256(run / "receiver/metadata.csv"),
                "received_keyframe_sha256": {str(k): sha256(target / f"{k-start}.png") for k in selected}})
    write_json(root / "pairs.json", pairs)


def reconstruct(run):
    from semantic_transmission.codec_transport import decoder_config_text
    from semantic_transmission.decoder_runner import run as decode
    cfg = read_json(run / "run_config.json")
    text = decoder_config_text(cfg, REPO, read_json(run / "receiver/decoder_inputs.json"))
    text += f"\ncpu_video_storage=True\nconditioning_alignment='official_release'\nalign={cfg['diagnostic_align']!r}\n"
    path = run / "receiver/decoder_config.py"
    path.write_text(text)
    env = dict(os.environ, PYTHONPATH=str(REPO / ".local/vendor/Open-Sora"),
               ETRI_CONDITION_TRACE=str(run / "receiver/conditioning_trace.json"))
    decode(run / "receiver/metadata.csv", run / "receiver/reconstruction", "key_frames_received",
           run / "receiver", decoder=REPO / "scripts/etri_conditioning_probe.py", config=path, environment=env)


def summarize(root):
    rows = []
    for name in WINDOWS:
        values, traces = {}, {}
        for case in CASES:
            run = root / name / case
            quality = read_json(run / "quality.json")
            traces[case] = read_json(run / "receiver/conditioning_trace.json")
            values[case] = quality["delivered_mp4"]
        a, b = traces["baseline"]["mask"], traces["no_rounding"]["mask"]
        assert len(a) == len(b) == len(WINDOWS[name]) - 1
        assert all(x["noise_sha256"] == y["noise_sha256"] for x, y in zip(a, b)), "paired noise differs"
        for x, y in zip(a, b):
            # Original received keyframes always precede appended generated refs.
            rx_a = [v["reference_sha256"] for v in x["active"] if v["reference"] < len(WINDOWS[name])]
            rx_b = [v["reference_sha256"] for v in y["active"] if v["reference"] < len(WINDOWS[name])]
            assert rx_a == rx_b, "received keyframe latents differ"
        rows.append({"window": name, "metrics": values, "identical_initial_noise_all_segments": True,
                     "identical_received_keyframe_latents": True,
                     "delta_after_minus_before": {k: values['no_rounding'][k]-values['baseline'][k]
                       for k in ("psnr_db", "ssim", "lpips_vgg", "clip", "dists")}})
    write_json(root / "RESULT.json", {"status": "PAIRED_DIAGNOSTICS_COMPLETE", "windows": rows,
        "intervention": "Only mask alignment rounding disabled (align=5 to None).",
        "scope": "Three restarted short contexts; not a full 60-second mitigation run.",
        "additional_channel_uses": 0, "semantic_error_review": "PENDING"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=REPO / "outputs/etri_conditioning_diagnosis_20260928")
    parser.add_argument("--worker", choices=["reconstruct"])
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.worker:
        reconstruct(args.run_dir.resolve())
        return
    root, base = args.output.resolve(), args.baseline.resolve()
    cfg = read_json(base / "baseline/run_config.json")
    evidence = read_json(base / "RESULT.json")
    if evidence["status"] != "PASS_60S_RECONSTRUCTION":
        raise ValueError("baseline incomplete")
    identity = {"baseline": str(base), "baseline_result_sha256": sha256(base / "RESULT.json"),
        "windows": WINDOWS, "cases": CASES, "execution": execution_identity(REPO, cfg),
        "scripts": {p: sha256(REPO / p) for p in ["scripts/diagnose_etri_conditioning.py", "scripts/etri_conditioning_probe.py"]}}
    signature = fingerprint(identity)
    root.mkdir(parents=True, exist_ok=True)
    with lock(REPO / ".local/etri_60s_check.lock"), lock(root / ".lock"):
        protocol = root / "protocol.json"
        if protocol.exists() and read_json(protocol)["signature"] != signature:
            raise ValueError("diagnostic code/settings changed; use a new output directory")
        if not protocol.exists():
            write_json(protocol, dict(identity, signature=signature))
        pipeline = Stages(root, signature)
        owned = ["inputs", "pairs.json", *WINDOWS]
        # Later stage outputs must not join the immutable initialization snapshot.
        def required():
            return [str(p.relative_to(root)) for d in [root / 'inputs', *[root / w for w in WINDOWS]]
                    for p in d.rglob('*') if p.is_file()] + ['pairs.json']
        pipeline.step("initialize", owned, required, lambda _: initialize(root, base))
        local, env = settings(REPO), environment(cfg["seed"])
        for window in WINDOWS:
            for case in CASES:
                relative = f"{window}/{case}"
                run = root / relative
                for stage in ("reconstruct", "evaluate"):
                    name = f"{window}.{case}.{stage}"
                    command = ([local['python'], str(Path(__file__).resolve()), '--worker', 'reconstruct', '--run-dir', str(run)]
                               if stage == 'reconstruct' else [local['python'], '-m', 'semantic_transmission.research_quality', 'evaluate', str(run)])
                    files = (["receiver/reconstruction", "receiver/decoder_config.py", "receiver/conditioning_trace.json"]
                             if stage == 'reconstruct' else ["quality.json", "quality_lossless_frames.csv", "quality_delivered_mp4.csv"])
                    paths = [f"{relative}/{p}" for p in files]
                    metrics, timing, progress = f"resources/{name}.json", f"resources/{name}.time.txt", f"progress/{name}.json"
                    pipeline.step(name, paths + [metrics, timing, progress], paths + [metrics, timing],
                        lambda log, cmd=command, m=metrics, p=progress: run_command(REPO, cmd, log, env, root/m, root/p))
        summarize(root)
        print(f"PAIRED_DIAGNOSTICS_COMPLETE: {root}", flush=True)


if __name__ == "__main__":
    main()
