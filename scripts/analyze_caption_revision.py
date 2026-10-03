"""Audit completed caption v1/v2 runs and prepare CPU-only visual evidence.

Preserves all frozen execution outputs. Writes only v2/analysis. No model calls.
"""
import csv
import datetime
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from semantic_transmission import caption_revision
from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.etri_60s_check import pts_audit
from semantic_transmission.video_io import probe
from semantic_transmission.webvid5 import fingerprint, read_json
from semantic_transmission.webvid_ablation import snapshot

ROOT = REPO / "outputs/etri_hybrid_keys_tv_low_08_v1"
OLD = ROOT / "reconstruction_assistant_captions"
NEW = ROOT / "reconstruction_assistant_captions_v2"
DEST = NEW / "analysis"
METRICS = ("psnr_db", "ssim", "lpips_vgg", "clip", "dists")


def read_metrics(run):
    with (run / "quality_delivered_mp4.csv").open() as stream:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(stream)]
    assert [r["frame"] for r in rows] == list(range(1440))
    return rows


def mean(rows, indices):
    return {k: float(np.mean([rows[i][k] for i in indices])) for k in METRICS}


def audit_run(root):
    execution = read_json(root / "execution_protocol.json")
    assert fingerprint({k: v for k, v in execution.items() if k != "signature"}) == execution["signature"]
    result = read_json(root / "RESULT.json")
    dependencies, artifacts = {}, {}
    for name in result["stage_seconds"]:
        stage = read_json(root / f"stages/{name}.json")
        assert stage["status"] == "PASSED" and stage["identity"] == execution["signature"], name
        assert stage["dependencies"] == {k: fingerprint(v) for k, v in dependencies.items()}, name
        assert snapshot(root, stage["required"]) == stage["artifacts"], name
        dependencies[name] = stage
        artifacts.update(stage["artifacts"])
    audit = read_json(root / "run/output_audit.json")
    assert audit["source_indices"] == list(range(1440)) and audit["cross_segment_transfers"] == 77
    q = read_json(root / "run/quality.json")
    video = root / "run/receiver/reconstruction/sample_0000.mp4"
    assert q["status"] == "PASSED" and q["video_sha256"] == sha256(video)
    assert q["source_sha256"] == sha256(root / "run/data/normalized.mp4")
    cfg = read_json(root / "run/run_config.json")
    assert cfg == execution["config"]
    info = probe(video)
    assert all(info[k] == cfg[k] for k in ("width", "height", "frames", "fps"))
    return dict(stages=len(dependencies), unique_artifacts=len(artifacts),
                video_sha256=sha256(video), video=info, pts=pts_audit(video, 1440, 24))


def sheets(group, indices, values):
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    folders = [NEW / "run/data/frames/sample", *[r / "run/receiver/reconstruction/sample_0000_frames" for r in (OLD, NEW)]]
    labels = ["SOURCE", "CAPTION v1", "CAPTION v2"]
    result = []
    for page, start in enumerate(range(0, len(indices), 4)):
        these = indices[start:start+4]
        canvas = Image.new("RGB", (1440, len(these)*301), "#202020")
        draw = ImageDraw.Draw(canvas)
        for row, i in enumerate(these):
            for col, (folder, label) in enumerate(zip(folders, labels)):
                p = folder / (f"{i}.png" if col == 0 else f"{i:05d}.png")
                with Image.open(p) as im:
                    canvas.paste(im.convert("RGB").resize((480, 267)), (col*480, row*301+34))
                extra = "" if col == 0 else f" LPIPS={values[col-1][i]['lpips_vgg']:.3f}"
                draw.text((col*480+5, row*301+3), f"{label} | f{i} | {i/24:.3f}s", font=font, fill="white")
                draw.text((col*480+5, row*301+18), extra, font=font, fill="white")
        path = DEST / f"{group}_{page:02d}.jpg"
        canvas.save(path, quality=95)
        result.append(dict(path=path.name, frames=these, sha256=sha256(path), viewed=False,
            scope="Lossless pre-MP4 frames; displayed scores are delivered-MP4 LPIPS."))
    return result


def main():
    if (DEST / "AI_REVIEW.json").exists():
        raise SystemExit("Preserve completed review; refusing to overwrite evidence.")
    identity, _ = caption_revision.preflight(ROOT)
    pairing = caption_revision.verify_pair(OLD, NEW)
    verified = {label: audit_run(root) for label, root in (("v1", OLD), ("v2", NEW))}
    compare = read_json(NEW / "CAPTION_COMPARISON.json")
    assert compare["comparison_sha256"] == sha256(NEW / "caption_comparison.mp4")
    stage = read_json(NEW / "stages/caption-revision-comparison.json")
    assert stage["status"] == "PASSED" and stage["identity"] == fingerprint(identity)
    assert snapshot(NEW, stage["required"]) == stage["artifacts"]
    pts_audit(NEW / "caption_comparison.mp4", 1440, 24)
    values = [read_metrics(r / "run") for r in (OLD, NEW)]
    keys = read_json(NEW / "run/keyframes.json")["indices"]
    keyset = set(keys)
    plan = read_json(REPO / "outputs/etri_visual_keys_20260929/quality_plan.json")
    conditions = {}
    for label, root, rows in zip(("v1", "v2"), (OLD, NEW), values):
        q = read_json(root / "run/quality.json")
        aggregate = mean(rows, range(1440))
        assert all(abs(aggregate[k]-q["delivered_mp4"][k]) < 1e-8 for k in METRICS)
        assert q["delivered_mp4"] == compare[f"quality_{label}"]
        run = read_json(root / "RESULT.json")
        caps = read_json(root / "run/captions.json")
        conditions[label] = dict(quality=q["delivered_mp4"], lossless_quality=q["lossless_frames"],
            key_position_quality=mean(rows, keys), generated_position_quality=mean(rows, [i for i in range(1440) if i not in keyset]),
            total_channel_uses=run["channel"]["total_complex_channel_uses"],
            visual_channel_uses=run["channel"]["visual_complex_channel_uses"],
            digital_channel_uses=run["channel"]["digital_complex_channel_uses"],
            caption_utf8_bytes=sum(len(r["text"].encode()) for r in caps),
            caption_mean_words=float(np.mean([len(r["text"].split()) for r in caps])),
            pipeline_minutes=sum(run["stage_seconds"].values())/60,
            generation_minutes=run["stage_seconds"]["reconstruct"]/60,
            uniform_windows=[dict(frames=[a,b], quality=mean(rows, range(a,b+1))) for a,b in plan["uniform_windows"]],
            event_windows={k: mean(rows, range(a,b+1)) for k,(a,b) in plan["source_event_windows"].items()})
    delta = {k: conditions["v2"]["quality"][k]-conditions["v1"]["quality"][k] for k in METRICS}
    segments = []
    for j, (a, b) in enumerate(zip(keys, keys[1:])):
        # endpoint_exact retains the preceding segment's last frame at a boundary.
        indices = list(range(a if j == 0 else a+1, b+1))
        q = [mean(rows, indices) for rows in values]
        segments.append(dict(segment=j, owned_frames=[indices[0],indices[-1]], v1=q[0], v2=q[1],
                             lpips_delta=q[1]["lpips_vgg"]-q[0]["lpips_vgg"]))
    differences = [b["lpips_vgg"]-a["lpips_vgg"] for a,b in zip(*values)]
    extreme = {}
    for label, reverse in (("metric_regressions", True), ("metric_improvements", False)):
        chosen = []
        for i in sorted(range(1440), key=lambda i: differences[i], reverse=reverse):
            if all(abs(i-j) >= 24 for j in chosen): chosen.append(i)
            if len(chosen) == 8: break
        extreme[label] = chosen
    previous = read_json(OLD / "analysis/AI_REVIEW.json")["frames"]
    groups = {"prior_review": previous, **extreme}
    DEST.mkdir(exist_ok=True)
    evidence = [s for name, indices in groups.items() for s in sheets(name, indices, values)]
    baseline = read_json(REPO / "outputs/etri_60s_tv_low_08_42057b2ee8ed/RESULT.json")
    result = dict(status="NUMERIC_AUDIT_COMPLETE_VISUAL_REVIEW_PENDING", source="tv_low_08",
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), verification=verified,
        pairing=pairing, conditions=conditions, quality_delta=delta,
        total_channel_percent_change=100*(conditions["v2"]["total_channel_uses"]/conditions["v1"]["total_channel_uses"]-1),
        segment_comparison=segments, segments_lpips_improved=sum(r["lpips_delta"] < 0 for r in segments),
        segments_lpips_regressed=sum(r["lpips_delta"] > 0 for r in segments),
        metric_extremes=extreme, evidence=evidence,
        requested_review_frames=sorted(set(i for indices in groups.values() for i in indices)),
        baseline_result=str(REPO / "outputs/etri_60s_tv_low_08_42057b2ee8ed/RESULT.json"),
        limitations=["One development video and one seed; captions revised after inspecting v1 failures.",
            "Generated history changes with captions; initial noise tensors were not recorded.",
            "Review frames reuse the prior plan plus post hoc metric extremes; not independent or blinded.",
            "Pipeline timing excludes keyframe selection and offline caption authoring; runtime difference is not a controlled speed benchmark.",
            "Existing baseline uses different keys/captions and is not a caption-only comparison; no paper-level superiority claim."],
        independent_semantic_review="PENDING", hallucination_mitigation_verified=False,
        analysis_code_sha256=sha256(Path(__file__)), result_sha256=sha256(NEW / "RESULT.json"))
    write_json(DEST / "analysis.json", result)
    print(dict(status=result["status"], quality_delta=delta,
        channel_percent_change=result["total_channel_percent_change"],
        segments_lpips_improved=result["segments_lpips_improved"], segments_lpips_regressed=result["segments_lpips_regressed"],
        extremes=extreme, sheets=len(evidence), frames=len(result["requested_review_frames"])))


if __name__ == "__main__":
    main()
