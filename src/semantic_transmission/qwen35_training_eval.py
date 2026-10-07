"""Audit source-grounded caption review and freeze a pretraining baseline.

Caption recall is reviewed against semantic reference items, not string overlap
or Qwen self-scores. Omitted facts and unsupported assertions stay separate.
"""
import argparse
import html
import os
from pathlib import Path
from urllib.parse import quote

from .qwen35_training_data import DEFAULT_DATA, read_json, write_json, sha256, verify


def finalize(data, baseline, review):
    verify(data)
    if (baseline / "freeze.json").exists():
        verify_baseline(data)
        if read_json(review) != read_json(baseline / "caption_review.json"):
            raise ValueError("Baseline review is frozen; create a new version for corrections")
        return read_json(baseline / "RESULT.json")
    generated = read_json(baseline / "generation_freeze.json")
    if generated["protocol_sha256"] != sha256(baseline / "protocol.json"):
        raise ValueError("Baseline protocol changed")
    for name, digest in generated["records"].items():
        if sha256(baseline / name) != digest:
            raise ValueError("Baseline output changed")
    judgments = read_json(review)
    if judgments["generation_freeze_sha256"] != sha256(baseline / "generation_freeze.json"):
        raise ValueError("Review is for different baseline outputs")
    rows = [r for split in ["validation", "test"] for r in read_json(data / f"{split}.json")["records"]]
    if set(judgments["records"]) != {r["id"] for r in rows}:
        raise ValueError("Caption review must cover every frozen evaluation window")
    result = {"status": "FROZEN_AI_REVIEWED_BASELINE", "human_ground_truth": False,
              "provenance": judgments["provenance"], "splits": {}}
    for split in ["validation", "test"]:
        stats = {"windows": 0, "keys": 0, "event_hits": 0, "event_count": 0,
                 "facts_recalled": 0, "fact_count": 0, "error_windows": 0, "error_claims": 0,
                 "selection_failures": 0, "caption_failures": 0, "processing_seconds": 0.}
        for row in [r for r in rows if r["split"] == split]:
            actual = read_json(baseline / "records" / f"{row['id']}.json")
            judged = judgments["records"][row["id"]]
            valid = {f["id"] for f in row["facts"]}
            recalled = judged["facts_recalled"]
            if len(set(recalled)) != len(recalled) or not set(recalled) <= valid:
                raise ValueError("Unknown/duplicate fact judgment")
            for error in judged["errors"]:
                if error["quote"] not in actual["caption"] or not error["reason"] or not error["evidence"]:
                    raise ValueError("Error judgment lacks exact quote or source evidence")
                if not set(error["evidence"]) <= set(range(8)):
                    raise ValueError("Invalid error evidence indices")
            stats["windows"] += 1
            stats["keys"] += len(actual["keys"])
            stats["event_hits"] += actual["coverage"]["event_hits"]
            stats["event_count"] += actual["coverage"]["event_count"]
            stats["facts_recalled"] += len(recalled)
            stats["fact_count"] += len(row["facts"])
            stats["error_windows"] += bool(judged["errors"])
            stats["error_claims"] += len(judged["errors"])
            stats["selection_failures"] += actual["selection_error"] is not None
            stats["caption_failures"] += actual["caption_call"]["status"] != "CAPTION_COMPLETE"
            stats["processing_seconds"] += actual["total_seconds"]
        stats["mean_seconds_per_window"] = stats["processing_seconds"] / stats["windows"]
        result["splits"][split] = stats
    result["metrics"] = {
        "event_capture": "At least one selected frame in EACH frozen before/after evidence set; counts sampled events only",
        "fact_recall": "AI source review: core reference claim must be expressed, temporal order retained; extra correct facts receive no reference-recall credit",
        "error_windows": "At least one explicit contradicted/unsupported generated assertion, independent of omitted reference facts",
        "time": "Measured wall time for image IO, Qwen proposal/selection and caption; model loading excluded; not full video/transmission/reconstruction time",
        "selection_failure": "Invalid JSON/key budget/order is reported; fixed uniform [0,2,5,7] fallback included in all quality/time counts"}
    write_json(baseline / "caption_review.json", judgments)
    write_json(baseline / "RESULT.json", result)
    freeze = {"files": {p: sha256(baseline / p) for p in ["protocol.json", "generation_freeze.json", "caption_review.json", "RESULT.json"]}}
    write_json(baseline / "freeze.json", freeze)
    write_json(data / "baseline_reference.json", {"path": str(baseline.resolve()), "freeze_sha256": sha256(baseline / "freeze.json")})
    return result


def verify_baseline(data):
    reference = read_json(data / "baseline_reference.json")
    folder = Path(reference["path"])
    if sha256(folder / "freeze.json") != reference["freeze_sha256"]:
        raise ValueError("Baseline freeze changed")
    for name, digest in read_json(folder / "freeze.json")["files"].items():
        if sha256(folder / name) != digest:
            raise ValueError("Frozen baseline changed")
    protocol = read_json(folder / "protocol.json")
    if protocol["data_freeze_sha256"] != sha256(data / "freeze.json"):
        raise ValueError("Baseline is for a different dataset")
    for name, digest in read_json(folder / "generation_freeze.json")["records"].items():
        if sha256(folder / name) != digest:
            raise ValueError("Frozen generated caption changed")
    return protocol


def render_review(data, baseline):
    """A source-first inspection page; never a substitute for the frozen JSON."""
    judged = read_json(baseline / "caption_review.json")["records"]
    parts = ["<!doctype html><meta charset='utf-8'><title>Qwen training pilot source review</title>",
        "<style>body{font:16px system-ui;max-width:1500px;margin:30px auto;padding:0 20px}article{border-top:2px solid #777;margin-top:32px}.frames{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.frames img{width:100%}.note{background:#f5f1e4;padding:12px}li{margin:6px 0}code{font-size:13px}</style>",
        "<h1>Qwen3.5 multi-task training pilot</h1><p class='note'>96 windows from 24 original WebVid videos. AI source review, not human or GPT6-Astra ground truth. Eight sampled frames per quarter; event times are observed before/after brackets. Training, validation and test originals are disjoint. Baseline labels were frozen before generation. Error quotes are a later non-blinded AI review.</p>"]
    for split in ["train", "validation", "test"]:
        parts.append(f"<h2>{split}</h2>")
        for row in read_json(data / f"{split}.json")["records"]:
            parts.append(f"<article id='{row['id']}'><h3>{row['id']}</h3><p>{row['start']/24:.3f}–{row['end_exclusive']/24:.3f}s</p><div class='frames'>")
            for i, frame in enumerate(row["frames"]):
                url = quote(os.path.relpath(frame["path"], data))
                parts.append(f"<div><a href='{url}'><img loading='lazy' src='{url}'></a>i{i} · f{frame['index']} · {frame['index']/24:.3f}s</div>")
            parts.append(f"</div><p><b>Reference:</b> {html.escape(row['caption'])}</p><ul>")
            for fact in row["facts"]:
                parts.append(f"<li>{html.escape(fact['text'])} <code>evidence {fact['evidence']}</code></li>")
            for event in row["events"]:
                parts.append(f"<li><b>Event:</b> {html.escape(event['description'])} <code>{event['before']} → {event['after']}; bracket {event['start_frame']/24:.3f}–{event['end_frame']/24:.3f}s</code></li>")
            parts.append("</ul>")
            if row["id"] in judged:
                actual = read_json(baseline / "records" / f"{row['id']}.json")
                r = judged[row["id"]]
                parts.append(f"<p><b>Untrained Qwen:</b> {html.escape(actual['caption'])}</p><p>Keys: {actual['keys']}; selection fallback: {actual['selection_error'] is not None}. Recalled items: {len(r['facts_recalled'])}/{len(row['facts'])}.</p><p>{html.escape(r['note'])}</p><ul>")
                for e in r["errors"]:
                    parts.append(f"<li><q>{html.escape(e['quote'])}</q> — {html.escape(e['reason'])} <code>{e['evidence']}</code></li>")
                parts.append("</ul>")
            parts.append("</article>")
    (data / "review.html").write_text("\n".join(parts))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--review", type=Path, required=True)
    args = p.parse_args()
    import json
    print(json.dumps(finalize(args.data, args.baseline, args.review), indent=2, ensure_ascii=False))
    render_review(args.data, args.baseline)


if __name__ == "__main__":
    main()
