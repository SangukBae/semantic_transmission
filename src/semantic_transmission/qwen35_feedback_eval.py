"""Frozen Astra-reference evaluation; this module never supplies labels to Qwen.

Caption judgments are written by the explicitly requested GPT6-Astra evaluator,
not by string similarity or Qwen self-review. Reports validate the exact quoted
claims, every interval and the frozen artifact chain before aggregating results.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path
import random

from . import auto_extraction as ae, qwen35_feedback as fb
from .artifacts import sha256, write_json

HISTORICAL = fb.REPO / "outputs/caption_context_qwen_tv_low_08_20261003"
LABELS = dict(historical="이전 Qwen 문맥 캡션 + 기존 InternVL 키프레임",
              forward="Qwen 공유 관찰 · 피드백 전", feedback="Qwen 공유 관찰 · 피드백 후")


def audit_feedback_requests(root):
    """Verify applied feedback is exactly the bounded subset of actual responses."""
    plan = ae.read(root / "plan.json")
    first, final = (ae.unseal(root / f"{arm}.json") for arm in ("forward", "feedback"))
    requests = ae.unseal(root / "feedback_requests.json")
    calls = [ae.unseal(p) for p in (root / "calls/feedback").glob("*.json")]
    observed = ae.unseal(root / "ledger.json")["observations"]
    for row, observation in zip(requests["windows"], observed):
        a, b = observation["indices"][0], observation["indices"][-1]
        caps = [r for r in first["captions"] if r["start"] <= b and a < r["end_exclusive"]]
        used = set(observation["indices"]) | {i for r in caps for i in r["source_indices"]}
        if row["indices"] != fb.additional_samples(a,b,used) or row["window"] != observation["window"]:
            raise ValueError("feedback source sampling mismatch")
        caps = [r for r in caps if any(r["start"] <= i < r["end_exclusive"] for i in row["indices"])]
        if row["model_response"] is None:
            failed = [ae.unseal(p) for p in (root / "failed_attempts/feedback").glob("*.json")]
            if len([f for f in failed if f["identity"]["ids"] == row["indices"]]) < 3:
                raise ValueError("failed audit lacks retained attempts")
            if row["result"]["events"] or row["result"]["caption_issues"]:
                raise ValueError("failed audit applied an unsupported repair")
            continue
        if not any(c["identity"]["ids"] == row["indices"] and c["result"] == row["model_response"] for c in calls):
            raise ValueError("feedback has no matching actual model response")
        valid, rejected = fb.filter_feedback_requests(row["model_response"],row["indices"],caps)
        if row["result"] != valid or row["rejected_requests"] != rejected:
            raise ValueError("applied requests differ from source validation")
    if len(requests["windows"]) != len(observed):
        raise ValueError("incomplete feedback audit windows")
    accepted, deferred = fb.bounded_repairs(requests["windows"],first["candidates"],plan["max_added_candidates"])
    if requests["accepted_events"] != accepted or requests["deferred_events"] != deferred:
        raise ValueError("repair budget replay differs")
    if final["events"] != first["events"] + accepted:
        raise ValueError("repaired event ledger differs from source feedback")
    candidates, mandatory = set(first["candidates"]), set(first["mandatory"])
    for event in accepted:
        candidates.update((event["before"],event["after"]))
        if fb.protected(event):
            mandatory.update((event["before"],event["after"]))
    if final["candidates"] != sorted(candidates) or final["mandatory"] != sorted(mandatory):
        raise ValueError("repaired candidates differ from validated requests")
    return dict(status="PASS_SOURCE_GROUNDED_FEEDBACK_TRACE",windows=len(observed),
        accepted_events=len(accepted),deferred_events=len(deferred),
        rejected_requests=sum(len(r["rejected_requests"]) for r in requests["windows"]),
        failed_audit_windows=[r["window"] for r in requests["windows"] if r["model_response"] is None])


def event_coverage(keys, reference):
    rows = []
    for event in reference["events"]:
        before = [k for k in keys if event["before_range"][0] <= k <= event["before_range"][1]]
        after = [k for k in keys if event["after_range"][0] <= k <= event["after_range"][1]]
        rows.append(dict(event_id=event["id"], description=event["description"],
                         before_keys=before, after_keys=after, covered=bool(before and after)))
    return dict(covered=sum(r["covered"] for r in rows), total=len(rows), events=rows)


def tolerant_matching(keys, reference_keys, tolerance=6):
    """Maximum one-to-one matching within a frozen temporal tolerance."""
    matches, used = [], {}
    def assign(i, seen):
        options = sorted((j for j, r in enumerate(reference_keys) if abs(keys[i] - r) <= tolerance),
                         key=lambda j: abs(keys[i] - reference_keys[j]))
        for j in options:
            if j in seen:
                continue
            seen.add(j)
            if j not in used or assign(used[j], seen):
                used[j] = i
                return True
        return False
    for i in range(len(keys)):
        assign(i, set())
    matches = [dict(predicted=keys[i], reference=reference_keys[j]) for j, i in sorted(used.items())]
    p = len(matches) / len(keys) if keys else 0.
    r = len(matches) / len(reference_keys) if reference_keys else 0.
    return dict(matches=matches, tolerance_frames=tolerance, precision=p, recall=r,
                f1=2*p*r/(p+r) if p+r else 0.)


def freeze_protocol(root):
    dest = root / "evaluation"
    freeze = ae.read(root / "astra_reference/freeze.json")
    if sha256(root / "astra_reference/reference.json") != freeze["reference_sha256"]:
        raise ValueError("Astra reference changed after freeze")
    protocol = dict(evaluator_model="gpt-6-astra", created_utc=ae.now(),
        reference_sha256=freeze["reference_sha256"], source_only_reference=True,
        reference_method="Independent source-only AI annotation; not human ground truth",
        primary_keyframe_metric="Fraction of frozen events with at least one key in EACH acceptable before/after range",
        secondary_keyframe_metric="One-to-one matching against one recommended reference set, +/-6 frames (0.25s); alternative valid frames and time anchors can lower precision",
        caption_metrics=["one-second bins without clear unsupported/contradictory claims", "fraction of frozen essential facts recalled"],
        caption_judgment="Method names hidden and order shuffled; judge checks actual source and frozen facts; absence from reference alone is not an error",
        facts_policy="Essential fact is covered only when a candidate caption describes it in a temporally overlapping interval with source evidence; broad synonyms accepted",
        errors_policy="Only clear source contradictions or unsupported specificity; ambiguous claims are listed separately, never automatically counted as errors",
        temporal_domain=[0, 1439], bin_frames=24, bins=60, fps=24,
        reference_source_inspection=freeze["inspected_frame_count"],
        historical_caveat="Historical keyframes came from the preexisting assistant/InternVL pipeline, NOT Qwen; historical caption prompt/sampling/boundaries differ",
        causal_caveat="Forward/feedback share exact first pass but feedback adds source images and compute; compares complete workflows, not equal-compute mechanisms",
        test_scope="One development video; no unseen-video generalization, human validation, or reconstructed-video hallucination claim",
        judged_output_tuning_forbidden=True)
    path = dest / "protocol.json"
    if path.exists():
        old = ae.read(path)
        if old["reference_sha256"] != protocol["reference_sha256"]:
            raise ValueError("reference mismatch")
        return old
    write_json(path, protocol)
    return protocol


def prepare(root):
    protocol = freeze_protocol(root)
    plan = fb.check_plan(root)
    fb.verify(root)
    write_json(root / "feedback_trace_audit.json",audit_feedback_requests(root))
    # Exact-request reuse after the schema parser fix retains the original response.
    for path in (root / "calls").glob("*/*.json"):
        record = ae.unseal(path)
        if "reused_from" in record:
            parent = record["reused_from"]
            if sha256(parent["path"]) != parent["sha256"]:
                raise ValueError("parent inference changed after exact-request reuse")
            original = ae.unseal(parent["path"])
            if any(record[k] != original[k] for k in ("raw_text", "result", "model_runtime", "actual_prompt")):
                raise ValueError("reused inference differs from original response")
            if {k:v for k,v in record["identity"].items() if k != "plan_signature"} != {k:v for k,v in original["identity"].items() if k != "plan_signature"}:
                raise ValueError("reused generation input changed")
    reference = ae.read(root / "astra_reference/reference.json")
    past_plan = ae.read(HISTORICAL / "plan.json")
    if past_plan["source_video_sha256"] != plan["normalized_sha256"]:
        raise ValueError("historical video differs from paired source")
    historical = [ae.read(p) for p in sorted((HISTORICAL / "captions/context").glob("*.json"))]
    if len(historical) != 78 or any(r["model_id"] != fb.MODEL_ID or r["revision"] != fb.REVISION
                                    or r["truncated"] for r in historical):
        raise ValueError("incomplete or incorrectly attributed historical captions")
    arms = {arm: ae.unseal(root / f"{arm}.json") for arm in ("forward", "feedback")}
    arms["historical"] = dict(keys=[r["start"] for r in historical] + [historical[-1]["end_exclusive"]], captions=historical)
    mapping = dict(zip(random.Random(20261003).sample(["M17", "M43", "M82"], 3), ["historical", "forward", "feedback"]))
    dest = root / "evaluation"
    write_json(dest / "private_method_map.json", mapping)
    code_for = {v: k for k, v in mapping.items()}
    packets = []
    for lo in range(0, 60, 10):
        bins = []
        for ref in reference["bins"][lo:lo+10]:
            candidates = []
            for arm, bundle in arms.items():
                caps = [dict(caption_id=f"{code_for[arm]}_{i:03d}", start=r["start"],
                             end_exclusive=r["end_exclusive"], caption=r["caption"])
                        for i, r in enumerate(bundle["captions"])
                        if r["start"] < ref["end_exclusive"] and ref["start"] < r["end_exclusive"]]
                candidates.append(dict(method=code_for[arm], captions=caps))
            random.Random(901 + ref["id"]).shuffle(candidates)
            bins.append(dict(reference=ref, candidates=candidates))
        packet = dict(evaluator_model="gpt-6-astra", source_frames=plan["source_frames"], fps=24,
            frames=plan["frames"], reference_sha256=protocol["reference_sha256"], bins=bins,
            instructions=protocol)
        path = dest / "packets" / f"{lo:02d}_{lo+9:02d}.json"
        if path.exists() and ae.read(path) != packet:
            raise ValueError("frozen evaluation packet changed")
        write_json(path, packet)
        packets.append(path)
    key_results = {}
    for arm, bundle in arms.items():
        keys = bundle["keys"]
        uniform = sorted({round(i * (plan["frames"] - 1) / (len(keys) - 1)) for i in range(len(keys))})
        key_results[arm] = dict(keyframes=len(keys), keys=keys,
            event_coverage=event_coverage(keys, reference),
            recommended_frame_matching=tolerant_matching(keys, reference["recommended_keyframes"]),
            equal_count_uniform_event_coverage=event_coverage(uniform, reference))
    write_json(dest / "keyframe_metrics.json", key_results)
    prepare_feedback_review(root, packets, mapping)
    files = [root / "plan.json", root / "forward.json", root / "feedback.json",
             root / "model_bytes_verification.json",
             root / "feedback_trace_audit.json",
             root / "astra_reference/reference.json", root / "astra_reference/freeze.json",
             dest / "protocol.json", dest / "private_method_map.json", dest / "keyframe_metrics.json",
             dest / "agent_provenance.json",
             *packets, *sorted((dest / "early_packets").glob("*.json")),
             *sorted((dest / "judgments").glob("*phase1.json")),
             *sorted((dest / "judgments").glob("*phase2.json")),
             *(HISTORICAL / "captions/context").glob("*.json")]
    manifest = {str(p): sha256(p) for p in files}
    path = dest / "input_manifest.json"
    if path.exists() and ae.read(path) != manifest:
        raise ValueError("evaluation inputs changed")
    write_json(path, manifest)
    return dict(status="READY_FOR_BLINDED_ASTRA_REVIEW", packets=[str(p) for p in packets])


def candidate_identity(method):
    return [(c["start"], c["end_exclusive"], c["caption"]) for c in method["captions"]]


def prepare_feedback_review(root, packet_paths, mapping):
    """Reuse Astra labels ONLY for byte-identical captions and identical intervals."""
    dest = root / "evaluation"
    codes = {v:k for k,v in mapping.items()}
    prior = {}
    for p in (dest / "judgments").glob("*phase2.json"):
        for row in ae.read(p)["judgments"]:
            if row["method"] == codes["forward"]:
                prior[row["bin_id"]] = (row, p)
    all_bins = [r for p in packet_paths for r in ae.read(p)["bins"]]
    pending, reused = [], []
    for row in all_bins:
        ref = row["reference"]
        methods = {m["method"]:m for m in row["candidates"]}
        before, after = methods[codes["forward"]], methods[codes["feedback"]]
        if ref["id"] in prior and candidate_identity(before) == candidate_identity(after):
            original, path = prior[ref["id"]]
            ids = {a["caption_id"]:b["caption_id"] for a,b in zip(before["captions"],after["captions"])}
            value = dict(original, method=codes["feedback"],
                unsupported_claims=[dict(err, caption_id=ids[err["caption_id"]]) for err in original["unsupported_claims"]],
                reuse_of=dict(path=str(path),sha256=sha256(path),method=codes["forward"],bin_id=ref["id"],caption_id_mapping=ids),
                evaluation_mode="exact caption/interval reuse of an existing GPT6-Astra judgment")
            reused.append(value)
        else:
            pending.append(dict(reference=ref,candidates=[after]))
    template = ae.read(packet_paths[0])
    for lo in (0,30):
        packet = dict(template,bins=[r for r in pending if lo <= r["reference"]["id"] < lo+30])
        path = dest / "early_packets" / f"{lo:02d}_{lo+29:02d}_final.json"
        if path.exists() and ae.read(path) != packet:
            raise ValueError("final review delivery changed")
        write_json(path,packet)
    if reused:
        write_json(dest / "judgments/exact_input_reuse.json",dict(evaluator_model="gpt-6-astra",
            reference_sha256=template["reference_sha256"],packet_sha256s=[sha256(p) for p in packet_paths],
            evaluation_mode="Reused existing Astra labels for exact same captions/intervals; no new model judgment",
            judgments=reused))
    write_json(dest / "feedback_review_delivery.json",dict(new_judgments=len(pending),
        exact_input_reuse=len(reused),reused_bins=[r["bin_id"] for r in reused],
        changed_bins=[r["reference"]["id"] for r in pending]))


def validate_judgments(root):
    dest = root / "evaluation"
    for p, digest in ae.read(dest / "input_manifest.json").items():
        if sha256(p) != digest:
            raise ValueError(f"frozen evaluation input changed: {p}")
    truth = {}
    for p in sorted((dest / "packets").glob("*.json")):
        for row in ae.read(p)["bins"]:
            ref = row["reference"]
            for method in row["candidates"]:
                truth[(ref["id"], method["method"])] = (ref, method)
    annotations = []
    packet_by_hash = {sha256(p): ae.read(p) for folder in ("packets", "early_packets")
                      for p in (dest / folder).glob("*.json")}
    files = sorted((dest / "judgments").glob("*.json"))
    for p in files:
        data = ae.read(p)
        if data.get("evaluator_model") != "gpt-6-astra" or data.get("reference_sha256") != ae.read(dest / "protocol.json")["reference_sha256"]:
            raise ValueError("judge identity or reference mismatch")
        packet_hashes = data.get("packet_sha256s", [data.get("packet_sha256")])
        if not packet_hashes or any(h not in packet_by_hash for h in packet_hashes):
            raise ValueError("judge did not identify the exact reviewed packet")
        delivered = {}
        for digest in packet_hashes:
            for row in packet_by_hash[digest]["bins"]:
                for method in row["candidates"]:
                    delivered[(row["reference"]["id"], method["method"])] = (row["reference"], method)
        for r in data["judgments"]:
            key = (r["bin_id"], r["method"])
            if key not in truth or any((x["bin_id"], x["method"]) == key for x in annotations):
                raise ValueError("duplicate or unknown judgment")
            ref, method = truth[key]
            if delivered.get(key) != (ref, method):
                raise ValueError("judged candidate differs from final frozen evaluation inputs")
            if "reuse_of" in r:
                reuse = r["reuse_of"]
                if sha256(reuse["path"]) != reuse["sha256"]:
                    raise ValueError("original Astra judgment changed after reuse")
                original = next(x for x in ae.read(reuse["path"])["judgments"]
                                if x["bin_id"] == r["bin_id"] and x["method"] == reuse["method"])
                if candidate_identity(truth[(r["bin_id"], reuse["method"])][1]) != candidate_identity(method):
                    raise ValueError("reused judgment inputs differ")
                expected_errors = [dict(e,caption_id=reuse["caption_id_mapping"][e["caption_id"]]) for e in original["unsupported_claims"]]
                if (r["covered_fact_ids"] != original["covered_fact_ids"] or r["unsupported_claims"] != expected_errors
                        or r["uncertain_claims"] != original["uncertain_claims"]):
                    raise ValueError("reused Astra labels were altered")
            facts = {f["id"] for f in ref["facts"]}
            if not isinstance(r["covered_fact_ids"], list) or not set(r["covered_fact_ids"]).issubset(facts):
                raise ValueError("unknown reference fact")
            if len(set(r["covered_fact_ids"])) != len(r["covered_fact_ids"]):
                raise ValueError("duplicate recalled fact")
            captions = {c["caption_id"]: c for c in method["captions"]}
            for error in r["unsupported_claims"]:
                if error["caption_id"] not in captions or not error["quote"].strip() or error["quote"] not in captions[error["caption_id"]]["caption"]:
                    raise ValueError("error is not an exact candidate quote")
                if error.get("type") not in ("added", "distorted") or not error.get("reason"):
                    raise ValueError("invalid source error classification")
                fb.validate_evidence(error["evidence_frames"], list(range(1440)))
            if not isinstance(r["uncertain_claims"], list):
                raise ValueError("uncertain claims must be separate")
            annotations.append(dict(r, total_facts=len(facts),
                missed_fact_ids=sorted(facts - set(r["covered_fact_ids"]))))
    if len(annotations) != len(truth):
        raise ValueError(f"incomplete Astra review: {len(annotations)}/{len(truth)}")
    return sorted(annotations, key=lambda r: (r["bin_id"], r["method"]))


def summarize(root):
    dest = root / "evaluation"
    rows = validate_judgments(root)
    mapping = ae.read(dest / "private_method_map.json")
    keys = ae.read(dest / "keyframe_metrics.json")
    extraction = ae.read(root / "EXTRACTION_RESULT.json")
    metrics = {}
    for code, arm in mapping.items():
        part = [r for r in rows if r["method"] == code]
        recalled = sum(len(r["covered_fact_ids"]) for r in part)
        total = sum(r["total_facts"] for r in part)
        errors = [r["bin_id"] for r in part if r["unsupported_claims"]]
        metrics[arm] = dict(label=LABELS[arm], bins=len(part), error_bins=errors,
            error_free_bins=len(part) - len(errors), error_free_bin_rate=(len(part) - len(errors)) / len(part),
            recalled_facts=recalled, reference_facts=total, essential_fact_recall=recalled/total,
            missing_fact_bins=[r["bin_id"] for r in part if r["missed_fact_ids"]],
            keyframes=keys[arm], costs=extraction["arms"].get(arm))
    changes = {}
    final = metrics["feedback"]
    for comparator in ("historical", "forward"):
        previous = metrics[comparator]
        prev_errors, new_errors = set(previous["error_bins"]), set(final["error_bins"])
        changes[comparator + "_to_feedback"] = dict(
            resolved_error_bins=sorted(prev_errors - new_errors), new_error_bins=sorted(new_errors - prev_errors),
            net_error_free_bins=final["error_free_bins"] - previous["error_free_bins"],
            recalled_fact_delta=final["recalled_facts"] - previous["recalled_facts"],
            captured_event_delta=final["keyframes"]["event_coverage"]["covered"] - previous["keyframes"]["event_coverage"]["covered"],
            keyframe_count_delta=final["keyframes"]["keyframes"] - previous["keyframes"]["keyframes"])
    result = dict(status="COMPLETE_ASTRA_REFERENCE_DEVELOPMENT_EVALUATION", video_id="tv_low_08",
        duration_seconds=60, model_id=fb.MODEL_ID, revision=fb.REVISION, evaluator_model="gpt-6-astra",
        reference_sha256=ae.read(dest / "protocol.json")["reference_sha256"],
        metrics=metrics, paired_changes=changes,
        human_ground_truth=False, unseen_video_validation=False, reconstruction_evaluated=False,
        judge_delivery="Method names hidden; reference frozen first; candidate methods delivered in stages as available, not a fully simultaneous randomized blinded study",
        identical_input_policy="Exact same caption texts and intervals reuse the earlier Astra judgment; avoids attributing evaluator variation to the workflow",
        execution_caveat="Development run resumed after schema-validation fixes with exact-input successful inference checkpoints retained, including raw retry prompts. Not a fresh uninterrupted wall-time or deterministic replay benchmark; inference seconds exclude failed calls, loading and IO.",
        feedback_review_delivery=ae.read(dest / "feedback_review_delivery.json"),
        feedback_validation=ae.read(root / "feedback_trace_audit.json"),
        protocol=ae.read(dest / "protocol.json"))
    monitor = root / "device_monitor.jsonl"
    if monitor.exists():
        samples = [json.loads(line) for line in monitor.read_text().splitlines() if line.strip()]
        result["sampled_device_usage"] = dict(samples=len(samples), interval_seconds=2,
            peak_used_mib=max(r["used_mib"] for r in samples), min_free_mib=min(r["free_mib"] for r in samples),
            scope="Device-wide, includes display/other processes; starts after resumed model load")
    write_json(dest / "RESULT.json", result)
    write_json(dest / "validated_judgments.json", rows)
    write_report(root, result, rows, mapping)
    write_json(dest / "report_manifest.json", {str(p): sha256(p) for p in
        [dest / "RESULT.json", dest / "validated_judgments.json", dest / "report.html",
         dest / "keyframe_timeline.svg",
         *sorted((dest / "judgments").glob("*.json"))]})
    return result


def write_report(root, result, judgments, mapping):
    """Standalone readable report; neutral wording handles negative results too."""
    e = html.escape
    table = []
    selection_table = []
    for arm in ("historical", "forward", "feedback"):
        m = result["metrics"][arm]
        k = m["keyframes"]
        table.append(f"<tr><td>{e(m['label'])}</td><td>{k['keyframes']}</td>"
            f"<td>{k['event_coverage']['covered']}/{k['event_coverage']['total']}</td>"
            f"<td>{m['error_free_bins']}/{m['bins']}</td>"
            f"<td>{m['recalled_facts']}/{m['reference_facts']} ({100*m['essential_fact_recall']:.1f}%)</td></tr>")
        uniform = k['equal_count_uniform_event_coverage']
        match = k['recommended_frame_matching']
        selection_table.append(f"<tr><td>{e(m['label'])}</td><td>{k['keyframes']}</td>"
            f"<td>{k['event_coverage']['covered']}/{k['event_coverage']['total']}</td>"
            f"<td>{uniform['covered']}/{uniform['total']}</td>"
            f"<td>{match['precision']:.3f} / {match['recall']:.3f} / {match['f1']:.3f}</td></tr>")
    packets = [ae.read(p) for p in sorted((root / "evaluation/packets").glob("*.json"))]
    bins = [r for p in packets for r in p["bins"]]
    cards = []
    source = Path(ae.read(root / "plan.json")["source_frames"])
    lookup = {(r["bin_id"], r["method"]): r for r in judgments}
    for row in bins:
        ref = row["reference"]
        ids = sorted({i for f in ref["facts"] for i in f["evidence_frames"]})
        images = "".join(f'<figure><img loading="lazy" src="data:image/png;base64,{base64.b64encode((source / f"{i}.png").read_bytes()).decode()}"><figcaption>{i} / {i/24:.3f}s</figcaption></figure>' for i in ids[:4])
        methods = []
        for candidate in sorted(row["candidates"], key=lambda c: ["historical", "forward", "feedback"].index(mapping[c["method"]])):
            judge = lookup[(ref["id"], candidate["method"])]
            caps = "".join(f"<p><small>[{c['start']/24:.3f}, {c['end_exclusive']/24:.3f})s</small> {e(c['caption'])}</p>" for c in candidate["captions"])
            errors = "".join(f"<li><q>{e(x['quote'])}</q> — {e(x['reason'])}</li>" for x in judge["unsupported_claims"])
            missing = "".join(f"<li>{e(f['text'])}</li>" for f in ref["facts"] if f["id"] not in judge["covered_fact_ids"])
            uncertain = "".join(f"<li>{e(str(x))}</li>" for x in judge["uncertain_claims"])
            methods.append(f"<article><h3>{e(LABELS[mapping[candidate['method']]])}</h3>{caps}"
                f"<p>기준 사실 {len(judge['covered_fact_ids'])}/{judge['total_facts']} · 명확한 오류 {'있음' if errors else '없음'}</p><ul>{errors}</ul>"
                f"<p>누락된 기준 사실</p><ul>{missing or '<li>없음</li>'}</ul>"
                f"<details><summary>오류에 포함하지 않은 불확실성</summary><ul>{uncertain or '<li>없음</li>'}</ul></details></article>")
        cards.append(f"<details><summary>{ref['id']}–{ref['id']+1}초</summary><div class=frames>{images}</div>"
            f"<p><b>Astra 기준</b> {e(ref['caption'])}</p><div class=methods>{''.join(methods)}</div></details>")
    costs = {arm: result["metrics"][arm]["costs"] for arm in ("forward", "feedback")}
    timeline = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 210" role="img" aria-label="시간별 키프레임 분포"><rect width="1200" height="210" fill="white"/>']
    colors = dict(historical="#72809b", forward="#3177ad", feedback="#d76c28")
    for n, arm in enumerate(("historical", "forward", "feedback")):
        y = 48 + n * 58
        timeline.append(f'<text x="12" y="{y+5}" font-family="sans-serif" font-size="14">{e(LABELS[arm])}</text><line x1="350" y1="{y}" x2="1170" y2="{y}" stroke="#d2d8e0"/>')
        for i in result["metrics"][arm]["keyframes"]["keys"]:
            x = 350 + i / 1439 * 820
            timeline.append(f'<circle cx="{x:.2f}" cy="{y}" r="3" fill="{colors[arm]}"><title>{i} / {i/24:.3f}s</title></circle>')
    for t in range(0, 61, 10):
        timeline.append(f'<text x="{350+t/60*820:.2f}" y="200" font-family="sans-serif" font-size="13" text-anchor="middle">{t}s</text>')
    timeline.append('</svg>')
    timeline_svg = ''.join(timeline)
    (root / "evaluation/keyframe_timeline.svg").write_text(timeline_svg)
    event_rows = []
    ref = ae.read(root / "astra_reference/reference.json")
    for i, event in enumerate(ref["events"]):
        cells = []
        for arm in ("historical", "forward", "feedback"):
            record = result["metrics"][arm]["keyframes"]["event_coverage"]["events"][i]
            cells.append(f"<td>{'포착' if record['covered'] else '누락'}<br><small>전 {record['before_keys']} / 후 {record['after_keys']}</small></td>")
        event_rows.append(f"<tr><td>{e(event['id'])}: {e(event['description'])}<br><small>전 {event['before_range']} / 후 {event['after_range']}</small></td>{''.join(cells)}</tr>")
    markup = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen3.5 상호 피드백 · Astra 기준 평가</title><style>body{{font:16px/1.6 system-ui;max-width:1500px;margin:30px auto;padding:0 24px;background:#f6f8fb;color:#1d2838}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{padding:12px;border-bottom:1px solid #ddd;text-align:left}}details{{margin:12px 0;padding:16px;background:white;border-radius:10px}}summary{{cursor:pointer;font-weight:bold}}.frames,.methods{{display:flex;gap:12px;flex-wrap:wrap}}figure{{margin:8px 0;max-width:24%}}img{{width:100%}}article{{flex:1;min-width:260px;padding:12px;background:#f4f6fa}}small{{color:#607080}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}li{{color:#9a302a}}h1{{font-size:28px}}</style>
<h1>Qwen3.5-9B 상호 피드백 평가</h1>
<p>tv_low_08 전체 60초 · 동일 Qwen NF4 한 개 · 수정 최대 1회 · GPT6-Astra 원본 기준 443프레임 / 60구간 / 18사건</p>
<p>원본 관찰 → 순차 키프레임 선택 → 캡션 → 추가 원본 검토 → 필요한 후보·선택·캡션 수정. 평가 기준은 모델 입력에서 제외했습니다.</p>
<table><thead><tr><th>방법</th><th>키프레임</th><th>사건 전후 포착</th><th>명확한 캡션 오류 없는 1초 구간</th><th>기준 핵심 사실 재현</th></tr></thead><tbody>{''.join(table)}</tbody></table>
{timeline_svg}
<p>‘오류 없음’은 완전한 정확도를 뜻하지 않으며 사실 누락은 별도 재현율에 반영합니다. 사건 포착은 Astra가 정한 전후 허용 구간의 시간 일치입니다. 이전 키프레임은 기존 InternVL 경로 산출물이며 Qwen이 고른 것이 아닙니다.</p>
<p>피드백 전후는 같은 1차 출력을 공유하지만 추가 이미지와 계산량이 다릅니다. 단일 개발 영상에 대한 AI 기준 평가이며 사람 정답·미관측 영상·복원 영상 평가가 아닙니다.</p>
<p>Astra에는 방법 이름을 숨겼지만, 준비된 결과부터 순차적으로 전달했습니다. 모든 방법을 동시에 무작위로 보여 준 완전한 맹검 실험은 아닙니다.</p>
<p>캡션 문장과 시간 구간이 완전히 같은 피드백 전후 입력은 기존 Astra 판정을 그대로 재사용합니다. 평가자의 판정 변동을 모델의 개선으로 세지 않기 위한 처리입니다.</p>
<h2>선택 개수를 맞춘 비교</h2><table><tr><th>방법</th><th>키프레임 수</th><th>선택 결과 사건 포착</th><th>동일 개수 균등 간격 사건 포착</th><th>권장 프레임 일치 P / R / F1</th></tr>{''.join(selection_table)}</table>
<p>키프레임을 늘리면 사건 전후를 포착하기 쉬워집니다. 같은 개수의 균등 간격 선택을 함께 확인해야 선택 효율을 구분할 수 있습니다. 권장 프레임 일치는 ±6프레임의 일대일 대응이며, 다른 유효한 프레임이나 시간 간격 보호용 프레임도 정밀도를 낮출 수 있습니다.</p>
<h2>전후 변화</h2><pre>{e(json.dumps(result['paired_changes'],ensure_ascii=False,indent=2))}</pre>
<h2>추출 비용</h2><p>추론 시간은 성공한 고유 호출의 생성 시간 합계이며 모델 로딩·실패 재시도·입출력을 제외합니다. 키프레임 PNG 크기는 실제 무선 전송량이 아닙니다.</p><pre>{e(json.dumps(costs,ensure_ascii=False,indent=2))}</pre>
<p>개발 중 응답 형식 검증을 수정하고, 동일 입력의 성공한 추론 원문과 실제 재시도 프롬프트를 보존해 재개했습니다. 새로 한 번에 실행한 시간 벤치마크는 아닙니다.</p>
<h2>피드백 적용 검증</h2><pre>{e(json.dumps(result['feedback_validation'],ensure_ascii=False,indent=2))}</pre>
<h2>사건별 키프레임 확인</h2><table><tr><th>Astra 사건과 허용 프레임 구간</th><th>이전</th><th>피드백 전</th><th>피드백 후</th></tr>{''.join(event_rows)}</table>
<h2>원본·기준·캡션·오류 근거</h2>{''.join(cards)}</html>'''
    (root / "evaluation/report.html").write_text(markup)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("freeze", "prepare", "summarize"))
    p.add_argument("--root", type=Path, default=fb.DEFAULT_ROOT)
    args = p.parse_args(argv)
    action = {"freeze": freeze_protocol, "prepare": prepare, "summarize": summarize}[args.action]
    print(json.dumps(action(args.root.resolve()), ensure_ascii=False))


if __name__ == "__main__":
    main()
