"""Source-grounded, bounded feedback between three roles of one resident Qwen.

The forward arm and the feedback arm share their exact first pass. Reference
annotations are never read here. All generation is pinned, cached and auditable.
This is inference-time coordination, not weight training or InternVL PSSS.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import time

from . import auto_extraction as ae
from .artifacts import sha256, write_json
from .assisted_captions import sample_indices
from .qwen35_backend import LocalQwen35, MODEL_ID, REVISION, SNAPSHOT

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / "outputs/qwen35_feedback_20261003"
DEFAULT_SOURCE = REPO / "outputs/etri_60s_tv_low_08_42057b2ee8ed/baseline"
POLICY = "qwen35_shared_evidence_bounded_feedback_v1"
KIND_ALIASES = {"entry": "entry_exit", "exit": "entry_exit", "entry/exit": "entry_exit", "exit_exit": "entry_exit"}

OBSERVE = '''Inspect the chronological SOURCE images for faithful reconstruction.
Record visible states and meaningful changes, especially small or briefly visible
people/animals, entry/exit, occlusion, pose, and camera viewpoint. Do not invent
unseen events or exact onset times between samples. Use only supplied frame IDs.
Return ONLY JSON: {"facts":[{"text":"short directly visible fact",
"evidence_frames":[0]}], "events":[{"before":0,"after":4,
"kind":"entry_exit|occlusion|pose|viewpoint|scene_change|appearance",
"description":"short visible change"}],"uncertainties":[]}.
Use at most 5 facts and 4 events; replace example IDs by actual supplied IDs.
Use COMPACT JSON, no indentation. kind must be exactly one of the listed values,
for example entry_exit (never entry or exit).'''

SELECT = '''Select keyframes for faithful reconstruction, not highlight summaries.
The first SOURCE image is the latest accepted reference; the second is a later
candidate. Select meaningful visible differences in objects, their visibility,
pose, position or camera-revealed content; skip near duplicates. The shared notes
are fallible model observations, NOT ground truth: verify them against these
images. A note about another time does not establish presence in these images.
Return ONLY JSON: {"select":true,"reason":"at most 18 words",
"needs_check":false}. Set needs_check=true if extra intermediate source images
are needed to resolve visibility or a possible missed event.
Shared observations: '''

CAPTION = '''Describe ONLY the visible TARGET interval in faithful English.
Preserve objects, appearance, position, pose, occlusion and evidenced changes.
Never guess identity, purpose, emotion, a room subtype, hidden body parts, or
motion direction. Do not claim no motion. Use generic terms if uncertain.
Inspect image edges for briefly visible people or animals. Distinguish camera
movement from object movement only when TARGET evidence establishes it.
Shared notes from other roles are fallible; verify claims against TARGET images.
Do not carry an object from another time into this interval. Metadata is not
scene content. Return ONLY JSON: {"caption":"1-4 sentences, <=79 words",
"uncertainties":[],"needs_check":false,"evidence_frames":[0]}.
needs_check requests additional source inspection if this interval may omit a
brief event or a selected boundary fails to describe a change. evidence_frames
must be actual supplied TARGET IDs supporting the caption.
'''

REVIEW = '''Review the supplied draft against the same TARGET source images.
Correct unsupported claims, including motion, emotion, holding/physical relations,
inferred room types, facing direction or absence of an entity. Retain visible
useful details and changes; mention partial visibility when relevant. If source
coverage is insufficient, set needs_check=true instead of guessing. Shared notes
are fallible. Return ONLY the caption JSON schema requested above.
Draft: '''

FEEDBACK = '''Audit the three roles against ADDITIONAL chronological SOURCE images.
The supplied observations, selection decisions and captions are fallible outputs
from the same model. Look for source-evidenced missed brief entities/changes,
incorrect captions, or keyframes that omit a visible state. A request for a check
is not itself evidence. Return ONLY JSON:
{"events":[{"before":0,"after":4,
"kind":"entry_exit|occlusion|pose|viewpoint|scene_change|appearance",
"description":"visible missing change"}],
"caption_issues":[{"caption_id":"c00000",
"issue":"specific unsupported or missing caption content",
"evidence_frames":[4]}], "uncertainties":[]}.
Only propose missing events, not changes already represented adequately by the
selected frames. At most 2 events and 4 caption issues. Event IDs must refer to
supplied SOURCE images; caption issues must use a supplied caption_id and choose
evidence ONLY from that caption's allowed_evidence_frames list. Never assign an
event seen later to an earlier caption. If no valid evidence supports an issue,
omit the issue. At most FOUR caption_issues total. A caption covering only part of this
window must not describe events outside its interval. Do not rewrite captions
here. Requests return to detection/selection, followed by affected caption repair.
'''


def target_samples(a, b):
    if not 0 <= a < b:
        raise ValueError("invalid half-open caption interval")
    return sorted({a, b - 1, *sample_indices(a, b)})


def additional_samples(a, b, previous, count=12):
    """Evenly cover unseen source frames; retain both window endpoints."""
    pool = [i for i in range(a, b + 1) if i not in set(previous)]
    if len(pool) > count - 2:
        pool = [pool[round(i * (len(pool) - 1) / (count - 3))]
                for i in range(count - 2)]
    return sorted({a, b, *pool})


def validate_evidence(ids, allowed):
    if (not isinstance(ids, list) or not ids or
            any(type(i) is not int or i not in allowed for i in ids)):
        raise ValueError("evidence must refer to supplied source frames")


def validate_observation(obj, ids):
    try:
        validate_events(obj, ids)
    except ValueError as exc:
        raise ValueError(f"{exc}; allowed frame IDs={ids}; before<after; allowed kinds={sorted(ae.KINDS)}") from exc
    facts = obj.get("facts")
    if not isinstance(facts, list) or len(facts) > 5 or len(obj["events"]) > 4:
        raise ValueError("invalid bounded observation ledger")
    for fact in facts:
        if not isinstance(fact.get("text"), str) or not fact["text"].strip():
            raise ValueError("empty observation")
        validate_evidence(fact.get("evidence_frames"), ids)
    return obj


def validate_events(obj, ids):
    """Accept explicit spelling aliases without changing the saved raw result."""
    if not isinstance(obj.get("events"), list):
        raise ValueError("invalid event list")
    normalized = dict(obj, events=[dict(e, kind=KIND_ALIASES.get(e.get("kind"), e.get("kind")))
                                  for e in obj["events"]])
    ae.validate_events(normalized, ids)
    return obj


def protected(event):
    return KIND_ALIASES.get(event["kind"], event["kind"]) in ae.PROTECTED


def validate_selection(obj):
    if (type(obj.get("select")) is not bool or ("needs_check" in obj and type(obj["needs_check"]) is not bool)
            or not isinstance(obj.get("reason"), str) or not obj["reason"].strip()
            or len(obj["reason"].split()) > 18):
        raise ValueError("selection requires select:boolean, reason:string <=18 words, optional needs_check:boolean")
    # A missing uncertainty flag requests inspection conservatively; preserve raw JSON.
    return dict(obj, needs_check=obj.get("needs_check", True))


def validate_caption(obj, ids):
    ae.validate_caption(obj)
    if len(obj["caption"].split()) > 79 or type(obj.get("needs_check")) is not bool:
        raise ValueError("caption requires <=79 words and an explicit needs_check boolean; also include uncertainties:[] and evidence_frames using supplied IDs")
    validate_evidence(obj.get("evidence_frames"), ids)
    return obj


def validate_feedback(obj, ids, captions):
    try:
        validate_events(obj, ids)
    except ValueError as exc:
        raise ValueError(f"{exc}; events must use supplied IDs {ids} and kinds {sorted(ae.KINDS)}") from exc
    if len(obj["events"]) > 2:
        raise ValueError("feedback event budget exceeded")
    issues = obj.get("caption_issues")
    by_id = {f"c{r['segment']:05d}": r for r in captions}
    if not isinstance(issues, list) or len(issues) > 4:
        raise ValueError("caption_issues must be a list of at most FOUR issues; do not repeat one issue across every caption")
    for issue in issues:
        row = by_id.get(issue.get("caption_id"))
        if (row is None
                or not isinstance(issue.get("issue"), str) or not issue["issue"].strip()):
            raise ValueError(f"each issue needs caption_id from {list(by_id)} and a nonempty issue string")
        allowed = [i for i in ids if row["start"] <= i < row["end_exclusive"]]
        try:
            validate_evidence(issue.get("evidence_frames"), allowed)
        except ValueError as exc:
            raise ValueError(f"{issue['caption_id']} evidence must be a nonempty subset of {allowed}; an event outside that caption cannot be assigned to it") from exc
    return obj


def feedback_envelope(obj):
    if not isinstance(obj.get("events"), list) or not isinstance(obj.get("caption_issues"), list):
        raise ValueError("feedback requires events:[] and caption_issues:[]")
    ae.validate_strings(obj, "uncertainties")
    return obj


def filter_feedback_requests(raw, ids, captions):
    """Apply only individually valid source-grounded requests; retain all rejects.

    Never invent or snap a frame ID to make a request valid. A malformed event
    must not discard an unrelated valid caption correction or stop the video.
    The original model response stays unchanged in its inference record.
    """
    feedback_envelope(raw)
    result = dict(events=[], caption_issues=[], uncertainties=raw["uncertainties"])
    rejected = []
    for field, limit in (("events", 2), ("caption_issues", 4)):
        for request in raw[field]:
            test = dict(events=[], caption_issues=[], uncertainties=[])
            test[field] = [request]
            try:
                validate_feedback(test, ids, captions)
                if len(result[field]) >= limit:
                    raise ValueError("per-window request budget exceeded")
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                rejected.append(dict(type=field, request=request, reason=str(exc)))
                continue
            result[field].append(request)
    return result, rejected


def choose_keys(frames, candidates, mandatory, scorer, max_gap=24):
    """Stateful selection; replay uses the actual latest accepted reference."""
    candidates, required = sorted(set(candidates)), set(mandatory) | {0, frames - 1}
    if (frames < 2 or max_gap < 1 or not required.issubset(candidates)
            or any(type(i) is not int or not 0 <= i < frames for i in candidates)):
        raise ValueError("invalid candidate timeline")
    keys, trace = [0], [dict(frame=0, reference=None, selected=True, reason="endpoint")]
    for i in candidates[1:]:
        while keys[-1] + max_gap <= i:
            ref = keys[-1]
            keys.append(ref + max_gap)
            trace.append(dict(frame=keys[-1], reference=ref, selected=True,
                              reason="mandatory" if keys[-1] in required else "max_gap"))
        if keys[-1] == i:
            continue
        ref = keys[-1]
        if i in required:
            result = dict(select=True, reason="mandatory", needs_check=False)
        else:
            result = validate_selection(scorer(ref, i))
        trace.append(dict(frame=i, reference=ref, selected=result["select"],
                          reason=result["reason"], needs_check=result["needs_check"]))
        if result["select"]:
            keys.append(i)
    return keys, trace


def bounded_repairs(feedback, original_candidates, budget):
    """Accept complete evidence pairs within a global added-candidate budget."""
    accepted, deferred, known = [], [], set(original_candidates)
    added = set()
    for row in feedback:
        for event in row["result"]["events"]:
            fresh = {event["before"], event["after"]} - known - added
            if len(added | fresh) > budget:
                deferred.append(dict(event, window=row["window"], reason="repair_budget"))
                continue
            added |= fresh
            accepted.append(dict(event, window=row["window"], origin="source_feedback"))
    return accepted, deferred


def prepare(root, baseline, video_id):
    root.mkdir(parents=True, exist_ok=True)
    if (root / "plan.json").exists():
        raise FileExistsError("preserve the frozen plan; use run/check or a new root")
    prep = ae.read(baseline / "prepare.json")
    source = baseline / "data/frames/sample"
    count, fps = prep["frames"], prep["fps"]
    if fps != 24 or count < 2:
        raise ValueError("expected a prepared 24fps video")
    video = baseline / "data/normalized.mp4"
    if sha256(video) != prep["normalized_sha256"]:
        raise ValueError("normalized video differs from preparation")
    prompts = dict(observe=OBSERVE, select=SELECT, caption=CAPTION, review=REVIEW, feedback=FEEDBACK)
    code = [Path(__file__), REPO / "src/semantic_transmission/qwen35_backend.py",
            REPO / "src/semantic_transmission/auto_extraction.py",
            REPO / "src/semantic_transmission/assisted_captions.py",
            REPO / "scripts/qwen35_advanced_caption.py", REPO / "scripts/qwen35_caption.py"]
    model_manifest = REPO / "outputs/qwen35_setup/model_manifest.json"
    plan = dict(policy=POLICY, created_utc=ae.now(), video_id=video_id,
        source_frames=str(source.resolve()), frames=count, fps=fps,
        normalized_video=str(video.resolve()), normalized_sha256=sha256(video),
        source_frame_hashes={str(i): sha256(source / f"{i}.png") for i in range(count)},
        model_id=MODEL_ID, model_revision=REVISION, model_path=str(SNAPSHOT),
        quantization="nf4", max_pixels=262144, seed=2025, do_sample=False,
        candidate_stride=4, window_frames=48, optional_stride=12, max_gap=24,
        max_repair_rounds=1, max_added_candidates=24, max_images=16,
        prompts=prompts, code_sha256={str(p): sha256(p) for p in code},
        model_manifest=str(model_manifest), model_manifest_sha256=sha256(model_manifest),
        arms=["forward", "feedback"],
        control="Exact shared first pass; second arm adds source-grounded bounded feedback",
        reference_access="No reference annotations or prior captions are inputs",
        feedback_policy="One additional-source audit per 2s window; repair only evidenced issues; no second feedback round",
        limitations=["Inference workflow, not weight training", "New direct Qwen selector, not InternVL PSSS",
                     "Feedback consumes additional observations and compute; no equal-compute causality claim",
                     "Captions cover half-open keyframe intervals; final keyframe is an endpoint condition"])
    plan["signature"] = ae.digest(plan)
    write_json(root / "plan.json", plan)
    return plan


def check_plan(root):
    plan = ae.read(root / "plan.json")
    if plan["signature"] != ae.digest({k: v for k, v in plan.items() if k != "signature"}):
        raise ValueError("plan changed")
    for path, expected in plan["code_sha256"].items():
        if sha256(path) != expected:
            raise ValueError(f"code changed after freeze: {path}")
    for path, expected in [(plan["normalized_video"], plan["normalized_sha256"]),
                           (plan["model_manifest"], plan["model_manifest_sha256"])]:
        if sha256(path) != expected:
            raise ValueError(f"input changed: {path}")
    for i, expected in plan["source_frame_hashes"].items():
        if sha256(Path(plan["source_frames"]) / f"{i}.png") != expected:
            raise ValueError(f"source frame changed: {i}")
    return plan


class Session:
    def __init__(self, root, plan, factory=LocalQwen35):
        self.root, self.plan, self.factory = root, plan, factory
        self.model = None
        self.calls = []

    def call(self, role, ids, prompt, validator, tokens=640):
        from PIL import Image
        if not 1 <= len(ids) <= self.plan["max_images"]:
            raise ValueError("image budget exceeded")
        prompt += "\nVALID SOURCE FRAME IDS: " + json.dumps(ids) + ". Use these exact IDs, never image positions or example numbers. Use compact JSON without indentation."
        labels = [f"SOURCE frame {i}, {i/self.plan['fps']:.6f}s" for i in ids]
        identity = dict(plan_signature=self.plan["signature"], role=role, ids=ids,
            frame_hashes=[self.plan["source_frame_hashes"][str(i)] for i in ids],
            prompt=prompt, labels=labels, tokens=tokens)
        key = ae.digest(identity)
        path = self.root / "calls" / role / f"{key}.json"
        if path.exists():
            record = ae.unseal(path)
            if record["identity"] != identity:
                raise ValueError("cache identity changed")
            self.validate_record(record, validator)
            self.calls.append(dict(role=role, path=str(path), sha256=sha256(path), reused=True))
            return record["result"]
        if self.model is None:
            started = time.monotonic()
            self.model = self.factory(self.plan)
            write_json(self.root / "runtime.json", dict(**self.model.runtime,
                setup_seconds=time.monotonic() - started, created_utc=ae.now()))
        images = []
        try:
            for i in ids:
                with Image.open(Path(self.plan["source_frames"]) / f"{i}.png") as im:
                    images.append(im.convert("RGB"))
            error = None
            for attempt in range(3):
                actual = prompt + (f"\nYour previous response failed validation: {error}. Correct this error; return only the requested compact JSON." if attempt else "")
                response = self.model.generate(images, labels, actual, min(1152, tokens + attempt * 192))
                record = dict(identity=identity, attempt=attempt, actual_prompt=actual,
                              created_utc=ae.now(), **response)
                try:
                    record["result"] = ae.parse_json(response["raw_text"])
                    self.validate_record(record, validator)
                except (ValueError, TypeError, KeyError) as exc:
                    error = str(exc)
                    write_json(self.root / "failed_attempts" / role / f"{key}_{attempt}.json",
                               ae.seal(dict(record, error=str(exc))))
                    continue
                write_json(path, ae.seal(record))
                self.calls.append(dict(role=role, path=str(path), sha256=sha256(path), reused=False))
                print(json.dumps(dict(role=role, ids=[ids[0], ids[-1]], seconds=round(response["seconds"], 2))), flush=True)
                return record["result"]
            raise RuntimeError(f"three invalid responses: {role}/{key}")
        finally:
            for im in images:
                im.close()

    def validate_record(self, record, validator):
        runtime = record.get("model_runtime", {})
        if (runtime.get("model_id"), runtime.get("revision"), runtime.get("quantization")) != (MODEL_ID, REVISION, "nf4"):
            raise ValueError("model provenance mismatch")
        if record.get("truncated") is not False or ae.parse_json(record["raw_text"]) != record["result"]:
            raise ValueError("invalid raw response provenance")
        validator(record["result"])


def notes_for(observations, a, b, events=()):
    facts = [f for row in observations for f in row["result"]["facts"]
             if any(a <= i <= b for i in f["evidence_frames"])]
    changes = [e for e in events if a <= e["after"] and e["before"] <= b]
    return dict(facts=facts, events=changes)


def extract(session):
    plan, root = session.plan, session.root
    windows = list(ae.proposal_windows(plan["frames"], plan["candidate_stride"], plan["window_frames"]))
    observations = []
    for w, ids in enumerate(windows):
        result = session.call("observe", ids, plan["prompts"]["observe"],
                              lambda x: validate_observation(x, ids), 768)
        observations.append(dict(window=w, indices=ids, result=result))
    write_json(root / "ledger.json", ae.seal(dict(signature=plan["signature"], observations=observations)))
    events = [dict(e, window=r["window"], origin="observe") for r in observations for e in r["result"]["events"]]
    candidates = {0, plan["frames"] - 1, *range(0, plan["frames"], plan["optional_stride"])}
    mandatory = {0, plan["frames"] - 1}
    for e in events:
        candidates.update((e["before"], e["after"]))
        if protected(e):
            mandatory.update((e["before"], e["after"]))

    def selection(candidate_set, required, current_events):
        def score(a, b):
            notes = notes_for(observations, a, b, current_events)
            return session.call("select", [a, b], plan["prompts"]["select"] + json.dumps(notes), validate_selection, 192)
        return choose_keys(plan["frames"], candidate_set, required, score, plan["max_gap"])

    def captions(keys, issues=(), additional=()):
        result = []
        for n, (a, b) in enumerate(zip(keys, keys[1:])):
            local = [i for i in issues if i["start"] < b and a < i["end_exclusive"]]
            ids = target_samples(a, b)
            if local:
                extra = {i for row in local for i in row["evidence_frames"] if a <= i < b}
                extra.update(i for i in additional if a <= i < b)
                # Endpoints and original target samples are never replaced.
                for i in sorted(extra):
                    if len(ids) >= 16:
                        break
                    ids = sorted(set(ids) | {i})
            notes = notes_for(observations, a, b - 1, events)
            prompt = plan["prompts"]["caption"] + json.dumps(dict(target=[a, b], shared=notes, source_review_issues=local))
            draft = session.call("caption", ids, prompt, lambda x: validate_caption(x, ids), 384)
            reviewed = session.call("caption_review", ids,
                prompt + "\n" + plan["prompts"]["review"] + json.dumps(draft),
                lambda x: validate_caption(x, ids), 384)
            result.append(dict(segment=n, start=a, end_exclusive=b, source_indices=ids, **reviewed))
        return result

    started = time.monotonic()
    keys, trace = selection(candidates, mandatory, events)
    first = captions(keys)
    first_calls = list(session.calls)
    write_json(root / "forward.json", ae.seal(dict(signature=plan["signature"], keys=keys,
        candidates=sorted(candidates), mandatory=sorted(mandatory), events=events, selection=trace,
        captions=first, calls=first_calls, selection_caption_wall_seconds=time.monotonic() - started)))

    feedback = []
    extra_ids = set()
    for row in observations:
        w, observed = row["window"], row["indices"]
        a, b = observed[0], observed[-1]
        local_caps = [r for r in first if r["start"] <= b and a < r["end_exclusive"]]
        used = set(observed) | {i for r in local_caps for i in r["source_indices"]}
        ids = additional_samples(a, b, used)
        extra_ids.update(ids)
        local_caps = [r for r in local_caps if any(r["start"] <= i < r["end_exclusive"] for i in ids)]
        data = dict(window=[a, b], observations=row["result"],
                    selected_frames=[i for i in keys if a <= i <= b],
                    selection_requests=[r for r in trace if a <= r["frame"] <= b and r.get("needs_check")],
                    captions=[dict(caption_id=f"c{r['segment']:05d}",
                        allowed_evidence_frames=[i for i in ids if r["start"] <= i < r["end_exclusive"]],
                        **{k: r[k] for k in ("start", "end_exclusive", "caption", "uncertainties", "needs_check")}) for r in local_caps])
        try:
            raw = session.call("feedback", ids, plan["prompts"]["feedback"] + json.dumps(data),
                               feedback_envelope, 768)
        except RuntimeError as exc:
            if not str(exc).startswith("three invalid responses:"):
                raise
            # The failure is explicit system metadata, never attributed to Qwen.
            feedback.append(dict(window=w, indices=ids, model_response=None,
                status="AUDIT_FAILED_UNAPPLIED", validation_error=str(exc),
                result=dict(events=[], caption_issues=[], uncertainties=[]), rejected_requests=[]))
            continue
        result, rejected = filter_feedback_requests(raw, ids, local_caps)
        feedback.append(dict(window=w, indices=ids, model_response=raw, result=result,
                             rejected_requests=rejected, status="VALIDATED_REQUESTS_ONLY"))
    accepted, deferred = bounded_repairs(feedback, candidates, plan["max_added_candidates"])
    write_json(root / "feedback_requests.json", ae.seal(dict(signature=plan["signature"],
        windows=feedback, accepted_events=accepted, deferred_events=deferred)))
    repaired_candidates, repaired_mandatory = set(candidates), set(mandatory)
    for e in accepted:
        repaired_candidates.update((e["before"], e["after"]))
        if protected(e):
            repaired_mandatory.update((e["before"], e["after"]))
    events += accepted
    repaired_keys, repaired_trace = selection(repaired_candidates, repaired_mandatory, events)
    by_id = {f"c{r['segment']:05d}": r for r in first}
    issues = [dict(i, start=by_id[i["caption_id"]]["start"],
                   end_exclusive=by_id[i["caption_id"]]["end_exclusive"])
              for row in feedback for i in row["result"]["caption_issues"]]
    # New boundaries also require source-checked captions even without an explicit caption issue.
    old_pairs = {(r["start"], r["end_exclusive"]) for r in first}
    for a, b in zip(repaired_keys, repaired_keys[1:]):
        if (a, b) not in old_pairs:
            issues.append(dict(start=a, end_exclusive=b, issue="keyframe boundary changed after source feedback",
                               evidence_frames=[i for i in extra_ids if a <= i < b] or [a]))
    repaired_caps = captions(repaired_keys, issues, extra_ids)
    value = dict(signature=plan["signature"], keys=repaired_keys,
        candidates=sorted(repaired_candidates), mandatory=sorted(repaired_mandatory), events=events,
        selection=repaired_trace, captions=repaired_caps, calls=list(session.calls),
        repair_rounds=1, accepted_feedback_events=len(accepted), deferred_feedback_events=len(deferred),
        rejected_feedback_requests=sum(len(r["rejected_requests"]) for r in feedback),
        failed_audit_windows=[r["window"] for r in feedback if r["status"] == "AUDIT_FAILED_UNAPPLIED"],
        changed_caption_requests=len(issues),
        added_keys=sorted(set(repaired_keys) - set(keys)), removed_keys=sorted(set(keys) - set(repaired_keys)))
    write_json(root / "feedback.json", ae.seal(value))
    return value


def verify(root):
    plan = check_plan(root)
    ledger = ae.unseal(root / "ledger.json")
    if ledger["signature"] != plan["signature"]:
        raise ValueError("ledger plan mismatch")
    observations = ledger["observations"]
    outputs = {}
    for arm in plan["arms"]:
        value = ae.unseal(root / f"{arm}.json")
        if value["signature"] != plan["signature"]:
            raise ValueError("output plan mismatch")
        keys = value["keys"]
        if (keys != sorted(set(keys)) or keys[0] != 0 or keys[-1] != plan["frames"] - 1
                or max(b - a for a, b in zip(keys, keys[1:])) > plan["max_gap"]
                or not set(value["mandatory"]).issubset(keys)):
            raise ValueError("invalid keyframe timeline")
        if len(value["captions"]) != len(keys) - 1:
            raise ValueError("incomplete captions")
        referenced = {}
        for call in value["calls"]:
            path = Path(call["path"])
            if sha256(path) != call["sha256"]:
                raise ValueError("inference artifact changed")
            record = ae.unseal(path)
            if record["identity"]["plan_signature"] != plan["signature"]:
                raise ValueError("foreign inference record")
            Session(root, plan).validate_record(record, lambda x: x)
            identity = record["identity"]
            if identity["frame_hashes"] != [plan["source_frame_hashes"][str(i)] for i in identity["ids"]]:
                raise ValueError("inference source provenance mismatch")
            referenced[ae.digest(record["identity"])] = record
        records = list(referenced.values())
        for row in observations:
            validate_observation(row["result"], row["indices"])
            if not any(c["identity"]["role"] == "observe" and c["identity"]["ids"] == row["indices"]
                       and c["result"] == row["result"] for c in records):
                raise ValueError("ledger has no matching inference")
        def replay_score(a, b):
            prompt = plan["prompts"]["select"] + json.dumps(notes_for(observations, a, b, value["events"]))
            prompt += "\nVALID SOURCE FRAME IDS: " + json.dumps([a, b]) + ". Use these exact IDs, never image positions or example numbers. Use compact JSON without indentation."
            matches = [c for c in records if c["identity"]["role"] == "select"
                       and c["identity"]["ids"] == [a, b] and c["identity"]["prompt"] == prompt]
            if len(matches) != 1:
                raise ValueError("missing exact sequential selection call")
            return matches[0]["result"]
        replay_keys, replay_trace = choose_keys(plan["frames"], value["candidates"],
                                               value["mandatory"], replay_score, plan["max_gap"])
        if (keys, value["selection"]) != (replay_keys, replay_trace):
            raise ValueError("sequential selection trace mismatch")
        for r, a, b in zip(value["captions"], keys, keys[1:]):
            if (r["start"], r["end_exclusive"]) != (a, b):
                raise ValueError("caption boundary mismatch")
            if not set(target_samples(a, b)).issubset(r["source_indices"]):
                raise ValueError("lost target endpoint or frozen sample")
            validate_caption(r, r["source_indices"])
            if any(not a <= i < b for i in r["source_indices"]):
                raise ValueError("caption source outside target")
            if not any(c["identity"]["role"] == "caption_review" and
                       c["identity"]["ids"] == r["source_indices"] and
                       f'"target": [{a}, {b}]' in c["identity"]["prompt"] and
                       c["result"]["caption"] == r["caption"] for c in referenced.values()):
                raise ValueError("caption has no matching raw inference")
        used = {str(p): ae.unseal(p) for p in {Path(c["path"]) for c in value["calls"]}}
        outputs[arm] = dict(keyframes=len(keys), captions=len(value["captions"]),
            unique_calls=len(used), inference_seconds=sum(c["seconds"] for c in used.values()),
            input_tokens=sum(c["input_tokens"] for c in used.values()),
            output_tokens=sum(c["generated_tokens"] for c in used.values()),
            peak_allocated_gib=max(c["peak_allocated_bytes"] for c in used.values()) / 1024 ** 3,
            keyframe_png_bytes=sum((Path(plan["source_frames"]) / f"{i}.png").stat().st_size for i in keys),
            caption_utf8_bytes=sum(len(c["caption"].encode()) for c in value["captions"]))
    result = dict(status="PASS_FEEDBACK_EXTRACTION_INTEGRITY", policy=POLICY,
                  video_id=plan["video_id"], duration_seconds=plan["frames"] / plan["fps"], arms=outputs,
                  model_id=MODEL_ID, revision=REVISION, semantic_accuracy_verified=False,
                  reconstruction_executed=False, signature=plan["signature"])
    write_json(root / "EXTRACTION_RESULT.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "check"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--video-id", default="tv_low_08")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.action == "prepare":
        result = prepare(root, args.baseline.resolve(), args.video_id)
        print(json.dumps(dict(status="FROZEN", signature=result["signature"])))
        return
    plan = check_plan(root)
    if args.action == "check":
        print(json.dumps(verify(root)))
        return
    with (root / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        session = Session(root, plan)
        started = time.monotonic()
        try:
            extract(session)
            result = verify(root)
            result.update(session_seconds=time.monotonic() - started,
                          new_calls=sum(not c["reused"] for c in session.calls))
            write_json(root / "last_execution.json", result)
            print(json.dumps(result), flush=True)
        except BaseException as exc:
            write_json(root / "last_failure.json", dict(error=str(exc), created_utc=ae.now(),
                completed_calls=len(session.calls), seconds=time.monotonic() - started))
            raise


if __name__ == "__main__":
    main()
