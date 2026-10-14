"""Feedback must preserve causal selector state, source evidence and budgets."""
import json

import pytest

from semantic_transmission import qwen35_feedback as fb


def test_rejection_does_not_advance_reference():
    seen = []
    def score(a, b):
        seen.append((a, b))
        return dict(select=b == 8, reason="change", needs_check=False)
    keys, _ = fb.choose_keys(16, [0, 4, 8, 12, 15], [0, 15], score)
    assert seen == [(0, 4), (0, 8), (8, 12)]
    assert keys == [0, 8, 15]


def test_feedback_insertion_replays_downstream_with_new_reference():
    seen = []
    def score(a, b):
        seen.append((a, b))
        return dict(select=True, reason="change", needs_check=False)
    keys, _ = fb.choose_keys(16, [0, 4, 6, 8, 12, 15], [0, 6, 15], score)
    assert (6, 8) in seen and (4, 8) not in seen
    assert keys == [0, 4, 6, 8, 12, 15]


def test_time_anchors_update_reference_before_optional_candidate():
    seen = []
    def score(a, b):
        seen.append((a, b))
        return dict(select=False, reason="same", needs_check=False)
    keys, _ = fb.choose_keys(90, [0, 52, 89], [0, 89], score)
    assert keys == [0, 24, 48, 72, 89]
    assert seen == [(48, 52)]


@pytest.mark.parametrize("a,b", [(0, 1), (8, 10), (20, 44), (1416, 1439)])
def test_caption_samples_preserve_target_endpoints_and_never_leak(a, b):
    ids = fb.target_samples(a, b)
    assert a in ids and b - 1 in ids
    assert all(a <= i < b for i in ids)
    assert set(fb.sample_indices(a, b)).issubset(ids)


def test_additional_evidence_covers_unseen_frames_with_bounded_count():
    old = list(range(0, 49, 4))
    ids = fb.additional_samples(0, 48, old)
    assert len(ids) <= 12 and 0 in ids and 48 in ids
    assert all(i not in old for i in ids if i not in (0, 48))


def test_budget_defers_entire_event_instead_of_inventing_partial_bracket():
    events = [dict(before=2, after=5), dict(before=8, after=9)]
    rows = [dict(window=0, result=dict(events=events))]
    accepted, deferred = fb.bounded_repairs(rows, [0, 12], 3)
    assert len(accepted) == len(deferred) == 1
    assert deferred[0]["before"] == 8


def test_caption_feedback_cannot_borrow_evidence_from_another_interval():
    value = dict(events=[], uncertainties=[], caption_issues=[dict(caption_id="c00000",
        issue="object missing", evidence_frames=[30])])
    with pytest.raises(ValueError, match="evidence"):
        fb.validate_feedback(value, [0, 12, 30], [dict(segment=0,start=0, end_exclusive=24)])


def test_observer_cannot_invent_frame_ids():
    value = dict(events=[], uncertainties=[], facts=[dict(text="dog", evidence_frames=[99])])
    with pytest.raises(ValueError, match="evidence"):
        fb.validate_observation(value, [0, 4, 8])


def test_spelling_alias_preserves_raw_model_response_and_event_protection():
    event = dict(before=0, after=4, kind="exit", description="dog leaves frame")
    value = dict(events=[event], facts=[], uncertainties=[])
    assert fb.validate_observation(value, [0, 4]) is value
    assert value["events"][0]["kind"] == "exit" and fb.protected(event)


def test_missing_selector_uncertainty_requests_conservative_inspection():
    raw = dict(select=True, reason="visible change")
    result = fb.validate_selection(raw)
    assert result["needs_check"] is True
    assert "needs_check" not in raw


def test_invalid_event_cannot_discard_valid_caption_feedback_or_invent_a_frame():
    raw=dict(events=[dict(before=0,after=7,kind="entry_exit",description="dog appears")],
        caption_issues=[dict(caption_id="c00000",issue="dog omitted",evidence_frames=[5])],uncertainties=[])
    result,rejected=fb.filter_feedback_requests(raw,[0,5,10],[dict(segment=0,start=0,end_exclusive=12)])
    assert result["events"] == [] and result["caption_issues"] == raw["caption_issues"]
    assert len(rejected)==1 and rejected[0]["request"]["after"]==7
    assert raw["events"][0]["after"]==7  # No snapping or rewriting the raw source claim.


def test_raw_output_and_runtime_are_required_for_cache_reuse(tmp_path):
    session = fb.Session(tmp_path, {})
    result = dict(select=True, reason="change", needs_check=False)
    record = dict(raw_text=json.dumps(result), result=result, truncated=False,
        model_runtime=dict(model_id=fb.MODEL_ID, revision=fb.REVISION, quantization="nf4"))
    session.validate_record(record, fb.validate_selection)
    record["result"] = dict(result, select=False)
    with pytest.raises(ValueError, match="raw response"):
        session.validate_record(record, fb.validate_selection)


def test_current_backend_is_lazy_single_resident_instance(tmp_path):
    from PIL import Image
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (8, 8), "white").save(source / "0.png")
    loads = []
    class FakeModel:
        def __init__(self, config):
            loads.append(1)
            self.runtime = dict(model_id=fb.MODEL_ID, revision=fb.REVISION, quantization="nf4")
        def generate(self, images, labels, prompt, tokens):
            obj = dict(select=True, reason="change", needs_check=False)
            return dict(raw_text=json.dumps(obj), truncated=False, model_runtime=self.runtime, seconds=.1)
    plan = dict(max_images=16, fps=24, signature="s", source_frames=str(source),
                source_frame_hashes={"0": fb.sha256(source / "0.png")})
    session = fb.Session(tmp_path, plan, factory=FakeModel)
    first = session.call("select", [0], "prompt1", fb.validate_selection)
    session.call("select", [0], "prompt2", fb.validate_selection)
    assert session.call("select", [0], "prompt1", fb.validate_selection) == first
    assert len(loads) == 1 and session.calls[-1]["reused"]
    # A complete resume must not load a GPU model at all.
    resumed = fb.Session(tmp_path, plan, factory=FakeModel)
    resumed.call("select", [0], "prompt1", fb.validate_selection)
    assert resumed.model is None and len(loads) == 1


def test_full_feedback_round_recomputes_boundaries_and_verifies_raw_provenance(tmp_path, monkeypatch):
    from PIL import Image
    source = tmp_path / "frames"
    source.mkdir()
    for i in range(20):
        Image.new("RGB", (8, 8), (i, i, i)).save(source / f"{i}.png")
    plan = dict(max_images=16, fps=24, frames=20, signature="synthetic",
        source_frames=str(source), source_frame_hashes={str(i):fb.sha256(source/f"{i}.png") for i in range(20)},
        candidate_stride=4, window_frames=48, optional_stride=12, max_gap=24,
        max_added_candidates=24, arms=["forward", "feedback"], video_id="synthetic",
        prompts=dict(observe=fb.OBSERVE, select=fb.SELECT, caption=fb.CAPTION, review=fb.REVIEW, feedback=fb.FEEDBACK))
    class Model:
        runtime = dict(model_id=fb.MODEL_ID, revision=fb.REVISION, quantization="nf4")
        def __init__(self, config):
            pass
        def generate(self, images, labels, prompt, tokens):
            ids = [int(label.split()[2].rstrip(',')) for label in labels]
            if prompt.startswith(fb.OBSERVE):
                obj = dict(facts=[], events=[], uncertainties=[])
            elif prompt.startswith(fb.SELECT):
                obj = dict(select=False, reason="near duplicate", needs_check=True)
            elif prompt.startswith(fb.FEEDBACK):
                obj = dict(events=[dict(before=ids[1], after=ids[-2], kind="entry_exit", description="dog appears")],
                    caption_issues=[dict(caption_id="c00000", issue="dog omitted", evidence_frames=[ids[1]])], uncertainties=[])
            else:
                obj = dict(caption="A dog is visible." if 'dog omitted' in prompt else "Snow covers the ground.",
                           uncertainties=[], needs_check=False, evidence_frames=[ids[0]])
            return dict(raw_text=json.dumps(obj), truncated=False, model_runtime=self.runtime,
                        seconds=.1, input_tokens=100, generated_tokens=30, peak_allocated_bytes=1024)
    monkeypatch.setattr(fb, "check_plan", lambda root:plan)
    session = fb.Session(tmp_path, plan, factory=Model)
    value = fb.extract(session)
    result = fb.verify(tmp_path)
    fb.write_json(tmp_path / "plan.json",plan)
    from semantic_transmission.qwen35_feedback_eval import audit_feedback_requests
    assert audit_feedback_requests(tmp_path)["status"] == "PASS_SOURCE_GROUNDED_FEEDBACK_TRACE"
    assert result["status"] == "PASS_FEEDBACK_EXTRACTION_INTEGRITY"
    assert value["added_keys"] and len(value["captions"]) > 1
    assert all(r["caption"] == "A dog is visible." for r in value["captions"])
    assert session.model is not None
    # Corrupt a causal reference while maintaining JSON checksum: replay rejects it.
    value["selection"][-1]["reference"] = 999
    fb.write_json(tmp_path / "feedback.json", fb.ae.seal(value))
    with pytest.raises(ValueError, match="selection trace"):
        fb.verify(tmp_path)


def retry_session(tmp_path, responses):
    from PIL import Image
    source = tmp_path / "source"
    source.mkdir()
    for i in (0, 16):
        Image.new("RGB", (8, 8), "white").save(source / f"{i}.png")
    prompts = []
    class Model:
        runtime = dict(model_id=fb.MODEL_ID, revision=fb.REVISION, quantization="nf4")
        def __init__(self, config):
            pass
        def generate(self, images, labels, prompt, tokens):
            prompts.append(prompt)
            return dict(raw_text=responses[len(prompts)-1], truncated=False,
                        model_runtime=self.runtime, seconds=.1)
    plan = dict(max_images=16, fps=24, signature="s", source_frames=str(source),
                source_frame_hashes={str(i):fb.sha256(source / f"{i}.png") for i in (0, 16)})
    return fb.Session(tmp_path, plan, factory=Model), prompts


ENTRY_RESPONSE = ('{"decision":"changed","kind":"entry",'
    '"reference_state":"field with no visible vehicle",'
    '"candidate_state":"green tractor enters bottom of frame","evidence_frames":[0,16]}')


def test_remote_entry_failure_succeeds_without_rewriting_raw_or_loading_on_resume(tmp_path):
    from semantic_transmission.qwen35_semantic_selection import validate_decision
    session, prompts = retry_session(tmp_path, [ENTRY_RESPONSE])
    validator = lambda obj: validate_decision(obj, 0, 16)
    result = session.call("semantic_select", [0, 16], "source-only judgment", validator, 256)
    assert result == json.loads(ENTRY_RESPONSE) and len(prompts) == 1
    record = fb.ae.unseal(next((session.root / "calls/semantic_select").glob("*.json")))
    assert record["raw_text"] == ENTRY_RESPONSE and record["result"] == result
    resumed = fb.Session(tmp_path, session.plan, factory=lambda _: pytest.fail("cache loaded GPU"))
    assert resumed.call("semantic_select", [0, 16], "source-only judgment", validator, 256) == result
    assert resumed.calls[-1]["reused"] and resumed.model is None


def test_repair_includes_failed_output_exact_error_and_all_previous_attempts(tmp_path):
    from semantic_transmission.qwen35_semantic_selection import validate_decision
    bad = ENTRY_RESPONSE.replace('"entry"', '"motion"')
    session, prompts = retry_session(tmp_path, [bad, bad, ENTRY_RESPONSE])
    validator = lambda obj: validate_decision(obj, 0, 16)
    assert session.call("semantic_select", [0, 16], "source-only judgment", validator)["kind"] == "entry"
    assert prompts[0].startswith("source-only judgment") and "FORMAT REPAIR" not in prompts[0]
    assert len(set(prompts)) == 3
    assert all("entry_exit" in p and "validation_error" in p and '\\"motion\\"' in p for p in prompts[1:])
    records = list((tmp_path / "failed_attempts/semantic_select").glob("*.json"))
    assert len(records) == 2 and all(fb.ae.unseal(p)["raw_text"] == bad for p in records)


def test_invalid_source_evidence_exhausts_bounded_retries_and_never_creates_success(tmp_path):
    from semantic_transmission.qwen35_semantic_selection import validate_decision
    bad = ENTRY_RESPONSE.replace('[0,16]', '[0,99]')
    session, prompts = retry_session(tmp_path, [bad]*3)
    with pytest.raises(RuntimeError, match="three invalid responses:.*actual endpoints"):
        session.call("semantic_select", [0, 16], "source-only judgment",
                     lambda obj: validate_decision(obj, 0, 16))
    assert len(prompts) == 3 and session.calls == []
    assert len(list((tmp_path / "failed_attempts/semantic_select").glob("*.json"))) == 3
    assert not list((tmp_path / "calls").glob("*/*.json"))


def test_repair_does_not_hide_truncation_or_model_failure(tmp_path):
    session, prompts = retry_session(tmp_path, [ENTRY_RESPONSE])
    record = dict(raw_text=ENTRY_RESPONSE, result=json.loads(ENTRY_RESPONSE), truncated=True,
                  model_runtime=dict(model_id=fb.MODEL_ID, revision=fb.REVISION, quantization="nf4"))
    with pytest.raises(ValueError, match="token limit"):
        session.validate_record(record, lambda obj: obj)
    class BrokenModel:
        runtime = {}
        def __init__(self, config):
            raise RuntimeError("CUDA out of memory")
    session.factory = BrokenModel
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        session.call("semantic_select", [0, 16], "source-only judgment", lambda obj: obj)
    assert not prompts
