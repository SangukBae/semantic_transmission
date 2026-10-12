import json

import numpy as np
import pytest

from semantic_transmission import auto_extraction as ae
from semantic_transmission import qwen35_pairs_comparison as run
from semantic_transmission import qwen35_pairs_review as review
from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.qwen35_caption_adapter import task_runtime
from semantic_transmission.webvid_ablation import Stages


def test_read_only_check_never_starts_workers_or_creates_output(tmp_path, monkeypatch, capsys):
    root = tmp_path / "not-created"
    p = dict(source=dict(frames=1440), keys={a: list(range(79)) for a in run.ARMS},
             caption=dict(caption_adapter=dict(id="fixture")), scope="test", semantic_review="non-independent")
    monkeypatch.setattr(run, "preflight", lambda _: p)
    monkeypatch.setattr(run, "execute", lambda *_: pytest.fail("--check ran experiment"))
    run.main(["--root", str(root), "--check"])
    assert not root.exists()
    assert json.loads(capsys.readouterr().out)["model_inference_started"] is False


def test_default_launch_and_partial_resume_use_same_root(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(run, "preflight", lambda _: {"fixture": True})
    monkeypatch.setattr(run, "execute", lambda *args: calls.append(args))
    run.main(["--root", str(tmp_path)])
    run.main(["--root", str(tmp_path), "--stop-after", "captions"])
    assert calls == [(tmp_path, {"fixture": True}, None), (tmp_path, {"fixture": True}, "captions")]


@pytest.mark.parametrize("output", ["source", "source/new", "."])
def test_output_cannot_overlap_historical_inputs(tmp_path, output):
    with pytest.raises(ValueError, match="protected"):
        run.forbid_overlap(tmp_path / output, [tmp_path / "source"])


@pytest.mark.parametrize("keys", [[0, 24, 48], [0, 0, 48], [0, 25, 48], [0, 23, 47], [0, 24.0, 48]])
def test_key_budget_and_endpoints(keys):
    if keys == [0, 24, 48] and all(type(i) is int for i in keys):
        run.validate_keys(keys, 49, 3)
    else:
        with pytest.raises(ValueError, match="ordered keys"):
            run.validate_keys(keys, 49, 3)


class FakeVLM:
    def __init__(self):
        self.config = dict(caption_adapter=dict(id="synthetic"))
        self.calls = 0

    def generate(self, images, labels, prompt, tokens, task):
        self.calls += 1
        return dict(raw_text='{"caption":"A car is visible.","uncertainties":[]}', truncated=False,
            seconds=1., model_runtime=dict(model_id=run.MODEL_ID, revision=run.REVISION, quantization="nf4",
                                          **task_runtime(self.config, task)))


def test_inference_resume_rejects_changed_prompt_raw_and_adapter_role(tmp_path):
    vlm, path = FakeVLM(), tmp_path / "draft.json"
    def call(prompt="Describe"):
        return run.infer(vlm, path, dict(task="draft"), [], [], prompt, ae.caption_validator({}))
    expected = call()
    assert call() == expected and vlm.calls == 1
    with pytest.raises(ValueError, match="identity"):
        call("Different prompt")
    record = ae.unseal(path)
    original = dict(record)
    record["raw_text"] = '{"caption":"A dog.","uncertainties":[]}'
    write_json(path, ae.seal(record))
    with pytest.raises(ValueError, match="raw inference"):
        call()
    original["model_runtime"]["caption_adapter"] = None
    write_json(path, ae.seal(original))
    with pytest.raises(ValueError, match="adapter/task"):
        call()


def test_failed_inference_retries_do_not_overwrite_previous_attempts(tmp_path):
    vlm = FakeVLM()
    def broken(*args, **kw):
        return dict(raw_text="not json", truncated=True, model_runtime={})
    vlm.generate = broken
    for _ in range(2):
        with pytest.raises(RuntimeError, match="three invalid"):
            run.infer(vlm, tmp_path / "draft.json", dict(task="draft"), [], [], "prompt", ae.caption_validator({}))
    assert len(list((tmp_path / "failed_attempts").glob("*.json"))) == 6
    assert not (tmp_path / "draft.json").exists()


def test_interrupted_caption_stage_keeps_successful_calls_and_verifies_resume(tmp_path):
    def fail(_):
        write_json(tmp_path / "calls/one.json", {"success": True})
        raise RuntimeError("interrupted")
    required = ["calls", "captions"]
    with pytest.raises(RuntimeError):
        Stages(tmp_path, "same").step("captions", ["captions"], required, fail)
    digest = sha256(tmp_path / "calls/one.json")
    Stages(tmp_path, "same").step("captions", ["captions"], required,
        lambda _: write_json(tmp_path / "captions/complete.json", {"success": True}))
    assert sha256(tmp_path / "calls/one.json") == digest
    Stages(tmp_path, "same").step("captions", ["captions"], required,
        lambda _: pytest.fail("completed caption stage reran"))
    write_json(tmp_path / "calls/one.json", {"success": False})
    with pytest.raises(ValueError, match="artifacts changed"):
        Stages(tmp_path, "same").step("captions", [], required, lambda _: None)


def test_frame_noise_is_keyed_by_source_frame_not_selection_position():
    a = {i: run.frame_noise(i, 12, 42, 10) for i in [0, 12, 24]}
    b = {i: run.frame_noise(i, 12, 42, 10) for i in [0, 8, 12, 24]}
    for i in a:
        np.testing.assert_array_equal(a[i], b[i])
    assert not np.array_equal(a[0], a[12])
    assert not np.array_equal(a[0], run.frame_noise(0, 12, 43, 10))


def test_real_channel_serialization_and_common_frame_audit(tmp_path, monkeypatch):
    from semantic_transmission import metadata_channel
    from semantic_transmission.packets import accounting
    from semantic_transmission.wire import pack
    monkeypatch.setattr(metadata_channel, "transmit", lambda packet, *_: (packet, dict(accounting(len(packet)), bit_errors=0)))
    selections = dict(internvl=[0, 12, 24], event_pairs=[0, 8, 24])
    for arm, keys in selections.items():
        dest = tmp_path / f"reconstruction/{arm}/run"
        write_json(dest / "run_config.json", dict(snr_db=10, channel_seed=42, width=2, height=2, frames=25,
                                                   visual_channel="source_frame_seeded_numpy_awgn_v1"))
        tx = dest / "transmitter"
        tx.mkdir()
        header = dict(video=dict(width=2, height=2, frames=25), decoder={},
            segments=[dict(text="A scene.", flow=0.) for _ in keys[1:]], keyframes=[
                dict(index=k, complex_offset=2*n, complex_count=2, rate_offset=n, rate_bytes=1, average_power=1.)
                for n, k in enumerate(keys)])
        (tx / "metadata.bin").write_bytes(pack(header, bytes(keys)))
        np.array([v for k in keys for v in (k+1, k+2)], dtype="<c8").tofile(tx / "visual.c64")
        run.paired_channel(dest)
        report = ae.read(dest / "channel_accounting.json")
        assert report["metadata_exact_match"] and report["total_complex_channel_uses"] > 6
    audit = review.check_common_channel(tmp_path, dict(keys=selections))
    assert audit["common_frames"] == [0, 24]
    received = tmp_path / "reconstruction/event_pairs/run/received/visual.c64"
    x = np.fromfile(received, dtype="<c8"); x[0] += 1; x.tofile(received)
    with pytest.raises(ValueError, match="common transmitted/received"):
        review.check_common_channel(tmp_path, dict(keys=selections))


def test_job_plan_keeps_transport_receiver_repairs_and_evaluation():
    jobs = list(run.reconstruction_jobs())
    names = [j[0] for j in jobs]
    assert names == ["input-audit", "semantic-clips", "caption", "flow", "send", "channel", "receive",
                     "prepare-text", "reconstruct", "output-audit", "evaluate"]
    for name in ("caption", "channel", "prepare-text", "reconstruct", "output-audit"):
        assert next(m for n, m, _ in jobs if n == name) == run.MODULE
    assert {"receiver/condition_collision_trace.json", "receiver/short_schedule_trace.json"} <= set(
        next(products for name, _, products in jobs if name == "reconstruct"))


def judgment():
    return dict(caption_facts=[dict(id="f0", status="reproduced")], caption_issues=[],
        reconstruction_issues=[dict(type="Missing", claim="The person is absent.", evidence_frames=[0], reason="Visible in source only.")],
        uncertainties=[])


def test_semantic_review_requires_real_evidence_and_caption_quotes():
    value = judgment()
    facts, ids, caps = [dict(id="f0")], [0, 8], [dict(caption="A person walks.")]
    review.validate_judgment(value, facts, ids, caps)
    value["caption_issues"] = [dict(type="Added", claim="A cat", evidence_frames=[0], reason="No cat")]
    with pytest.raises(ValueError, match="quote"):
        review.validate_judgment(value, facts, ids, caps)
    value["caption_issues"] = []
    value["reconstruction_issues"][0]["evidence_frames"] = [999]
    with pytest.raises(ValueError, match="evidence"):
        review.validate_judgment(value, facts, ids, caps)
    value = judgment(); value["caption_facts"].append(dict(id="f0", status="missing"))
    with pytest.raises(ValueError, match="duplicate"):
        review.validate_judgment(value, facts, ids, caps)


def test_summary_keeps_unobservable_facts_out_of_missing_count():
    a, b = judgment(), judgment()
    b["caption_facts"][0]["status"] = "unobservable"
    metrics = review.summarize([dict(arm="internvl", judgment=a), dict(arm="event_pairs", judgment=b)])
    assert metrics["event_pairs"]["facts_missing"] == 0
    assert metrics["event_pairs"]["facts_unobservable"] == 1
    assert metrics["internvl"]["reconstruction_issue_counts"] == dict(Added=0, Missing=1, Distorted=0)


@pytest.fixture
def caption_comparison(tmp_path, monkeypatch):
    from PIL import Image
    from semantic_transmission import qwen35_backend
    source = tmp_path / "source"
    source.mkdir()
    for i in range(80):
        Image.new("RGB", (4, 4), (i, 0, 0)).save(source / f"{i}.png")
    video = source / "source.mp4"
    video.write_bytes(b"synthetic video")
    p = dict(signature="test", source=dict(source_frames=str(source), frames=80,
        normalized_video=str(video), normalized_sha256=sha256(video),
        source_frame_hashes={str(i): sha256(source / f"{i}.png") for i in range(80)}),
        keys=dict(internvl=[*range(78), 79], event_pairs=[*range(39), *range(40, 80)]),
        caption=dict(caption_adapter=dict(id="synthetic"), prompts=dict(caption="Describe", review="Review: "),
                     caption_max_words=79))
    calls = []
    class CaptionModel(FakeVLM):
        def __init__(self, config):
            self.config = config
        def generate(self, images, labels, prompt, tokens, task):
            calls.append((task, labels, prompt, tokens))
            result = dict(caption="A car is visible.", uncertainties=[])
            if task == "visual_review":
                result["removed_claims"] = []
            return dict(raw_text=json.dumps(result), truncated=False, seconds=.1,
                model_runtime=dict(model_id=run.MODEL_ID, revision=run.REVISION, quantization="nf4",
                                   **task_runtime(self.config, task)))
    monkeypatch.setattr(qwen35_backend, "LocalQwen35", CaptionModel)
    monkeypatch.setattr(run, "load_protocol", lambda _: p)
    root = tmp_path / "experiment"
    run.caption_worker(root)
    return root, p, calls


def test_caption_both_arms_same_roles_and_import_without_relabeling(caption_comparison):
    root, p, calls = caption_comparison
    assert len(calls) == 78*2*2
    assert {tokens for _, _, _, tokens in calls} == {384}
    assert {prompt for role, _, prompt, _ in calls if role == "draft"} == {"Describe"}
    for arm in run.ARMS:
        bundle = run.validate_bundle(root, arm, p)
        assert len(bundle["captions"]) == 78
        dest = root / f"reconstruction/{arm}"
        run.initialize(dest, p["source"], dict(indices=p["keys"][arm], selector=arm), {})
        write_json(dest / "run/semantic_clips_audit.json", dict(status="PASSED", policy="frame_exact",
                                                              source=dict(keyframes=p["keys"][arm])))
        run.import_captions(root, arm)
        rows = ae.read(dest / "run/captions.json")
        assert [r["text"] for r in rows] == [r["caption"] for r in bundle["captions"]]
        provenance = ae.read(dest / "run/caption_provenance.json")
        assert provenance["caption_adapter"] == p["caption"]["caption_adapter"]
        assert provenance["assistant_authored"] is False and provenance["pllava_inference_executed"] is False
    run.caption_worker(root)
    assert len(calls) == 78*2*2  # Restart reuses all validated individual calls.


@pytest.mark.parametrize("mutation", ["caption", "sample", "prompt", "raw", "adapter", "model"])
def test_import_rejects_tampered_caption_chain(caption_comparison, mutation):
    root, p, _ = caption_comparison
    path = root / ("captions/event_pairs.json" if mutation in {"caption", "sample"}
                   else "calls/event_pairs/review/00000.json")
    obj = ae.unseal(path)
    if mutation == "caption":
        obj["captions"][0]["caption"] = "An invented person."
    elif mutation == "sample":
        obj["captions"][0]["source_indices"] = [7, 7, 7, 7]
    elif mutation == "prompt":
        obj["identity"]["prompt_sha256"] = "changed"
    elif mutation == "raw":
        obj["raw_text"] = "{}"
    elif mutation == "adapter":
        obj["model_runtime"]["caption_adapter"] = p["caption"]["caption_adapter"]
    else:
        obj["model_runtime"]["model_id"] = "a different model"
    write_json(path, ae.seal(obj))
    with pytest.raises(ValueError):
        run.validate_bundle(root, "event_pairs", p)


def test_model_bytes_verification_rejects_same_size_corruption(tmp_path):
    file = tmp_path / "weights"
    file.write_bytes(b"good")
    p = dict(base_model_files={str(file): dict(sha256=sha256(file))})
    run.verify_model_bytes(tmp_path, p)
    file.write_bytes(b"evil")
    with pytest.raises(ValueError, match="weight hash"):
        run.verify_model_bytes(tmp_path, p)


def test_semantic_input_identity_changes_with_actual_video_or_caption(tmp_path, monkeypatch):
    video = tmp_path / "candidate.mp4"
    video.write_bytes(b"original")
    reference = tmp_path / "reference.json"
    write_json(reference, {})
    for arm in run.ARMS:
        write_json(tmp_path / f"captions/{arm}.json", {"caption": "first"})
    p = dict(signature="test", reference=str(reference))
    monkeypatch.setattr(review, "reconstructed_video", lambda *_: video)
    before = review.semantic_inputs(tmp_path, p)
    video.write_bytes(b"different video")
    assert before != review.semantic_inputs(tmp_path, p)
    video.write_bytes(b"original")
    write_json(tmp_path / "captions/event_pairs.json", {"caption": "changed caption"})
    assert before != review.semantic_inputs(tmp_path, p)


def test_caption_only_report_does_not_claim_completed_reconstruction(caption_comparison):
    root, p, _ = caption_comparison
    p["reference"] = str(root / "reference.json")
    p["scope"], p["noise_scope"] = "fixture", "fixture"
    write_json(root / "reference.json", dict(events=[]))
    review.report(root, p)
    result = ae.read(root / "RESULT.json")
    assert result["status"] == "CAPTIONS_COMPLETE_RECONSTRUCTION_PENDING"
    assert result["independent_semantic_review"] == "PENDING"
    assert result["hallucination_mitigation_verified"] is False
    assert result["semantic_diagnostic"] is None
    assert (root / "report.html").exists()
