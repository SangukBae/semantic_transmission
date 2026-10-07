import json
from pathlib import Path

import pytest

from semantic_transmission import auto_extraction as extraction
from semantic_transmission import qwen35_conditions as conditions
from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.assisted_captions import sample_indices
from semantic_transmission.hybrid_selection import choose_keys
from semantic_transmission.qwen35_backend import MODEL_ID, REVISION


@pytest.fixture(params=[False, True], ids=["base", "caption_adapter"])
def extracted(tmp_path, request, monkeypatch):
    """Real signed extraction chain with synthetic frame bytes; no GPU required."""
    root, prepared = tmp_path / "extracted", tmp_path / "prepared"
    source_dir = prepared / "webvid/test"
    frames = source_dir / "frames"
    frames.mkdir(parents=True)
    for i in range(25):
        (frames / f"{i}.png").write_bytes(f"source frame {i}".encode())
    video = source_dir / "video.mp4"
    video.write_bytes(b"synthetic source video")
    source = dict(id="test", dataset="WebVid", fps=24, frames=25,
        source_frames=str(frames), source_frame_hashes={str(i): sha256(frames / f"{i}.png") for i in range(25)},
        source_path=str(video), source_sha256=sha256(video),
        normalized_video=str(video), normalized_sha256=sha256(video))
    prepared_record = source_dir / "source_prepared.json"
    write_json(prepared_record, source)
    config = dict(backend="qwen35", model_id=MODEL_ID, model_revision=REVISION, quantization="nf4",
        caption_max_words=79, code={}, transnet={}, max_gap_frames=24, psss_threshold=.35,
        prepared_root=str(prepared), videos=[dict(id="test", dataset="WebVid", prepared_sha256=sha256(prepared_record))])
    if request.param:
        from semantic_transmission import qwen35_caption_adapter as adapter
        # File pinning is tested separately; this fixture tests the signed chain.
        monkeypatch.setattr(adapter, "verify_adapter", lambda _: None)
        config.update(caption_adapter=dict(id="synthetic-caption", files={"weights": "test"}),
            candidate_stride=4, candidate_window_frames=48)
    config["signature"] = extraction.digest(config)
    write_json(root / "run_config.json", config)
    dest = root / "webvid/test"
    candidates = [dict(frame=0, mandatory=True), dict(frame=24, mandatory=True)]
    write_json(dest / "proposals.json", extraction.seal(dict(config_signature=config["signature"], candidates=candidates)))
    if request.param:
        from semantic_transmission.qwen35_caption_adapter import task_runtime
        ids = next(extraction.proposal_windows(25))
        identity = dict(config_signature=config["signature"], task="proposals", indices=ids,
            frame_hashes=[source["source_frame_hashes"][str(i)] for i in ids])
        write_json(dest / "observations/00000.json", extraction.seal(dict(identity=identity,
            model_runtime=task_runtime(config, "proposals"))))
    keys, records = choose_keys(25, [0, 24], [0, 24], lambda *_: pytest.fail("unexpected SKEM call"))
    selection_id = dict(config_signature=config["signature"], proposals_sha256=sha256(dest / "proposals.json"))
    write_json(dest / "selection.json", extraction.seal(dict(identity=selection_id, indices=keys,
        records=records, keyframe_png_bytes=1)))
    write_json(dest / "keyframes.json", dict(indices=keys))
    selected = dest / "selected_frames"
    selected.mkdir()
    for i in keys:
        (selected / f"{i:05d}.png").write_bytes((frames / f"{i}.png").read_bytes())
    identity = dict(config_signature=config["signature"], selection_sha256=sha256(dest / "selection.json"))
    ids = sample_indices(0, 24)
    item_id = dict(identity, segment=0, start=0, end_exclusive=24, source_indices=ids,
                   frame_hashes=[source["source_frame_hashes"][str(i)] for i in ids])
    draft = dict(caption="A red object is visible on the left.", uncertainties=[])
    review = dict(draft, removed_claims=[])
    runtime = dict(model_id=MODEL_ID, revision=REVISION, quantization="nf4")
    for name, task_id, result in [
        ("captions_draft", dict(item_id, task="draft"), draft),
        ("captions_review", dict(item_id, task="visual_review", draft_sha256=extraction.digest(draft)), review)]:
        role_runtime = dict(runtime)
        if request.param:
            role_runtime.update(task_runtime(config, task_id["task"]))
        write_json(dest / name / "00000.json", extraction.seal(dict(identity=task_id, result=result,
            raw_text=json.dumps(result), truncated=False, model_runtime=role_runtime)))
    row = dict(segment=0, start=0, end_exclusive=24, source_indices=ids, **review)
    write_json(dest / "captions.json", extraction.seal(dict(identity=identity, author=MODEL_ID,
        model_revision=REVISION, caption_adapter=config.get("caption_adapter"), captions=[row])))
    return root, config, source, dest


def test_qwen_import_preserves_text_sampling_and_actual_authorship(extracted, tmp_path):
    root, config, source, dest = extracted
    before = {p: sha256(p) for p in root.rglob("*") if p.is_file()}
    conditions.validated(root, "test")
    assert before == {p: sha256(p) for p in root.rglob("*") if p.is_file()}
    run = tmp_path / "run"
    frames = run / "data/frames/sample"
    frames.mkdir(parents=True)
    for i in sample_indices(0, 24):
        (frames / f"{i}.png").symlink_to(Path(source["source_frames"]) / f"{i}.png")
    (run / "data/normalized.mp4").symlink_to(source["normalized_video"])
    write_json(run / "run_config.json", dict(caption_provider=conditions.PROVIDER,
        condition_extraction_root=str(root), condition_video_id="test", condition_signature=config["signature"],
        condition_captions_sha256=sha256(dest / "captions.json")))
    write_json(run / "keyframes.json", dict(indices=[0, 24]))
    write_json(run / "semantic_clips_audit.json", dict(status="PASSED", policy="frame_exact", source=dict(keyframes=[0, 24])))
    conditions.import_captions(run)
    provenance = extraction.read(run / "caption_provenance.json")
    assert provenance["provider"] == "local_qwen35"
    assert provenance["model_id"] == MODEL_ID and provenance["quantization"] == "nf4"
    assert provenance["assistant_authored"] is False and provenance["pllava_inference_executed"] is False
    assert provenance["caption_adapter"] == config.get("caption_adapter")
    assert extraction.read(run / "captions.json")[0]["text"] == "A red object is visible on the left."
    assert extraction.read(run / "caption_sampling.json")[0]["source_indices"] == sample_indices(0, 24)
    write_json(run / "keyframes.json", dict(indices=[0, 12, 24]))
    with pytest.raises(ValueError, match="keyframes differ"):
        conditions.import_captions(run)


@pytest.mark.parametrize("mutation,match", [
    ("author", "caption model provenance"), ("runtime", "inference model provenance"),
    ("raw", "raw model response"), ("truncated", "raw model response"),
    ("timeline", "timeline mismatch"), ("source", "source frame changed")])
def test_relabeling_and_corrupt_model_evidence_are_rejected(extracted, mutation, match):
    root, config, source, dest = extracted
    if mutation in {"author", "timeline"}:
        path = dest / "captions.json"
        row = extraction.unseal(path)
        if mutation == "author":
            row["author"] = "assistant_provided"
        else:
            row["captions"][0]["source_indices"] = [0, 1, 2, 3]
    elif mutation == "source":
        i = sample_indices(0, 24)[0]
        (Path(source["source_frames"]) / f"{i}.png").write_bytes(b"changed frame")
        with pytest.raises(ValueError, match=match):
            conditions.validated(root, "test")
        return
    else:
        path = dest / "captions_review/00000.json"
        row = extraction.unseal(path)
        if mutation == "runtime":
            row["model_runtime"]["model_id"] = "Qwen/Qwen3-VL-8B-Instruct"
        elif mutation == "raw":
            row["raw_text"] = '{"caption":"Another object.","uncertainties":[],"removed_claims":[]}'
        else:
            row["truncated"] = True
    write_json(path, extraction.seal(row))
    with pytest.raises(ValueError, match=match):
        conditions.validated(root, "test")


def test_qwen_word_limit_is_under_80_without_changing_legacy_policy():
    value = dict(caption=" ".join(["word"] * 80), uncertainties=[])
    extraction.caption_validator({})(value)
    with pytest.raises(ValueError, match="word limit"):
        extraction.caption_validator(dict(caption_max_words=79))(value)


def test_full_job_plan_uses_qwen_import_and_both_receiver_repairs():
    from semantic_transmission.fc_lgvsc_qwen35 import MODULE, job_plan
    jobs = {name: (module, products) for name, module, products in job_plan()}
    assert jobs["caption"][0] == MODULE
    assert "caption_provenance.json" in jobs["caption"][1]
    assert jobs["prepare-text"][0] == MODULE
    assert {"receiver/condition_collision_trace.json", "receiver/short_schedule_trace.json"} <= set(jobs["reconstruct"][1])


@pytest.mark.parametrize("stage", ["captions_draft", "captions_review", "observations"])
def test_wrong_adapter_role_or_weight_identity_rejected(extracted, stage):
    root, config, source, dest = extracted
    if not config.get("caption_adapter"):
        return
    path = dest / stage / "00000.json"
    record = extraction.unseal(path)
    record["model_runtime"]["caption_adapter"] = (
        None if stage == "captions_draft" else config["caption_adapter"])
    write_json(path, extraction.seal(record))
    with pytest.raises(ValueError, match="adapter/task provenance"):
        conditions.validated(root, "test")
