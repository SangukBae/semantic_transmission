"""Data leakage, exact hard selection and the caption-to-selector gradient path."""
import copy
import json
from pathlib import Path

import pytest

from semantic_transmission.qwen35_training import parsed_selection
from semantic_transmission.qwen35_training_data import validate_record


def row():
    return {"start": 0, "end_exclusive": 8, "frames": [{"index": i} for i in range(8)],
            "caption": "A hand briefly appears.", "review_status": "AI_SOURCE_REVIEWED",
            "events": [{"before": [1, 2], "after": [3, 4], "start_frame": 2, "end_frame": 3}],
            "facts": [{"text": "A hand is visible.", "evidence": [3, 4]}]}


def test_annotations_require_chronological_evidence_and_review():
    good = row()
    validate_record(good)
    for mutate in [lambda r: r.update(review_status="AI_UNREVIEWED"),
                   lambda r: r["facts"][0].update(evidence=[]),
                   lambda r: r["events"][0].update(after=[0]),
                   lambda r: r["events"][0].update(end_frame=8),
                   lambda r: r["frames"].reverse()]:
        bad = copy.deepcopy(good)
        mutate(bad)
        with pytest.raises(ValueError):
            validate_record(bad)


@pytest.mark.parametrize("keys", [[0, 1, 1, 7], [0, 2, 5, 8], [1, 2, 5, 7], [0, 5, 2, 7], [0, 2, 7], [0, True, 5, 7]])
def test_generated_selection_never_silently_changes_budget(keys):
    with pytest.raises(ValueError):
        parsed_selection(json.dumps({"keys": keys}))


def test_caption_gradient_reaches_both_heads_without_soft_frames_forward():
    torch = pytest.importorskip("torch")
    from semantic_transmission.qwen35_multitask import TemporalSelector
    torch.manual_seed(14)
    head = TemporalSelector(width=16, hidden=8)
    visual = torch.randn(8, 5, 16)
    pred = head(visual.mean(1))
    selected = head.select_tokens(visual, pred)
    assert torch.equal(selected, visual[pred["indices"]])
    assert pred["indices"].tolist() == sorted(set(pred["indices"].tolist()))
    assert pred["indices"][0] == 0 and pred["indices"][-1] == 7
    # A downstream text objective sees selected visual evidence; no auxiliary loss.
    decoder = torch.nn.Linear(16, 11)
    logits = decoder(selected.mean(1)).mean(0, keepdim=True)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([3]))
    grads = torch.autograd.grad(loss, [head.selection.weight, head.candidate.weight])
    assert all(torch.isfinite(g).all() and g.norm() > 0 for g in grads)


def test_event_reward_requires_both_sides_and_optimizes_nonendpoint_evidence():
    torch = pytest.importorskip("torch")
    from semantic_transmission.qwen35_multitask import TemporalSelector, coverage, supervised_head_loss
    r = row()
    assert coverage(r, [0, 1, 2, 7])["event_hits"] == 0
    assert coverage(r, [0, 2, 3, 7])["event_hits"] == 1
    torch.manual_seed(8)
    head = TemporalSelector(width=16, hidden=8)
    features = torch.randn(8, 16)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.01)
    initial = float(supervised_head_loss(head, head(features), r)[0].detach())
    for _ in range(15):
        optimizer.zero_grad()
        loss, _ = supervised_head_loss(head, head(features), r)
        loss.backward(); optimizer.step()
    assert float(supervised_head_loss(head, head(features), r)[0].detach()) < initial
    assert coverage(r, head(features)["indices"].tolist())["event_hits"] == 1


def test_checkpoint_restores_optimizer_rng_and_detects_corruption(tmp_path):
    torch = pytest.importorskip("torch")
    from semantic_transmission.qwen35_training import save_checkpoint, checkpoint_path
    torch.manual_seed(9)
    head = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.03)
    def step(h, opt):
        opt.zero_grad(); loss = h(torch.randn(4, 3)).square().mean(); loss.backward(); opt.step()
    step(head, optimizer)
    state = {"updates": 1, "next_epoch": 0, "next_offset": 4}
    ckpt = save_checkpoint(tmp_path, head, None, optimizer, state, best=True)
    assert checkpoint_path(tmp_path) == ckpt
    clone = torch.nn.Linear(3, 2)
    clone.load_state_dict(torch.load(ckpt / "head.pt", weights_only=True))
    other = torch.optim.AdamW(clone.parameters(), lr=.03)
    other.load_state_dict(torch.load(ckpt / "optimizer.pt", weights_only=True))
    rng = torch.load(ckpt / "rng.pt", weights_only=True)
    torch.set_rng_state(rng["torch"]); step(head, optimizer)
    torch.set_rng_state(rng["torch"]); step(clone, other)
    assert all(torch.equal(a, b) for a, b in zip(head.parameters(), clone.parameters()))
    (ckpt / "state.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        checkpoint_path(tmp_path)


def test_verify_rejects_original_video_overlap(tmp_path, monkeypatch):
    import semantic_transmission.qwen35_training_data as data
    monkeypatch.setattr(data, "exclusion_keys", lambda: (set(), set(), []))
    source = tmp_path / "source.mp4"; source.write_bytes(b"original video placeholder")
    digest = data.sha256(source)
    sheet = tmp_path / "review" / "v.jpg"; sheet.parent.mkdir(); sheet.write_bytes(b"review")
    r = row() | {"source_id": "same-original", "source_sha256": digest, "source_video": str(source),
                 "normalized_video": str(source), "normalized_sha256": digest, "video_id": "v",
                 "review_sheet_sha256": data.sha256(sheet)}
    for frame in r["frames"]:
        frame.update(path=str(source), sha256=digest)
    data.write_json(tmp_path / "source_split.json", {"exclusion_evidence": []})
    files = ["source_split.json"]
    for split in ["train", "validation", "test"]:
        data.write_json(tmp_path / f"{split}.json", {"records": [r | {"split": split}] if split != "test" else []})
        files.append(f"{split}.json")
    data.write_json(tmp_path / "freeze.json", {"files": {f: data.sha256(tmp_path / f) for f in files}, "counts": {}})
    with pytest.raises(ValueError, match="overlap"):
        data.verify(tmp_path)
