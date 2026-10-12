import math

import pytest

from semantic_transmission import qwen35_pairs_video as video
from semantic_transmission.artifacts import write_json, sha256


@pytest.mark.parametrize("frames,expected", [(2, 2), (25, 2), (494, 28), (1440, 79)])
def test_scaled_budget_preserves_original_density_and_receiver_coverage(frames, expected):
    count = video.key_budget(frames)
    assert count == expected
    assert math.ceil((frames - 1) / 24) + 1 <= count <= frames


def test_receiver_protocol_uses_new_source_and_selected_keys(tmp_path, monkeypatch):
    template = tmp_path / "template.json"
    write_json(template, dict(receiver=dict(frames=1440, max_frames=1440,
        selector="skem", selector_environment="internvl_release", internvl_8bit=False,
        method="old_internvl", input="old.mp4", snr_db=10, steps=30),
        interpreters={}, base_model_files={}))
    monkeypatch.setattr(video, "TEMPLATE", template)
    monkeypatch.setattr(video.pc, "load_protocol", lambda root: video.ae.read(root / "protocol.json"))
    for name in ("plan.json", "selection.json", "ledger.json"):
        write_json(tmp_path / name, {})
    source = dict(normalized_video="new.mp4", normalized_sha256="new-hash", frames=494,
                  fps=24, width=576, height=320)
    p = dict(source=source, policy="new", caption=dict(caption_adapter={}), code={},
             source_metadata="source.json", source_metadata_sha256="source-hash",
             template=str(template), template_sha256=sha256(template))
    keys = [0, 24, 493]
    result = video.make_protocol(tmp_path, p, keys)
    cfg = result["receiver"]
    assert (cfg["frames"], cfg["max_frames"], cfg["input"]) == (494, 494, "new.mp4")
    assert cfg["selector"] == video.SELECTOR
    assert "selector_environment" not in cfg and "internvl_8bit" not in cfg
    assert result["keys"] == {video.ARM: keys}
    assert (cfg["snr_db"], cfg["steps"]) == (10, 30)


def test_worker_uses_frozen_deployment_template_and_rejects_edits(tmp_path, monkeypatch):
    template = tmp_path / "deployment-template.json"
    write_json(template, dict(receiver=dict(snr_db=10), interpreters={"python": "new-env"},
                             base_model_files={}))
    monkeypatch.setattr(video, "TEMPLATE", tmp_path / "absent-historical-template.json")
    monkeypatch.setattr(video.pc, "load_protocol", lambda root: video.ae.read(root / "protocol.json"))
    for name in ("plan.json", "selection.json", "ledger.json"):
        write_json(tmp_path / name, {})
    p = dict(source=dict(normalized_video="new.mp4", normalized_sha256="hash", frames=25,
                         fps=24, width=576, height=320), policy="new", caption=dict(caption_adapter={}),
             code={}, source_metadata="source.json", source_metadata_sha256="source-hash",
             template=str(template), template_sha256=sha256(template))
    result = video.make_protocol(tmp_path, p, [0, 24])
    assert result["interpreters"]["python"] == "new-env"
    write_json(template, dict(receiver=dict(snr_db=8)))
    with pytest.raises(ValueError, match="frozen receiver template changed"):
        video.make_protocol(tmp_path, p, [0, 24])


def test_resume_rejects_changed_frozen_source_or_settings(tmp_path):
    path = tmp_path / "plan.json"
    video.freeze(path, {"source": "first"})
    video.freeze(path, {"source": "first"})
    with pytest.raises(ValueError, match="frozen input/code changed"):
        video.freeze(path, {"source": "different"})


def test_resume_accepts_json_tuple_roundtrip_but_rejects_changed_dimensions(tmp_path):
    path = tmp_path / "plan.json"
    value = {"generator": {"image_size": (320, 576)}, "code": {"decoder": "sha"}}
    video.freeze(path, value)
    original = path.read_bytes()
    video.freeze(path, value)
    video.freeze(path, {"generator": {"image_size": [320, 576]}, "code": {"decoder": "sha"}})
    with pytest.raises(ValueError, match="frozen input/code changed"):
        video.freeze(path, {"generator": {"image_size": (320, 512)}, "code": {"decoder": "sha"}})
    with pytest.raises(ValueError, match="frozen input/code changed"):
        video.freeze(path, {"generator": {"image_size": (320, 576)}, "code": {"decoder": "changed"}})
    assert path.read_bytes() == original
