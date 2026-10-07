from contextlib import contextmanager
import json
from types import SimpleNamespace

import pytest

from semantic_transmission import auto_extraction as extraction
from semantic_transmission import qwen35_caption_adapter as adapter
from semantic_transmission.artifacts import sha256
from semantic_transmission.qwen35_backend import LocalQwen35


@pytest.fixture
def pinned(tmp_path, monkeypatch):
    path = tmp_path / "adapter"
    path.mkdir()
    (path / "adapter_model.safetensors").write_bytes(b"synthetic test weights")
    (path / "adapter_config.json").write_text(json.dumps(dict(r=32, lora_alpha=64,
        peft_type="LORA", bias="none", modules_to_save=None)))
    monkeypatch.setattr(adapter, "ADAPTER", path)
    monkeypatch.setattr(adapter, "FILES", {p.name: sha256(p) for p in path.iterdir()})
    return adapter.default_adapter()


def test_pinned_adapter_rejects_changed_weights_or_roles(pinned):
    with pytest.raises(ValueError, match="policy"):
        adapter.verify_adapter(dict(pinned, active_tasks=["draft", "proposals"]))
    (adapter.ADAPTER / "adapter_model.safetensors").write_bytes(b"replaced")
    with pytest.raises(ValueError, match="bytes changed"):
        adapter.verify_adapter(pinned)


def test_init_enables_adapter_by_default_and_base_is_explicit(pinned, monkeypatch, tmp_path):
    from semantic_transmission import fc_lgvsc_qwen35 as fc
    calls = []
    monkeypatch.setattr(extraction, "initialize", lambda *a, **kw: calls.append((a, kw)))
    fc.main(["init"])
    assert calls[-1][0][0] == fc.DEFAULT_ROOT
    assert calls[-1][1]["caption_adapter"] == pinned
    fc.main(["init", "--root", str(tmp_path), "--base-caption"])
    assert calls[-1][1]["caption_adapter"] is None
    with pytest.raises(SystemExit):
        fc.main(["extract", "--base-caption"])


def test_one_backbone_roles_restore_norms_and_disable_lora_even_on_failure(pinned):
    import torch

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16), requires_grad=False)
            self.enabled = True

        def set_adapter(self, name):
            assert name == adapter.NAME
            self.enabled = True
            self.requires_grad_(True)  # PEFT's set_adapter behavior.

        @contextmanager
        def disable_adapter(self):
            self.enabled = False
            try:
                yield
            finally:
                self.enabled = True
                self.requires_grad_(True)

    vlm = LocalQwen35.__new__(LocalQwen35)
    vlm.config = dict(caption_adapter=pinned)
    vlm.adapter_loaded = True
    vlm.model = Model()
    for task in ("proposals", "draft", "visual_review", "draft", "proposals"):
        with vlm.inference_role(task) as runtime:
            active = task == "draft"
            assert vlm.model.enabled == active
            assert vlm.model.norm.dtype == (torch.float32 if active else torch.bfloat16)
            assert runtime["caption_adapter"] == (pinned if active else None)
            assert not any(p.requires_grad for p in vlm.model.parameters())
        assert vlm.model.norm.dtype == torch.bfloat16
        assert not any(p.requires_grad for p in vlm.model.parameters())
    with pytest.raises(RuntimeError):
        with vlm.inference_role("draft"):
            raise RuntimeError("failed generation")
    assert vlm.model.norm.dtype == torch.bfloat16
    with pytest.raises(ValueError, match="explicit inference task"):
        with vlm.inference_role(None):
            pass


def test_cache_rejects_base_caption_mislabeled_as_adapter(pinned, tmp_path):
    vlm = SimpleNamespace(config=dict(caption_adapter=pinned))
    identity = dict(task="draft")
    path = tmp_path / "draft.json"
    record = dict(identity=identity, result=dict(caption="A car.", uncertainties=[]),
        model_runtime=adapter.task_runtime(vlm.config, "draft"))
    record["model_runtime"]["caption_adapter"] = None
    extraction.write_json(path, extraction.seal(record))
    with pytest.raises(ValueError, match="adapter/task provenance"):
        extraction.infer_json(vlm, path, identity, [], [], "prompt", extraction.validate_caption)
