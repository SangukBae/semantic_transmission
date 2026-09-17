import torch
import json
from pathlib import Path
import pytest

from semantic_transmission.internvl_memory import compact_attention_output, compact_kv_cache


def test_prefill_cache_releases_fused_projection_storage_without_changing_values():
    # InternLM2 uses grouped query heads; V initially shares the entire QKV buffer.
    fused = torch.arange(2 * 7 * 3 * 6 * 4, dtype=torch.float32).reshape(2, 7, 3, 6, 4)
    value = fused[:, :, :, -1, :].transpose(1, 2)
    key = torch.randn_like(value).contiguous()
    compact_key, compact_value = compact_kv_cache((key, value))
    assert compact_key is key
    assert torch.equal(compact_value, value)
    assert compact_value.untyped_storage().nbytes() == value.numel() * value.element_size()
    assert compact_value.untyped_storage().nbytes() < value.untyped_storage().nbytes()
    assert compact_value.untyped_storage().data_ptr() != value.untyped_storage().data_ptr()
    next_value = torch.zeros(2, 3, 1, 4)
    assert torch.equal(torch.cat((value, next_value), dim=2),
                       torch.cat((compact_value, next_value), dim=2))


def test_attention_output_and_ordinary_decode_cache_are_not_copied():
    key = torch.randn(1, 2, 4, 3, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    attention = torch.randn(1, 4, 6, dtype=torch.bfloat16)
    weights = object()
    output, same_weights, (new_key, new_value) = compact_attention_output(None, (), (attention, weights, (key, value)))
    assert output is attention and same_weights is weights
    assert new_key is key and new_value is value
    assert compact_kv_cache(None) is None
    assert compact_attention_output(None, (), (attention, None, None)) == (attention, None, None)


@pytest.mark.parametrize("enabled", [False, True])
def test_selector_worker_forwards_cache_option_only_when_requested(tmp_path, monkeypatch, enabled):
    from semantic_transmission import cli, workers
    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs/webvid5.json").read_text())
    cfg.update(models={"internvl": "/fixture/model"}, frames=3, internvl_compact_kv_cache=enabled)
    monkeypatch.setattr(cli, "settings", lambda _: {"internvl_python": "/fixture/python"})
    commands = []
    def fake_selector(command, **kwargs):
        commands.append(command)
        target = tmp_path / "data/frames/sample" / cfg["method"]
        target.mkdir(parents=True)
        for i in (0, 2):
            (target / f"{i}.png").write_bytes(b"fixture")
    monkeypatch.setattr(workers.subprocess, "run", fake_selector)
    workers.select(cfg, tmp_path, tmp_path)
    assert ("--compact-kv-cache" in commands[0]) is enabled
    assert json.loads((tmp_path / "keyframes.json").read_text())["indices"] == [0, 2]
