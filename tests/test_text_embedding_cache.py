import importlib.util
from pathlib import Path
import sys
import types

import pytest
torch = pytest.importorskip("torch")

from semantic_transmission import text_embedding_cache as cache
from semantic_transmission.artifacts import write_json
from semantic_transmission.webvid5 import fingerprint


@pytest.fixture
def prepared(tmp_path):
    contract = dict(policy=cache.POLICY, max_length=3, output_dim=2)
    calls = []
    class Encoder:
        def encode(self, prompts):
            calls.extend(prompts)
            return dict(y=torch.full((1, 1, 3, 2), len(prompts[0]), dtype=torch.bfloat16),
                        mask=torch.tensor([[1, 1, 0]], dtype=torch.int64))
    report = cache.populate(["first", "second", "first"], contract, tmp_path, Encoder)
    report["status"] = "PASSED"
    return report, contract, calls, Encoder


def test_cold_and_warm_cache_reuse_exact_tensors_without_model_or_rng_changes(prepared, tmp_path):
    report, contract, calls, _ = prepared
    assert calls == ["first", "second"]
    assert report["cache_misses"] == 2 and report["cache_hits"] == 0
    assert report["legacy_cpu_fp32_equivalence_verified"] is False
    state = torch.random.get_rng_state().clone()
    warm = cache.populate(report["prompts"], contract, tmp_path,
                          lambda: pytest.fail("warm cache loaded the T5 model"))
    assert warm["cache_hits"] == 2 and warm["cache_misses"] == 0
    assert warm["model_loaded"] is False and warm["model_load_seconds"] == 0
    assert torch.equal(state, torch.random.get_rng_state())
    assert warm["entries"] == report["entries"]


def test_cache_miss_precision_and_model_changes_invalidate_reuse(prepared, tmp_path):
    report, contract, calls, factory = prepared
    second = cache.populate(["first", "new"], contract, tmp_path, factory)
    assert second["cache_hits"] == 1 and second["cache_misses"] == 1
    assert calls == ["first", "second", "new"]
    changed = cache.populate(["first"], dict(contract, checkpoint="new hash"), tmp_path, factory)
    assert changed["cache_misses"] == 1 and changed["signature"] != report["signature"]


def test_interruption_resumes_completed_prompts_without_reencoding_them(tmp_path):
    calls = []
    class Encoder:
        def encode(self, texts):
            calls.extend(texts)
            if texts == ["second"]:
                raise RuntimeError("simulated interruption")
            return dict(y=torch.ones((1,1,3,2), dtype=torch.bfloat16), mask=torch.ones((1,3), dtype=torch.int64))
    contract = dict(max_length=3, output_dim=2)
    with pytest.raises(RuntimeError, match="interruption"):
        cache.populate(["first", "second"], contract, tmp_path, Encoder)
    report = cache.populate(["first"], contract, tmp_path, lambda: pytest.fail("lost completed prompt"))
    assert report["cache_hits"] == 1


def test_corrupt_cache_fails_before_decoder_can_use_it(prepared):
    report, _, _, _ = prepared
    Path(report["entries"][0]["path"]).write_bytes(b"damaged tensor file")
    with pytest.raises(ValueError, match="hash mismatch"):
        cache.CachedTextEncoder(report)


@pytest.mark.parametrize("change", ["nan", "shape", "mask", "dtype"])
def test_invalid_encoder_outputs_are_never_saved(tmp_path, change):
    tensors = dict(y=torch.ones((1,1,3,2), dtype=torch.bfloat16), mask=torch.ones((1,3), dtype=torch.int64))
    if change == "nan":
        tensors["y"][0,0,0,0] = float("nan")
    elif change == "shape":
        tensors["y"] = tensors["y"][:,:,:2]
    elif change == "mask":
        tensors["mask"][0,0] = 2
    else:
        tensors["y"] = tensors["y"].float()
    with pytest.raises(ValueError, match="invalid T5"):
        cache.populate(["x"], dict(max_length=3, output_dim=2), tmp_path,
                       lambda: types.SimpleNamespace(encode=lambda _: tensors))
    assert not list(tmp_path.rglob("*.json"))


def test_adapter_keeps_cfg_null_condition_and_prevents_scheduler_mutation(prepared, tmp_path):
    report, _, _, _ = prepared
    adapter = cache.CachedTextEncoder(report, tmp_path / "trace.json")
    adapter.y_embedder = types.SimpleNamespace(y_embedding=torch.arange(6).reshape(3,2))
    assert adapter.null(2).shape == (2,1,3,2)
    assert torch.equal(adapter.null(1)[0,0], adapter.y_embedder.y_embedding)
    args = adapter.encode(["second", "first"])
    assert args["y"].shape == (2,1,3,2)
    args["y"].zero_()
    assert adapter.encode(["second"])["y"].sum() > 0
    with pytest.raises(ValueError, match="unprepared"):
        adapter.encode(["unknown"])
    trace = cache.read_json(tmp_path / "trace.json")
    assert trace["calls"] == 3 and trace["t5_model_loaded_in_decoder"] is False


def test_full_usage_audit_requires_every_prompt_in_order(prepared, tmp_path):
    report, _, _, _ = prepared
    csv = tmp_path / "receiver/metadata.csv"
    csv.parent.mkdir()
    csv.write_text("path,text,flow\nclips/a/0.mp4,caption,0.0\n")
    report["metadata_sha256"] = cache.sha256(csv)
    write_json(tmp_path / cache.REPORT, report)
    trace = dict(policy=cache.POLICY, calls=3, prompt_sha256=[fingerprint(p) for p in report["prompts"]],
                 t5_model_loaded_in_decoder=False)
    write_json(tmp_path / cache.TRACE, trace)
    cache.validate_usage(tmp_path)
    trace["prompt_sha256"].reverse()
    trace["prompt_sha256"][0] = "wrong"
    write_json(tmp_path / cache.TRACE, trace)
    with pytest.raises(ValueError, match="in order"):
        cache.validate_usage(tmp_path)


def test_registry_adapter_never_builds_t5_and_forwards_other_models(prepared, tmp_path, monkeypatch):
    report, _, _, _ = prepared
    report["contract"].update(model_path=str(tmp_path), model_inventory={}, code={})
    report["signature"] = fingerprint(report["contract"])
    for entry in report["entries"]:
        entry["contract"] = report["signature"]
    write_json(tmp_path / "report.json", report)
    monkeypatch.setattr(cache, "model_inventory", lambda _: {})
    calls = []
    registry = types.SimpleNamespace(build_module=lambda *a, **k: calls.append((a,k)))
    monkeypatch.setitem(sys.modules, "opensora", types.SimpleNamespace(registry=registry))
    cache.install_cached_encoder(tmp_path / "report.json", tmp_path / "trace.json", tmp_path)
    config = dict(type="t5", from_pretrained=str(tmp_path), model_max_length=3)
    with pytest.raises(ValueError, match="configuration"):
        registry.build_module(dict(config, model_max_length=2), None, device="cpu")
    with pytest.raises(ValueError, match="CPU tensor"):
        registry.build_module(config, None, device="cuda")
    adapter = registry.build_module(config, None, device="cpu")
    assert adapter.output_dim == 2 and not calls
    registry.build_module({"type": "STDiT3-XL/2"}, None, device="cuda")
    assert len(calls) == 1


def test_script_installs_cache_before_entering_existing_tail17_decoder(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("cached_entry", repo / "scripts/etri_t5_cached_decoder.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    monkeypatch.setenv("ETRI_TEXT_EMBEDDINGS", str(tmp_path / "cache.json"))
    monkeypatch.setenv("ETRI_TEXT_EMBEDDING_TRACE", str(tmp_path / "trace.json"))
    calls = []
    monkeypatch.setattr(entry, "install_cached_encoder", lambda *args: calls.append("cache"))
    def run(path, run_name):
        assert calls == ["cache"] and Path(path).name == "etri_tail17_decoder.py"
        calls.append("decoder")
    monkeypatch.setattr(entry.runpy, "run_path", run)
    entry.main()
    assert calls == ["cache", "decoder"]
