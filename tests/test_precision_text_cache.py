import types

import pytest
torch = pytest.importorskip("torch")

from semantic_transmission import precision_text_cache as cache


def contract(dtype="fp32"):
    return dict(policy=cache.POLICY, compute_dtype=dtype, max_length=3, output_dim=2)


def encoder(dtype, calls):
    def encode(texts):
        calls.extend(texts)
        return dict(y=torch.full((1,1,3,2), 0.12345, dtype=dtype),
                    mask=torch.tensor([[1,1,0]], dtype=torch.int64))
    return types.SimpleNamespace(encode=encode)


def test_fp32_is_calculated_stored_and_reused_without_bf16_rounding(tmp_path):
    calls = []
    report = cache.populate(["one", "two", "one"], contract(), tmp_path,
                            lambda: encoder(torch.float32, calls))
    assert calls == ["one", "two"]
    warm = cache.populate(report["prompts"], contract(), tmp_path,
                          lambda: pytest.fail("warm FP32 cache reloaded model"))
    assert warm["cache_hits"] == 2 and warm["cache_misses"] == 0
    adapter = cache.Encoder(report, tmp_path / "trace.json")
    output = adapter.encode(["one"])
    assert output["y"].dtype == torch.float32
    assert not torch.equal(output["y"], output["y"].bfloat16().float())
    output["y"].zero_()
    assert adapter.encode(["one"])["y"].sum() > 0
    # The old decoder also converted FP32 CPU conditions to its BF16 dtype.
    assert adapter.encode(["two"])["y"].to(torch.bfloat16).dtype == torch.bfloat16


def test_compute_precisions_use_separate_entries(tmp_path):
    fp32 = cache.populate(["one"], contract(), tmp_path, lambda: encoder(torch.float32, []))
    bf16 = cache.populate(["one"], contract("bf16"), tmp_path, lambda: encoder(torch.bfloat16, []))
    assert fp32["signature"] != bf16["signature"] and bf16["cache_misses"] == 1
    with pytest.raises(ValueError, match="precision"):
        cache.populate(["wrong"], contract(), tmp_path, lambda: encoder(torch.bfloat16, []))


def test_corrupt_cache_or_invalid_values_are_rejected(tmp_path):
    report = cache.populate(["one"], contract(), tmp_path, lambda: encoder(torch.float32, []))
    from pathlib import Path
    Path(report["entries"][0]["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        cache.Encoder(report, tmp_path / "trace.json")
    tensors = encoder(torch.float32, []).encode(["one"])
    tensors["y"].fill_(float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        cache.validate(tensors, contract())


def test_failed_preparation_keeps_completed_prompts(tmp_path):
    base = encoder(torch.float32, [])
    def fail(texts):
        if texts == ["two"]:
            raise RuntimeError("interrupted")
        return base.encode(texts)
    with pytest.raises(RuntimeError, match="interrupted"):
        cache.populate(["one", "two"], contract(), tmp_path, lambda: types.SimpleNamespace(encode=fail))
    result = cache.populate(["one"], contract(), tmp_path, lambda: pytest.fail("lost cached condition"))
    assert result["cache_hits"] == 1
