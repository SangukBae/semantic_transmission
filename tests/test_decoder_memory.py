"""Regression checks against the pinned decoder's actual attention and MLP."""
import sys
from pathlib import Path

import pytest
import torch

from semantic_transmission import decoder_memory as memory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".local/vendor/Open-Sora"))
blocks = pytest.importorskip("opensora.models.layers.blocks")


def install(monkeypatch, **kwargs):
    monkeypatch.setattr(blocks.Attention, "forward", blocks.Attention.forward)
    monkeypatch.setattr(blocks.Mlp, "forward", blocks.Mlp.forward)
    return memory.install(**kwargs)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_batched_attention_keeps_full_context_kernel_and_rng(monkeypatch, dtype, causal, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA parity check requires a GPU")
    from rotary_embedding_torch import RotaryEmbedding
    torch.manual_seed(42)
    rope = RotaryEmbedding(8).to(device=device, dtype=dtype)
    model = blocks.Attention(32, num_heads=4, qk_norm=True,
        rope=rope.rotate_queries_or_keys, enable_flash_attn=True).eval().to(device=device, dtype=dtype)
    model.is_causal = causal
    x = torch.randn(17, 13, 32, dtype=dtype, device=device)
    with torch.no_grad():
        expected = model(x)
        state = torch.random.get_rng_state()
        record = install(monkeypatch, limit_bytes=4096, chunk_bytes=2048)
        actual = model(x)
    # GEMM batch sizes can change rounding. Bound it to dtype precision.
    eps = torch.finfo(dtype).eps
    torch.testing.assert_close(actual, expected, rtol=8*eps, atol=eps)
    assert torch.equal(state, torch.random.get_rng_state())
    assert model.enable_flash_attn is True
    assert record["attention_calls"] == 1
    assert record["max_attention_batch"] == 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mlp_token_chunks_match_original_and_keep_rng(monkeypatch, dtype, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA parity check requires a GPU")
    torch.manual_seed(42)
    model = blocks.Mlp(32, hidden_features=128, act_layer=blocks.approx_gelu).eval().to(device=device, dtype=dtype)
    x = torch.randn(2, 71, 32, dtype=dtype, device=device)
    with torch.no_grad():
        expected = model(x)
        state = torch.random.get_rng_state()
        record = install(monkeypatch, limit_bytes=4096, chunk_bytes=2048)
        actual = model(x)
    eps = torch.finfo(dtype).eps
    torch.testing.assert_close(actual, expected, rtol=8*eps, atol=eps)
    assert torch.equal(state, torch.random.get_rng_state())
    assert record["mlp_calls"] == 1


def test_small_inputs_keep_original_execution(monkeypatch):
    attention = blocks.Attention(32, num_heads=4).eval()
    mlp = blocks.Mlp(32, hidden_features=64).eval()
    x = torch.randn(2, 13, 32)
    with torch.no_grad():
        expected = mlp(attention(x))
        record = install(monkeypatch)
        actual = mlp(attention(x))
    assert torch.equal(actual, expected)
    assert record["attention_calls"] == record["mlp_calls"] == 0


def test_training_and_gradients_do_not_use_inference_chunks(monkeypatch):
    attention = blocks.Attention(32, num_heads=4)
    mlp = blocks.Mlp(32, hidden_features=64)
    record = install(monkeypatch, limit_bytes=32, chunk_bytes=16)
    x = torch.randn(2, 13, 32, requires_grad=True)
    mlp(attention(x)).sum().backward()
    assert torch.isfinite(x.grad).all()
    attention.eval(); mlp.eval()
    mlp(attention(x)).sum().backward()
    assert record["attention_calls"] == record["mlp_calls"] == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("legacy_norm", [False, True])
def test_flash_attention_chunks_projection_and_norm_without_splitting_context(
        monkeypatch, dtype, causal, legacy_norm):
    if not torch.cuda.is_available():
        pytest.skip("FlashAttention requires a GPU")
    flash_attn = pytest.importorskip("flash_attn")
    from rotary_embedding_torch import RotaryEmbedding
    torch.manual_seed(42)
    rope = RotaryEmbedding(8).cuda().to(dtype=dtype)
    model = blocks.Attention(32, num_heads=4, qk_norm=True,
        rope=rope.rotate_queries_or_keys, qk_norm_legacy=legacy_norm,
        enable_flash_attn=True).eval().cuda().to(dtype=dtype)
    model.is_causal = causal
    x = torch.randn(5, 17, 32, dtype=dtype, device="cuda")
    calls = []
    original = flash_attn.flash_attn_func

    def tracked(q, k, v, **kwargs):
        calls.append((q.shape[0], q.shape[1], kwargs["causal"]))
        return original(q, k, v, **kwargs)

    monkeypatch.setattr(flash_attn, "flash_attn_func", tracked)
    with torch.no_grad():
        expected = model(x)
        assert calls == [(5, 17, causal)]
        calls.clear()
        cpu_state, cuda_state = torch.random.get_rng_state(), torch.cuda.get_rng_state()
        record = install(monkeypatch, limit_bytes=20000, chunk_bytes=10000)
        actual = model(x)
    eps = torch.finfo(dtype).eps
    torch.testing.assert_close(actual, expected, rtol=8*eps, atol=eps)
    assert calls == [(1, 17, causal)] * 5
    assert torch.equal(cpu_state, torch.random.get_rng_state())
    assert torch.equal(cuda_state, torch.cuda.get_rng_state())
    assert model.enable_flash_attn is True
    assert record["attention_calls"] == record["flash_attention_calls"] == 1
    assert record["max_attention_batch"] == 1


def test_attention_kernel_setting_is_restored_after_chunk_failure(monkeypatch):
    def failed(module, x):
        assert module.enable_flash_attn is False
        raise RuntimeError("injected chunk failure")

    monkeypatch.setattr(blocks.Attention, "forward", failed)
    model = blocks.Attention(32, num_heads=4, enable_flash_attn=True).eval()
    install(monkeypatch, limit_bytes=4096, chunk_bytes=2048)
    with torch.no_grad(), pytest.raises(RuntimeError, match="injected chunk failure"):
        model(torch.randn(17, 13, 32))
    assert model.enable_flash_attn is True


def test_dense_attention_also_bounds_projection_when_scores_are_small(monkeypatch):
    model = blocks.Attention(128, num_heads=4, qk_norm=True,
        enable_flash_attn=True).eval()
    x = torch.randn(17, 13, 128)
    calls = []
    handle = model.qkv.register_forward_pre_hook(lambda _, args: calls.append(args[0].shape))
    with torch.no_grad():
        expected = model(x)
        calls.clear()
        record = install(monkeypatch, limit_bytes=300000, chunk_bytes=150000)
        actual = model(x)
    handle.remove()
    torch.testing.assert_close(actual, expected)
    assert len(calls) > 1
    assert all(shape[1:] == (13, 128) for shape in calls)
    assert record["attention_calls"] == 1
    assert record["flash_attention_calls"] == 0
