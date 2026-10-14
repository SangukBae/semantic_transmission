"""Bound inference temporaries without splitting a video's temporal context.

Attention batches are independent spatial positions; MLP tokens are independent.
Keep the upstream attention kernel choice, full sequence, weights and precision.
"""
from functools import wraps

POLICY = "independent_attention_batches_and_mlp_tokens_v1"
TRACE = "receiver/memory_execution.json"
LIMIT_BYTES = 512 * 1024**2
CHUNK_BYTES = 128 * 1024**2


def install(limit_bytes=LIMIT_BYTES, chunk_bytes=CHUNK_BYTES):
    import torch
    from opensora.models.layers.blocks import Attention, Mlp

    if not 0 < chunk_bytes <= limit_bytes:
        raise ValueError("invalid decoder memory limits")
    original_attention, original_mlp = Attention.forward, Mlp.forward
    record = dict(policy=POLICY, limit_bytes=limit_bytes, chunk_bytes=chunk_bytes,
                  attention_calls=0, mlp_calls=0, max_attention_batch=0, max_mlp_tokens=0)

    @wraps(original_attention)
    def attention(module, x):
        batch, length, _ = x.shape
        flash = module.enable_flash_attn and length > batch
        score_bytes = batch * module.num_heads * length * length * 4
        if module.training or torch.is_grad_enabled() or flash or score_bytes <= limit_bytes:
            return original_attention(module, x)
        size = max(1, chunk_bytes // (module.num_heads * length * length * 4))
        output = torch.empty_like(x)
        enabled = module.enable_flash_attn
        # A smaller batch must not silently switch the upstream N > B heuristic
        # from ordinary attention to FlashAttention.
        module.enable_flash_attn = False
        try:
            for start in range(0, batch, size):
                output[start:start+size] = original_attention(module, x[start:start+size])
        finally:
            module.enable_flash_attn = enabled
        record["attention_calls"] += 1
        record["max_attention_batch"] = max(record["max_attention_batch"], min(size, batch))
        return output

    @wraps(original_mlp)
    def mlp(module, x):
        if (module.training or torch.is_grad_enabled() or x.ndim != 3
                or not isinstance(module.fc1, torch.nn.Linear)
                or not isinstance(module.norm, torch.nn.Identity)):
            return original_mlp(module, x)
        batch, tokens, _ = x.shape
        bytes_per_token = batch * module.fc1.out_features * x.element_size()
        if tokens * bytes_per_token <= limit_bytes:
            return original_mlp(module, x)
        size = max(1, chunk_bytes // bytes_per_token)
        output = x.new_empty(batch, tokens, module.fc2.out_features)
        for start in range(0, tokens, size):
            output[:, start:start+size] = original_mlp(module, x[:, start:start+size])
        record["mlp_calls"] += 1
        record["max_mlp_tokens"] = max(record["max_mlp_tokens"], min(size, tokens))
        return output

    Attention.forward, Mlp.forward = attention, mlp
    return record
