"""Differentiable chunks of independent batches/tokens, with frozen dropout.

Never splits the temporal or spatial attention sequence. The model stays in
eval mode, but gradients propagate from its output into the input adapter.
"""
from contextlib import contextmanager


@contextmanager
def independent_chunks(limit_bytes=512*1024**2, chunk_bytes=128*1024**2):
    import torch
    from opensora.models.layers.blocks import Attention, Mlp
    if not 0 < chunk_bytes <= limit_bytes:
        raise ValueError('invalid chunk sizes')
    old_attention, old_mlp = Attention.forward, Mlp.forward
    record = dict(attention_calls=0, mlp_calls=0, preserves_full_sequence=True)

    def attention(module, x):
        batch, length, channels = x.shape
        flash = module.enable_flash_attn and length > batch
        cost = max(length*channels*(3*x.element_size()+12),
                   0 if flash else module.num_heads*length*length*4)
        if module.training or batch*cost <= limit_bytes:
            return old_attention(module, x)
        size = max(1, chunk_bytes//cost); enabled = module.enable_flash_attn
        module.enable_flash_attn = flash
        try:
            result = torch.cat([old_attention(module, part) for part in x.split(size, dim=0)], dim=0)
        finally:
            module.enable_flash_attn = enabled
        record['attention_calls'] += 1
        return result

    def mlp(module, x):
        if module.training or x.ndim != 3 or not isinstance(module.norm, torch.nn.Identity):
            return old_mlp(module, x)
        cost = x.shape[0]*module.fc1.out_features*x.element_size()
        if x.shape[1]*cost <= limit_bytes:
            return old_mlp(module, x)
        size = max(1, chunk_bytes//cost)
        record['mlp_calls'] += 1
        return torch.cat([old_mlp(module, part) for part in x.split(size, dim=1)], dim=1)

    Attention.forward, Mlp.forward = attention, mlp
    try:
        yield record
    finally:
        Attention.forward, Mlp.forward = old_attention, old_mlp
