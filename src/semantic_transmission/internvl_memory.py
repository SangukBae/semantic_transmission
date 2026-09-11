"""Bound InternVL memory while retaining the pretrained vocabulary weights.

The official profile streams BF16 vocabulary rows to CUDA; the legacy profile
projects on CPU in FP32. Both consume only final-position generation logits.
"""
import torch


class CpuVocabulary(torch.nn.Module):
    def __init__(self, module, *, last_token_only=False, dtype=torch.float32):
        super().__init__()
        self.module = module.to(device="cpu", dtype=dtype)
        self.last_token_only = last_token_only

    @property
    def weight(self):
        return self.module.weight

    def forward(self, values):
        device = values.device
        if self.last_token_only:
            if self.training:
                raise RuntimeError("last-token vocabulary projection is inference-only")
            values = values[:, -1:]
        dtype = torch.float32 if values.is_floating_point() else values.dtype
        return self.module(values.to(device="cpu", dtype=dtype)).to(device=device, dtype=torch.bfloat16)


class LastTokenHead(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module.cpu()
        self.module.weight.data = self.module.weight.data.pin_memory()

    @property
    def weight(self):
        return self.module.weight

    def forward(self, values):
        if self.training:
            raise RuntimeError("last-token vocabulary projection is inference-only")
        # Vocabulary rows are independent; retain the released BF16 CUDA dot products
        # while streaming the 724 MiB output matrix in bounded blocks.
        logits = []
        for start in range(0, self.weight.shape[0], 16384):
            weight = self.weight[start:start + 16384].to(values.device, non_blocking=True)
            bias = self.module.bias
            if bias is not None:
                bias = bias[start:start + 16384].to(values.device)
            logits.append(torch.nn.functional.linear(values[:, -1:], weight, bias))
        return torch.cat(logits, dim=-1)


def place_internvl(model, *, gpu_head=False):
    language = model.language_model
    class GpuExecution(type(language)):
        @property
        def device(self):
            return next(self.model.layers.parameters()).device
    # Generation initializes token bookkeeping on model.device, independent of embeddings.
    language.__class__ = GpuExecution
    language.model.tok_embeddings = CpuVocabulary(language.model.tok_embeddings, dtype=torch.bfloat16 if gpu_head else torch.float32)
    language.output = (LastTokenHead(language.output) if gpu_head else
                       CpuVocabulary(language.output, last_token_only=True))
    # Stage ViT once per pair, then release its GPU weights before language decoding.
    model.vision_model.cpu()
    model.mlp1.cuda()
    for layer in language.model.layers:
        rotary = layer.attention.rotary_emb
        # The checkpoint allocates 32k positions per layer although this input needs <4k.
        # Slice existing values exactly; the original implementation still grows on demand.
        rotary.cos_cached = rotary.cos_cached[:4096].clone()
        rotary.sin_cached = rotary.sin_cached[:4096].clone()
        rotary.max_seq_len_cached = min(rotary.max_seq_len_cached, 4096)
    language.model.layers.cuda()
    language.model.norm.cuda()
    for layer in language.model.layers:
        original_ffn = layer.feed_forward.forward
        def chunked_ffn(values, forward=original_ffn):
            if values.shape[1] <= 128:
                return forward(values)
            return torch.cat([forward(chunk) for chunk in values.split(128, dim=1)], dim=1)
        layer.feed_forward.forward = chunked_ffn
    original_extract = model.extract_feature
    cached_pixels = cached_features = None
    def extract_sequential(pixel_values):
        nonlocal cached_pixels, cached_features
        if pixel_values is cached_pixels:
            return cached_features
        # ViT samples are independent in eval mode; avoid six simultaneous attention maps.
        model.vision_model.cuda()
        features = torch.cat([original_extract(image.unsqueeze(0)) for image in pixel_values], dim=0)
        model.vision_model.cpu()
        torch.cuda.empty_cache()
        cached_pixels, cached_features = pixel_values, features
        return features
    model.extract_feature = extract_sequential
    model.eval()
    return model
