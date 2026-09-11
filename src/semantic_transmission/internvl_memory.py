"""Keep InternVL transformer/vision BF16 on GPU and vocabulary tables on CPU.

CPU FP32 computation uses the original BF16 checkpoint values, without low-bit
quantization. Autoregressive inference consumes only the final-position logits.
"""
import torch


class CpuVocabulary(torch.nn.Module):
    def __init__(self, module, *, last_token_only=False):
        super().__init__()
        self.module = module.to(device="cpu", dtype=torch.float32)
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


def place_internvl(model):
    language = model.language_model
    class GpuExecution(type(language)):
        @property
        def device(self):
            return next(self.model.layers.parameters()).device
    # Generation initializes token bookkeeping on model.device, independent of embeddings.
    language.__class__ = GpuExecution
    language.model.tok_embeddings = CpuVocabulary(language.model.tok_embeddings)
    language.output = CpuVocabulary(language.output, last_token_only=True)
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
