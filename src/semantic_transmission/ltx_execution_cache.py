"""Opt-in, bounded execution caches; unchanged weights, kernels and sampling.

Keep a prefix of the denoiser's streamed blocks resident on GPU for the duration
of one denoising stage. Other blocks retain upstream's event-guarded streaming.
Repeated, unenhanced prompts reuse exact CPU-cached embeddings. Nothing is
persisted as a replacement model checkpoint, and vendor files are untouched.
"""
from collections import OrderedDict
import copy

VERSION = 'ltx_exact_execution_cache_v1'


def provider_class(base, stats, cache_bytes, reserve_bytes):
    import torch
    from ltx_core.block_streaming.utils import carve_buffer, layout_nbytes

    class ResidentPrefixProvider(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._resident = {}
            self._resident_bytes = 0
            # Gemma is used for one forward pass: retaining its weights brings
            # no reuse. Cache only the repeatedly evaluated diffusion model.
            self._eligible = self._blocks_prefix == 'transformer_blocks'
            self._original_schedule = getattr(self._source, '_ensure_scheduled', None)
            if self._eligible and self._original_schedule is not None:
                original = self._original_schedule
                def schedule(index):
                    if index not in self._resident:
                        original(index)
                self._source._ensure_scheduled = schedule

        def get(self, index):
            if index in self._resident:
                stats['gpu_cache_hits'] += 1
                return self._resident[index][1]
            layout = self._source.block_layout(index)
            size = layout_nbytes(layout)
            room = self._resident_bytes + size <= cache_bytes
            if self._eligible and room and self._target_device.type == 'cuda':
                free, _ = torch.cuda.mem_get_info(self._target_device)
                room = room and free - size >= reserve_bytes
            if not self._eligible or not room:
                stats['streamed_block_requests'] += 1
                return super().get(index)
            raw = torch.empty(size, dtype=torch.uint8, device=self._target_device)
            views = carve_buffer(raw, layout)
            source = self._source.get(index)
            event = self._copy_to_gpu(index, raw, views, source, size)
            self._source.release(index, event=event)
            self._resident[index] = (raw, views)
            self._resident_bytes += size
            stats['cached_block_loads'] += 1
            stats['gpu_cache_peak_bytes'] = max(stats['gpu_cache_peak_bytes'], self._resident_bytes)
            return views

        def mark_block_done(self, index):
            # Resident buffers are immutable until cleanup's synchronization.
            if index not in self._resident:
                super().mark_block_done(index)

        def cleanup(self):
            super().cleanup()
            self._resident.clear()
            self._resident_bytes = 0
            if self._original_schedule is not None:
                self._source._ensure_scheduled = self._original_schedule

    return ResidentPrefixProvider


def prompt_call(original, stats, max_bytes):
    """Per-encoder LRU cache, CPU tensors only; return fresh tensors on hits."""
    import torch

    def call(encoder, prompts, **kwargs):
        prompts = list(prompts)
        if kwargs.get('enhance_first_prompt', False):
            return original(encoder, prompts, **kwargs)
        # Complete argument key avoids sharing embeddings across enhancements.
        key = (tuple(prompts), tuple(sorted(kwargs.items())))
        cache = getattr(encoder, '_bsu_embedding_cache', None)
        if cache is None:
            cache = OrderedDict()
            encoder._bsu_embedding_cache = cache
            encoder._bsu_embedding_bytes = 0
        if key in cache:
            tensors, size = cache.pop(key)
            cache[key] = (tensors, size)
            stats['prompt_cache_hits'] += 1
            return [type(row)(*(t.to(encoder._device).clone() if torch.is_tensor(t) else copy.deepcopy(t)
                               for t in row)) for row in tensors]
        result = original(encoder, prompts, **kwargs)
        stats['prompt_cache_misses'] += 1
        tensors = [type(row)(*(t.detach().to('cpu').clone() if torch.is_tensor(t) else copy.deepcopy(t)
                              for t in row)) for row in result]
        size = sum(t.numel()*t.element_size() for row in tensors for t in row if torch.is_tensor(t))
        if size <= max_bytes:
            while cache and encoder._bsu_embedding_bytes + size > max_bytes:
                _, (_, old_size) = cache.popitem(last=False)
                encoder._bsu_embedding_bytes -= old_size
            cache[key] = (tensors, size)
            encoder._bsu_embedding_bytes += size
        return result
    return call


class Installation:
    def __init__(self, gpu_cache_bytes, reserve_bytes, prompt_cache_bytes):
        if min(gpu_cache_bytes, reserve_bytes, prompt_cache_bytes) < 0:
            raise ValueError('cache and reserve limits must be nonnegative')
        from ltx_core.block_streaming import builder
        from ltx_pipelines.utils.blocks import PromptEncoder
        if getattr(builder.WeightsProvider, '_bsu_cache_installed', False):
            raise RuntimeError('execution cache already installed')
        self.stats = dict(gpu_cache_hits=0, cached_block_loads=0, streamed_block_requests=0,
                          gpu_cache_peak_bytes=0, prompt_cache_hits=0, prompt_cache_misses=0)
        self._builder, self._prompt = builder, PromptEncoder
        self._provider, self._call = builder.WeightsProvider, PromptEncoder.__call__
        provider = provider_class(self._provider, self.stats, gpu_cache_bytes, reserve_bytes)
        provider._bsu_cache_installed = True
        builder.WeightsProvider = provider
        PromptEncoder.__call__ = prompt_call(self._call, self.stats, prompt_cache_bytes)

    def close(self):
        self._builder.WeightsProvider = self._provider
        self._prompt.__call__ = self._call


def install(gpu_cache_bytes=4*2**30, reserve_bytes=3*2**30, prompt_cache_bytes=256*2**20):
    return Installation(gpu_cache_bytes, reserve_bytes, prompt_cache_bytes)
