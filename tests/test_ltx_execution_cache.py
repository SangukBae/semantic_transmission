from collections import Counter, namedtuple

import pytest
import torch

from semantic_transmission.ltx_execution_cache import provider_class, prompt_call


def statistics():
    return dict(gpu_cache_hits=0, cached_block_loads=0, streamed_block_requests=0,
                gpu_cache_peak_bytes=0, prompt_cache_hits=0, prompt_cache_misses=0)


def test_resident_weights_are_exact_and_reused_without_recycling():
    pytest.importorskip('ltx_core')
    from ltx_core.block_streaming.provider import WeightsProvider
    from ltx_core.block_streaming.pool import BufferPool
    from ltx_core.block_streaming.stream_sync import SynchronousStreamSync
    from ltx_core.block_streaming.utils import carve_buffer, layout_nbytes
    layout = {'weight': (torch.Size([4]), torch.float32)}
    size = layout_nbytes(layout)

    class Source:
        def __init__(self):
            self.loads = Counter()
            self.scheduled = []
            self.closed = False
        def block_layout(self, index):
            return layout
        def get(self, index):
            self.loads[index] += 1
            raw = torch.zeros(size, dtype=torch.uint8)
            carve_buffer(raw, layout)['weight'].fill_(index + .25)
            return raw
        def release(self, index, event):
            pass
        def _ensure_scheduled(self, index):
            self.scheduled.append(index)
        def cleanup(self):
            self.closed = True
    source = Source()
    sync = SynchronousStreamSync()
    pool = BufferPool(size, 2, torch.device('cpu'), sync.reuse_barrier)
    stats = statistics()
    cls = provider_class(WeightsProvider, stats, cache_bytes=2*size, reserve_bytes=0)
    provider = cls(pool, sync, torch.device('cpu'), source, blocks_prefix='transformer_blocks')
    for repeat in range(4):
        for index in range(5):
            assert torch.equal(provider.get(index)['weight'], torch.full((4,), index+.25))
            provider.mark_block_done(index)
    assert source.loads == {0: 1, 1: 1, 2: 4, 3: 4, 4: 4}
    source._ensure_scheduled(0)
    source._ensure_scheduled(2)
    assert source.scheduled == [2]
    assert stats['gpu_cache_hits'] == 6 and stats['gpu_cache_peak_bytes'] == 2*size
    provider.cleanup()
    assert source.closed and provider._resident == {}
    source._ensure_scheduled(0)
    assert source.scheduled == [2, 0]


def test_prompt_cache_returns_independent_exact_values_and_is_bounded():
    result_type = namedtuple('Embeddings', 'video audio mask')
    class Encoder:
        _device = torch.device('cpu')
        calls = 0
    encoder = Encoder()
    def original(self, prompts, **kwargs):
        self.calls += 1
        return [result_type(torch.tensor([float(len(prompts[0]))]), None, torch.tensor([1]))]
    stats = statistics()
    call = prompt_call(original, stats, max_bytes=12)
    first = call(encoder, ['first'])
    first[0].video.fill_(-1)
    second = call(encoder, ['first'])
    assert second[0].video.item() == 5 and encoder.calls == 1
    second[0].video.fill_(999)
    assert call(encoder, ['first'])[0].video.item() == 5
    call(encoder, ['second'])
    call(encoder, ['first'])
    assert encoder.calls == 3
    assert encoder._bsu_embedding_bytes <= 12
    call(encoder, ['first'], enhance_first_prompt=True)
    assert encoder.calls == 4
