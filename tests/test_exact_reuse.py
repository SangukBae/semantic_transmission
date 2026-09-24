from types import SimpleNamespace

import pytest

from semantic_transmission.exact_reuse import DiffusionModelReuse, FrameTensorCache


def test_frame_reference_survives_repeated_comparisons_and_switches():
    calls = []
    def loader(path):
        calls.append(path)
        return object()
    cache = FrameTensorCache(loader)
    first = cache.get("a")
    cache.get("b")
    assert cache.get("a") is first
    current = cache.get("c")
    assert cache.get("c") is current  # c becomes the new keyframe
    cache.get("d")
    assert calls == ["a", "b", "c", "d"]
    assert len(cache._items) == 2


@pytest.mark.parametrize("enabled", [False, True])
def test_dynamic_shape_reuse_and_legacy_switch(enabled):
    cfg = {"type": "STDiT3-XL/2", "from_pretrained": "local-checkpoint"}
    cache = DiffusionModelReuse(cfg, "cuda:0", enabled)
    def build():
        return SimpleNamespace(config=SimpleNamespace(input_size=(3, 16, 16)))
    first = cache.get(build, (3, 16, 16))
    second = cache.get(build, (5, 16, 16))
    assert (first is second) == enabled
    assert cache.builds == (1 if enabled else 2)
    if enabled:
        assert second.config.input_size == (5, 16, 16)


@pytest.mark.parametrize("device,config", [
    ("cpu", {"type": "STDiT3-XL/2", "from_pretrained": "checkpoint"}),
    ("cuda", {"type": "OtherModel", "from_pretrained": "checkpoint"}),
    ("cuda", {"type": "STDiT3-XL/2"}),
])
def test_unknown_or_randomly_initialized_models_keep_legacy_behavior(device, config):
    assert not DiffusionModelReuse(config, device).enabled
