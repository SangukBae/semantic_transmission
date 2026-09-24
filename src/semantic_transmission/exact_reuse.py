"""Bounded, exact reuse of immutable inference inputs and STDiT3 weights.

No diffusion-step caching, token reduction, or approximate features are used.
Objects are scoped to one inference run; source images must remain immutable.
"""

from collections import OrderedDict


class FrameTensorCache:
    """Retain just the latest reference/current frame tensors on their device."""

    def __init__(self, loader, enabled=True):
        self.loader = loader
        self.enabled = enabled
        self._items = OrderedDict()
        self.loads = 0
        self.hits = 0

    def get(self, path):
        if self.enabled and path in self._items:
            self.hits += 1
            self._items.move_to_end(path)
            return self._items[path]
        value = self.loader(path)
        self.loads += 1
        if self.enabled:
            self._items[path] = value
            if len(self._items) > 2:
                self._items.popitem(last=False)
        return value


class DiffusionModelReuse:
    """Reuse pretrained CUDA STDiT3, whose forward derives shape from its input.

Keep the legacy construction path for other architectures, untrained models,
and CPU inference (construction can change the CPU sampling RNG stream).
The caller must keep checkpoint, dtype, and model options fixed for this run.
Schedulers and per-segment conditioning are deliberately not cached.
"""

    def __init__(self, model_config, device, enabled=True):
        self.enabled = bool(
            enabled
            and str(device).split(":")[0] == "cuda"
            and model_config.get("type") == "STDiT3-XL/2"
            and model_config.get("from_pretrained")
        )
        self.model = None
        self.builds = 0
        self.hits = 0

    def get(self, factory, latent_size):
        if not self.enabled or self.model is None:
            model = factory()
            self.builds += 1
            if self.enabled:
                self.model = model
        else:
            model = self.model
            self.hits += 1
            # STDiT3 derives positional embeddings/padding from x each forward.
            # Keep informational config in sync with the current segment too.
            model.config.input_size = latent_size
        return model
