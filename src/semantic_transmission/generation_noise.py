"""Isolate and fingerprint every inference noise draw for paired T5 runs.

Seeds belong to VAE calls, initial latents and RFLOW sampling separately.
The trace stores reproducible seeds and tensor hashes, not multi-GB tensors.
It cannot recover noise from older runs which did not use this protocol.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path

from .artifacts import write_json
from .webvid5 import read_json

POLICY = "scoped_vae_initial_rflow_noise_v1"
TRACE = "receiver/generation_noise.json"


def tensor_hash(tensor):
    import torch
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def validate_trace(trace, contract, segments, steps):
    if trace.get("status") != "PASSED" or trace.get("policy") != POLICY or trace.get("contract") != contract:
        raise ValueError("incomplete or incompatible generation noise trace")
    records = trace["records"]
    for kind, count in (("vae", 2 * segments), ("initial", segments), ("sampling", segments)):
        selected = [r for r in records if r["kind"] == kind]
        if [r["index"] for r in selected] != list(range(count)):
            raise ValueError(f"missing/duplicate {kind} noise scopes")
        for row in selected:
            expected = steps if kind == "sampling" else 1 if kind == "initial" else None
            if not row["draws"] or (expected is not None and len(row["draws"]) != expected):
                raise ValueError(f"missing {kind} noise draws")
    if any(r["kind"] not in {"vae", "initial", "sampling"} for r in records):
        raise ValueError("unknown generation noise scope")


class NoiseControl:
    def __init__(self, path, contract, runtime, expected=None):
        self.path, self.contract = Path(path), contract
        self.runtime, self.records, self.active = runtime, [], False
        self.expected = expected
        if expected is not None and (expected.get("status") != "PASSED"
                or expected.get("policy") != POLICY or expected.get("contract") != contract
                or expected.get("runtime") != runtime):
            raise ValueError("noise reference input/runtime contract differs")
        self.save("RUNNING")

    def save(self, status):
        write_json(self.path, dict(status=status, policy=POLICY, contract=self.contract,
            runtime=self.runtime, records=self.records,
            matched_reference=self.expected is not None and status == "PASSED",
            matches_legacy_unrecorded_run=False))

    @contextmanager
    def scope(self, kind, index, tensor):
        import torch
        if self.active:
            raise ValueError("nested noise scopes are unsupported")
        label = f"{self.contract['seed']}:{kind}:{index}"
        seed = int(hashlib.sha256(label.encode()).hexdigest()[:15], 16)
        device = tensor.device
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        record = dict(kind=kind, index=index, seed=seed, input_shape=list(tensor.shape),
                      dtype=str(tensor.dtype), device=device.type, draws=[])
        originals = {name: getattr(torch, name) for name in ("randn", "randn_like")}

        def intercept(name):
            def draw(*args, **kwargs):
                value = originals[name](*args, **kwargs)
                record["draws"].append(dict(api=name, shape=list(value.shape), dtype=str(value.dtype),
                    device=value.device.type, sha256=tensor_hash(value)))
                return value
            return draw

        self.active = True
        try:
            with torch.random.fork_rng(devices=devices):
                torch.random.default_generator.manual_seed(seed)
                for number in devices:
                    torch.cuda.default_generators[number].manual_seed(seed)
                for name in originals:
                    setattr(torch, name, intercept(name))
                try:
                    yield
                finally:
                    for name, original in originals.items():
                        setattr(torch, name, original)
            if self.expected is not None:
                position = len(self.records)
                if position >= len(self.expected["records"]) or record != self.expected["records"][position]:
                    raise ValueError(f"generation noise differs at {kind}:{index}")
            self.records.append(record)
            self.save("RUNNING")
        except BaseException:
            self.save("FAILED")
            raise
        finally:
            self.active = False

    def finish(self, segments, steps):
        if self.expected is not None and len(self.records) != len(self.expected["records"]):
            raise ValueError("noise reference was not fully consumed")
        candidate = dict(status="PASSED", policy=POLICY, contract=self.contract, records=self.records)
        validate_trace(candidate, self.contract, segments, steps)
        self.save("PASSED")


def install(control):
    """Hook only inference operations; retain VAE, mask and RFLOW mathematics."""
    import torch
    from opensora.models.vae.vae import VideoAutoencoderPipeline
    from opensora.schedulers.rf import RFLOW
    from opensora.utils import inference_utils
    original_encode = VideoAutoencoderPipeline.encode
    original_mask = inference_utils.apply_mask_strategy
    original_sample = RFLOW.sample
    vae_calls = sampling_calls = 0

    def encode(vae, x):
        nonlocal vae_calls
        index = vae_calls
        vae_calls += 1
        with control.scope("vae", index, x):
            return original_encode(vae, x)

    def apply(z, refs, strategies, loop_i, align=None):
        # Replace the initial draw before any received reference is injected.
        with control.scope("initial", loop_i, z):
            z.copy_(torch.randn_like(z))
        return original_mask(z, refs, strategies, loop_i, align=align)

    def sample(scheduler, model, text_encoder, z, *args, **kwargs):
        nonlocal sampling_calls
        index = sampling_calls
        sampling_calls += 1
        with control.scope("sampling", index, z):
            return original_sample(scheduler, model, text_encoder, z, *args, **kwargs)

    VideoAutoencoderPipeline.encode = encode
    inference_utils.apply_mask_strategy = apply
    RFLOW.sample = sample
