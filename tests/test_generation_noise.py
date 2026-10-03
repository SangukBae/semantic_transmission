import copy
import sys
import types

import pytest
torch = pytest.importorskip("torch")

from semantic_transmission import generation_noise as noise
from semantic_transmission.webvid5 import read_json

CONTRACT = dict(seed=2025, segments=2, steps=3, input="fixture")


def produce(path, expected=None, extra_random_work=0):
    control = noise.NoiseControl(path, CONTRACT, {"runtime": "test"}, expected)
    tensor = torch.zeros((1, 2, 3))
    for index in range(4):
        torch.randn(extra_random_work + index)
        with control.scope("vae", index, tensor):
            torch.randn(tensor.shape)
            torch.randn_like(tensor)
    for index in range(2):
        with control.scope("initial", index, tensor):
            torch.randn_like(tensor)
        with control.scope("sampling", index, tensor):
            for _ in range(3):
                torch.randn_like(tensor)
    control.finish(2, 3)
    return read_json(path)


def test_all_noise_is_identical_despite_unrelated_global_rng_work(tmp_path):
    first = produce(tmp_path / "first.json")
    torch.manual_seed(91831)
    second = produce(tmp_path / "second.json", first, 90)
    assert first["records"] == second["records"]
    assert second["matched_reference"] is True
    assert second["matches_legacy_unrecorded_run"] is False
    assert len({r["seed"] for r in first["records"]}) == len(first["records"])


def test_scope_restores_rng_and_functions_even_on_failure(tmp_path):
    control = noise.NoiseControl(tmp_path / "trace.json", CONTRACT, {})
    state, original = torch.get_rng_state().clone(), torch.randn
    with pytest.raises(RuntimeError, match="failure"):
        with control.scope("vae", 0, torch.zeros(3)):
            torch.randn(5)
            raise RuntimeError("failure")
    assert torch.equal(torch.get_rng_state(), state)
    assert torch.randn is original
    assert read_json(tmp_path / "trace.json")["status"] == "FAILED"


@pytest.mark.parametrize("change", ["hash", "shape", "extra_draw", "runtime"])
def test_mismatched_noise_reference_fails_closed(tmp_path, change):
    first = produce(tmp_path / "first.json")
    altered = copy.deepcopy(first)
    if change == "hash":
        altered["records"][0]["draws"][0]["sha256"] = "bad"
    elif change == "shape":
        altered["records"][0]["input_shape"] = [99]
    elif change == "extra_draw":
        altered["records"][0]["draws"].append(altered["records"][0]["draws"][0])
    else:
        altered["runtime"] = {"runtime": "different"}
    with pytest.raises(ValueError, match="noise"):
        produce(tmp_path / "second.json", altered)


def test_missing_scope_or_sampling_steps_cannot_pass(tmp_path):
    trace = produce(tmp_path / "trace.json")
    trace["records"].pop()
    with pytest.raises(ValueError, match="sampling"):
        noise.validate_trace(trace, CONTRACT, 2, 3)


def test_hooks_cover_received_and_generated_vae_initial_and_every_sampling_draw(tmp_path, monkeypatch):
    class VAE:
        def encode(self, x):
            return x + torch.randn_like(x) + torch.randn(x.shape)
    class Scheduler:
        def sample(self, model, text_encoder, z):
            for _ in range(3):
                z = z + torch.randn_like(z)
            return z
    def mask(z, refs, strategies, loop_i, align=None):
        assert align == 5
        z[:, :, 0] = refs[0][:, :, 0]
        return "mask"
    utils = types.SimpleNamespace(apply_mask_strategy=mask)
    monkeypatch.setitem(sys.modules, "opensora.models.vae.vae", types.SimpleNamespace(VideoAutoencoderPipeline=VAE))
    monkeypatch.setitem(sys.modules, "opensora.schedulers.rf", types.SimpleNamespace(RFLOW=Scheduler))
    monkeypatch.setitem(sys.modules, "opensora.utils", types.SimpleNamespace(inference_utils=utils))
    control = noise.NoiseControl(tmp_path / "trace.json", CONTRACT, {})
    noise.install(control)
    vae, scheduler = VAE(), Scheduler()
    refs = [vae.encode(torch.zeros(1, 2, 3)) for _ in range(3)]
    for index in range(2):
        if index:
            refs.append(vae.encode(z))
        z = torch.zeros(1, 2, 3)
        assert utils.apply_mask_strategy(z, refs, [], index, align=5) == "mask"
        assert torch.equal(z[:, :, 0], refs[0][:, :, 0])
        z = scheduler.sample(None, None, z)
    control.finish(2, 3)
    trace = read_json(tmp_path / "trace.json")
    assert trace["status"] == "PASSED"
    assert sum(len(r["draws"]) for r in trace["records"]) == 4 * 2 + 2 + 2 * 3


def test_cuda_scope_replays_both_cpu_and_gpu_draws_and_restores_both_states(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    tensor = torch.zeros((1, 4, 5), device="cuda", dtype=torch.bfloat16)
    original_cpu = torch.get_rng_state().clone()
    original_cuda = torch.cuda.get_rng_state().clone()
    control = noise.NoiseControl(tmp_path / "trace.json", CONTRACT, {})
    with control.scope("vae", 0, tensor):
        first_cpu = torch.randn(tensor.shape)
        first_cuda = torch.randn_like(tensor)
    assert torch.equal(torch.get_rng_state(), original_cpu)
    assert torch.equal(torch.cuda.get_rng_state(), original_cuda)
    torch.randn(39)
    torch.randn(53, device="cuda")
    with control.scope("vae", 0, tensor):
        second_cpu = torch.randn(tensor.shape)
        second_cuda = torch.randn_like(tensor)
    assert torch.equal(first_cpu, second_cpu) and torch.equal(first_cuda, second_cuda)
