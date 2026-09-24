"""Actual released masking behavior on the ETRI latent lengths, without model loading."""
import ast
from pathlib import Path

import pytest

from semantic_transmission.temporal import conditioning_indices


@pytest.fixture
def masking():
    torch = pytest.importorskip("torch")
    source = Path(__file__).resolve().parents[1] / ".local/vendor/Open-Sora/opensora/utils/inference_utils.py"
    if not source.exists():
        pytest.skip("pinned Open-Sora checkout is not installed")
    names = {"parse_mask_strategy", "find_nearest_point", "apply_mask_strategy"}
    tree = ast.parse(source.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = {"torch": torch, "MASK_DEFAULT": ["0", "0", "0", "0", "1", "0"]}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), scope)
    return torch, scope["apply_mask_strategy"]


@pytest.mark.parametrize("length,rounded", [(53, 50), (23, 20)])
def test_exact_endpoint_is_really_conditioned(masking, length, rounded):
    torch, apply = masking
    refs = [[torch.full((4, 1, 1, 1), 7.)]]
    old = torch.zeros(1, 4, length, 1, 1)
    fixed = old.clone()
    strategy = ["0,0,0,-1,1"]
    old_mask = apply(old, refs, strategy, 0, align=5)
    fixed_mask = apply(fixed, refs, strategy, 0, align=None)
    assert old_mask[0, -1] == 1 and old_mask[0, rounded] == 0
    assert fixed_mask[0, -1] == 0
    assert torch.all(fixed[0, :, -1] == 7)


def test_overlap_uses_last_real_frames_and_short_clips_are_padded():
    assert conditioning_indices(180, 17) == list(range(163, 180))
    assert conditioning_indices(3, 17) == [0] * 15 + [1, 2]


def test_endpoint_and_overlap_do_not_overwrite_one_another(masking):
    torch, apply = masking
    z = torch.zeros(1, 4, 23, 1, 1)
    refs = [[torch.full((4, 1, 1, 1), 7.), torch.arange(5.).reshape(1, 5, 1, 1).expand(4, 5, 1, 1)]]
    mask = apply(z, refs, ["1,0,0,-1,1;1,1,-5,0,5,0"], 1, align=None)
    assert torch.equal(z[0, 0, :5, 0, 0], torch.arange(5.))
    assert torch.all(z[0, :, -1] == 7)
    assert torch.where(mask[0] == 0)[0].tolist() == [0, 1, 2, 3, 4, 22]


@pytest.mark.parametrize("length", [2, 8, 16, 17, 35])
def test_official_short_first_segment_supplies_five_overlap_latents(masking, length):
    """Exercise the actual decoder branch and released mask, without loading models."""
    torch, apply = masking
    decoder = Path(__file__).resolve().parents[1] / "04_semantic_decoder/scripts/mydemo_new_align_sh.py"
    tree = ast.parse(decoder.read_text())
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                          and c.func.id == "conditioning_indices" for c in ast.walk(n))
                  and "previous.shape" in ast.unparse(n.test))
    previous = torch.arange(float(length)).reshape(1, 1, length, 1, 1)
    scope = dict(previous=previous, conditioning_alignment="official_release",
                 cfg={"decoder_policy":"official_release"}, condition_frame_length=5,
                 dframe_to_frame=lambda n: n//5*17, conditioning_indices=conditioning_indices)
    exec(compile(ast.Module(body=[branch], type_ignores=[]), str(decoder), "exec"), scope)
    result = scope["previous"]
    if length >= 17:
        assert result is previous  # previously successful paths are untouched
    else:
        assert result.shape[2] == 17
        assert torch.equal(result[:, :, -length:], previous)
        assert torch.all(result[:, :, :17-length] == previous[:, :, :1])
    # The real VAE maps a full 17-frame block to five latents. The actual
    # short-clip VAE path is also exercised by the GPU repair probe.
    ref = torch.arange(5.).reshape(1, 5, 1, 1).expand(4, 5, 1, 1)
    z = torch.zeros(1, 4, 8, 1, 1)
    apply(z, [[ref]], ["1,0,-5,0,5,0"], 1, align=5)
    assert torch.equal(z[0, 0, :5, 0, 0], torch.arange(5.))


def test_short_reference_reproduces_original_empty_slice_failure(masking):
    torch, apply = masking
    ref = torch.zeros(4, 2, 1, 1)
    with pytest.raises(RuntimeError, match="expanded size"):
        apply(torch.zeros(1, 4, 8, 1, 1), [[ref]], ["1,0,-5,0,5,0"], 1, align=5)
