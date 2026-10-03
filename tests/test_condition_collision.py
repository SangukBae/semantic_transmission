import copy
from pathlib import Path

import pytest
torch = pytest.importorskip("torch")

from semantic_transmission import condition_collision as fix


@pytest.fixture
def functions():
    result = fix.upstream_functions(Path(__file__).resolve().parents[1])
    result.apply_mask_strategy.__globals__["torch"] = torch
    return result


@pytest.mark.parametrize("length", [3, 4])
def test_real_short_segment_regressions_keep_both_keys(functions, length):
    refs = [[torch.full((1, 1, 1, 1), 11.), torch.full((1, 1, 1, 1), 22.)]]
    z = torch.full((1, 1, length, 1, 1), -99.)
    old = z.clone()
    functions.apply_mask_strategy(old, refs, ["0;0,1,0,-1,1"], 0, align=5)
    assert old[0, 0, 0].item() == 22
    state = torch.get_rng_state().clone()
    masks, rows = fix.apply_guard(z, refs, ["0;0,1,0,-1,1"], 0, 5, 2, functions)
    assert torch.equal(state, torch.get_rng_state())
    assert z[0, 0, 0].item() == 11 and z[0, 0, -1].item() == 22
    assert masks[0, 0] == masks[0, -1] == 0
    assert torch.all(z[0, 0, 1:-1] == -99)
    assert rows[0]["repaired"]


@pytest.mark.parametrize("first_frames", [5, 8, 9, 13, 16, 17, 18, 21, 24, 25, 34, 35, 60, 121, 241, 600])
def test_variable_segments_preserve_noncolliding_upstream_tensors(functions, first_frames):
    keys = [0, first_frames-1, first_frames, first_frames+7, first_frames+31, first_frames+90]
    strategy = "0;" + ";".join(f"{i},{i+1},0,-1,1" for i in range(len(keys)-1))
    refs = [[torch.full((2, 1, 2, 2), float(i+1)) for i in range(len(keys))]]
    for expected in fix.preview(keys, functions):
        loop = expected["loop"]
        if loop:
            refs[0].append(torch.full((2, 5, 2, 2), float(100+loop)))
            strategy += f";{loop},{len(keys)+loop-1},-5,0,5,0"
        z = torch.randn(1, 2, expected["latent_frames"], 2, 2)
        original = z.clone()
        old_mask = functions.apply_mask_strategy(original, refs, [strategy], loop, align=5)
        mask, records = fix.apply_guard(z, refs, [strategy], loop, 5, len(keys), functions)
        if not expected["repaired"]:
            assert torch.equal(z, original) and torch.equal(mask, old_mask)
        else:
            assert loop == 0
            assert torch.equal(z[0, :, 0:1], refs[0][0])
            assert torch.equal(z[0, :, -1:], refs[0][1])
        if loop:
            assert torch.equal(z[0, :, :5], refs[0][-1])
        assert records[0]["effective_strategy"] == expected["effective_strategy"]


def test_impossible_single_latent_fails_before_any_batch_changes(functions):
    refs = [[torch.ones(1, 1, 1, 1), torch.ones(1, 1, 1, 1)*2]]
    z = torch.full((1, 1, 1, 1, 1), -99.)
    with pytest.raises(ValueError, match="insufficient separate"):
        fix.apply_guard(z, refs, ["0;0,1,0,-1,1"], 0, 5, 2, functions)
    assert z.item() == -99
    for end in (1, 2, 3):
        with pytest.raises(ValueError, match="insufficient separate"):
            fix.preview([0, end], functions)


def test_bad_history_and_truncated_references_are_rejected(functions):
    with pytest.raises(ValueError, match="truncated"):
        fix.plan("1,2,0,-1,1;1,3,-4,0,5,0", [1, 1, 1, 4], 8, 1, 3, 5, functions)
    with pytest.raises(ValueError, match="strategy"):
        fix.plan("0", [1, 1], 3, 0, 2, 5, functions)


def test_missing_changed_or_duplicated_runtime_records_fail(functions):
    expected = fix.preview([0, 8, 32], functions)
    records = [[{k: v for k, v in row.items() if k not in {"start_frame", "end_frame"}}] for row in expected]
    fix.validate_trace(records, expected)
    bad = copy.deepcopy(records)
    bad[0][0]["operations"][-1]["target"] = 0
    for value in (bad, records[:-1], records + records[-1:]):
        with pytest.raises(ValueError, match="placement"):
            fix.validate_trace(value, expected)

def test_repair_replays_baseline_noise_without_changing_draws(tmp_path, functions):
    from semantic_transmission import generation_noise as noise
    import json
    contract = dict(seed=2025, segments=1, steps=2)
    def produce(path, expected=None, repair=False):
        control = noise.NoiseControl(path, contract, {"device": "cpu-test"}, expected)
        refs = [[]]
        for index in range(2):
            ref = torch.ones(1, 1, 1, 1) * (index+1)
            with control.scope("vae", index, ref):
                torch.randn_like(ref)
            refs[0].append(ref)
        z = torch.zeros(1, 1, 3, 1, 1)
        with control.scope("initial", 0, z):
            z.copy_(torch.randn_like(z))
        if repair:
            fix.apply_guard(z, refs, ["0;0,1,0,-1,1"], 0, 5, 2, functions)
        else:
            functions.apply_mask_strategy(z, refs, ["0;0,1,0,-1,1"], 0, align=5)
        with control.scope("sampling", 0, z):
            for _ in range(2):
                torch.randn_like(z)
        control.finish(1, 2)
        return json.loads(path.read_text())
    before = produce(tmp_path / "before.json")
    after = produce(tmp_path / "after.json", before, True)
    assert after["matched_reference"]
    assert before["records"] == after["records"]
