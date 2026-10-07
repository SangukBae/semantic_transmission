import itertools

import pytest

from semantic_transmission.qwen35_state_delta import (
    apply_matches, budget_keys, caption_text, local_notes, uniform_keys,
    validate_caption, validate_matches,
    explicit_source_examples,
    caption_prompt, Session,
)


def draft():
    return dict(anchor=dict(text="A dog is visible.", evidence_frames=[10]),
        changes=[dict(text="It leaves the image.", before=10, after=15)],
        uncertainties=[], needs_check=False)


def test_selection_matches_bruteforce_and_holds_budget_gap_endpoints():
    scores = [0., 2., .3, 4., 1., .8, 2., 0.]
    result = budget_keys(scores, 4, max_gap=3, short_gap=2)
    def score(keys):
        return sum(scores[k] for k in keys) - sum(.5 * max(0, 2-(b-a)) for a,b in zip(keys,keys[1:]))
    feasible = [[0,*mid,7] for mid in itertools.combinations(range(1,7),2)
                if max(b-a for a,b in zip([0,*mid],[*mid,7])) <= 3]
    assert score(result) == max(map(score, feasible))
    assert len(set(result)) == 4 and result[0] == 0 and result[-1] == 7


def test_infeasible_budget_is_rejected_instead_of_silently_adding_keys():
    with pytest.raises(ValueError, match="infeasible"):
        budget_keys([0.] * 100, 4, max_gap=24)
    with pytest.raises(ValueError):
        budget_keys([0., float("nan"), 0.], 2)


def test_selection_budget_and_coverage_on_full_timeline():
    keys = budget_keys([0.] * 1440, 79)
    assert len(keys) == 79 and max(b-a for a,b in zip(keys,keys[1:])) <= 24
    assert min(b-a for a,b in zip(keys,keys[1:])) >= 6
    assert len(set(uniform_keys(1440,79))) == 79


def test_caption_evidence_cannot_cross_target_or_forget_first_frame():
    value = draft()
    validate_caption(value, [10,12,15])
    value["changes"][0]["after"] = 20
    with pytest.raises(ValueError):
        validate_caption(value, [10,12,15])
    value = draft(); value["anchor"]["evidence_frames"] = [12]
    with pytest.raises(ValueError, match="first TARGET"):
        validate_caption(value, [10,12,15])


def test_notes_never_copy_outside_or_unseen_frame_claims():
    rows = [dict(result=dict(facts=[dict(text="dog",visibility="visible",evidence_frames=[9,10]),
        dict(text="snow",visibility="visible",evidence_frames=[10])],
        events=[dict(before=9,after=10),dict(before=10,after=12)]))]
    result = local_notes(rows,[10,12])
    assert [f["text"] for f in result["facts"]] == ["snow"]
    assert result["events"] == [dict(before=10,after=12)]


def test_correspondence_cannot_use_neighbor_as_target_evidence():
    result = dict(verdicts=[dict(id="anchor",match="target",evidence_frames=[9]),
        dict(id="change_0",match="target",evidence_frames=[10,15])])
    with pytest.raises(ValueError, match="context"):
        validate_matches(result,[9,10,15,20],[10,15],draft())


def test_repair_only_removes_clear_mismatch_and_never_invents_replacement():
    result = dict(verdicts=[dict(id="anchor",match="target"),dict(id="change_0",match="neighbor")])
    revised, decision = apply_matches(draft(),result)
    assert caption_text(revised) == "A dog is visible." and decision["removed"] == [0]
    result["verdicts"][1]["match"] = "uncertain"
    assert apply_matches(draft(),result)[0] == draft()
    result["verdicts"][0]["match"] = "neither"
    assert apply_matches(draft(),result)[0] == draft()


def test_schema_repair_examples_use_real_target_ids_without_remapping_output():
    prompt='{"evidence_frames":[0],"before":0,"after":4} target data: {"target_ids": [100, 104]}'
    text=explicit_source_examples(prompt,[94,100,104,110])
    assert '"evidence_frames":[100]' in text and '"before":100,"after":104' in text
    assert '"target_ids": [100, 104]' in text


def test_caption_schema_follows_metadata_and_single_image_forbids_changes():
    prompt=caption_prompt(dict(prompts=dict(caption='OUTPUT {"anchor":{"evidence_frames":[0]},"changes":[]}')),[],100,101)
    assert prompt.index('"target"') < prompt.index('OUTPUT')
    assert '"evidence_frames":[100]' in prompt and 'changes MUST be []' in prompt


def test_schema_fallback_is_bounded_and_does_not_mask_cuda_errors(tmp_path,monkeypatch):
    from semantic_transmission import qwen35_feedback as fb
    calls=[]
    def fake(self,role,ids,prompt,validator,tokens):
        calls.append(role)
        if role.endswith('_schema'):return {'ok':True}
        raise RuntimeError('three invalid responses: example')
    monkeypatch.setattr(fb.Session,'call',fake)
    session=Session(tmp_path,{})
    assert session.call('caption',[100,104],'"evidence_frames":[0]',lambda x:x)=={'ok':True}
    assert calls==['caption','caption_schema']
    def cuda(*args):raise RuntimeError('CUDA out of memory')
    monkeypatch.setattr(fb.Session,'call',cuda)
    with pytest.raises(RuntimeError,match='CUDA'):
        session.call('caption',[100,104],'prompt',lambda x:x)


def test_syntax_recovery_preserves_claims_and_ids_despite_missing_brace():
    from semantic_transmission.qwen35_state_delta_resume import semantic_fields, normalized_json
    malformed='{"anchor":{"text":"A van is visible.","evidence_frames":[1128],"changes":[],"uncertainties":[],"needs_check":false}'
    valid='{"anchor":{"text":"A van is visible.","evidence_frames":[1128]},"changes":[],"uncertainties":[],"needs_check":false}'
    assert semantic_fields(malformed)==semantic_fields(valid)
    assert semantic_fields(malformed)!=semantic_fields(valid.replace('van','car'))
    assert semantic_fields(malformed)!=semantic_fields(valid.replace('1128','1130'))
    assert normalized_json(malformed)[0] == __import__('json').loads(valid)
    nested=malformed+'}'
    assert normalized_json(nested)[0] == __import__('json').loads(valid)
    with pytest.raises(ValueError):
        normalized_json('{"anchor":{"text":"broken text')
