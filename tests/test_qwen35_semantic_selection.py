import pytest

from semantic_transmission import qwen35_semantic_selection as semantic
from semantic_transmission import qwen35_pairs_video as video


def decision(reference, candidate, changed=False, uncertain=False):
    return dict(decision="uncertain" if uncertain else "changed" if changed else "same",
                kind="unknown" if uncertain else "position" if changed else "none",
                reference_state="Person stands at the left.",
                candidate_state="Person stands at the right." if changed else "Person stands at the left.",
                evidence_frames=[reference, candidate])


class Session:
    def __init__(self, judge):
        self.judge, self.calls = judge, []

    def call(self, role, ids, prompt, validator, tokens):
        self.calls.append((role, ids))
        value = self.judge(role, ids)
        validator(value)
        return value


def test_same_length_static_and_changing_sources_have_different_key_counts():
    stable = Session(lambda _, ids: decision(ids[0], ids[-1]))
    changing = Session(lambda _, ids: decision(ids[0], ids[-1], ids[-1] in [12, 28, 44]))
    a, _ = semantic.select(49, stable)
    b, _ = semantic.select(49, changing)
    assert a == [0, 48]
    assert b == [0, 12, 28, 44, 48]
    assert changing.calls[3][1] == [12, 16]  # Reference advances only after acceptance.
    assert changing.calls[5][1] == [12, 24]  # A skipped candidate is not the new reference.


def test_stable_long_video_never_gets_time_gap_or_count_filler_keys():
    for stride in (1, 4, 12):
        session = Session(lambda _, ids: decision(ids[0], ids[-1]))
        keys, trace = semantic.select(384, session, stride=stride)
        assert keys == [0, 383]
        assert all(r["reference"] == 0 and not r["selected"] for r in trace)


def test_uncertainty_gets_intermediate_evidence_before_acceptance():
    def judge(role, ids):
        return decision(ids[0], ids[-1], changed=role.endswith("review"), uncertain=not role.endswith("review"))
    session = Session(judge)
    keys, trace = semantic.select(5, session)
    assert keys == [0, 4] and trace[0]["reviewed"]
    assert session.calls == [("semantic_select", [0, 4]), ("semantic_select_review", [0, 1, 2, 3, 4])]


def test_unresolved_uncertainty_fails_without_silently_forcing_or_skipping():
    session = Session(lambda _, ids: decision(ids[0], ids[-1], uncertain=True))
    with pytest.raises(RuntimeError, match="selection incomplete"):
        semantic.select(9, session)
    assert len(session.calls) == 2


def test_same_state_cannot_be_selected_as_change_and_ids_must_match():
    value = decision(4, 8)
    with pytest.raises(ValueError, match="identical"):
        semantic.validate_decision(dict(value, decision="changed", kind="viewpoint"), 4, 8)
    with pytest.raises(ValueError, match="contradict"):
        semantic.validate_decision(dict(value, kind="viewpoint"), 4, 8)
    with pytest.raises(ValueError, match="endpoints"):
        semantic.validate_decision(dict(value, evidence_frames=[0, 1]), 4, 8)


def test_semantic_defaults_never_derive_budget_from_duration():
    for frames in (2, 384, 1440):
        settings = video.selection_settings(frames)
        assert settings["key_budget"] is None and settings["max_gap"] is None
        assert settings["selector"] == semantic.POLICY
    with pytest.raises(ValueError, match="no key budget"):
        video.selection_settings(384, budget=22)
    legacy = video.selection_settings(384, "fixed_budget", 22)
    assert legacy["key_budget"] == 22 and legacy["selector"] == video.LEGACY_SELECTOR


def test_endpoints_and_very_short_sources():
    session = Session(lambda _, ids: decision(ids[0], ids[-1]))
    assert semantic.select(2, session)[0] == [0, 1]
    assert semantic.candidates(10) == [4, 8, 9]
    semantic.validate_keys([0, 383], 384)  # No one-second constraint.
    for keys in ([0, 5, 4, 9], [0, 4, 4, 9], [0, 4, 8]):
        with pytest.raises(ValueError):
            semantic.validate_keys(keys, 10)


def test_selection_replays_actual_reference_pairs_on_resume():
    # A changed acceptance affects later input identities, so those calls cannot
    # be reused using the candidate alone as a cache key.
    cache, inferred = {}, []
    class CachedSession(Session):
        def call(self, role, ids, prompt, validator, tokens):
            key = (role, tuple(ids), prompt, tokens)
            if key not in cache:
                inferred.append(key)
                cache[key] = self.judge(role, ids)
            validator(cache[key])
            return cache[key]
    judge = lambda _, ids: decision(ids[0], ids[-1], ids[-1] in [8, 20])
    first = semantic.select(25, CachedSession(judge))
    count = len(inferred)
    assert semantic.select(25, CachedSession(judge)) == first
    assert len(inferred) == count
