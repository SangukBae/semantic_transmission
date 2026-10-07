import copy
import sys
from collections import Counter

import pytest
import torch

from semantic_transmission import qwen35_evidence_repair as runtime
from semantic_transmission.qwen35_evidence_repair_data import (
    ordered_paths, informative_queries, validate_patch, shortcut_audit, allocate_development, expand_reviewed_queries,
)


def query(groups):
    return dict(text='A dog lies down and sits again.',
                phases=[dict(text=f'Visible pose {n}', positions=p) for n, p in enumerate(groups)])


def test_middle_phase_is_required_even_with_correct_endpoints():
    good = torch.tensor([[9., 0., 0., 0.], [0., 9., 0., 0.], [0., 0., 0., 9.]])
    wrong = good.clone(); wrong[1] = torch.tensor([0., 0., 0., 9.])
    paths = [(0, 1, 3)]
    assert runtime.ordered_evidence_loss(wrong, paths) > runtime.ordered_evidence_loss(good, paths) + 2


def test_valid_alternatives_are_marginalized_without_requiring_one_arbitrary_frame():
    logits = torch.tensor([[8., 0., 0., 0.], [0., 0., 8., 0.], [0., 0., 0., 8.]])
    q = query([[0], [1, 2], [3]])
    assert ordered_paths(q) == [(0, 1, 3), (0, 2, 3)]
    assert runtime.ordered_evidence_loss(logits, ordered_paths(q)) < .01
    assert runtime.ordered_evidence_loss(logits, [(0, 1, 3)]) > 2


@pytest.mark.parametrize('paths', [[], [(0, 3)], [(0, 0, 3)], [(0, 4, 3)], [(3, 1, 0)]])
def test_invalid_full_paths_are_rejected(paths):
    with pytest.raises(ValueError):
        runtime.ordered_evidence_loss(torch.zeros(3, 4), paths)


def test_three_classes_preserve_unknown_instead_of_relabeling_as_false():
    row = dict(claims=[dict(verdict='supported', text='visible'),
                      dict(verdict='supported', text='also visible'),
                      dict(verdict='contradicted', text='opposed'),
                      dict(verdict='insufficient', text='unobserved')])
    assert Counter(c['verdict'] for c in runtime.selected_claims(row, 0)) == Counter(runtime.VERDICTS)
    assert runtime.selected_claims(row, 0)[0] != runtime.selected_claims(row, 1)[0]
    assert runtime.CHOICES['insufficient'] != runtime.CHOICES['contradicted']


def test_preference_gradient_rewards_correct_and_penalizes_rejected_caption():
    positive = torch.tensor(2., requires_grad=True); negative = torch.tensor(1., requires_grad=True)
    loss = runtime.preference_loss(positive, negative, .1)
    gradients = torch.autograd.grad(loss, (positive, negative))
    assert [float(g) for g in gradients] == [1., -1.]
    assert runtime.preference_loss(torch.tensor(.5), torch.tensor(2.), .1) == 0


def test_streamed_gradient_equals_joint_graph_including_dropout_and_middle_phase():
    torch.manual_seed(42)
    weights = torch.nn.Parameter(torch.randn(4, 4))
    inputs = [torch.randn(4) for _ in range(3)]
    before = weights.detach().clone(); state = torch.get_rng_state()
    def forward(x):
        return torch.nn.functional.dropout(x @ weights, p=.25, training=True)
    complete = runtime.ordered_evidence_loss(torch.stack([forward(x) for x in inputs]), [(0, 1, 3), (0, 2, 3)])
    (complete * .3).backward(); expected = weights.grad.clone(); expected_rng = torch.get_rng_state()
    weights.grad = None; torch.set_rng_state(state)
    value = runtime.streamed_backward([lambda x=x: forward(x) for x in inputs],
        lambda xs: runtime.ordered_evidence_loss(torch.stack(xs), [(0, 1, 3), (0, 2, 3)]), scale=.3)
    assert value == pytest.approx(float(complete.detach()), abs=1e-6)
    assert torch.allclose(weights.grad, expected, atol=1e-6)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    # Gradient calculation only: no optimizer or weight mutation in this test.
    assert torch.equal(weights.detach(), before)


def test_streamed_preference_matches_direct_gradient():
    weights = torch.nn.Parameter(torch.tensor([2., 1.]))
    runtime.streamed_backward([lambda: weights[0] ** 2, lambda: weights[1] ** 2],
                             lambda xs: runtime.preference_loss(xs[0], xs[1], .1), scale=.2)
    assert torch.allclose(weights.grad, torch.tensor([.8, -.4]))


def test_streamed_replay_fails_before_optimizer_if_forward_changes():
    weight = torch.nn.Parameter(torch.tensor(1.)); counter = [0]
    def forward():
        counter[0] += 1
        return weight + counter[0]
    with pytest.raises(ValueError, match='replay changed'):
        runtime.streamed_backward([forward], lambda xs: xs[0])
    assert weight.detach() == 1


def fixture():
    row = dict(id='r', split='train', source_sha256='source', keys=[0, 2, 4, 7],
               frames=[dict(sha256=f'h{i}') for i in range(8)])
    patch = dict(id='r', split='train', source_sha256='source', frame_sha256=['h0', 'h2', 'h4', 'h7'],
        caption='The dog lies down and sits again.', evidence_queries=[query([[0], [1, 2], [3]])],
        claims=[dict(text='A statement.', verdict=v, observations=[0, 1, 2, 3], reason='Reviewed evidence.') for v in runtime.VERDICTS],
        preference_pairs=[dict(rejected='A dog is visible.', kind='omission', reason='Missing action.')],
        review=dict(source_only=True, viewed_positions=[0, 1, 2, 3]))
    return row, patch


@pytest.mark.parametrize('mutation', ['test', 'hash', 'phase', 'reverse', 'unknown', 'subset', 'missing_review', 'duplicate_position'])
def test_data_rejects_leakage_unseen_frames_and_unreviewed_claims(mutation):
    row, patch = fixture(); validate_patch(patch, row)
    if mutation == 'test': row['split'] = patch['split'] = 'test'
    if mutation == 'hash': patch['frame_sha256'][1] = 'different'
    if mutation == 'phase': patch['evidence_queries'][0]['phases'][1]['positions'] = [8]
    if mutation == 'reverse': patch['evidence_queries'] = [query([[3], [1], [0]])]
    if mutation == 'unknown': patch['claims'][2]['verdict'] = 'contradicted'
    if mutation == 'subset': patch['claims'][0]['observations'] = [0, 3, 1]
    if mutation == 'missing_review': patch['review']['viewed_positions'] = [0, 3]
    if mutation == 'duplicate_position': patch['evidence_queries'][0]['phases'][1]['positions'] = [1, 1]
    with pytest.raises(ValueError): validate_patch(patch, row)


def test_shortcut_audit_counts_middle_information_and_skips_trivial_states():
    row = dict(evidence_queries=[query([[0], [1, 2], [3]]), query([[1, 2]]), query([[0, 1, 2, 3]])], claims=[])
    assert len(informative_queries(row)) == 2
    audit = shortcut_audit([row])
    assert audit['constant_AD_hits'] == 0
    assert audit['requires_middle_phase'] == 2
    assert audit['multi_phase_three_or_four'] == 1


def test_prompt_does_not_embed_support_or_verdict_answer():
    row, patch = fixture(); q = patch['evidence_queries'][0]
    before = runtime.evidence_prompt(row, q, 1)
    q['phases'][1]['positions'] = [2]
    assert runtime.evidence_prompt(row, q, 1) == before
    c = patch['claims'][0]; before = runtime.judgment_prompt(row, c)
    c['verdict'] = 'insufficient'; c['reason'] = 'Different hidden answer.'
    assert runtime.judgment_prompt(row, c) == before


@pytest.mark.parametrize('command', ['prepare', 'verify', 'dry-run', 'audit-inputs'])
def test_nontraining_commands_cannot_dispatch_to_train(monkeypatch, command):
    called = []
    for name in ('prepare', 'verify', 'dry_run', 'audit_inputs'):
        monkeypatch.setattr(runtime, name, lambda name=name: called.append(name) or {})
    monkeypatch.setattr(runtime, 'train', lambda *a: pytest.fail('No training permitted'))
    monkeypatch.setattr(sys, 'argv', ['repair', command]); runtime.main()
    assert called == [command.replace('-', '_')]


def test_train_requires_explicit_arm(monkeypatch):
    monkeypatch.setattr(runtime, 'train', lambda *a: pytest.fail('Missing explicit arm'))
    monkeypatch.setattr(sys, 'argv', ['repair', 'train'])
    with pytest.raises(SystemExit): runtime.main()


def test_development_allocation_moves_entire_source_and_excludes_parent_training():
    records = dict(train=[dict(id=f'r{i}', split='train', source_id='new', source_sha256='h', origin='grounded_expansion') for i in range(4)], validation=[])
    moved = allocate_development(records, ['new'], [])
    assert not moved['train'] and len(moved['validation']) == 4
    assert all(r['split'] == 'validation' for r in moved['validation'])
    assert len(records['train']) == 4  # Preserve the old frozen partition.
    with pytest.raises(ValueError):
        allocate_development(records, ['new'], [dict(source_id='different_alias', source_sha256='h')])


def test_adjacent_reviewed_transitions_preserve_middle_evidence():
    q = query([[0], [1, 2], [3]])
    expanded = expand_reviewed_queries([q])
    assert expanded[0] == q and len(expanded) == 3
    assert ordered_paths(expanded[1]) == [(0, 1), (0, 2)]
    assert ordered_paths(expanded[2]) == [(1, 3), (2, 3)]
    assert all((0, 3) not in ordered_paths(x) for x in expanded)
