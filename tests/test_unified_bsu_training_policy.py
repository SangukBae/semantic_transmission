import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from semantic_transmission import unified_bsu_core as core
from semantic_transmission import unified_bsu_training_policy as policy
from semantic_transmission.artifacts import sha256, write_json


def config():
    return policy.config(core.REPO / 'configs/bsu_unified_training_etri.json')


def context(frames=300, fps=24):
    keys = [0, frames//3, 2*frames//3, frames-1]
    return dict(video=dict(frames=frames, fps=fps, width=64, height=96), indices=keys,
        segments=[dict(path=f'clips/sample/{a}_{b}.mp4', text=f'scene {i}', flow=float(i))
                  for i, (a, b) in enumerate(zip(keys, keys[1:]))], decoder={})


def test_cropped_captions_preserve_changes_and_actual_local_times():
    cropped = policy.crop_context(context(), 90, 53)
    assert cropped['indices'] == [0, 10, 52]
    assert [r['text'] for r in cropped['segments']] == ['scene 0', 'scene 1']
    assert cropped['video']['frames'] == 53
    core.validate_context(cropped)
    with pytest.raises(ValueError):
        policy.crop_context(context(), 290, 53)


def test_long_timeline_sampling_keeps_tail_middle_and_transitions():
    starts = policy.sequence_starts(14400, 53, 12, [0, 401, 3251, 8200, 14399])
    assert starts[0] == 0 and starts[-1] == 14400-53
    assert any(6000 < s < 9000 for s in starts)
    assert 401-53//2 in starts
    assert len(starts) == len(set(starts)) == 12
    assert policy.sequence_starts(17, 53, 12, [0, 16]) == [0]


def test_expanded_plan_uses_original_split_varied_lengths_rates_and_long_validation(tmp_path, monkeypatch):
    sources = []
    for n, role in enumerate(['train', 'validation', 'validation']):
        directory = tmp_path / str(n); directory.mkdir()
        for i in range(301):
            (directory / f'{i}.png').write_bytes(f'{n}:{i}'.encode())
        sources.append(dict(source_id=f'video_{n}', source_sha256=str(n)*64, role=role,
            dataset='tvsum', directory=str(directory), context=context(301, fps=1), caption_provenance='test'))
    monkeypatch.setattr(policy, 'source_catalog', lambda *args: (sources, {}, []))
    examples, quality, summary, _, _ = policy.expanded_plan(None, config(), core.profile())
    train = [e for e in examples if e['role'] == 'train']
    assert {len(e['source_frames']) for e in train} == {53, 101, 197}
    assert {e['budget_per_frame'] for e in train} == {768, 1536}
    assert all(e['source_sha256'] != train[0]['source_sha256'] for e in quality)
    assert summary['quality_seconds'] == [60, 120]
    assert quality[-1]['source_start'] == 181
    assert summary['long_term_generator_memory'] is False


def test_balanced_sampler_covers_every_sequence_and_restores_deterministically():
    examples = [dict(dataset=d, source_sha256=s) for d, s, n in
                [('webvid', 'a', 4), ('webvid', 'b', 2), ('tvsum', 'c', 8)] for _ in range(n)]
    rng = np.random.default_rng(4); saved = copy.deepcopy(rng.bit_generator.state)
    order = policy.balanced_order(examples, rng)
    assert set(order) == set(range(len(examples)))
    assert sum(examples[i]['dataset'] == 'webvid' for i in order) == sum(examples[i]['dataset'] == 'tvsum' for i in order)
    rng.bit_generator.state = saved
    assert policy.balanced_order(examples, rng) == order


def test_overlap_proxy_detaches_previous_window_and_has_current_gradient():
    previous = torch.zeros(1, 4, 5, 2, 2, requires_grad=True)
    current = torch.ones(1, 4, 5, 2, 2, requires_grad=True)
    value = policy.overlap_error_loss(previous, current, 0, 12)
    assert float(value) == 1.
    value.backward()
    assert previous.grad is None and current.grad.abs().sum() > 0
    assert float(policy.overlap_error_loss(current, current, 0, 12)) == 0.


def test_stopping_requires_joint_updates_and_completed_coverage_epochs():
    c = config(); s = dict(step=2200, epoch=2, offset=4, order=list(range(5)), loss_monitor=dict(bad=8))
    assert not policy.plateau_ready(s, c, 200)
    s['offset'] = 5
    assert policy.plateau_ready(s, c, 200)
    s['step'] = 2199
    assert not policy.plateau_ready(s, c, 200)
    m, improved = policy.update_monitor(None, 1., 2200, c)
    assert improved and m['step'] == 2200
    m, improved = policy.update_monitor(m, .999, 2300, c)
    assert not improved and m['bad'] == 1 and m['step'] == 2200


def test_sequential_gradients_average_windows_and_accumulation():
    from semantic_transmission.unified_bsu_training import Trainer
    t = Trainer.__new__(Trainer); t.p = dict(training_config=dict(overlap_consistency_weight=0.))
    weight = torch.nn.Parameter(torch.tensor(2.))
    t.load = lambda e: dict(windows=[dict(start=0), dict(start=12)])
    t.loss = lambda e, seed, **kw: (weight*(kw['window_index']+1), {'loss': float(weight.detach())}, None)
    value, _ = t.backward_sequence({}, 1, accumulation=2)
    assert value == 3. and float(weight.grad) == .75


def test_time_metric_matches_full_array_formula_without_full_float_video():
    from semantic_transmission.unified_bsu_evaluate import metrics
    rng = np.random.default_rng(5)
    source = rng.integers(0, 256, (9, 16, 16, 3), dtype=np.uint8)
    output = rng.integers(0, 256, source.shape, dtype=np.uint8)
    result = metrics(source, output, fps=3)
    expected = np.abs(np.diff(source.astype(float), axis=0)-np.diff(output.astype(float), axis=0)).mean()
    assert result['temporal_rgb_mae'] == pytest.approx(expected)
    assert len(result['temporal_timeline']) == 3


def test_evaluator_compares_decoded_mp4_not_png_intermediates(tmp_path, monkeypatch):
    from semantic_transmission import unified_bsu_evaluate as evaluate, research_quality, etri_60s_check
    source_dir = tmp_path / 'source'; source_dir.mkdir()
    for i in range(3):
        (source_dir / f'{i}.png').write_bytes(bytes([i]))
    movie = tmp_path / 'refined/reconstruction.mp4'; movie.parent.mkdir(); movie.write_bytes(b'encoded')
    (movie.parent / 'frames').mkdir()
    write_json(tmp_path / 'protocol.json', dict(signature='test', source=str(source_dir),
        source_hashes={f'{i}.png': sha256(source_dir / f'{i}.png') for i in range(3)}, context=context(3)))
    write_json(tmp_path / 'RESULT.json', dict(request_signature='test', movie=str(movie),
        movie_sha256=sha256(movie), channel_uses=100, budget=100, budget_met=True))
    source = np.zeros((3, 16, 16, 3), dtype=np.uint8)
    def load(path, count=None):
        assert Path(path) == source_dir, 'must not evaluate PNG intermediate output'
        return source
    from pathlib import Path
    monkeypatch.setattr(evaluate.u, 'load_rgb', load)
    monkeypatch.setattr(research_quality, 'read_video', lambda path: source+10)
    monkeypatch.setattr(etri_60s_check, 'pts_audit', lambda *args: None)
    result = evaluate.evaluate(tmp_path)
    assert result['metric_input'] == 'decoded_final_mp4'
    assert result['rows']['refined']['psnr_db'] == pytest.approx(28.1308036)


def test_quality_gate_detects_short_temporal_failure_hidden_by_mean():
    from semantic_transmission.unified_bsu_training_quality import summarize_quality
    base = dict(official_quality=dict(psnr_db=30., lpips_vgg=.2), temporal_rgb_mae=1., worst_second_temporal_rgb_mae=2.)
    final = copy.deepcopy(base); final['worst_second_temporal_rgb_mae'] = 5.
    row = dict(name='late-event', dataset='tvsum', quality=dict(rows=dict(refined=final),
        comparison=dict(reference=base), budget_met=True))
    result = summarize_quality([row], config())
    assert result['score'] == 2. and not result['guard_passed']
    assert result['guards'][0]['temporal'] and not result['guards'][0]['worst_second']


def test_catalog_excludes_etri_parent_and_reserved_test_and_keeps_old_roles(tmp_path, monkeypatch):
    from semantic_transmission.wire import pack
    c = config()
    monkeypatch.setattr(core, 'REPO', tmp_path)
    packet = tmp_path / 'metadata.bin'; ctx = context(200)
    packet.write_bytes(pack(dict(video=ctx['video'], keyframes=[dict(index=i) for i in ctx['indices']],
                                 segments=ctx['segments'])))
    existing = [dict(id='old_'+role, dataset='webvid', role=role, source_sha256=role*8,
        source_directory=str(tmp_path), packet=str(packet)) for role in ('train', 'validation')]
    selection = tmp_path / 'selection.json'; write_json(selection, dict(examples=existing))
    manifest = tmp_path / 'reserved.csv'; manifest.write_text('source_id,source_sha256\nparent,blocked\n')
    usage = tmp_path / 'usage.json'
    write_json(usage, dict(newly_used_evaluation_sources=[dict(source_sha256='testhash', split='test')]))
    inventory = tmp_path / 'inventory.json'
    videos = [dict(id='reserved-extension', dataset='TVSum', source_id='parent', source_sha256='different-clip', benchmark_memberships=[]),
              dict(id='previous-test', dataset='TVSum', source_id='heldout', source_sha256='testhash', benchmark_memberships=[])]
    for n in range(4):
        source = tmp_path / f'original{n}'; source.write_bytes(str(n).encode()); digest = sha256(source)
        folder = tmp_path / 'tvsum' / f'v{n}'
        write_json(folder / 'source_prepared.json', dict(source_sha256=digest, normalized_sha256=digest,
            normalized_video=str(source), source_frames=str(tmp_path), source_frame_hashes={},
            frames=200, fps=24, width=64, height=96))
        write_json(folder / 'extraction/assistant_captions/captions_bundle.json', dict(input_sha256=digest,
            records=[dict(start=0, end_exclusive=200, text='Test caption.')], source_frame_hashes={}))
        videos.append(dict(id=f'v{n}', dataset='TVSum', source_id=f'original{n}', source_sha256=digest,
                           source_path=str(source), benchmark_memberships=[]))
    write_json(inventory, dict(videos=videos))
    c.update(excluded_manifests=['reserved.csv'], previous_source_usage='usage.json',
                            additional_inventory='inventory.json')
    sources, _, excluded = policy.source_catalog(selection, c)
    assert len(excluded) == 2
    assert {s['source_id']: s['role'] for s in sources if s['dataset'] == 'webvid'} == {
        'old_train': 'train', 'old_validation': 'validation'}
    assert {s['role'] for s in sources if s['dataset'] == 'tvsum'} == {'train', 'validation'}
    assert not {'different-clip', 'testhash'} & {s['source_sha256'] for s in sources}


@pytest.mark.parametrize('quality_every', [2, 4])
def test_controller_resumes_pending_quality_before_training_and_selects_guarded_best(tmp_path, monkeypatch, quality_every):
    from semantic_transmission import unified_bsu_training_quality as q
    c = config(); c.update(quality_every=quality_every, quality_patience=1, min_joint_steps=2, max_steps=6)
    config_path = tmp_path / 'config.json'; write_json(config_path, c)
    selection = tmp_path / 'selection.json'; write_json(selection, {})
    write_json(tmp_path / 'protocol.json', {}); write_json(tmp_path / 'prepared.json', {})
    profile = core.REPO / 'configs/bsu_unified_sender.json'
    p = dict(signature='controller-test', training_config=c, profile=core.profile(profile),
        selection_sha256=sha256(selection), train=dict(innovation_warmup_steps=0), runtime=dict(python='fake'))
    monkeypatch.setattr(q.training, 'verify_plan', lambda root: p)
    saved = {}; calls = []; fail_quality = [True]
    monkeypatch.setattr(q, 'load_checkpoint', lambda *args: saved.copy() or None)
    def launch(command, log):
        assert command[3] == 'train'
        calls.append('train')
        step = saved.get('step', 0)+2
        saved.update(step=step, state=dict(step=step, epoch=2, order=[0], offset=1,
            elapsed_seconds=step, loss_monitor=dict(bad=8)))
        weights = tmp_path / f'{step}.pt'; weights.write_bytes(str(step).encode())
        write_json(tmp_path / 'export.json', dict(step=step, path=str(weights), sha256=sha256(weights)))
    def evaluate(root, protocol, export):
        calls.append('quality')
        write_json(root / 'quality' / f'step_{export["step"]:07d}' / 'protocol.json', dict(step=export['step']))
        if fail_quality[0]:
            fail_quality[0] = False
            raise RuntimeError('simulated interrupted decoder')
        value = dict(step=export['step'], score=1. if export['step'] == 2 else .9,
                     guard_passed=export['step'] != 2)
        write_json(root / 'quality' / f'step_{export["step"]:07d}' / 'RESULT.json', value)
        return value
    monkeypatch.setattr(q, 'launch', launch); monkeypatch.setattr(q, 'evaluate_checkpoint', evaluate)
    args = SimpleNamespace(root=tmp_path, training_config=config_path, profile=profile, selection=selection,
                           codec_warmup_steps=0, initialize=None, seconds=0)
    with pytest.raises(RuntimeError, match='interrupted decoder'):
        q.run(args)
    calls.clear(); q.run(args)
    assert calls[0] == 'quality' and calls.count('train') == 2
    assert core.read(tmp_path / 'controller.json')['status'] == 'VALIDATION_PLATEAU'
    best = core.read(tmp_path / 'best_quality.json')
    assert best['step'] == 4 and best['guard_passed']
    assert len(core.read(tmp_path / 'controller.json')['history']) == 3
