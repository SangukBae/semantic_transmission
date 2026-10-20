import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from semantic_transmission import bsu_allocation as model_lib
from semantic_transmission import bsu_allocation_data as data
from semantic_transmission import bsu_allocation_training as training
from semantic_transmission import bsu_allocation_transport as transport
from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.packets import accounting
from semantic_transmission.wire import pack, unpack


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def make_candidate(root, identifier, keys, etas, frames=9, *, extra_bytes=0):
    root.mkdir(parents=True)
    values = np.ones(len(keys)*2, dtype='<c8')
    values.tofile(root/'visual.c64')
    header = dict(video=dict(frames=frames), visual_codec=dict(eta=.2), keyframes=[
        dict(index=k, allocation_eta=e, complex_count=2, complex_offset=i*2)
        for i, (k, e) in enumerate(zip(keys, etas))])
    (root/'metadata.bin').write_bytes(pack(header))
    extras = []
    if extra_bytes:
        (root/'extra.bin').write_bytes(b'x'*extra_bytes)
        extras.append(root/'extra.bin')
    return dict(id=identifier, keys=keys, etas=etas,
                bundle=transport.bundle(root/'visual.c64', root/'metadata.bin', extras))


def make_dataset(root, *, splits=('train', 'validation'), feature_dim=5):
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, split in enumerate(splits):
        case = root/f'case{index}'
        case.mkdir()
        frames = 9+index
        rng = np.random.default_rng(index)
        features = rng.normal(size=(frames, feature_dim)).astype(np.float32)
        cache = case/'features.npz'
        np.savez(cache, features=features, frame_indices=np.arange(frames),
                 times=np.arange(frames, dtype=np.float32)/24)
        candidates = [
            make_candidate(case/'base', 'baseline', [0, 4, frames-1], [.2]*3, frames),
            make_candidate(case/'more', 'more', [0, 2, 4, frames-1], [.12, .15, .18, .12], frames),
            make_candidate(case/'costly', 'costly', [0, frames-1], [.8, .7], frames, extra_bytes=4000)]
        labels = [(30, .25, .3), (32, .22, .2), (40, .01, .01)]
        for c, metrics in zip(candidates, labels):
            c['metrics'] = dict(zip(data.METRICS, metrics))
        rows.append(dict(id=f'row{index}', source_id=f'original{index}', source_sha256=f'{index:064x}',
            split=split, frames=frames, budget=10000, snr_db=10., baseline_id='baseline',
            feature=dict(**transport.receipt(cache), source_only=True), candidates=candidates))
    document = dict(schema=data.SCHEMA, feature_schema='synthetic_source_features_v1',
        feature_dim=feature_dim, pipeline_id='synthetic_test_only', records=rows)
    path = root/'dataset.json'
    write_json(path, document)
    return path


def config(steps=12):
    result = training.read(training.CONFIG)
    result['model'].update(feature_dim=5, hidden=32, heads=4, temporal_layers=1,
                           candidate_layers=1, dropout=.1)
    result['training'].update(max_steps=steps, batch_size=2, warmup_steps=2,
        save_every=2, validate_every=4, learning_rate=.003, cpu_threads=1)
    return result


def test_label_builder_validates_best_checkpoint_without_claiming_independent_test(tmp_path):
    from semantic_transmission.bsu_allocation_labels import validate_predictor
    dataset=make_dataset(tmp_path)
    training.train(dataset,tmp_path/'predictor',config(steps=4),device='cpu')
    result=validate_predictor(tmp_path)
    assert result['sources']==1 and result['budget_violations']==0
    assert result['checkpoint_selected_using_this_validation_set'] is True
    assert result['untouched_test'] is False
    assert set(result['mean_metric_delta_vs_baseline'])==set(data.METRICS)
    assert len(result['utility_gain_bootstrap_95_percent'])==2


def test_exact_budget_counts_visual_and_independent_fec_packets(tmp_path):
    c = make_candidate(tmp_path/'c', 'c', [0,8], [.1,.2], extra_bytes=768)
    before = transport.wire_cost(c['bundle'], c, 9)
    extra = tmp_path/'c/extra.bin'
    extra.write_bytes(b'x'*769)
    c['bundle']['digital_packets'][-1] = transport.receipt(extra)
    after = transport.wire_cost(c['bundle'], c, 9)
    assert after['total_uses']-before['total_uses'] == 2304
    assert after['visual_uses'] == 4
    assert after['digital_uses'] == sum(accounting(Path(p['path']).stat().st_size)['complex_channel_uses']
                                         for p in c['bundle']['digital_packets'])


def test_feasibility_int64_boundary_and_no_feasible_plan():
    # Above float32 AND float64 exact integer ranges.
    budget = torch.tensor([2**54], dtype=torch.long)
    costs = torch.tensor([[2**54, 2**54+1]], dtype=torch.long)
    assert model_lib.choose(torch.tensor([[1., 99.]]), costs, budget, torch.ones(1,2,dtype=torch.bool)).item() == 0
    with pytest.raises(ValueError, match='INFEASIBLE'):
        model_lib.choose(torch.zeros(1,2), costs, budget-1, torch.ones(1,2,dtype=torch.bool))
    with pytest.raises(ValueError, match='int64'):
        model_lib.feasible_mask(costs.float(), budget, torch.ones(1,2,dtype=torch.bool))


def test_selection_ignores_labels_and_rechecks_actual_bundle(tmp_path):
    path = make_dataset(tmp_path/'data')
    _, records = data.load_dataset(path)
    row = records[0]['row']
    chosen = transport.select_verified(row['candidates'], [0., 1., 1000.], row['budget'], row['frames'])
    assert chosen['candidate_id'] == 'more'
    assert chosen['ledger']['total_uses'] <= row['budget']
    file = Path(row['candidates'][1]['bundle']['visual']['path'])
    file.write_bytes(file.read_bytes()+b'12345678')
    with pytest.raises(ValueError, match='hash'):
        transport.select_verified(row['candidates'], [0.,1.,2.], row['budget'], row['frames'])


@pytest.mark.parametrize('keys,etas', [([1,8],[.2,.2]),([0,4,4,8],[.2]*4),([0,8],[.2]),
                                     ([0,8],[0.,.2]),([0,8],[float('nan'),.2])])
def test_invalid_joint_allocations_rejected(keys, etas):
    with pytest.raises(ValueError):
        transport.validate_candidate(dict(keys=keys,etas=etas), 9)


def test_candidate_grid_includes_nonuniform_rates_and_bounds_space():
    grid = transport.candidate_grid([[0,8],[0,4,8]], [.1,.2], 9)
    assert len(grid) == 12
    assert any(c['etas'] == [.1,.2,.1] for c in grid)
    with pytest.raises(ValueError, match='exceeds'):
        transport.candidate_grid([[0,4,8]], [.1,.2], 9, max_candidates=4)


@pytest.mark.parametrize('identity', ['source_id', 'source_sha256'])
def test_source_leakage_is_rejected_across_originals_and_hashes(tmp_path, identity):
    path = make_dataset(tmp_path/'data')
    value = training.read(path)
    value['records'][1][identity] = value['records'][0][identity]
    write_json(path, value)
    with pytest.raises(ValueError, match='leakage'):
        data.load_dataset(path)


def test_inference_never_reads_target_metrics(tmp_path):
    path = make_dataset(tmp_path/'data')
    _, records = data.load_dataset(path, require_labels=False)
    first = data.collate(records, config()['objective'], labels=False)
    value = training.read(path)
    for row in value['records']:
        for candidate in row['candidates']:
            candidate['metrics'] = {'forbidden_or_missing': 'must not be read'}
    write_json(path, value)
    _, records = data.load_dataset(path, require_labels=False)
    second = data.collate(records, config()['objective'], labels=False)
    assert set(first['inputs']) == set(data.MODEL_INPUTS)
    assert all(torch.equal(v, second['inputs'][k]) for k,v in first['inputs'].items())


def test_variable_lengths_padding_and_candidate_order_invariance(tmp_path):
    path = make_dataset(tmp_path/'data')
    _, records = data.load_dataset(path)
    spec = model_lib.ModelConfig(**config()['model'])
    model = model_lib.JointAllocationPredictor(spec).eval()
    batch = data.collate(records, config()['objective'])
    with torch.no_grad():
        result = model(**batch['inputs'])
        alone = model(**data.collate(records[:1], config()['objective'])['inputs'])
        assert torch.allclose(result['scores'][0], alone['scores'][0], atol=1e-6)
        inputs = copy.deepcopy(batch['inputs'])
        inputs['features'][~inputs['frame_mask']] = float('nan')
        inputs['times'][~inputs['frame_mask']] = float('nan')
        masked = model(**inputs)
        assert torch.allclose(result['scores'], masked['scores'], atol=1e-6)
        order = [2,0,1]
        for key in ('key_positions','etas','key_mask','candidate_mask'):
            inputs[key] = inputs[key][:,order]
        inputs['baseline_index'] = torch.ones(2, dtype=torch.long)
        permuted = model(**inputs)
        assert torch.allclose(result['scores'][:,order], permuted['scores'], atol=1e-6)
        assert torch.allclose(result['quality'][:,order], permuted['quality'], atol=1e-6)
        assert torch.count_nonzero(result['quality'][:,0]) == 0


def test_variable_candidate_count_padding_has_no_effect(tmp_path):
    path = make_dataset(tmp_path/'data')
    document = training.read(path)
    document['records'][1]['candidates'].pop()
    write_json(path, document)
    _, records = data.load_dataset(path)
    model = model_lib.JointAllocationPredictor(model_lib.ModelConfig(**config()['model'])).eval()
    batch = data.collate(records, config()['objective'])
    with torch.no_grad():
        together = model(**batch['inputs'])
        alone = model(**data.collate(records[1:], config()['objective'])['inputs'])
    assert torch.allclose(together['scores'][1,:2], alone['scores'][0], atol=1e-6)
    assert torch.isfinite(together['scores']).all()
    assert not batch['inputs']['candidate_mask'][1,2]


def test_joint_predictor_uses_time_rate_budget_and_feature_context(tmp_path):
    path = make_dataset(tmp_path/'data')
    _, records = data.load_dataset(path)
    model = model_lib.JointAllocationPredictor(model_lib.ModelConfig(**config()['model'])).eval()
    original = data.collate(records[:1], config()['objective'])['inputs']
    reference = model(**original)['scores']
    for key in ('times', 'etas', 'budget', 'snr', 'features'):
        inputs = copy.deepcopy(original)
        inputs[key] = inputs[key]*2
        assert not torch.allclose(model(**inputs)['scores'], reference)


def test_rank_and_regret_gradients_reward_feasible_good_decisions():
    scores = torch.tensor([[0.,0.,100.]], requires_grad=True)
    q = torch.zeros(1,3,3, requires_grad=True)
    valid = torch.tensor([[True,True,False]])
    losses = model_lib.allocation_loss(dict(scores=scores,quality=q), torch.zeros_like(q),
        torch.tensor([[0.,2.,1000.]]), valid, torch.ones_like(valid))
    losses['loss'].backward()
    assert scores.grad[0,1] < 0 < scores.grad[0,0]
    assert scores.grad[0,2] == 0
    # One feasible candidate and tied utilities remain finite and differentiable.
    loss = model_lib.allocation_loss(dict(scores=scores,quality=q), torch.zeros_like(q),
        torch.zeros(1,3), torch.tensor([[True,False,False]]), torch.ones_like(valid))
    assert float(loss['ranking']) == 0 and float(loss['regret']) == 0


def test_training_resume_matches_uninterrupted_optimizer_and_sampling(tmp_path):
    dataset = make_dataset(tmp_path/'data')
    cfg = config(8)
    full = training.train(dataset, tmp_path/'full', cfg)
    partial = training.train(dataset, tmp_path/'resume', cfg, stop_after=4)
    assert partial['status'] == 'INTERRUPTED_RESUMABLE'
    resumed = training.train(dataset, tmp_path/'resume', cfg)
    def snapshot(folder):
        receipt = training.read(tmp_path/folder/'latest.json')
        return torch.load(receipt['path'], weights_only=True)
    a,b = snapshot('full'), snapshot('resume')
    assert all(torch.equal(v,b['model'][k]) for k,v in a['model'].items())
    assert a['training']['history'] == b['training']['history']
    assert a['scheduler'] == b['scheduler']
    assert torch.equal(a['sampling_rng'], b['sampling_rng'])
    assert full['status'] == resumed['status'] == 'COMPLETE_ALLOCATION_TRAINING'
    chosen = training.select(dataset, tmp_path/'resume/best.json')
    assert len(chosen['selections']) == 2
    assert all(r['ledger']['total_uses'] <= r['budget'] for r in chosen['selections'])
    cfg['training']['learning_rate'] *= 2
    with pytest.raises(ValueError, match='frozen'):
        training.train(dataset, tmp_path/'resume', cfg)


def test_development_data_cannot_be_silently_trained_or_deployed(tmp_path):
    dataset = make_dataset(tmp_path/'data', splits=('development','development'))
    with pytest.raises(ValueError, match='source-disjoint'):
        training.train(dataset, tmp_path/'bad', config(2))
    result = training.train(dataset, tmp_path/'smoke', config(2), development=True)
    assert result['development_only'] and not result['quality_improvement_proven']
    with pytest.raises(ValueError, match='development-only'):
        training.select(dataset, tmp_path/'smoke/best.json')
    selected = training.select(dataset, tmp_path/'smoke/best.json', allow_development=True)
    assert len(selected['selections']) == 2


def test_source_feature_extractor_and_tamper_detection(tmp_path):
    paths = []
    for i in range(3):
        frame = tmp_path/f'{i}.png'
        Image.new('RGB',(16,16),(i*60,0,0)).save(frame)
        paths.append(frame)
    feature = data.extract_features(paths, 24., tmp_path/'source.npz')
    with np.load(feature['path']) as values:
        assert values['features'].shape == (3,71)
        assert np.isfinite(values['features']).all()
        assert values['features'][1,48:64].mean() > 0
    assert feature['source_only']
    with pytest.raises(FileExistsError):
        data.extract_features(paths,24.,tmp_path/'source.npz')


def test_per_key_encoding_preserves_rates_and_restores_codec_state(tmp_path, monkeypatch):
    from semantic_transmission import ntscc_plus_codec as codec
    source = tmp_path/'source'
    source.mkdir()
    for i in (0,4,8):
        Image.new('RGB',(64,64),(i,0,0)).save(source/f'{i:05d}.png')
    seen = []
    def encode(model, image):
        seen.append(model.eta)
        return torch.tensor([1+2j,3+4j]), torch.tensor([1,1]), .5
    monkeypatch.setattr(codec, 'encode_symbols', encode)
    model = SimpleNamespace(eta=.2)
    candidate = dict(keys=[0,4,8], etas=[.1,.3,.15])
    row = dict(video=dict(frames=9, width=64, height=64, fps=24), keys=[0,8],
               segments=[dict(text='A person moves.',flow=1.)])
    state = torch.get_rng_state().clone()
    value = transport.encode_candidate(model, row, candidate, source, tmp_path/'encoded', device='cpu')
    assert seen == candidate['etas'] and model.eta == .2
    assert torch.equal(state, torch.get_rng_state())
    assert transport.wire_cost(value, candidate, 9)['visual_uses'] == 6
    header,_ = unpack(Path(value['digital_packets'][0]['path']).read_bytes())
    assert header['schema'] == transport.SCHEMA
    assert [k['allocation_eta'] for k in header['keyframes']] == candidate['etas']
    # Source files need not exist at the receiver.
    received = tmp_path/'received'
    received.mkdir()
    for name in ('metadata.bin','visual.c64'):
        (received/name).write_bytes((tmp_path/'encoded'/name).read_bytes())
    for path in source.iterdir():
        path.unlink()
    def decode(model, values, indices, power, height, width):
        assert values.tolist() == [1+2j,3+4j]
        assert indices.tolist() == [1,1]
        return torch.ones(1,3,height,width)*.5
    monkeypatch.setattr(codec, 'decode_symbols', decode)
    report = transport.decode_received(received, tmp_path/'decoded', model=model, device='cpu')
    assert not report['source_pixels_read'] and report['keys'] == candidate['keys']
    assert sorted(p.name for p in (tmp_path/'decoded').iterdir()) == ['0.png','4.png','8.png']
