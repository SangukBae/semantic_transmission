from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from semantic_transmission import bsu_ltx_pipeline as pipeline, ltx_receiver as ltx
from semantic_transmission import qwen35_pairs_video as runner
from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.wire import pack, unpack
from semantic_transmission.webvid_ablation import Stages


class Generator:
    calls = []
    fail = False
    fail_start = 3

    def __init__(self, assets):
        assert assets == {}

    def __call__(self, window, request):
        assert not request['source_pixels_available']
        assert not request['packet_decoder_overridden_by_registered_profile']
        image_paths = {i['path'] for w in request['windows'] for i in w['images']}
        assert len(request['inputs']) == len(image_paths) + 1  # packet and received keys only
        self.calls.append(window['start'])
        for i in range(window['generated_frames']):
            if self.fail and window['start'] == self.fail_start:
                raise RuntimeError('interrupted GPU')
            yield np.full((320, 576, 3), window['start']*10+i, dtype=np.uint8)


@pytest.fixture
def run(tmp_path, monkeypatch):
    return make_run(tmp_path, monkeypatch)


def make_run(tmp_path, monkeypatch, indices=(0, 3, 8), colors=None, mode=None):
    run = tmp_path / 'run'
    keys = run / 'receiver/frames/sample/key_frames_received'; keys.mkdir(parents=True)
    for n, i in enumerate(indices):
        Image.new('RGB', (576, 320), colors[n] if colors else (i, 0, 0)).save(keys / f'{i}.png')
    cfg = dict(generation_conditioning=mode) if mode else {}
    decoder = dict(pipeline.decoder_fields(cfg), seed=2025, steps=11, policy='endpoint_exact',
                   concatenation_policy='endpoint_exact')
    header = dict(video=dict(frames=indices[-1]+1, fps=24, width=576, height=320), decoder=decoder,
        keyframes=[dict(index=i) for i in indices],
        segments=[dict(text=f'A person moves at step {i}.', flow=1.) for i in range(len(indices)-1)])
    (run / 'received').mkdir(); (run / 'received/metadata.bin').write_bytes(pack(header))
    write_json(run / 'receiver/decoder_inputs.json', dict(video=header['video'], decoder=decoder, indices=list(indices)))
    source = tmp_path / 'source.mp4'; source.write_bytes(b'source audit identity only')
    write_json(run / 'run_config.json', dict(header['video'], **cfg, input=str(source), input_sha256=sha256(source)))
    write_json(run / 'input_audit.json', dict(source_sha256=sha256(source)))
    write_json(run / 'channel_accounting.json', dict(status='PASSED', metadata_exact_match=True))
    write_json(run / 'keyframes.json', dict(indices=list(indices)))
    monkeypatch.setattr(pipeline, 'assets', lambda cfg: {})
    Generator.calls=[]; Generator.fail=False; Generator.fail_start=3
    pipeline.prepare(run)
    return run


def test_ltx_dispatch_excludes_opensora_preparation_and_uses_isolated_runtime(monkeypatch):
    from semantic_transmission import qwen35_pairs_comparison as pc
    jobs = list(pipeline.jobs())
    assert 'prepare-text' not in [stage for stage, _, _ in jobs]
    assert 'prepare-generator' in [stage for stage, _, _ in jobs]
    reconstruction = next(p for stage, _, p in jobs if stage == 'reconstruct')
    assert reconstruction == ['receiver/ltx_generation', 'receiver/reconstruction']
    p = dict(receiver=dict(generator_backend='ltx25', visual_codec='NTSCC+'),
             interpreters=dict(python='general', channel_python='channel', ltx_python='ltx', codec_python='codec'))
    assert [runner.stage_interpreter(p, s) for s in ('send','channel','receive','reconstruct','evaluate')] == [
        'codec','channel','codec','ltx','general']
    p['receiver'].pop('generator_backend')
    assert runner.stage_interpreter(p, 'reconstruct') == 'general'
    p['receiver']['generator_backend'] = 'ltx25'
    monkeypatch.setattr(pc, 'load_protocol', lambda root: p)
    monkeypatch.setattr(pc.fc, 'reconstruct', lambda run: pytest.fail('Open-Sora dispatched'))
    calls=[]
    monkeypatch.setattr(pipeline, 'reconstruct', lambda run: calls.append(run))
    runner.worker(Path('/synthetic/base'), 'reconstruct')
    assert calls == [Path('/synthetic/base/reconstruction/event_pairs/run')]


def test_received_only_generation_publication_audit_and_outer_stage_resume(run):
    root = run.parent
    required = ['run/receiver/ltx_generation', 'run/receiver/reconstruction']
    owned = pipeline.owned_products('reconstruct', required)
    Generator.fail=True
    def reconstruct(_):
        pipeline.reconstruct(run, Generator)
    with pytest.raises(RuntimeError, match='interrupted GPU'):
        Stages(root, 'test').step('reconstruct', owned, required, reconstruct)
    assert (run / 'receiver/ltx_generation/stages/interval_00000.json').is_file()
    Generator.fail=False
    Stages(root, 'test').step('reconstruct', owned, required, reconstruct)
    assert Generator.calls == [0, 3, 3]
    pipeline.audit(run)
    audit = ltx.read(run / 'output_audit.json')
    assert audit['source_indices'] == list(range(9))
    assert audit['generator_backend'] == 'ltx25' and not audit['generator_source_pixels_read']
    frames = sorted((run / 'receiver/reconstruction/sample_0000_frames').glob('*.png'))
    assert [np.asarray(Image.open(p))[0,0,0] for p in frames] == [0,1,2,30,31,32,33,34,35]
    Stages(root, 'test').step('reconstruct', owned, required, reconstruct)
    assert Generator.calls == [0, 3, 3]
    frames[1].write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='artifacts changed'):
        Stages(root, 'test').step('reconstruct', owned, required, reconstruct)


def test_received_opensora_schedule_cannot_be_silently_overridden(run):
    packet = run / 'received/metadata.bin'
    header, payload = unpack(packet.read_bytes()); header['decoder']['steps'] = 30
    packet.write_bytes(pack(header, payload))
    with pytest.raises(ValueError, match='decoder profile'):
        pipeline.reconstruct(run, Generator)
    assert Generator.calls == []


def test_received_key_tampering_stops_before_generation(run):
    key = run / 'receiver/frames/sample/key_frames_received/3.png'
    Image.new('RGB', (576, 320), (255, 255, 255)).save(key)
    with pytest.raises(ValueError, match='request changed'):
        pipeline.reconstruct(run, Generator)
    assert Generator.calls == []


def test_ltx_manifest_or_vendor_change_is_rejected(monkeypatch):
    monkeypatch.setattr(ltx, 'runtime', lambda *args: {'vendor_revision':'new'})
    with pytest.raises(ValueError, match='frozen LTX runtime'):
        pipeline.assets(dict(ltx_model_dir='models', ltx_vendor='vendor',
                             generator_runtime={'vendor_revision':'old'}))


def test_default_multikey_received_only_generation_and_audit(tmp_path, monkeypatch):
    from semantic_transmission.bsu_gen import profile
    cfg = profile()
    run = make_run(tmp_path, monkeypatch, mode=cfg['generation_conditioning'])
    request = ltx.read(run / 'receiver/ltx_request.json')
    assert len(request['windows']) == 1
    window = request['windows'][0]
    assert (window['start'], window['end'], window['generated_frames']) == (0, 8, 9)
    assert [i['frame_idx'] for i in window['images']] == [0, 3, 8]
    assert [sha256(i['path']) for i in window['images']] == [
        sha256(run / f'receiver/frames/sample/key_frames_received/{i}.png') for i in (0, 3, 8)]
    assert window['prompt'] == ('From 0.000 to 0.125 seconds: A person moves at step 0. motion score: 1.0. '
                                'From 0.125 to 0.333 seconds: A person moves at step 1. motion score: 1.0.')
    assert request['receiver_improvements']['flags'] == dict(
        postprocess=False, trajectory=False, context=True, prompt=False)
    assert request['receiver_improvements']['additional_transmitted_bytes'] == 0
    products = next(p for stage, _, p in pipeline.jobs(cfg) if stage == 'prepare-generator')
    assert 'receiver/ltx_conditions' in products
    assert pipeline.condition_plan([0, 3, 8], cfg)['grouping'] == 'deferred_until_received_keys_available'
    pipeline.reconstruct(run, Generator)
    pipeline.reconstruct(run, Generator)
    assert Generator.calls == [0]
    pipeline.audit(run)
    audit = ltx.read(run / 'output_audit.json')
    assert audit['conditioning']['mode'] == pipeline.MULTIKEY
    assert audit['generation_windows'] == 1 and audit['source_indices'] == list(range(9))


@pytest.mark.parametrize('indices,colors,windows', [
    ((0, 3, 8), ((0, 0, 0), (255, 255, 255), (255, 255, 255)), [(0, 3), (3, 8)]),
    ((0, 48, 96), None, [(0, 48), (48, 96)]),
    ((0, 47, 95), ((8, 8, 8),)*3, [(0, 95)]),
    ((0, 8), None, [(0, 8)]),
])
def test_multikey_scene_cut_duration_boundary_and_pair_fallback(tmp_path, monkeypatch, indices, colors, windows):
    run = make_run(tmp_path, monkeypatch, indices, colors, pipeline.MULTIKEY)
    request = ltx.read(run / 'receiver/ltx_request.json')
    assert [(w['start'], w['end']) for w in request['windows']] == windows
    assert sorted({w['start']+i['frame_idx'] for w in request['windows'] for i in w['images']}) == list(indices)


def test_multikey_interrupted_generation_resumes_completed_group(tmp_path, monkeypatch):
    run = make_run(tmp_path, monkeypatch, indices=(0, 3, 8, 11, 16), mode=pipeline.MULTIKEY)
    required = ['run/receiver/ltx_generation', 'run/receiver/reconstruction']
    owned = pipeline.owned_products('reconstruct', required)
    Generator.fail=True; Generator.fail_start=8
    def reconstruct(_):
        pipeline.reconstruct(run, Generator)
    with pytest.raises(RuntimeError, match='interrupted GPU'):
        Stages(run.parent, 'multikey-test').step('reconstruct', owned, required, reconstruct)
    Generator.fail=False
    Stages(run.parent, 'multikey-test').step('reconstruct', owned, required, reconstruct)
    assert Generator.calls == [0, 8, 8]
    pipeline.audit(run)
    frames = sorted((run / 'receiver/reconstruction/sample_0000_frames').glob('*.png'))
    assert [np.asarray(Image.open(p))[0, 0, 0] for p in frames] == list(range(8)) + list(range(80, 89))
    key = run / 'receiver/frames/sample/key_frames_received/3.png'
    Image.new('RGB', (576, 320), (255, 255, 255)).save(key)
    with pytest.raises(ValueError):
        pipeline.reconstruct(run, Generator)
    assert Generator.calls == [0, 8, 8]


def test_multikey_wire_registration_adds_no_bytes_and_rejects_silent_downgrade(tmp_path, monkeypatch):
    run = make_run(tmp_path, monkeypatch, mode=pipeline.MULTIKEY)
    packet = run / 'received/metadata.bin'
    header, payload = unpack(packet.read_bytes())
    size = packet.stat().st_size
    header['decoder'].update(pipeline.decoder_fields())
    legacy = pack(header, payload)
    assert len(legacy) == size
    assert pipeline.decoder_fields()['generator_profile_sha256'] == ltx.fingerprint(ltx.profile())
    packet.write_bytes(legacy)
    with pytest.raises(ValueError, match='decoder profile'):
        pipeline.reconstruct(run, Generator)
    assert Generator.calls == []


def test_unknown_conditioning_is_rejected():
    with pytest.raises(ValueError, match='unknown LTX generation conditioning'):
        pipeline.decoder_fields(dict(generation_conditioning='unregistered'))
