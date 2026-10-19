import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from semantic_transmission import bsu_gen as model
from semantic_transmission import bsu_webvid as batch
from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.packets import accounting
from semantic_transmission.wire import pack


def test_profile_status_and_help_need_no_models(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(model, 'preflight', lambda _: pytest.fail('unexpected preflight'))
    model.main(['profile'])
    p = json.loads(capsys.readouterr().out)
    assert p['name'] == 'BSU-Gen' and p['route'] == 'generative_only'
    assert not p['predictive_fallback'] and not p['adaptive_route_selection']
    model.main(['status', '--output', str(tmp_path / 'absent')])
    assert json.loads(capsys.readouterr().out)['status'] == 'NOT_STARTED'
    assert not (tmp_path / 'absent').exists()


@pytest.mark.parametrize('field,value', [('predictive_fallback', True), ('candidate_stride', 2),
    ('generator', 'LTX'), ('receiver_mode', 'none'), ('snr_db', 8)])
def test_fixed_profile_rejects_undeclared_model_changes(tmp_path, monkeypatch, field, value):
    p = model.profile(); p[field] = value
    path = tmp_path / 'profile.json'; write_json(path, p)
    monkeypatch.setattr(model, 'CONFIG', path)
    with pytest.raises(ValueError, match='fixed profile'):
        model.profile()


def test_dispatch_uses_only_generative_workers_and_restores_on_failure(monkeypatch):
    calls = []
    def command(cmd, *_):
        calls.append(cmd)
    original = batch.verify_plan
    monkeypatch.setattr(batch, 'command', command)
    with pytest.raises(RuntimeError):
        with model.generative_pipeline():
            assert batch.verify_plan is model.verify_plan
            batch.command(['python', '-m', 'semantic_transmission.bsu_webvid_base', '--root', 'base'], None, None)
            batch.command(['python', '-m', 'semantic_transmission.bsu_sequence_refinement', 'refine', '--job', 'job'], None, None)
            raise RuntimeError('interruption')
    assert batch.verify_plan is original and batch.command is command
    assert calls[0][2] == 'semantic_transmission.bsu_kinetics_base'
    assert calls[1][2:5] == [model.MODULE, '_worker', 'refine']


def test_video_normalization_preserves_entire_timeline_and_check_is_read_only(tmp_path, monkeypatch, capsys):
    source = tmp_path / 'input.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=96x64:rate=24',
        '-frames:v', '9', '-threads', '2', '-c:v', 'libx264', str(source)], check=True)
    monkeypatch.setattr(model, 'runtime_template', lambda _: {'test_double': True})
    output = tmp_path / 'output'
    argv = ['--video', str(source), '--output', str(output)]
    model.main(['check', *argv])
    assert json.loads(capsys.readouterr().out)['status'] == 'READY_FILES_VERIFIED'
    assert not output.exists()
    record = model.preflight(model.parser().parse_args(argv))
    output.mkdir(); model.prepare_source(record)
    prepared = output / 'source/source_prepared.json'
    result = model.validate_prepared(prepared)
    assert result['frames'] == 9 and result['fps'] == 24
    assert (result['width'], result['height']) == (576, 320)
    assert result['recipe']['duration_limit'] is None
    (Path(result['source_frames']) / '3.png').write_bytes(b'changed')
    with pytest.raises(ValueError):
        model.validate_prepared(prepared)


@pytest.mark.parametrize('kind', ['overlap', 'nonempty'])
def test_preflight_protects_existing_inputs_and_unknown_output(tmp_path, monkeypatch, kind):
    source = tmp_path / 'input.mp4'; source.write_bytes(b'source')
    from semantic_transmission import video_io
    monkeypatch.setattr(video_io, 'probe', lambda _: dict(duration=1))
    monkeypatch.setattr(model, 'runtime_template', lambda _: pytest.fail('checked models before path guard'))
    output = tmp_path if kind == 'overlap' else tmp_path / 'output'
    if kind == 'nonempty':
        output.mkdir(); (output / 'important').write_text('keep')
    args = model.parser().parse_args(['--video', str(source), '--output', str(output)])
    with pytest.raises(ValueError):
        model.preflight(args)
    assert source.read_bytes() == b'source'


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    """Synthetic generator/quality doubles; real ledger, stages, refinement and FFmpeg."""
    from semantic_transmission import qwen35_pairs_video, bsu_sequence_refinement as refinement
    from semantic_transmission import bsu_dense_router, bsu_adaptive_dense, bsu_predict
    for module, name in [(bsu_dense_router, 'choose'), (bsu_adaptive_dense, 'execute'), (bsu_predict, 'execute')]:
        monkeypatch.setattr(module, name, lambda *a, **kw: pytest.fail('predictive route was invoked'))
    root = tmp_path / 'output'; source_dir = tmp_path / 'source'; source_dir.mkdir()
    source_video = source_dir / 'normalized.mp4'; source_video.write_bytes(b'synthetic source identity')
    rng = np.random.default_rng(3); texture = rng.integers(0, 255, (320, 576, 3), dtype=np.uint8)
    for i in range(5):
        Image.fromarray(np.roll(texture, i, axis=1)).save(source_dir / f'{i}.png')
    source = dict(id='video', source_path=str(source_video), source_sha256=sha256(source_video),
        normalized_video=str(source_video), normalized_sha256=sha256(source_video),
        source_duration_seconds=5/24, source_frames=str(source_dir),
        source_frame_hashes={str(i): sha256(source_dir / f'{i}.png') for i in range(5)},
        frames=5, fps=24, width=576, height=320, recipe=dict(duration_limit=None, source_start_seconds=0))
    prepared = source_dir / 'source_prepared.json'; write_json(prepared, source)
    monkeypatch.setattr(qwen35_pairs_video, 'plan', lambda *a, **kw: {})
    template = dict(code={}, interpreters=dict(python=sys.executable, channel_python=sys.executable,
        qwen=sys.executable), receiver=dict(seed=2025, channel_seed=42, steps=30))
    record = dict(profile=model.profile(), output=str(root), template=template,
        inputs=dict(prepared=str(prepared), sha256=sha256(prepared)), signature='synthetic-contract')
    condition = root / 'benchmark/snr_10/video'; base = condition / 'base'
    run = base / 'reconstruction/event_pairs/run'
    movie = run / 'receiver/reconstruction/sample_0000.mp4'
    movie.parent.mkdir(parents=True); movie.write_bytes(b'synthetic generator output identity')
    frames = movie.with_name('sample_0000_frames'); frames.mkdir()
    keys = run / 'receiver/frames/sample/key_frames_received'; keys.mkdir(parents=True)
    for i in range(5):
        Image.fromarray(np.roll(texture, i, axis=1)).save(frames / f'{i:05d}.png')
        if i in (0, 4):
            Image.fromarray(255 - np.roll(texture, i, axis=1)).save(keys / f'{i}.png')
    header = dict(video=dict(frames=5, fps=24, width=576, height=320), decoder={},
        keyframes=[dict(index=i, rate_bytes=0, average_power=1., complex_count=10) for i in (0, 4)],
        segments=[dict(text='synthetic test', flow=0.)])
    packet = pack(header)
    for directory in ('transmitter', 'received'):
        (run / directory).mkdir(); (run / directory / 'metadata.bin').write_bytes(packet)
    np.zeros(20, dtype=np.complex64).tofile(run / 'transmitter/visual.c64')
    (run / 'data').mkdir(); (run / 'data/normalized.mp4').write_bytes(b'synthetic source identity')
    quality = dict(lossless_frames={'frames': 5}, delivered_mp4={'frames': 5})
    for path, data in [(base / 'RESULT.json', {}), (base / 'protocol.json', {}),
        (run / 'output_audit.json', dict(video=str(movie))), (run / 'quality.json', quality),
        (run / 'channel_accounting.json', accounting(len(packet)))]:
        write_json(path, data)
    monkeypatch.setattr(batch, 'verify_base', lambda *a: (dict(source=source), run, movie, frames, quality))
    def evaluate(job, output):
        write_json(output / 'quality.json', dict(quality, evaluator='test_double'))
        for name in ('quality_lossless_frames.csv', 'quality_delivered_mp4.csv'):
            (output / name).write_text('frame\n0\n')
    monkeypatch.setattr(refinement, 'evaluate', evaluate)
    calls = []
    def command(cmd, *_):
        assert cmd[2] == model.MODULE and cmd[3] == '_worker'
        calls.append(cmd[4]); model.worker(cmd[4], Path(cmd[-1]))
    monkeypatch.setattr(batch, 'command', command)
    return record, condition, calls


def test_generative_pipeline_delivers_received_anchors_accounts_rate_and_resumes(pipeline):
    record, condition, calls = pipeline
    result = model.execute(record)
    assert result['route'] == 'generative_only' and result['status'] == 'COMPLETE_BSU_GEN'
    assert calls == ['prepare', 'channel', 'refine', 'evaluate']
    assert result['total_channel_uses'] == result['base_channel_uses'] + result['trajectory_channel_uses']
    assert result['independent_semantic_review'] == 'PENDING'
    assert model.read(condition / 'refined/refinement.json')['timeline']['frames'] == 5
    key_dir = condition / 'base/reconstruction/event_pairs/run/receiver/frames/sample/key_frames_received'
    for i in (0, 4):
        assert np.array_equal(np.asarray(Image.open(condition / f'refined/frames/{i:05d}.png')),
                              np.asarray(Image.open(key_dir / f'{i}.png')))
    model.execute(record)
    assert len(calls) == 4
    (condition / 'refined/frames/00001.png').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='artifact changed'):
        model.execute(record)
    assert len(calls) == 4
    assert model.read(Path(record['output']) / 'status.json')['status'] == 'FAILED'
    assert not (Path(record['output']) / 'RESULT.json').exists()


def test_channel_failure_stops_without_predictive_fallback_and_resumes(pipeline, monkeypatch):
    from semantic_transmission import bsu_sequence_refinement as refinement
    record, condition, calls = pipeline
    original = refinement.channel
    def fail(*_):
        raise RuntimeError('channel failed')
    monkeypatch.setattr(refinement, 'channel', fail)
    with pytest.raises(RuntimeError, match='channel failed'):
        model.execute(record)
    assert calls == ['prepare', 'channel']
    assert not (Path(record['output']) / 'RESULT.json').exists()
    monkeypatch.setattr(refinement, 'channel', original)
    result = model.execute(record)
    assert calls == ['prepare', 'channel', 'channel', 'refine', 'evaluate']
    assert result['status'] == 'COMPLETE_BSU_GEN'
    assert list((condition / 'failed_attempts').iterdir())


def test_refinement_receives_no_source_paths(pipeline, monkeypatch):
    from semantic_transmission import bsu_receiver
    original = bsu_receiver.refine
    def refine(job, *args):
        assert set(job) == {'base', 'mode'}
        return original(job, *args)
    monkeypatch.setattr(bsu_receiver, 'refine', refine)
    model.execute(pipeline[0])


def test_model_entrypoints_are_independent():
    for script, name in [('run_bsu.sh', 'SPARC'), ('run_sparc.sh', 'SPARC'), ('run_bsu_gen.sh', 'BSU-Gen'),
                         ('run_bsu_predict.sh', 'BSU-Predict')]:
        value = json.loads(subprocess.check_output(['bash', str(model.REPO / 'scripts' / script), 'profile'], text=True))
        assert value['name'] == name
