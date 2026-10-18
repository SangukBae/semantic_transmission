import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

from semantic_transmission import bsu_predict as model
from semantic_transmission import predictive_codec as dcvc
from semantic_transmission import dense_temporal_receiver as temporal
from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.packets import accounting
from semantic_transmission.wire import pack, unpack


def codec_packet(indices, frames=None, width=16, height=16):
    return pack(dict(schema='bsu_dcvc_rt_keys_v1', indices=indices, width=width, height=height,
                     weights=dcvc.WEIGHTS, color='full_range_ycbcr444_bt709',
                     upstream='cbdae87a5445114cdc7f48816da63ea80bdeac40'),
                frames.tobytes() if frames is not None else b'encoded-test-stream')


def source_frames(tmp_path, count=5):
    source = tmp_path / 'source'; source.mkdir()
    for i in range(count):
        Image.fromarray(np.full((16, 16, 3), i * 10, dtype=np.uint8)).save(source / f'{i}.png')
    return source


def record(tmp_path, count=5):
    return dict(output=str(tmp_path / 'run'), signature='test-contract', profile=model.profile(),
                source=model.inspect_frames(source_frames(tmp_path, count), 24),
                runtimes=dict(codec=sys.executable, channel=sys.executable, receiver=sys.executable))


def test_packet_round_trip_has_only_paid_predictive_inputs():
    video = dict(frames=5, width=16, height=16, fps=24)
    coded = codec_packet([0, 2, 4])
    header, returned = model.unpack_packet(model.make_packet(video, coded))
    assert returned == coded
    assert header['video'] == video
    assert set(header) == {'schema', 'video', 'temporal'}
    # Even a two-frame input has a legal predictive stream, without semantic keys.
    model.make_packet({**video, 'frames': 2}, codec_packet([0, 1]))


@pytest.mark.parametrize('mutation', ['gap', 'dimensions', 'weights', 'fallback', 'temporal', 'fps', 'crc'])
def test_invalid_packet_is_rejected_before_codec(mutation):
    packet = model.make_packet(dict(frames=5, width=16, height=16, fps=24), codec_packet([0, 2, 4]))
    outer, coded = unpack(packet); inner, payload = unpack(coded)
    if mutation == 'gap': inner['indices'] = [0, 4]
    if mutation == 'dimensions': inner['width'] = 32
    if mutation == 'weights': inner['weights'] = {}
    if mutation == 'fallback': outer['schema'] = 'bsu_cost_routed_dense_v1'
    if mutation == 'temporal': outer['temporal']['weights'] = 'wrong'
    if mutation == 'fps': outer['video']['fps'] = 0
    packet = pack(outer, pack(inner, payload))
    if mutation == 'crc': packet = packet[:-1] + bytes([packet[-1] ^ 1])
    with pytest.raises(ValueError): model.unpack_packet(packet)


def test_profile_and_status_are_read_only_without_gpu(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(model, 'preflight', lambda _: pytest.fail('preflight ran'))
    model.main(['profile'])
    result = json.loads(capsys.readouterr().out)
    assert result['name'] == 'BSU-Predict' and result['generative_fallback'] is False
    model.main(['status', '--output', str(tmp_path / 'absent')])
    assert json.loads(capsys.readouterr().out)['status'] == 'NOT_STARTED'
    assert not (tmp_path / 'absent').exists()


def test_source_gaps_and_dimensions_are_rejected(tmp_path):
    source = source_frames(tmp_path)
    (source / '1.png').rename(source / '9.png')
    with pytest.raises(ValueError, match='contiguous'): model.inspect_frames(source, 24)
    (source / '9.png').rename(source / '1.png')
    Image.new('RGB', (32, 16)).save(source / '1.png')
    with pytest.raises(ValueError, match='dimensions'): model.inspect_frames(source, 24)


@pytest.fixture
def fake_models(monkeypatch):
    class Codec:
        def __init__(self, *_): pass
        def encode(self, frames, indices, qp):
            assert qp == 32
            return codec_packet(indices, frames)
        def decode(self, packet):
            h, data = unpack(packet)
            return h['indices'], np.frombuffer(data, np.uint8).reshape(-1, h['height'], h['width'], 3)
    def interpolate(anchors, indices, frames, *_):
        result = np.empty((frames, *anchors.shape[1:]), dtype=np.uint8)
        for k, (a, b) in enumerate(zip(indices, indices[1:])):
            for i in range(a, b+1):
                result[i] = (anchors[k].astype(float) * (b-i) + anchors[k+1].astype(float) * (i-a)) / (b-a)
        return result
    from semantic_transmission import metadata_channel
    monkeypatch.setattr(dcvc, 'Codec', Codec)
    monkeypatch.setattr(temporal, 'interpolate', interpolate)
    monkeypatch.setattr(metadata_channel, 'transmit', lambda data, *a, **kw:
                        (data, dict(**accounting(len(data)), bit_errors=0, backend='test_double')))


def test_pipeline_is_predictive_only_and_resumes_verified_stages(tmp_path, monkeypatch, fake_models):
    from semantic_transmission import bsu_adaptive_dense, bsu_dense_router
    monkeypatch.setattr(bsu_adaptive_dense, 'execute', lambda *_: pytest.fail('hybrid pipeline invoked'))
    monkeypatch.setattr(bsu_dense_router, 'choose', lambda *_: pytest.fail('route selection invoked'))
    request = record(tmp_path); root = Path(request['output']); calls = []
    def launch(stage, job, log, runtime):
        calls.append(stage)
        j = model.read(job)
        if stage in ('decode', 'restore'):
            assert 'source' not in j
            assert 'directory' not in j
        model.worker(j, stage)
    monkeypatch.setattr(model, 'launch', launch)
    model.execute(request)
    result = model.read(root / 'RESULT.json')
    assert result['status'] == 'COMPLETE_BSU_PREDICT'
    assert result['route'] == 'predictive_only' and not result['generative_fallback']
    assert calls == ['sender', 'channel', 'decode', 'restore']
    assert result['channel_uses'] == accounting((root / 'sender/packet.bin').stat().st_size)['complex_channel_uses']
    assert model.read(root / 'reconstruction/receiver.json')['video']['frames'] == 5
    model.execute(request)
    assert len(calls) == 4  # No model/channel/FFmpeg restart on verified resume.
    (root / 'received/packet.bin').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='artifacts changed'): model.execute(request)
    assert len(calls) == 4
    assert model.read(root / 'status.json')['status'] == 'FAILED'
    assert not (root / 'RESULT.json').exists()


def test_failed_channel_stops_without_decoder_or_fallback(tmp_path, monkeypatch, fake_models):
    from semantic_transmission import metadata_channel
    request = record(tmp_path, 2); calls = []
    monkeypatch.setattr(metadata_channel, 'transmit', lambda data, *a, **kw:
                        (b'corrupted', dict(bit_errors=1)))
    def launch(stage, job, *_):
        calls.append(stage); model.worker(model.read(job), stage)
    monkeypatch.setattr(model, 'launch', launch)
    with pytest.raises(ValueError, match='integrity failure'): model.execute(request)
    assert calls == ['sender', 'channel']
    assert not (Path(request['output']) / 'RESULT.json').exists()


def test_failed_stage_restarts_but_sender_is_reused(tmp_path, monkeypatch, fake_models):
    request = record(tmp_path); calls = []
    def launch(stage, job, *_):
        calls.append(stage)
        if stage == 'channel' and calls.count('channel') == 1:
            raise KeyboardInterrupt('test stop')
        model.worker(model.read(job), stage)
    monkeypatch.setattr(model, 'launch', launch)
    with pytest.raises(KeyboardInterrupt): model.execute(request)
    root = Path(request['output'])
    assert model.read(root / 'status.json')['status'] == 'INTERRUPTED'
    model.execute(request)
    assert calls == ['sender', 'channel', 'channel', 'decode', 'restore']
    assert list((root / 'failed_attempts').glob('channel_*'))


def test_prepare_checks_source_hashes(tmp_path):
    request = record(tmp_path)
    root = Path(request['output']); root.mkdir()
    source = Path(request['source']['directory'])
    Image.new('RGB', (16, 16), 'red').save(source / '0.png')
    with pytest.raises(ValueError, match='changed'): model.prepare(request, root)


def test_raw_video_normalization_uses_entire_timeline(tmp_path):
    import shutil
    import subprocess
    video = tmp_path / 'input.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=red:s=32x32:r=24',
                    '-frames:v', '7', '-pix_fmt', 'yuv420p', str(video)], check=True)
    root = tmp_path / 'run'; root.mkdir()
    model.prepare(dict(source=dict(video=str(video), sha256=sha256(video), fps=24, width=16, height=16),
                       binaries=dict(ffmpeg=dict(path=shutil.which('ffmpeg')))), root)
    source = model.read(root / 'source.json')
    assert source['video'] == dict(frames=7, fps=24, width=16, height=16)


def test_preflight_freezes_inputs_and_check_creates_no_output(tmp_path, monkeypatch, capsys):
    request = record(tmp_path)
    repo = tmp_path / 'repo'; repo.mkdir()
    config = repo / 'profile.json'; write_json(config, model.profile())
    monkeypatch.setattr(model, 'REPO', repo)
    monkeypatch.setattr(model, 'CONFIG', config)
    monkeypatch.setattr(model, 'CODE', ())
    names = ['dcvc_rt/dcvc_rt_intra.tar', 'dcvc_rt/dcvc_rt_inter.tar',
             'rife/v4.25/flownet.pkl', 'rife/v4.25/IFNet_HDv3.py']
    digests = []
    for name in names:
        path = repo / '.local/checkpoints' / name
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(name.encode())
        digests.append(sha256(path))
    monkeypatch.setattr(dcvc, 'WEIGHTS', dict(intra=digests[0], inter=digests[1]))
    monkeypatch.setattr(temporal, 'WEIGHT_SHA', digests[2])
    monkeypatch.setattr(temporal, 'NETWORK_SHA', digests[3])
    warp = repo / '.local/vendor/Practical-RIFE/model/warplayer.py'
    warp.parent.mkdir(parents=True); warp.write_text('test warp')
    monkeypatch.setattr(temporal, 'WARP_SHA', sha256(warp))
    vendor = repo / '.local/vendor/DCVC-RT/src/models/video_model.py'
    vendor.parent.mkdir(parents=True); vendor.write_text('test codec')
    argv = ['check', '--source-frames', request['source']['directory'], '--output', request['output']]
    for runtime in ('codec', 'receiver', 'channel'):
        argv += [f'--{runtime}-python', sys.executable]
    model.main(argv)
    assert json.loads(capsys.readouterr().out)['model_inference_started'] is False
    assert not Path(request['output']).exists()
    args = model.parser().parse_args(argv)
    saved = model.preflight(args); write_json(args.output / 'protocol.json', saved)
    Image.new('RGB', (16, 16), 'red').save(Path(request['source']['directory']) / '0.png')
    with pytest.raises(ValueError, match='frozen input/code changed'): model.preflight(args)
