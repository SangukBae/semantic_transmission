from pathlib import Path

import pytest

from semantic_transmission import hybrid_reconstruction as module
from semantic_transmission.artifacts import write_json
from semantic_transmission.webvid_ablation import Stages


def test_check_mode_never_runs_a_worker_or_creates_output(tmp_path, monkeypatch, capsys):
    output = tmp_path / "not-created"
    monkeypatch.setattr(module, "check_ready", lambda *args: (None,None,None,None,None,{"status":"READY_FOR_USER_EXECUTION"}))
    monkeypatch.setattr(module, "execute", lambda *args: pytest.fail("--check launched reconstruction"))
    module.main(["--selection-root", str(tmp_path), "--output", str(output), "--check"])
    assert "READY_FOR_USER_EXECUTION" in capsys.readouterr().out
    assert not output.exists()


def test_resume_keeps_initialization_after_later_clips_are_added(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for i in range(3):
        (source / f"{i}.png").write_bytes(bytes([i]))
    video = tmp_path / "source.mp4"
    video.write_bytes(b"fixture")
    protocol = dict(frames=3, source_frames=str(source), normalized_video=str(video))
    selection = dict(indices=[0,2], selector="hybrid")
    output = tmp_path / "output"
    stages = Stages(output, "fixture")
    stages.step("initialize", ["run"], module.initialization_products(3),
                lambda _: module.initialize(output, protocol, selection, {}))
    clips = output / "run/data/clips"
    clips.mkdir()
    (clips / "new.mp4").write_bytes(b"a later-stage product")
    Stages(output, "fixture").step("initialize", ["run"], module.initialization_products(3),
                                   lambda _: pytest.fail("immutable initialization was rerun"))
    (source / "1.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifacts changed"):
        Stages(output, "fixture").step("initialize", ["run"], module.initialization_products(3), lambda _: None)


def test_later_failed_stage_is_archived_and_only_that_stage_restarts(tmp_path):
    root = tmp_path / "execution"
    pipeline = Stages(root, "fixture")
    pipeline.step("caption", ["captions.json"], ["captions.json"], lambda _: write_json(root / "captions.json", {"complete":True}))
    def fail(_):
        write_json(root / "partial.json", {"complete":False})
        raise RuntimeError("simulated interrupted next stage")
    with pytest.raises(RuntimeError):
        pipeline.step("flow", ["partial.json"], ["partial.json"], fail)
    resumed = Stages(root, "fixture")
    resumed.step("caption", ["captions.json"], ["captions.json"], lambda _: pytest.fail("caption repeated"))
    resumed.step("flow", ["partial.json"], ["partial.json"], lambda _: write_json(root / "partial.json", {"complete":True}))
    assert list((root / "failed_attempts").glob("flow_*/partial.json"))


def test_reconstruction_plan_never_selects_keyframes_again():
    names = [name for name,_,_ in module.job_plan()]
    assert "select" not in names and "prepare" not in names
    assert names.index("caption") < names.index("channel") < names.index("reconstruct") < names.index("evaluate")


def test_hybrid_config_uses_own_caption_checkpoint_and_no_skem_checkpoint(tmp_path):
    cfg = module.build_config({"config":{"selector_checkpoint":"old-skem", "caption_checkpoint":"old-caption"}}, tmp_path, tmp_path / "reconstruction")
    assert "selector_checkpoint" not in cfg
    assert cfg["caption_checkpoint"] == str(tmp_path / "reconstruction/checkpoints/caption.json")


@pytest.fixture
def channel_fixture(tmp_path, monkeypatch):
    import numpy as np
    from semantic_transmission import metadata_channel
    from semantic_transmission.packets import accounting
    from semantic_transmission.wire import pack
    base, run = tmp_path / "baseline", tmp_path / "run"
    cfg = dict(hybrid_selection_root=str(tmp_path), snr_db=10, channel_seed=42,
               frames=3, width=2, height=2)
    write_json(run / "run_config.json", cfg)
    def header(indices):
        return dict(video={k:cfg[k] for k in ("frames", "width", "height")}, decoder={},
            segments=[dict(text="fixture", flow=0.0)],
            keyframes=[dict(index=i, complex_offset=n*2, complex_count=2,
                rate_offset=n, rate_bytes=1, average_power=1.0) for n,i in enumerate(indices)])
    for directory in (base / "transmitter", base / "received", run / "transmitter"):
        directory.mkdir(parents=True)
    (base / "received/metadata.bin").write_bytes(pack(header([0]), b"\x12"))
    (run / "transmitter/metadata.bin").write_bytes(pack(header([0,2]), b"\x12\x34"))
    np.array([1,2],dtype="<c8").tofile(base / "transmitter/visual.c64")
    np.array([1+.2j,2-.1j],dtype="<c8").tofile(base / "received/visual.c64")
    np.array([1,2,3,4],dtype="<c8").tofile(run / "transmitter/visual.c64")
    monkeypatch.setattr(module, "verify_selection", lambda *_: ({"baseline":str(base)}, {}))
    # Only LDPC inference is stubbed; serialization, paired AWGN, and accounting run.
    monkeypatch.setattr(metadata_channel, "transmit", lambda packet,*_:
        (packet, dict(accounting(len(packet)), bit_errors=0)))
    return base, run


def test_channel_replays_common_frames_and_reproduces_new_frame_noise(channel_fixture):
    import numpy as np
    import shutil
    base, run = channel_fixture
    module.matched_channel(run)
    received = np.fromfile(run / "received/visual.c64",dtype="<c8")
    original = np.fromfile(base / "received/visual.c64",dtype="<c8")
    np.testing.assert_array_equal(received[:2], original)
    assert not np.array_equal(received[2:], [3,4])
    report = module.read_json(run / "channel_accounting.json")
    assert report["common_frames_exact"] == [0] and report["new_noise_frames"] == [2]
    assert report["total_complex_channel_uses"] == 4 + report["digital_complex_channel_uses"]
    shutil.rmtree(run / "received")
    module.matched_channel(run)
    np.testing.assert_array_equal(np.fromfile(run / "received/visual.c64",dtype="<c8"), received)


def test_channel_rejects_changed_common_symbols_before_output(channel_fixture):
    import numpy as np
    _, run = channel_fixture
    np.array([9,2,3,4],dtype="<c8").tofile(run / "transmitter/visual.c64")
    with pytest.raises(ValueError, match="common encoded keyframe changed"):
        module.matched_channel(run)
    assert not (run / "received").exists()
