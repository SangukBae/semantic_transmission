"""Guard paired noise evidence and received-only short-window inputs."""
import csv
import importlib.util
from pathlib import Path
import subprocess

import pytest

from semantic_transmission.artifacts import write_json, sha256

SPEC = importlib.util.spec_from_file_location("diagnose", Path(__file__).resolve().parents[1] / "scripts/diagnose_etri_conditioning.py")
app = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(app)


def test_window_pair_uses_received_keys_and_identical_metadata(tmp_path, monkeypatch):
    base = tmp_path / "base/baseline"
    (base / "data").mkdir(parents=True)
    source = base / "data/normalized.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=24",
                    "-frames:v", "12", "-c:v", "libx264", "-crf", "0", str(source)], check=True)
    write_json(base / "run_config.json", {"models": {}, "frames": 12})
    write_json(base / "keyframes.json", {"indices": [0, 2, 5, 8, 11]})
    write_json(base / "receiver/decoder_inputs.json", {"indices": [0, 2, 5, 8, 11], "video": {"frames": 12}})
    folder = base / "receiver/frames/sample/key_frames_received"
    folder.mkdir(parents=True)
    for k in [0, 2, 5, 8, 11]:
        (folder / f"{k}.png").write_bytes(f"received-{k}".encode())
    with (base / "receiver/metadata.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "text", "flow"])
        writer.writeheader()
        writer.writerows({"path": f"clips/sample/{i:05d}.mp4", "text": f"caption {i}", "flow": i+.25} for i in range(4))
    monkeypatch.setattr(app, "WINDOWS", {"test": [2, 5, 8]})
    out = tmp_path / "out"
    app.initialize(out, base.parent)
    a, b = [out / "test" / case for case in app.CASES]
    for relative in ["data/normalized.mp4", "receiver/metadata.csv", "receiver/decoder_inputs.json"]:
        assert sha256(a / relative) == sha256(b / relative)
    for local, original in [(0, 2), (3, 5), (6, 8)]:
        for run in (a, b):
            assert sha256(run / f"receiver/frames/sample/key_frames_received/{local}.png") == sha256(folder / f"{original}.png")
    cfg_a, cfg_b = [app.read_json(run / "run_config.json") for run in (a, b)]
    assert cfg_a.pop("diagnostic_align") == 5 and cfg_b.pop("diagnostic_align") is None
    assert cfg_a == cfg_b


@pytest.mark.parametrize("mismatch", [False, True])
def test_summary_requires_identical_initial_noise(tmp_path, monkeypatch, mismatch):
    monkeypatch.setattr(app, "WINDOWS", {"test": [0, 1]})
    values = {key: .5 for key in ("psnr_db", "ssim", "lpips_vgg", "clip", "dists")}
    for case in app.CASES:
        run = tmp_path / "test" / case
        write_json(run / "quality.json", {"delivered_mp4": values})
        write_json(run / "receiver/conditioning_trace.json", {"mask": [{"noise_sha256": "different" if mismatch and case == "no_rounding" else "same", "active": [{"reference": 0, "reference_sha256": "key"}]}]})
    if mismatch:
        with pytest.raises(AssertionError, match="noise differs"):
            app.summarize(tmp_path)
    else:
        app.summarize(tmp_path)
        assert app.read_json(tmp_path / "RESULT.json")["status"] == "PAIRED_DIAGNOSTICS_COMPLETE"
