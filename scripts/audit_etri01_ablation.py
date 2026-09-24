"""Verify temporal contracts and paired conditioning for ETRI01 ablations."""
import json
from pathlib import Path

from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.video_io import probe
from semantic_transmission.wire import unpack

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "outputs/diagnostics/etri01_ablation_20260917"
BASE = REPO / "outputs/etri01_official_20260911_v2/01_person_walk"


def read(path):
    return json.loads(path.read_text())


def main():
    records = []
    baseline_hash = read(BASE / "quality.json")["video_sha256"]
    assert sha256(BASE / "receiver/reconstruction/sample_0000.mp4") == baseline_hash
    for name in ["replay", "aligned", "clean_keys", "action_caption", "dense_2s", "dense_1s", "combined"]:
        run = ROOT / name
        assert read(run / "complete.json")["status"] == "PASSED"
        cfg = read(run / "run_config.json")
        assert (cfg["frames"],cfg["width"],cfg["height"],cfg["fps"],cfg["seed"],cfg["steps"]) == (240,576,320,24,42,30)
        info = probe(run / "receiver/reconstruction/sample_0000.mp4")
        assert info["frames"] == (241 if name == "replay" else 240)
        experiment = read(run / "experiment.json")
        for file, expected in experiment["inputs"].items():
            assert sha256(run / file) == expected, (name,file)
        indices = read(run / "receiver/decoder_inputs.json")["indices"]
        expected_keys = [0,179,239]
        assert all(i in indices for i in expected_keys)
        if name in {"dense_2s", "dense_1s", "combined"}:
            assert max(b-a for a,b in zip(indices,indices[1:])) <= (48 if name == "dense_2s" else 24)
        for i in expected_keys:
            reference = BASE / (f"data/frames/sample/{i}.png" if name == "clean_keys"
                                else f"receiver/frames/sample/key_frames_received/{i}.png")
            assert sha256(run / f"receiver/frames/sample/key_frames_received/{i}.png") == sha256(reference)
        if (run / "received").exists():
            channel = read(run / "channel_accounting.json")
            assert channel["bit_errors"] == 0 and channel["metadata_exact_match"]
            assert (run / "received/metadata.bin").read_bytes() == (run / "transmitter/metadata.bin").read_bytes()
            for filename, record in channel["received_files"].items():
                assert sha256(run / "received" / filename) == record["sha256"]
            header,payload = unpack((run / "received/metadata.bin").read_bytes())
            count = sum(k["complex_count"] for k in header["keyframes"])
            assert count*8 == (run / "received/visual.c64").stat().st_size
            assert channel["total_complex_channel_uses"] == count + channel["digital_complex_channel_uses"]
        records.append({"name":name,"status":"PASSED","indices":indices,"video":info})
    a,b = ROOT / "dense_1s", ROOT / "combined"
    assert sha256(a / "received/visual.c64") == sha256(b / "received/visual.c64")
    for file in (a / "receiver/frames/sample/key_frames_received").glob("*.png"):
        assert sha256(file) == sha256(b / "receiver/frames/sample/key_frames_received" / file.name)
    assert sha256(ROOT / "replay/receiver/reconstruction/sample_0000.mp4") == baseline_hash
    for name in ["historical","replay","aligned","clean_keys","action_caption","dense_2s","dense_1s","combined"]:
        r = read(ROOT / f"evaluation_{name}.json")
        assert r["all_five_metrics"] and r["lossless_frames"]["frames"] == 240
        assert len(r["kept_output_positions"]) == 240
    write_json(ROOT / "AUDIT.json", {"status":"PASSED","records":records,
        "historical_video_unchanged":True,"replay_mp4_byte_identical":True,
        "original_three_keyframe_pngs_preserved":True,"dense_vs_combined_visual_inputs_identical":True})
    print("All temporal, source, wire, replay and paired-input checks passed")


if __name__ == "__main__":
    main()
