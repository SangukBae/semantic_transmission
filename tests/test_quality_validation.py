"""Fast campaign contracts; synthetic cases do not establish restoration quality."""
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from semantic_transmission.artifacts import write_json
from semantic_transmission import quality_validation as app
from semantic_transmission import quality_validation_worker as worker
from semantic_transmission.quality_methods import (
    anchor_correction, low_resolution_constraint, rank_insertions, choose_strength,
)
from semantic_transmission.wire import pack, unpack


class Terminal(io.StringIO):
    def isatty(self):
        return True


def test_terminal_has_one_active_line_and_one_completion_sentence():
    out = Terminal()
    line = app.ProgressLine(1, "검증", out)
    for fraction in (0, .1, .5, .2, .999, 1):
        line.update(fraction)
    assert "\n" not in out.getvalue()
    assert "100.0%" not in out.getvalue()
    assert " 20.0%" not in out.getvalue()  # never regress
    line.finish()
    assert out.getvalue().count("\n") == 1
    assert out.getvalue().count("검증이 끝났습니다") == 1
    assert "\r\033[2K" in out.getvalue()


def test_redirected_output_is_not_flooded_and_failure_is_not_completion():
    out = io.StringIO()
    line = app.ProgressLine(2, "비교", out)
    for i in range(101):
        line.update(i/100)
    assert out.getvalue() == ""
    line.finish(failed=True)
    assert out.getvalue().count("\n") == 1
    assert "중단되었습니다" in out.getvalue() and "100.0%" not in out.getvalue()


def test_receiver_methods_identity_bounded_change_and_geometry():
    frames = np.full((5, 32, 32, 3), 100, np.uint8)
    keys = {0: frames[0].copy(), 4: frames[4].copy()}
    assert np.array_equal(anchor_correction(frames, keys, .75), frames)
    keys = {0: frames[0]+10, 4: frames[4]+10}
    corrected = anchor_correction(frames, keys, .5)
    assert corrected.dtype == np.uint8 and np.all(corrected >= 100) and np.all(corrected <= 105)
    assert np.array_equal(anchor_correction(frames, keys, 0), frames)
    low = np.full((5, 8, 8, 3), 110, np.uint8)
    assert np.all(low_resolution_constraint(frames, low, .5) == 105)
    with pytest.raises(ValueError):
        low_resolution_constraint(frames, low[:-1])


def test_sender_ranking_targets_short_event_and_keeps_semantic_boundaries():
    frames = np.zeros((9, 32, 32, 3), np.uint8)
    frames[2] = 255
    assert rank_insertions(frames, [0, 8], 1, True) == [2]
    assert rank_insertions(frames, [0, 8], 1, False) == [4]
    selected = rank_insertions(frames, [0, 4, 8], 5, True)
    assert not set(selected) & {0, 4, 8}
    assert len(selected) == len(set(selected))


def test_quality_guard_can_reject_every_nonzero_strength():
    def result(lpips, psnr=20, temporal=.1):
        return dict(interior=dict(lpips_vgg=lpips, psnr_db=psnr), temporal_error=temporal)
    rows = {"0.0": result(.3), "0.25": result(.29, 19), "0.5": result(.28, temporal=.2)}
    assert choose_strength(rows) == 0
    rows["0.75"] = result(.29)
    assert choose_strength(rows) == .75


def test_wire_subset_retains_symbols_rates_and_enforces_real_budget():
    from semantic_transmission.packets import accounting
    header = dict(keyframes=[dict(index=i, complex_offset=i*2, complex_count=2, rate_offset=i,
                                 rate_bytes=1, rate_count=2, average_power=1) for i in range(5)],
                  segments=[], video={}, decoder={})
    symbols = np.arange(10).astype("<c8")
    rows = [dict(path="clips/sample/00000.mp4", text="same automatic caption", flow=.1)]
    build = lambda keys: worker.subset_wire(header, bytes(range(5)), symbols, keys, [0,4], rows)
    packet, subset = build([0,2,4])
    decoded, payload = unpack(packet)
    assert [k["index"] for k in decoded["keyframes"]] == [0,2,4]
    assert payload == bytes([0,2,4])
    assert subset.tolist() == symbols[[0,1,4,5,8,9]].tolist()
    budget = len(subset)+accounting(len(packet))["complex_channel_uses"]+100
    keys, cost = worker.choose_budget([0,4], [2,1,3], build, budget, 100)
    assert keys == [0,2,4] and cost == budget
    with pytest.raises(ValueError, match="fixed budget"):
        worker.choose_budget([0,4], [2], build, 1, 100)


def test_plan_contains_real_workers_for_all_validations_and_frozen_seeds(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    cfg = app.read_json(repo/"configs/quality_validation.json")
    for entry in cfg["heldout"]:
        entry["source"] = "/fixture/"+entry["filename"]
    jobs = app.build_plan(tmp_path, cfg, repo)
    assert {j["group"] for j in jobs} == {1,2,3}
    assert len(jobs) == len({j["name"] for j in jobs})
    assert all(callable(getattr(worker, j["spec"]["action"])) for j in jobs)
    held_reconstructions = [j for j in jobs if j["group"]==3 and j["spec"].get("stage")=="reconstruct"]
    assert len(held_reconstructions) == 5*3*4
    assert len([j for j in jobs if j["group"]==3 and j["spec"].get("stage")=="select"]) == 5
    assert len([j for j in jobs if j["spec"].get("freeze")]) == 1
    assert all(j["spec"].get("frozen", "").endswith("correction/development/frozen.json")
               for j in jobs if "frozen" in j["spec"])


def test_resume_preserves_partial_and_rejects_tamper(tmp_path, monkeypatch):
    product = tmp_path/"product"
    job = app.task("a", 1, {"action": "fixture"}, [product])
    calls = []
    def fail(*args):
        product.mkdir(); (product/"part").write_text("incomplete")
        raise RuntimeError("interrupted")
    monkeypatch.setattr(app, "launch", fail)
    with pytest.raises(RuntimeError):
        app.Runner(tmp_path, "v1", tmp_path, {}).execute(job, lambda _: None)
    def succeed(*args):
        calls.append(True)
        product.mkdir(); (product/"done").write_text("complete")
    monkeypatch.setattr(app, "launch", succeed)
    app.Runner(tmp_path, "v1", tmp_path, {}).execute(job, lambda _: None)
    assert next((tmp_path/"failed_attempts").glob("a_*/0_product/part")).read_text() == "incomplete"
    app.Runner(tmp_path, "v1", tmp_path, {}).execute(job, lambda _: None)
    assert len(calls) == 1
    (product/"done").write_text("changed")
    with pytest.raises(ValueError, match="변경"):
        app.Runner(tmp_path, "v1", tmp_path, {}).execute(job, lambda _: None)


def test_real_worker_suppresses_logs_and_records_progress(tmp_path):
    """Run actual correction worker + FFmpeg, small synthetic receiver boundary."""
    from PIL import Image
    run = tmp_path/"base"
    generated = run/"receiver/reconstruction/sample_0000_frames"
    keys = run/"receiver/frames/sample/key_frames_received"
    generated.mkdir(parents=True); keys.mkdir(parents=True)
    values = np.full((9,32,32,3), 100, np.uint8)
    for i, image in enumerate(values):
        Image.fromarray(image).save(generated/f"{i:05d}.png")
    for i in (0,8):
        Image.fromarray(values[i]+5).save(keys/f"{i}.png")
    (run/"data").mkdir()
    subprocess.run(["ffmpeg","-v","error","-framerate","4","-i",str(generated/"%05d.png"),
                    "-c:v","libx264","-pix_fmt","yuv420p",str(run/"data/normalized.mp4")], check=True)
    write_json(run/"receiver/decoder_inputs.json", dict(indices=[0,8],decoder=dict(policy="endpoint_exact"),
                                                       video=dict(fps=4)))
    dest = tmp_path/"corrected"
    spec = dict(action="correction", base=str(run), dest=str(dest), strengths=[0., .5], freeze=True,
                lowres_strength=.5, pixel_only=True)
    events=[]
    app.launch(spec, tmp_path/"spec.json", tmp_path/"worker.log", tmp_path/"progress.json",
               Path(__file__).resolve().parents[1], {"python":sys.executable}, events.append)
    assert (dest/"0.5/video.mp4").exists()
    assert app.read_json(dest/"frozen.json")["anchor_strength"] == 0  # pixel-only cannot claim LPIPS gains
    assert app.read_json(tmp_path/"progress.json") == {"done":1,"total":1}
    assert app.read_json(dest/"results.json")["0.5"]["interior_frames"] == 7


def test_lowres_actual_encode_decode_and_budget_planning(tmp_path):
    """Real FFmpeg byte payload, with channel substitute explicitly limited to this test."""
    from semantic_transmission.packets import accounting
    base = tmp_path/"base"
    (base/"data").mkdir(parents=True)
    subprocess.run(["ffmpeg","-v","error","-f","lavfi","-i","testsrc2=size=64x64:rate=4",
                    "-frames:v","9","-c:v","libx264",str(base/"data/normalized.mp4")], check=True)
    write_json(base/"run_config.json", dict(frames=9,fps=4))
    encoded=tmp_path/"encoded"
    worker.encode_side(dict(base=str(base),dest=str(encoded),lowres_size=[16,16],lowres_crf=30), tmp_path)
    header, payload = unpack((encoded/"packet.bin").read_bytes())
    assert header["frames"]==9 and payload==(encoded/"sender.mp4").read_bytes()
    assert accounting(len((encoded/"packet.bin").read_bytes()))["complex_channel_uses"] > 0


def test_budget_plan_materializes_three_valid_transmittable_variants(tmp_path, monkeypatch):
    from semantic_transmission.packets import accounting
    base, bank, side = [tmp_path/x for x in ("base", "bank", "side")]
    (base/"data").mkdir(parents=True)
    write_json(base/"run_config.json", dict(seed=42))
    write_json(base/"metadata_tx.json", [dict(path="clips/sample/0.mp4", text="automatic", flow=1.)])
    write_json(tmp_path/".local/model_paths.json", {})
    (bank/"transmitter").mkdir(parents=True)
    header = dict(keyframes=[dict(index=i, complex_offset=i*4, complex_count=4, rate_offset=i,
                                 rate_bytes=1, rate_count=2, average_power=1.) for i in range(9)],
                  decoder=dict(seed=42,steps=30,policy="official_release"),
                  video=dict(frames=9,fps=4,width=32,height=32), segments=[])
    (bank/"transmitter/metadata.bin").write_bytes(pack(header, bytes(range(9))))
    np.arange(36,dtype=np.float32).astype("<c8").tofile(bank/"transmitter/visual.c64")
    write_json(bank/"rankings.json", dict(original=[0,8],adaptive=[2,6,4],uniform=[4,2,6]))
    write_json(side/"channel.json", dict(complex_channel_uses=2304))
    dest = tmp_path/"variants"
    worker.plan_wires(dict(base=str(base),bank=str(bank),dest=str(dest),side=str(side),budget_multiplier=4.), tmp_path)
    ledger = app.read_json(dest/"budget.json")
    for name in ("uniform","adaptive","adaptive_lowres"):
        packet=(dest/name/"transmitter/metadata.bin").read_bytes()
        received_header, _ = unpack(packet)
        symbols=np.fromfile(dest/name/"transmitter/visual.c64",dtype="<c8")
        cost=len(symbols)+accounting(len(packet))["complex_channel_uses"]+(2304 if name=="adaptive_lowres" else 0)
        assert cost==ledger["variants"][name]["total_complex_channel_uses"] <= ledger["budget"]
        assert len(received_header["segments"])==len(received_header["keyframes"])-1
        assert (dest/name/"data").is_symlink()


def test_report_never_calls_completion_quality_success(tmp_path):
    for seed in (42,43,44):
        for video in range(5):
            for method in ("baseline","uniform","adaptive","adaptive_lowres","corrected"):
                path=tmp_path/f"heldout/v{video}/evaluations/{seed}/{method}"
                delta=0 if method in {"baseline","uniform"} else .1
                write_json(path/"metrics.json", dict(all=dict(psnr_db=20,ssim=.5,lpips_vgg=.3+delta),
                    interior=dict(psnr_db=20,lpips_vgg=.3),temporal_error=.1,hallucination_review="PENDING"))
    cfg=dict(development=[],heldout=[dict(id=f"v{i}",source="/fixture/source.mp4") for i in range(5)])
    app.build_report(tmp_path,cfg,[1,2,3])
    summary=app.read_json(tmp_path/"summary.json")
    assert summary["execution"]=="COMPLETED"
    assert summary["aggregates"]["adaptive"]["finding"]=="IMPROVEMENT_NOT_ESTABLISHED"
    assert summary["aggregates"]["adaptive"]["videos"]==5
    assert summary["hallucination_review"]=="PENDING"
