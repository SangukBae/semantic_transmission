"""Reconstruct all completed SKEM speed conditions with measured transmission.

Equal keyframe sets share identical downstream artifacts. Different selections
get freshly computed segment captions/flow. Common visual frames get identical
AWGN by original frame number; metadata is sent in full. Decoder settings/seed
are fixed, but different segment layouts do not imply identical diffusion noise.
"""
import argparse
import csv
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.webvid5 import fingerprint, read_json
from semantic_transmission.webvid_ablation import Stages, environment
from semantic_transmission.etri_60s_check import lock, run_command
from semantic_transmission.cli import settings


def baseline_reference(root):
    path = root / "baseline_reference.json"
    if not path.exists():
        return None
    policy = read_json(path)
    if policy["selection_protocol"] != read_json(root / "protocol.json")["signature"]:
        raise ValueError("historical baseline belongs to a different protocol")
    for source, digest in policy["source_files"].items():
        if sha256(Path(source)) != digest:
            raise ValueError(f"historical baseline source changed: {source}")
    for record in policy["records"].values():
        if sha256(root / record["path"]) != record["sha256"]:
            raise ValueError("imported baseline changed")
    return policy


def selection_path(root, mode, window):
    if mode == "baseline" and (root / "baseline_reference.json").exists():
        return root / read_json(root / "baseline_reference.json")["records"][window]["path"]
    return root / f"selection/{mode}/{window}.json"


def plan(root):
    protocol = read_json(root / "protocol.json")
    reference = baseline_reference(root)
    runs, aliases, selections = {}, {}, {}
    for window in protocol["windows"]:
        for mode in protocol["modes"]:
            path = selection_path(root, mode, window)
            if not path.exists() or read_json(path)["status"] != "COMPLETE":
                raise ValueError(f"selection incomplete: {mode}/{window}")
            result = read_json(path)
            keys = result["indices"]
            name = f"{window}_{fingerprint(keys)[:8]}"
            aliases[f"{window}/{mode}"] = name
            runs[name] = {"window": window, "indices": keys}
            selections[str(path.relative_to(root))] = sha256(path)
    return {"selection_signature": protocol["signature"], "selections": selections,
        "baseline_reference": reference,
        "runs": runs, "aliases": aliases, "seed": 2025, "channel_seed": 42,
        "visual_noise": "numpy PCG64 seeded by original video frame; same law, new paired realization",
        "generation_noise": "same configured seed; different segment layouts are not exact noise replay",
        "code": {str(p.relative_to(REPO)): sha256(p) for p in
            (Path(__file__).resolve(), REPO / "scripts/etri_conditioning_probe.py",
             REPO / "src/semantic_transmission/workers.py", REPO / "src/semantic_transmission/codec_transport.py",
             REPO / "04_semantic_decoder/scripts/mydemo_new_align_sh.py")}}


def initialize(root, spec):
    for name, run_spec in spec["runs"].items():
        folder = root / "inputs" / run_spec["window"]
        run = root / "reconstruction" / name
        frames = run / "data/frames/sample"
        frames.mkdir(parents=True)
        cfg = read_json(folder / "config.json")
        cfg.update(seed=spec["seed"], channel_seed=spec["channel_seed"],
                   visual_channel="paired_frame_numpy", profile="skem_speed_downstream_diagnostic_v1")
        shutil.copyfile(folder / "source.mp4", run / "data/normalized.mp4")
        for source in sorted((folder / "frames").glob("*.png")):
            shutil.copyfile(source, frames / source.name)
        write_json(run / "run_config.json", cfg)
        write_json(run / "keyframes.json", {"indices": run_spec["indices"]})
        write_json(run / "speed_window.json", read_json(root / "protocol.json")["windows"][run_spec["window"]])
    write_json(root / "downstream_plan.json", spec)


def caption_all(root):
    from semantic_transmission import workers
    spec = read_json(root / "downstream_plan.json")
    loader = workers._load_caption_inference
    inference = None
    def shared(cfg, repo):
        nonlocal inference
        if inference is None: inference = loader(cfg, repo)
        return inference
    workers._load_caption_inference = shared
    for name in spec["runs"]:
        run = root / "reconstruction" / name
        started = time.perf_counter()
        workers.caption(read_json(run / "run_config.json"), REPO, run)
        write_json(run / "caption_time.json", {"seconds": time.perf_counter()-started,
            "meaning": "shared model; first run includes model load"})
        print(f"CAPTION_COMPLETE {name}", flush=True)


def frame_paired_noise(sent, items, original_start, channel_seed, snr_db):
    import numpy as np
    values = sent.copy()
    sigma = math.sqrt(1/(2 * 10**(snr_db/10)))
    for item in items:
        a, n = item["complex_offset"], item["complex_count"]
        rng = np.random.default_rng(channel_seed + 100000 + original_start + item["index"])
        noise = (rng.normal(0,sigma,n) + 1j*rng.normal(0,sigma,n)).astype("<c8")
        values[a:a+n] += noise
    return values


def paired_channel(run):
    import numpy as np
    from semantic_transmission.metadata_channel import transmit
    from semantic_transmission.wire import unpack
    from semantic_transmission.transmission_accounting import channel_breakdown
    cfg = read_json(run / "run_config.json")
    window = read_json(run / "speed_window.json")
    packet = (run / "transmitter/metadata.bin").read_bytes()
    restored, report = transmit(packet, cfg["snr_db"], cfg["channel_seed"])
    if restored != packet or report["bit_errors"]: raise ValueError("metadata decoding error")
    header, _ = unpack(restored)
    sent = np.fromfile(run / "transmitter/visual.c64", dtype="<c8")
    values = frame_paired_noise(sent,header["keyframes"],window["start"],cfg["channel_seed"],cfg["snr_db"])
    output = run / "received"
    output.mkdir()
    (output / "metadata.bin").write_bytes(restored)
    values.tofile(output / "visual.c64")
    digital = report["complex_channel_uses"]
    report.update(status="PASSED", metadata_exact_match=True,
        visual_complex_channel_uses=len(sent), digital_complex_channel_uses=digital,
        total_complex_channel_uses=len(sent)+digital,
        cbr_complex_uses_per_source_scalar=(len(sent)+digital)/(3*cfg["width"]*cfg["height"]*cfg["frames"]),
        complete_sample_dependent_model_input_accounting=True, physical_link_overhead_included=False,
        visual_awgn_rng="numpy PCG64 channel_seed + 100000 + original frame index",
        channel_seed=cfg["channel_seed"], transmission_breakdown=channel_breakdown(packet,len(sent),header["video"],report),
        received_files={p.name: {"bytes":p.stat().st_size,"sha256":sha256(p)} for p in output.iterdir()})
    write_json(run / "channel_accounting.json", report)


def reconstruct(run):
    from semantic_transmission.codec_transport import decoder_config_text
    from semantic_transmission.decoder_runner import run as decode
    cfg = read_json(run / "run_config.json")
    inputs = read_json(run / "receiver/decoder_inputs.json")
    text = decoder_config_text(cfg, REPO, inputs)
    text += "\ncpu_video_storage=True\nconditioning_alignment='official_release'\nalign=5\n"
    path = run / "receiver/decoder_config.py"
    path.write_text(text)
    env = dict(os.environ, PYTHONPATH=str(REPO / ".local/vendor/Open-Sora"),
               ETRI_CONDITION_TRACE=str(run / "receiver/conditioning_trace.json"))
    decode(run / "receiver/metadata.csv", run / "receiver/reconstruction", "key_frames_received",
           run / "receiver", decoder=REPO / "scripts/etri_conditioning_probe.py", config=path, environment=env)


def summarize(root):
    import hashlib
    import numpy as np
    from semantic_transmission.wire import unpack
    from semantic_transmission.research_quality import read_video
    from semantic_transmission.quality_methods import temporal_error
    spec = read_json(root / "downstream_plan.json")
    result = {"status":"SHORT_ABLATIONS_COMPLETE", "windows":{}, "independent_semantic_review":"PENDING",
              "adopted_as_default":False, "scope":read_json(root / "protocol.json")["scope"],
              "paired_common_visual_frames":True, "generation_noise":spec["generation_noise"],
              "baseline_source":"VERIFIED_HISTORICAL_SELECTION" if spec["baseline_reference"] else "FRESH",
              "timing_scope":"Historical baseline is context only; unstable BF16 latency prevents a confirmed speedup ratio."
                  if spec["baseline_reference"] else "Single execution of each condition."}
    for window in read_json(root / "protocol.json")["windows"]:
        rows = {}
        common = {}
        union_keys=set()
        for mode in ("baseline","int8","brief","sparse"):
            union_keys.update(read_json(selection_path(root, mode, window))["indices"])
        for mode in ("baseline","int8","brief","sparse"):
            run = root / "reconstruction" / spec["aliases"][f"{window}/{mode}"]
            selection = read_json(selection_path(root, mode, window))
            quality = read_json(run / "quality.json")
            temporal = temporal_error(read_video(run / "data/normalized.mp4"),
                read_video(run / "receiver/reconstruction/sample_0000.mp4"))
            keys = selection["indices"]
            with (run / "quality_delivered_mp4.csv").open() as stream:
                per_frame = list(csv.DictReader(stream))
            inside = [r for r in per_frame if int(r["frame"]) not in keys]
            common_inside = [r for r in per_frame if int(r["frame"]) not in union_keys]
            header,_=unpack((run / "received/metadata.bin").read_bytes())
            received=np.fromfile(run / "received/visual.c64",dtype="<c8")
            for item in header["keyframes"]:
                a,n=item["complex_offset"],item["complex_count"]
                digest=hashlib.sha256(received[a:a+n].tobytes()).hexdigest()
                old=common.setdefault(item["index"],digest)
                if old!=digest: raise ValueError("common keyframe channel mismatch")
            rows[mode]={"keys":keys,"selection_seconds":selection["selection_seconds"],
                "timing_source":selection.get("timing_source", "current completed comparisons; perf_counter"),
                "comparisons":len(selection["records"]), "quality":quality["delivered_mp4"],
                "temporal_error":temporal,
                "interior_quality":{k:float(np.mean([float(r[k]) for r in inside])) for k in ("psnr_db","lpips_vgg")} if inside else None,
                "common_interior_frames":[int(r["frame"]) for r in common_inside],
                "common_interior_quality":{k:float(np.mean([float(r[k]) for r in common_inside])) for k in ("psnr_db","lpips_vgg")}
                    if common_inside else None,
                "channel_uses":read_json(run / "channel_accounting.json")["total_complex_channel_uses"],
                "run":str(run.relative_to(root))}
        for mode,row in rows.items():
            row["selection_speedup"]=(None if spec["baseline_reference"] else
                rows["baseline"]["selection_seconds"]/row["selection_seconds"])
            row["channel_use_ratio"]=row["channel_uses"]/rows["baseline"]["channel_uses"]
            row["delta_lpips"]=row["quality"]["lpips_vgg"]-rows["baseline"]["quality"]["lpips_vgg"]
            row["delta_psnr_db"]=row["quality"]["psnr_db"]-rows["baseline"]["quality"]["psnr_db"]
        result["windows"][window]=rows
    write_json(root / "RESULT.json",result)
    print("SHORT_ABLATIONS_COMPLETE",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=REPO / "outputs/skem_speed_20260929")
    p.add_argument("--worker",choices=["caption_all","channel","reconstruct"])
    p.add_argument("--run-dir",type=Path)
    args=p.parse_args()
    root=args.output.resolve()
    if args.worker:
        if args.worker=="caption_all": caption_all(root)
        elif args.worker=="channel": paired_channel(args.run_dir.resolve())
        else: reconstruct(args.run_dir.resolve())
        return
    spec=plan(root)
    signature=fingerprint(spec)
    local=settings(REPO)
    env=environment(2025)
    with lock(REPO / ".local/etri_60s_check.lock"),lock(root / ".downstream.lock"):
        protocol=root / "downstream_protocol.json"
        if protocol.exists() and read_json(protocol)["signature"]!=signature:
            raise ValueError("downstream code or selection changed")
        if not protocol.exists(): write_json(protocol,dict(spec,signature=signature))
        pipeline=Stages(root,signature)
        init=["downstream_plan.json"]
        for name in spec["runs"]:
            init.extend(f"reconstruction/{name}/{s}" for s in ("run_config.json","keyframes.json","speed_window.json","data/normalized.mp4","data/frames/sample"))
        pipeline.step("speed_initialize",["downstream_plan.json","reconstruction"],init,lambda _:initialize(root,spec))
        def launch(name,cmd,files,python=None):
            metrics,timing,progress=f"resources/{name}.json",f"resources/{name}.time.txt",f"progress/{name}.json"
            pipeline.step(name,files+[metrics,timing,progress],files+[metrics,timing],
                lambda log:run_command(REPO,cmd,log,env,root/metrics,root/progress))
        files=[f"reconstruction/{name}/{f}" for name in spec["runs"] for f in ("captions.json","caption_sampling.json","caption_time.json","semantic_clips_audit.json")]
        launch("speed_captions",[local["python"],str(Path(__file__).resolve()),"--worker","caption_all","--output",str(root)],files)
        for name in spec["runs"]:
            relative=f"reconstruction/{name}"
            run=root/relative
            for stage in ("flow","send","channel","receive","reconstruct","evaluate"):
                products={"flow":["metadata_tx.json","flow_sampling.json"],"send":["transmitter","sender_accounting.json"],
                    "channel":["received","channel_accounting.json"],"receive":["receiver/frames","receiver/metadata.csv","receiver/decoder_inputs.json","receiver_accounting.json"],
                    "reconstruct":["receiver/decoder_config.py","receiver/reconstruction","receiver/conditioning_trace.json"],
                    "evaluate":["quality.json","quality_delivered_mp4.csv","quality_lossless_frames.csv"]}[stage]
                if stage in ("channel","reconstruct"):
                    py=local["channel_python"] if stage=="channel" else local["python"]
                    cmd=[py,str(Path(__file__).resolve()),"--worker",stage,"--run-dir",str(run)]
                else:
                    module={"flow":"workers","send":"codec_transport","receive":"codec_transport","evaluate":"research_quality"}[stage]
                    cmd=[local["python"],"-m",f"semantic_transmission.{module}",stage,str(run)]
                launch(f"speed_{name}_{stage}",cmd,[f"{relative}/{f}" for f in products])
        summarize(root)


if __name__=="__main__": main()
