"""Resumable full-video BSU/WebVid reconstruction, refinement and paper-metric tables."""
import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from .artifacts import sha256, write_json
from .bsu_sequence_refinement import MODES, paths_for_run, read, validate_keys
from .bsu_webvid_base import digital_bits
from .cli import settings
from .etri_60s_check import lock
from .official_quality import KEYS
from .qwen35_pairs_video import freeze
from .transmission_accounting import channel_breakdown
from .webvid5 import fingerprint
from .webvid_ablation import Stages, environment, snapshot
from .wire import unpack

REPO = Path(__file__).resolve().parents[2]
POLICY = "bsu_webvid_full_benchmark_v1"
CODE = ("bsu_webvid.py", "bsu_webvid_base.py", "bsu_sequence_refinement.py", "sparse_trajectory.py",
        "trajectory_receiver.py", "official_quality.py", "stutter_metrics.py",
        "packets.py", "wire.py", "metadata_channel.py", "transmission_accounting.py")


def snr_name(value):
    return f"snr_{value:g}".replace("-", "m").replace(".", "p")


def verify_plan(plan):
    identity = {k:v for k,v in plan.items() if k != "signature"}
    if fingerprint(identity) != plan["signature"]:
        raise ValueError("benchmark protocol signature changed")
    for relative, digest in plan["code"].items():
        if sha256(REPO / relative) != digest:
            raise ValueError(f"frozen benchmark implementation changed: {relative}")
    for name in ("inventory", "template", "manifest"):
        if sha256(plan[name]) != plan[name+"_sha256"]:
            raise ValueError(f"frozen benchmark {name} changed")
    for video in plan["videos"]:
        if sha256(video["prepared"]) != video["prepared_sha256"]:
            raise ValueError("prepared source description changed")


def cohort(inventory, manifest, ids=None, require_complete=False):
    with manifest.open() as stream:
        expected = list(csv.DictReader(stream))
    hashes = {r["sha256"]: r["filename"] for r in expected}
    if len(hashes) != len(expected) or not expected:
        raise ValueError("manifest is empty or contains duplicate originals")
    videos = []
    for row in read(inventory)["videos"]:
        if row["dataset"].lower() != "webvid":
            continue
        if not re.fullmatch(r"[A-Za-z0-9_-]+", row["id"]):
            raise ValueError("unsafe inventory video ID")
        if hashes.get(row["source_sha256"]) != Path(row["source_path"]).name:
            raise ValueError(f"source is not an exact member of the published manifest: {row['id']}")
        prepared = inventory.parent / "webvid" / row["id"] / "source_prepared.json"
        source = read(prepared)
        if source["source_sha256"] != row["source_sha256"]:
            raise ValueError("inventory/prepared source identity mismatch")
        from .qwen35_pairs_video import validate_source_window
        validate_source_window(source)
        if source["recipe"].get("source_start_seconds") != 0 or (source["width"],source["height"],source["fps"]) != (576,320,24):
            raise ValueError("expected complete 576x320 24fps input starting at frame zero")
        videos.append(dict(id=row["id"], prepared=str(prepared), prepared_sha256=sha256(prepared),
            source_sha256=row["source_sha256"], filename=Path(row["source_path"]).name,
            frames=source["frames"], fps=source["fps"], width=source["width"], height=source["height"]))
    if len({v["id"] for v in videos}) != len(videos) or len({v["source_sha256"] for v in videos}) != len(videos):
        raise ValueError("duplicate video in inventory")
    missing = [r["filename"] for r in expected if r["sha256"] not in {v["source_sha256"] for v in videos}]
    if require_complete and missing:
        raise ValueError(f"full manifest required; missing {len(missing)} videos: {missing}")
    available = len(videos)
    if ids:
        if len(set(ids)) != len(ids) or set(ids)-{v["id"] for v in videos}:
            raise ValueError("unknown or duplicate --ids")
        videos = [v for v in videos if v["id"] in ids]
    if not videos:
        raise ValueError("no selected WebVid sources")
    return videos, dict(expected=len(expected), available=available, selected=len(videos), missing=missing,
        full_manifest_selected=len(videos)==len(expected), exact_paper_reproduction=False)


def make_plan(args):
    from . import qwen35_pairs_video as base
    inventory, template, manifest = args.inventory.resolve(), args.template.resolve(), args.manifest.resolve()
    root = args.root.resolve()
    if root.exists() and any(root.iterdir()) and not (root / "protocol.json").exists():
        raise ValueError("output directory is nonempty and is not a recognized benchmark; use a new --root")
    if not args.snr or len(set(args.snr)) != len(args.snr) or any(not math.isfinite(s) for s in args.snr):
        raise ValueError("--snr requires unique finite values")
    if len({snr_name(s) for s in args.snr}) != len(args.snr):
        raise ValueError("SNR values collide at the supported directory precision")
    if not math.isfinite(args.extra_rate_ratio) or not 0 <= args.extra_rate_ratio <= .25:
        raise ValueError("extra rate ratio must be between 0 and 0.25")
    videos, coverage = cohort(inventory, manifest, args.ids, args.require_complete_manifest)
    if args.reuse_base and (len(videos)!=1 or len(args.snr)!=1):
        raise ValueError("--reuse-base requires exactly one video and one SNR")
    # Read-only preflight. Each source, adapter and frozen base implementation
    # is checked by the actual video runner, without making an output directory.
    started = time.monotonic()
    for i, video in enumerate(videos, 1):
        base.plan(Path(video["prepared"]), root / "preflight" / video["id"], template_path=template)
        print(f"PREFLIGHT {i}/{len(videos)} ({100*i/len(videos):.1f}%) {video['id']} full={video['frames']/24:.3f}s elapsed={time.monotonic()-started:.1f}s", flush=True)
    local = settings(REPO)
    t = read(template)
    for path in (local["python"], local["channel_python"], t["interpreters"]["qwen"]):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    paper_path = REPO / "configs/bsu_webvid_paper.json"
    result = dict(policy=POLICY, root=str(root), inventory=str(inventory), inventory_sha256=sha256(inventory),
        template=str(template), template_sha256=sha256(template), manifest=str(manifest), manifest_sha256=sha256(manifest),
        paper=read(paper_path), paper_sha256=sha256(paper_path), coverage=coverage, videos=videos,
        snr_db=args.snr, mode=args.mode, extra_rate_ratio=args.extra_rate_ratio,
        digital_bits_per_symbol={f"{s:g}":digital_bits(s) for s in args.snr},
        normalization=dict(width=576,height=320,fps=24,full_source=True),
        interpreters=dict(python=local["python"],channel_python=local["channel_python"],qwen=t["interpreters"]["qwen"]),
        reuse_base=str(args.reuse_base.resolve()) if args.reuse_base else None,
        code={f"src/semantic_transmission/{n}":sha256(REPO / "src/semantic_transmission" / n) for n in CODE},
        generation_seed=t["receiver"]["seed"], channel_seed=t["receiver"]["channel_seed"],
        steps=t["receiver"]["steps"], metric_profile="lgvsc_official_metrics_v1",
        model_description="Qwen semantic selection + pinned caption LoRA + repaired Open-Sora; combined uses unguarded trajectory synthesis then cadence",
        independent_semantic_review="PENDING")
    result["signature"] = fingerprint(result)
    if (root / "protocol.json").exists() and read(root / "protocol.json") != result:
        raise ValueError("benchmark inputs/code/options changed; use a new --root")
    if args.reuse_base:
        verify_base(args.reuse_base.resolve(), videos[0], args.snr[0], result)
    return result


def verify_base(base, video, snr, plan):
    from . import qwen35_pairs_comparison as pc, qwen35_pairs_video as current
    current.load_plan(base)
    p = pc.load_protocol(base)
    r = read(base / "RESULT.json")
    template = read(plan["template"])
    if (not r["status"].startswith("COMPLETE_RECONSTRUCTION") or r["frames"]!=video["frames"]
            or p["source"]["source_sha256"]!=video["source_sha256"] or r["snr_db"]!=snr
            or p["receiver"]["selector"]!=current.SELECTOR or p["caption"]!=template["caption"]):
        raise ValueError("completed base run is not the requested full-source BSU condition")
    for key in ("seed", "channel_seed", "steps", "models", "evaluation_profile", "concatenation_policy"):
        if p["receiver"].get(key) != template["receiver"].get(key):
            raise ValueError(f"base receiver differs from template: {key}")
    # Compare prepared bytes, not just the name of the underlying original.
    source = read(video["prepared"])
    if p["source"]["normalized_sha256"] != source["normalized_sha256"] or p["source"]["source_frame_hashes"] != source["source_frame_hashes"]:
        raise ValueError("completed base used different prepared pixels")
    run, movie, frames = paths_for_run(base)
    audit, quality = read(run/"output_audit.json"), read(run/"quality.json")
    if (audit["status"]!="PASSED" or quality["status"]!="PASSED"
            or quality["evaluation_profile"]!="lgvsc_official_metrics_v1"
            or sha256(movie)!=audit["video_sha256"] or quality["video_sha256"]!=audit["video_sha256"]
            or audit["source_indices"]!=list(range(video["frames"]))):
        raise ValueError("base output or metric provenance is incomplete")
    if p["receiver"]["snr_db"] != snr or r["signature"] != p["signature"]:
        raise ValueError("base channel condition or result signature changed")
    if read(run/"channel_accounting.json")["bits_per_symbol"] != digital_bits(snr):
        raise ValueError("base metadata modulation differs from benchmark link policy")
    # Verify actual generated files against the original reconstruction receipt.
    stages = base / "reconstruction/event_pairs/stages"
    for stage in ("reconstruct", "evaluate", "output-audit", "channel", "receive"):
        receipt = read(stages / f"{stage}.json")
        if receipt["status"]!="PASSED": raise ValueError(f"base stage incomplete: {stage}")
        for name, digest in receipt["artifacts"].items():
            if sha256(stages.parent / name)!=digest: raise ValueError(f"base artifact changed: {name}")
    return p, run, movie, frames, quality


def command(cmd, log, env):
    """Stream terminal progress and keep every attempt; stop the whole worker group."""
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print(f"START {log.stem}; log={log}; elapsed=0s, ETA unknown", flush=True)
    with log.open("x") as stream:
        proc = subprocess.Popen(list(map(str,cmd)), cwd=REPO, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, start_new_session=True)
        try:
            for line in proc.stdout:
                print(line, end="", flush=True); stream.write(line); stream.flush()
            if proc.wait(): raise RuntimeError(f"worker exited {proc.returncode}; log={log}")
        except BaseException:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try: proc.wait(timeout=15)
                except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            raise
    print(f"DONE {log.stem}; elapsed={time.monotonic()-started:.1f}s", flush=True)


def run_one(plan, video, snr, root):
    from . import bsu_webvid_base as current
    verify_plan(plan)
    template = copy.deepcopy(read(plan["template"]))
    template["receiver"]["snr_db"] = snr
    template["receiver"]["benchmark_digital_bits_per_symbol"] = digital_bits(snr)
    wrapper = "src/semantic_transmission/bsu_webvid_base.py"
    template["code"][wrapper] = sha256(REPO / wrapper)
    template["benchmark_parent_sha256"] = plan["template_sha256"]
    template_path = Path(plan["root"]) / "templates" / f"{snr_name(snr)}.json"
    freeze(template_path, template)
    base = Path(plan["reuse_base"]) if plan["reuse_base"] else root / "base"
    env = dict(environment(plan["generation_seed"]), PYTHONPATH=str(REPO/"src"))
    if not (base / "RESULT.json").exists():
        command([plan["interpreters"]["qwen"], "-m", current.MODULE, "--prepared",video["prepared"],
                 "--root",base,"--template",template_path], root / f"logs/base_{time.time_ns()}.log",env)
    p, run, movie, frames, quality = verify_base(base,video,snr,plan)
    header, _ = unpack((run / "transmitter/metadata.bin").read_bytes())
    keys = [k["index"] for k in header["keyframes"]]; validate_keys(keys,video["frames"])
    ledger = channel_breakdown((run/"transmitter/metadata.bin").read_bytes(),
        (run/"transmitter/visual.c64").stat().st_size//8,header["video"],read(run/"channel_accounting.json"))
    base_files = [base/"RESULT.json",base/"protocol.json",run/"output_audit.json",run/"quality.json",movie,
        run/"transmitter/metadata.bin",run/"received/metadata.bin",run/"data/normalized.mp4",
        *sorted(frames.glob("*.png")),*sorted((run/"receiver/frames/sample/key_frames_received").glob("*.png"))]
    job = dict(benchmark_signature=plan["signature"], base=str(base),video_id=video["id"],
        source_frames=p["source"]["source_frames"],snr_db=snr,mode=plan["mode"],
        trajectory_channel_seed=plan["channel_seed"]+1,extra_rate_ratio=plan["extra_rate_ratio"],
        digital_bits_per_symbol=digital_bits(snr),
        base_channel_uses=ledger["complex_channel_uses"]["total"],base_artifacts={str(f):sha256(f) for f in base_files})
    root.mkdir(parents=True,exist_ok=True);freeze(root/"job.json",job)
    stages=Stages(root,fingerprint(job))
    with lock(REPO/".local/etri_60s_check.lock"):
        for stage, owned in (("prepare","sender"),("channel","received"),("refine","refined")):
            interpreter=plan["interpreters"]["channel_python" if stage=="channel" else "python"]
            required=["refined/frames","refined/reconstruction.mp4","refined/refinement.json"] if stage=="refine" else [owned]
            stages.step(stage,[owned],required,lambda log,s=stage,py=interpreter:command(
                [py,"-m","semantic_transmission.bsu_sequence_refinement",s,"--job",root/"job.json"],log,env))
        products=["refined/quality.json","refined/quality_lossless_frames.csv","refined/quality_delivered_mp4.csv"]
        stages.step("evaluate",products,products,lambda log:command([plan["interpreters"]["python"],"-m",
            "semantic_transmission.bsu_sequence_refinement","evaluate","--job",root/"job.json"],log,env))
    refined=read(root/"refined/quality.json");channel=read(root/"received/channel.json")
    sender=read(root/"sender/sender.json")
    extra=channel["complex_channel_uses"]
    if extra!=sender["extra_channel_uses"] or extra>sender["maximum_extra_channel_uses"]:
        raise ValueError("trajectory rate accounting disagrees")
    denominator=ledger["cbr"]["denominator_source_scalars"]
    result=dict(status="COMPLETE_RECONSTRUCTION_METRICS_REVIEW_PENDING",signature=plan["signature"],
        id=video["id"],snr_db=snr,mode=plan["mode"],frames=video["frames"],fps=24,
        source_sha256=video["source_sha256"],base=str(base),base_video=str(movie),video=str(root/"refined/reconstruction.mp4"),
        base_quality={k:quality[k] for k in ("lossless_frames","delivered_mp4")},
        refined_quality=refined,base_channel_uses=job["base_channel_uses"],trajectory_channel_uses=extra,
        total_channel_uses=job["base_channel_uses"]+extra,cbr_denominator=denominator,
        digital_bits_per_symbol=digital_bits(snr),
        base_cbr=job["base_channel_uses"]/denominator,total_cbr=(job["base_channel_uses"]+extra)/denominator,
        trajectory_audit=sender,base_transmission_ledger=ledger,independent_semantic_review="PENDING",
        artifacts=snapshot(root,["sender","received","refined","job.json"]))
    write_json(root/"RESULT.json",result)
    return result


def load_result(root, plan, verify=True):
    path=root/"RESULT.json"
    if not path.exists(): return None
    result=read(path)
    if result["signature"]!=plan["signature"]: raise ValueError("result belongs to another benchmark")
    if not result["status"].startswith("COMPLETE_RECONSTRUCTION_METRICS"):
        raise ValueError("result is not a completed reconstruction and evaluation")
    if not verify:
        return result
    for name,digest in result["artifacts"].items():
        if sha256(root/name)!=digest: raise ValueError(f"completed benchmark artifact changed: {root/name}")
    for name,digest in read(root/"job.json")["base_artifacts"].items():
        if sha256(name)!=digest: raise ValueError(f"completed base artifact changed: {name}")
    return result


def aggregate(results):
    if not results: return None
    frames=sum(r["frames"] for r in results)
    output=dict(videos=len(results),frames=frames)
    for variant,field,uses in (("base","base_quality","base_channel_uses"),("refined","refined_quality","total_channel_uses")):
        values={}
        for boundary in ("lossless_frames","delivered_mp4"):
            for r in results:
                row=r[field][boundary]
                if row["frames"]!=r["frames"] or any(not math.isfinite(row[k]) for k in KEYS):
                    raise ValueError("invalid or partial per-video metrics")
            values[boundary]=dict(
                video_mean={k:sum(r[field][boundary][k] for r in results)/len(results) for k in KEYS},
                frame_mean={k:sum(r[field][boundary][k]*r["frames"] for r in results)/frames for k in KEYS})
        values.update(total_channel_uses=sum(r[uses] for r in results),
            pooled_cbr=sum(r[uses] for r in results)/sum(r["cbr_denominator"] for r in results),
            video_mean_cbr=sum(r[uses]/r["cbr_denominator"] for r in results)/len(results))
        output[variant]=values
    return output


def report(plan, verify=True):
    root=Path(plan["root"]);rows=[];groups={};pending=[]
    for snr in plan["snr_db"]:
        values=[]
        for video in plan["videos"]:
            r=load_result(root/snr_name(snr)/video["id"],plan,verify)
            if r is None: pending.append(dict(id=video["id"],snr_db=snr));continue
            if r["id"] != video["id"] or r["snr_db"] != snr or r["source_sha256"] != video["source_sha256"]:
                raise ValueError("result identity does not match the requested video/SNR")
            values.append(r)
            for variant,field,cbr in (("base","base_quality","base_cbr"),("refined","refined_quality","total_cbr")):
                rows.append(dict(id=video["id"],snr_db=snr,variant=variant,mode=plan["mode"],frames=r["frames"],
                    **{k:r[field]["delivered_mp4"][k] for k in KEYS},cbr=r[cbr],
                    trajectory_channel_uses=r["trajectory_channel_uses"] if variant=="refined" else 0,
                    video=r["base_video"] if variant=="base" else r["video"]))
        groups[f"{snr:g}"]=aggregate(values)
    summary=dict(status="COMPLETE" if not pending else "PARTIAL",signature=plan["signature"],
        completed_video_snr_pairs=len(rows)//2,expected_video_snr_pairs=len(plan["videos"])*len(plan["snr_db"]),
        coverage=plan["coverage"],pending=pending,by_snr=groups,paper_reference=plan["paper"],
        same_protocol_paper_comparison=False,independent_semantic_review="PENDING")
    write_json(root/"SUMMARY.json",summary)
    fields=["id","snr_db","variant","mode","frames",*KEYS,"cbr","trajectory_channel_uses","video"]
    with (root/"per_video.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    lines=["BSU WebVid reconstruction — delivered MP4, equal-video mean",
           f"Status: {summary['status']}; complete video/SNR pairs {summary['completed_video_snr_pairs']}/{summary['expected_video_snr_pairs']}",
           f"Sources: {plan['coverage']['selected']}/{plan['coverage']['expected']} manifest videos; exact paper reproduction: false",
           "SNR\tVariant\tN\tPSNR\tSSIM\tLPIPS\tCLIP\tDISTS\tCBR (pooled)"]
    for snr,g in groups.items():
        if g:
            for variant in ("base","refined"):
                scores=g[variant]["delivered_mp4"]["video_mean"]
                lines.append("\t".join([snr,variant,str(g["videos"]),*[f"{scores[k]:.6f}" for k in KEYS],f"{g[variant]['pooled_cbr']:.8g}"]))
        paper_cbr=plan["paper"]["skem_dsa_cbr_table_i"].get(snr)
        if paper_cbr is not None:
            lines.append(f"{snr}\tLGVSC paper reference\t55\tNA\tNA\tNA\tNA\tNA\t{paper_cbr:.8g}")
    lines += ["",*plan["paper"]["comparison_limits"],plan["paper"]["quality_reference_note"],plan["paper"]["source"]]
    (root/"comparison.tsv").write_text("\n".join(lines)+"\n")
    print("\n".join(lines[:4]+[l for l in lines[4:] if "\t" in l]),flush=True)
    return summary


def run(plan, keep_going=False):
    root=Path(plan["root"]);root.mkdir(parents=True,exist_ok=True)
    with lock(root/"execution.lock"):
        freeze(root/"protocol.json",plan)
        report(plan)
        total=len(plan["videos"])*len(plan["snr_db"]);done=0;failures=[];started=time.monotonic()
        for video in plan["videos"]:
            for snr in plan["snr_db"]:
                directory=root/snr_name(snr)/video["id"]
                state=dict(status="RUNNING",current=dict(id=video["id"],snr_db=snr),completed=done,total=total,failures=failures)
                write_json(root/"status.json",state)
                print(f"BATCH {done}/{total} ({100*done/total:.1f}%) {video['id']} SNR={snr:g} mode={plan['mode']} elapsed={time.monotonic()-started:.1f}s; remaining ETA unknown",flush=True)
                try:
                    existing=load_result(directory,plan)
                    if existing is None: run_one(plan,video,snr,directory)
                    else: print("RESUME: complete source, output and metrics hashes verified; no inference repeated",flush=True)
                    done+=1
                    # Files from this job have just been validated; avoid rereading
                    # every previous PNG at each incremental table update.
                    report(plan,verify=False)
                except (Exception,KeyboardInterrupt) as error:
                    failures.append(dict(id=video["id"],snr_db=snr,error=str(error)))
                    write_json(root/"status.json",dict(state,status="INTERRUPTED" if isinstance(error,KeyboardInterrupt) else "FAILED",failures=failures))
                    # Preserve corrupted evidence; never silently omit it from a complete table.
                    if isinstance(error,KeyboardInterrupt) or not keep_going: raise
        summary=report(plan)
        write_json(root/"status.json",dict(status=summary["status"],completed=done,total=total,failures=failures,
            seconds=time.monotonic()-started))
        print(f"BATCH {done}/{total} ({100*done/total:.1f}%) {summary['status']}",flush=True)
        return 1 if failures or summary["pending"] else 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("check","run","status","report"))
    parser.add_argument("--root",type=Path,default=REPO/"outputs/bsu_webvid_paper")
    parser.add_argument("--inventory",type=Path,default=REPO/"outputs/remote_webvid_full_20261012/inventory.json")
    parser.add_argument("--template",type=Path,default=REPO/".local/deployment/bsu_template.json")
    parser.add_argument("--manifest",type=Path,default=REPO/"docs/eval_manifest_webvid.csv")
    parser.add_argument("--ids",nargs="+")
    parser.add_argument("--snr",nargs="+",type=float,default=[6.,8.,10.])
    parser.add_argument("--mode",choices=MODES,default="combined")
    parser.add_argument("--extra-rate-ratio",type=float,default=.05)
    parser.add_argument("--require-complete-manifest",action="store_true")
    parser.add_argument("--keep-going",action="store_true")
    parser.add_argument("--reuse-base",type=Path,help="verified completed BSU run; only with one selected video/SNR")
    args=parser.parse_args()
    if args.action=="status":
        print(json.dumps(read(args.root/"status.json"),ensure_ascii=False,indent=2));return
    if args.action=="report":
        plan=read(args.root/"protocol.json")
        with lock(args.root/"execution.lock"): report(plan)
        return
    plan=make_plan(args)
    if args.action=="check":
        print(json.dumps(dict(status="READY",coverage=plan["coverage"],snr_db=plan["snr_db"],mode=plan["mode"],
            jobs=len(plan["videos"])*len(plan["snr_db"]),model_inference_started=False),ensure_ascii=False,indent=2))
    else:
        signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))
        raise SystemExit(run(plan,args.keep_going))


if __name__=="__main__":
    main()
