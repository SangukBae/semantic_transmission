"""Run variable-count Qwen semantic selection and adopted captions on a video.

Only source normalization is reused. Observations, keys and captions are newly
inferred. Receiver policies match the completed pairs/caption experiment.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from . import auto_extraction as ae, qwen35_state_delta as sd, qwen35_event_pairs as ep
from . import qwen35_pairs_comparison as pc
from . import qwen35_semantic_selection as semantic
from .artifacts import sha256, write_json
from .validation_progress import emit

REPO = ae.REPO
MODULE = "semantic_transmission.qwen35_pairs_video"
ARM = "event_pairs"
SELECTOR = semantic.POLICY
LEGACY_SELECTOR = "qwen35_source_event_pairs"
DEFAULT_SOURCE = REPO / "outputs/fc_lgvsc_webvid_tvsum_full_20261001/webvid/webvid_004_10d5cd04/source_prepared.json"
DEFAULT_ROOT = REPO / "outputs/qwen35_semantic_zea_20261009"
TEMPLATE = pc.ROOT / "protocol.json"


def key_budget(frames):
    if frames < 2:
        raise ValueError("at least two source frames required")
    # Preserve the historical 78 intervals per 1439 source-frame gaps.
    return min(frames, max(math.ceil((frames - 1) / 24) + 1,
                           1 + round((frames - 1) * 78 / 1439)))


def freeze(path, value):
    # Compare the persisted JSON representation: configurations can contain
    # tuples, which become lists on disk without changing their meaning.
    persisted = json.loads(json.dumps(value))
    if path.exists():
        if ae.read(path) != persisted:
            raise ValueError(f"frozen input/code changed: {path}; use a new output")
    else:
        write_json(path, value)


def validate_source_window(source, duration_limit=None):
    """Clipped comparisons must opt in and retain their original-source provenance."""
    recipe = source["recipe"]
    if source["fps"] != 24 or recipe.get("duration_limit") != duration_limit:
        raise ValueError("requires a matching, explicitly declared 24 FPS source window")
    expected = source["source_duration_seconds"]
    if duration_limit is not None:
        if duration_limit <= 0 or recipe.get("source_start_seconds") != 0:
            raise ValueError("comparison window must start at zero with a positive duration")
        expected = min(expected, duration_limit)
    if abs(source["frames"] / 24 - expected) > 1 / 24:
        raise ValueError("prepared input does not match the declared source window")


def selection_settings(frames, selection_policy="semantic", budget=None, stride=4):
    semantic.candidates(frames, stride)
    if selection_policy == "semantic":
        if budget is not None:
            raise ValueError("semantic selection has no key budget; use fixed_budget only for an explicit ablation")
        return dict(selection_policy="semantic", selector=SELECTOR, key_budget=None,
                    key_budget_policy="determined_by_sequential_semantic_changes", max_gap=None,
                    candidate_stride=stride, selection_prompt=semantic.PROMPT, selection_tokens=256)
    if selection_policy != "fixed_budget":
        raise ValueError("unknown selection policy")
    count = key_budget(frames) if budget is None else budget
    sd.budget_keys([0.] * frames, count)
    return dict(selection_policy="fixed_budget", selector=LEGACY_SELECTOR, key_budget=count,
                key_budget_policy="explicit_fixed_budget_ablation", max_gap=24)


def plan(prepared, root, budget=None, *, duration_limit=None, selection_policy="semantic", stride=4,
         template_path=None):
    source = ae.read(prepared)
    pc.forbid_overlap(root, [prepared.parent, source["source_path"], pc.ROOT,
                            pc.SNAPSHOT, REPO / "outputs/qwen35_evidence_repair_v3"])
    validate_source_window(source, duration_limit)
    for path, digest in ((source["source_path"], source["source_sha256"]),
                         (source["normalized_video"], source["normalized_sha256"])):
        if sha256(path) != digest:
            raise ValueError(f"source changed: {path}")
    for i in range(source["frames"]):
        ae.check_frame(source, i)
    template_path = Path(template_path or TEMPLATE).resolve()
    template = ae.read(template_path)
    selection = selection_settings(source["frames"], selection_policy, budget, stride)
    code = dict(template["code"])
    for module in (sd, ep, semantic):
        p = Path(module.__file__).resolve()
        code[str(p.relative_to(REPO))] = sha256(p)
    for relative in ("src/semantic_transmission/qwen35_feedback.py",
                     "src/semantic_transmission/qwen35_pairs_video.py",
                     "scripts/run_qwen35_pairs_video.sh"):
        code[relative] = sha256(REPO / relative)
    for relative, digest in code.items():
        if sha256(REPO / relative) != digest:
            raise ValueError(f"adopted implementation changed: {relative}")
    caption = template["caption"]
    pc.verify_adapter(caption["caption_adapter"])
    value = dict(policy="qwen35_semantic_video_v2" if selection_policy == "semantic" else "qwen35_fixed_budget_ablation_v2", source=source,
        source_metadata=str(prepared), source_metadata_sha256=sha256(prepared),
        caption=caption, **selection, observation_stride=4, observation_window=48, max_images=16,
        observation_prompt=sd.OBSERVE, observation_tokens=768,
        template=str(template_path), template_sha256=sha256(template_path), code=code,
        training_executed=False, reused_selection=False, reused_captions=False)
    value["signature"] = ae.digest(value)
    return value


def make_protocol(root, p, keys):
    template_path = Path(p["template"])
    if sha256(template_path) != p["template_sha256"]:
        raise ValueError("frozen receiver template changed")
    t = ae.read(template_path)
    source = p["source"]
    cfg = dict(t["receiver"])
    for field in list(cfg):
        if field.startswith("internvl") or field in {
            "selector_environment", "threshold", "selection_stride", "skim_keyframes",
            "max_tiles", "caption_attention", "paper_caption", "max_gap", "key_budget"}:
            cfg.pop(field)
    cfg.update(profile=p["policy"], selector=p.get("selector", SELECTOR), method="key_frames_" + p.get("selector", SELECTOR),
        input=source["normalized_video"], input_sha256=source["normalized_sha256"],
        frames=source["frames"], max_frames=source["frames"], fps=source["fps"],
        width=source["width"], height=source["height"], caption_adapter=p["caption"]["caption_adapter"])
    value = dict(policy=p["policy"], source=source, keys={ARM: keys}, caption=p["caption"],
        receiver=cfg, interpreters=t["interpreters"], base_model_files=t["base_model_files"],
        inputs={str(root / name): sha256(root / name) for name in ("plan.json", "selection.json", "ledger.json")},
        code=p["code"], training_executed=False, independent_semantic_review="PENDING")
    value["inputs"].update({p["source_metadata"]: p["source_metadata_sha256"],
                            p["template"]: p["template_sha256"]})
    value["signature"] = ae.digest(value)
    freeze(root / "protocol.json", value)
    return pc.load_protocol(root)


def progress(stage, done, total, start):
    elapsed = time.monotonic() - start
    eta = f" | ETA {elapsed / done * (total-done) / 60:.1f}분" if done else ""
    print(f"{stage}: {100*done/total:.1f}% ({done}/{total}) | 경과 {elapsed/60:.1f}분{eta}", flush=True)
    emit(done, total)


def load_plan(root):
    p = ae.read(root / "plan.json")
    if ae.digest({k:v for k,v in p.items() if k != "signature"}) != p["signature"]:
        raise ValueError("selection plan signature changed")
    for path, digest in p["code"].items():
        if sha256(REPO / path) != digest:
            raise ValueError(f"frozen selection code changed: {path}")
    for i in range(p["source"]["frames"]):
        ae.check_frame(p["source"], i)
    return p


def select_keys(root, p):
    source = p["source"]
    obs_plan = dict(source, **{k: v for k, v in p["caption"].items()
                               if k not in {"caption_adapter", "prompts"}},
                    signature=p["signature"], max_images=16)
    session = sd.Session(root, obs_plan)
    if p.get("selection_policy") == "semantic":
        keys, decisions = semantic.select(source["frames"], session,
            stride=p["candidate_stride"], prompt=p["selection_prompt"], tokens=p["selection_tokens"],
            progress=lambda n,t,s: progress("Qwen 의미 변화 순차 판정", n,t,s))
        calls = [{k:v for k,v in c.items() if k != "reused"} for c in session.calls]
        freeze(root / "ledger.json", ae.seal(dict(signature=p["signature"], decisions=decisions, calls=calls)))
        freeze(root / "selection.json", ae.seal(dict(signature=p["signature"], keys=keys,
            selector=SELECTOR, key_budget=None, max_gap=None, candidate_stride=p["candidate_stride"],
            endpoint_keys=[0,source["frames"]-1], semantic_change_keys=[r["candidate"] for r in decisions if r["selected"]],
            decisions=decisions, calls=calls)))
        return keys, session
    windows = list(ae.proposal_windows(source["frames"], 4, 48))
    observations, started = [], time.monotonic()
    for n, ids in enumerate(windows):
        value = session.call("state_observe", ids, p["observation_prompt"],
            lambda obj: sd.validate_observation(obj, ids), p["observation_tokens"])
        observations.append(dict(window=n, indices=ids, result=value))
        progress("Qwen 원본 관찰", n+1, len(windows), started)
    freeze(root / "ledger.json", ae.seal(dict(signature=p["signature"], observations=observations)))
    scores, details = sd.selection_scores(source, observations)
    prior = sd.budget_keys(scores, p["key_budget"])
    pairs = ep.source_pairs(observations)
    selection_path = root / "selection.json"
    if selection_path.exists():
        saved = ae.unseal(selection_path)
        if saved["signature"] != p["signature"] or saved["pairs"] != pairs or saved["prior_keys"] != prior:
            raise ValueError("source selection inputs changed")
        keys = saved["keys"]
    else:
        keys, solver = ep.paired_keys(source["frames"], p["key_budget"], pairs, prior, details["pixel_novelty"])
        write_json(selection_path, ae.seal(dict(signature=p["signature"], keys=keys,
            pairs=pairs, prior_keys=prior, scores=scores, details=details, solver=solver)))
    ep.validate_keys(keys, source["frames"], p["key_budget"], 24)
    return keys, session


def select_only(root, p):
    root.mkdir(parents=True, exist_ok=True)
    with pc.lock(REPO / ".local/etri_60s_check.lock"), pc.lock(root / ".execution.lock"):
        freeze(root / "plan.json", p)
        keys, session = select_keys(root, load_plan(root))
        result = dict(status="SELECTION_COMPLETE_CAPTIONS_RECONSTRUCTION_PENDING", keys=keys,
            keyframes=len(keys), segments=len(keys)-1, selector=p["selector"], signature=p["signature"],
            independent_semantic_review="PENDING", new_model_calls=sum(not c["reused"] for c in session.calls),
            reused_model_calls=sum(c["reused"] for c in session.calls))
        write_json(root / "selection_status.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def extract(root):
    from PIL import Image
    from .qwen35_backend import LocalQwen35
    p = load_plan(root)
    source = p["source"]
    keys, session = select_keys(root, p)
    protocol = make_protocol(root, p, keys)
    vlm = session.model or LocalQwen35(p["caption"])
    vlm.config = p["caption"]  # Same backbone; LoRA is enabled only for draft calls.
    rows, started = [], time.monotonic()
    for n, (a, b) in enumerate(zip(keys, keys[1:])):
        ids = pc.sample_indices(a, b)
        paths = [ae.check_frame(source, i) for i in ids]
        identity = dict(protocol=protocol["signature"], arm=ARM, segment=n, start=a,
            end_exclusive=b, source_indices=ids, frame_hashes=[sha256(path) for path in paths])
        labels = [f"Source frame {i}, time {i/24:.6f} seconds" for i in ids]
        images = [Image.open(path).convert("RGB") for path in paths]
        try:
            draft = pc.infer(vlm, root / f"calls/{ARM}/draft/{n:05d}.json", dict(identity, task="draft"),
                images, labels, p["caption"]["prompts"]["caption"], ae.caption_validator(p["caption"]))
            review = pc.infer(vlm, root / f"calls/{ARM}/review/{n:05d}.json",
                dict(identity, task="visual_review", draft_sha256=ae.digest(draft)), images, labels,
                p["caption"]["prompts"]["review"] + draft["caption"], ae.caption_validator(p["caption"], True))
        finally:
            for im in images:
                im.close()
        rows.append(dict(segment=n, start=a, end_exclusive=b, source_indices=ids, **review))
        progress("개선 캡션 + 원본 재검토", n+1, len(keys)-1, started)
    freeze(root / f"captions/{ARM}.json", ae.seal(dict(protocol=protocol["signature"], arm=ARM,
        author=pc.MODEL_ID, model_revision=pc.REVISION, caption_adapter=p["caption"]["caption_adapter"],
        independent_verification=False, captions=rows)))
    pc.validate_bundle(root, ARM, protocol)


def launch(command, log, env, resource):
    """Retain full logs while displaying measured worker progress every 20s."""
    start = time.monotonic()
    ipc = resource.with_suffix(".progress.json")
    ipc.unlink(missing_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x") as stream:
        process = subprocess.Popen(command, cwd=REPO,
            env=dict(env, QUALITY_PROGRESS_FILE=str(ipc), PYTHONUNBUFFERED="1"),
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while True:
                try:
                    code = process.wait(timeout=20)
                    break
                except subprocess.TimeoutExpired:
                    value = ae.read(ipc) if ipc.exists() else dict(done=0, total=1)
                    progress(log.stem, value["done"], value["total"], start)
                    # Extraction has two successive counters; show its named stage.
                    if log.stem == "extract":
                        rows = [r for r in log.read_text(errors="replace").splitlines() if "% (" in r]
                        if rows:
                            print("  " + rows[-1], flush=True)
            if code:
                print(log.read_text(errors="replace")[-7000:], flush=True)
                raise RuntimeError(f"worker failed ({code}): {log}")
            progress(log.stem, 1, 1, start)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
        finally:
            write_json(resource, dict(command=command, seconds=time.monotonic()-start,
                                      returncode=process.returncode))


def prepare_text(run):
    # Instrument the existing cache routine; leave inference and cache contracts intact.
    cache = pc.text_cache
    original = cache.populate
    def populate(prompts, contract, root, factory):
        total = len(set(prompts))
        signature = ae.digest(contract)
        existing = sum((root / signature / f"{ae.digest(p)}.json").exists() for p in set(prompts))
        saved, start, count = cache.legacy.save_entry, time.monotonic(), existing
        def save(*args, **kwargs):
            nonlocal count
            result = saved(*args, **kwargs)
            count += 1
            progress("T5 캡션 인코딩", count, total, start)
            return result
        cache.legacy.save_entry = save
        try:
            return original(prompts, contract, root, factory)
        finally:
            cache.legacy.save_entry = saved
    cache.populate = populate
    try:
        pc.fc.prepare_text(run)
    finally:
        cache.populate = original


def worker(root, stage):
    if stage == "extract":
        return extract(root)
    p = pc.load_protocol(root)
    run = root / f"reconstruction/{ARM}/run"
    if stage == "caption":
        pc.import_captions(root, ARM)
        path = run / "caption_provenance.json"
        value = ae.read(path)
        value["selector"] = p["receiver"]["selector"]
        write_json(path, value)
    elif stage == "channel":
        pc.paired_channel(run)
    elif stage == "prepare-text":
        prepare_text(run)
    elif stage == "reconstruct":
        pc.fc.reconstruct(run)
    elif stage == "output-audit":
        pc.fc.audit(run)
        path = run / "output_audit.json"
        value = ae.read(path)
        value.update(selector=p["receiver"]["selector"], automatic_keyframe_insertion=False,
            automatic_keyframe_insertion_stage=None,
            full_source_reconstructed=p["source"]["recipe"].get("duration_limit") is None,
            prepared_source_reconstructed=True,
            normalized_source_frames=p["source"]["frames"])
        write_json(path, value)
    else:
        raise ValueError(stage)


def execute(root, p):
    root.mkdir(parents=True, exist_ok=True)
    with pc.lock(REPO / ".local/etri_60s_check.lock"), pc.lock(root / ".execution.lock"):
        freeze(root / "plan.json", p)
        env = dict(pc.environment(2025), PYTHONPATH=str(REPO / "src"))
        if not (root / f"captions/{ARM}.json").exists():
            stamp = time.time_ns()
            launch([str(REPO / ".local/qwen35_env/bin/python"), "-m", MODULE, "--root", str(root),
                "--worker", "extract"], root / f"logs/extract_{stamp}/extract.log", env,
                root / f"resources/extract_{stamp}.json")
        protocol = pc.load_protocol(root)
        pc.validate_bundle(root, ARM, protocol)
        dest = root / f"reconstruction/{ARM}"
        cfg, keys = protocol["receiver"], protocol["keys"][ARM]
        freeze(dest / "execution_protocol.json", dict(signature=protocol["signature"], config=cfg,
            condition_plan=pc.collision.preview(keys, pc.collision.upstream_functions(REPO))))
        stages = pc.Stages(dest, protocol["signature"])
        stages.step("initialize", ["run"], pc.initialization_products(p["source"]["frames"]),
            lambda _: pc.initialize(dest, p["source"], dict(indices=keys, selector=cfg["selector"]), cfg))
        jobs = list(pc.reconstruction_jobs())
        for n, (stage, module, products) in enumerate(jobs):
            print(f"수신·복원 단계: {n}/{len(jobs)} 완료 ({100*n/len(jobs):.1f}%, 단계 수 기준) | {stage}", flush=True)
            python = protocol["interpreters"]["channel_python" if stage == "channel" else "python"]
            if module == pc.MODULE:
                module = MODULE
                args = ["--worker", stage, "--root", str(root)]
            elif module == pc.fc.ETRI:
                args = ["--worker", stage, "--run-dir", str(dest / "run")]
            else:
                args = [stage, str(dest / "run")]
            required = [*[f"run/{v}" for v in products], f"resources/{stage}.json"]
            worker_env = dict(env, ETRI_RUN_SIGNATURE=protocol["signature"])
            if stage == "channel":
                worker_env["CUDA_VISIBLE_DEVICES"] = "-1"
            stages.step(stage, required, required,
                lambda log, cmd=[python, "-m", module, *args], e=worker_env, s=stage:
                    launch(cmd, log, e, dest / f"resources/{s}.json"))
        finalize(root, protocol, stages)


def finalize(root, p, stages):
    run = root / f"reconstruction/{ARM}/run"
    audit, quality = ae.read(run / "output_audit.json"), ae.read(run / "quality.json")
    if audit["status"] != "PASSED" or quality["status"] != "PASSED" or quality["video_sha256"] != audit["video_sha256"]:
        raise ValueError("reconstruction/evaluation audit mismatch")
    video = Path(audit["video"])
    if sha256(video) != audit["video_sha256"]:
        raise ValueError("output changed after audit")
    comparison = root / "source_vs_output.mp4"
    if not comparison.exists():
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", p["source"]["normalized_video"],
            "-i", str(video), "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]", "-map", "[v]",
            "-an", "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p",
            str(comparison)], check=True)
    from .etri_60s_check import pts_audit
    pts_audit(comparison, p["source"]["frames"], 24)
    result = dict(status="COMPLETE_RECONSTRUCTION_INDEPENDENT_REVIEW_PENDING", video=str(video),
        comparison=str(comparison), source=p["source"]["source_path"], frames=p["source"]["frames"],
        fps=24, duration_seconds=p["source"]["frames"]/24, keyframes=len(p["keys"][ARM]),
        captions=len(p["keys"][ARM])-1, model_id=pc.MODEL_ID, caption_adapter=p["caption"]["caption_adapter"]["id"],
        selector=p["receiver"]["selector"], snr_db=p["receiver"]["snr_db"], steps=p["receiver"]["steps"],
        metrics=quality["delivered_mp4"], training_executed=False,
        independent_semantic_review="PENDING", signature=p["signature"],
        source_window=p["source"]["recipe"],
        stage_seconds={name: value["seconds"] for name, value in stages.completed.items()})
    write_json(root / "RESULT.json", result)
    print("전체 작업 100% 완료", flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--template", type=Path, default=TEMPLATE,
                        help="receiver/caption template for this deployment; frozen in the new plan")
    parser.add_argument("--key-budget", type=int)
    parser.add_argument("--selection-policy", choices=("semantic", "fixed_budget"), default="semantic")
    parser.add_argument("--candidate-stride", type=int, default=4)
    parser.add_argument("--duration-limit", type=float)
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--worker", choices=("extract", "caption", "channel", "prepare-text", "reconstruct", "output-audit"))
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))
    root = args.root.resolve()
    if args.worker:
        return worker(root, args.worker)
    print("입력과 기존 구현 검증: 0%", flush=True)
    p = plan(args.prepared.resolve(), root, args.key_budget, duration_limit=args.duration_limit,
             selection_policy=args.selection_policy, stride=args.candidate_stride, template_path=args.template)
    count = "의미 변화에 따라 키 개수 결정" if p["key_budget"] is None else f"{p['key_budget']} keys (fixed-budget ablation)"
    print(f"입력 검증 100% | {p['source']['frames']} frames | {p['source']['frames']/24:.3f}s | {count}", flush=True)
    if args.check:
        print(json.dumps(dict(status="READY", model_inference_started=False, key_budget=p["key_budget"])))
        return
    if args.select_only:
        return select_only(root, p)
    execute(root, p)


if __name__ == "__main__":
    main()
