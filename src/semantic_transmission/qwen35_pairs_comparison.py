"""Resumable, frozen-selector comparison with the adopted caption adapter.

Default: caption both 79-key conditions, transmit/reconstruct both, score image
quality, then run a NON-INDEPENDENT source-grounded Qwen diagnostic. --check is
read-only and never loads a model. Historical selections/outputs are read-only.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time

from . import auto_extraction as ae, fc_lgvsc_qwen35 as fc
from . import condition_collision as collision, generation_noise as noise
from . import precision_text_cache as text_cache, short_schedule, tail_reference as tail
from .artifacts import sha256, write_json
from .assisted_captions import sample_indices
from .cli import settings
from .etri_60s_check import lock
from .hybrid_reconstruction import initialize, initialization_products, launch
from .qwen35_backend import MODEL_ID, REVISION, SNAPSHOT
from .qwen35_caption_adapter import default_adapter, verify_adapter, verify_record
from .webvid_ablation import Stages, environment

REPO = Path(__file__).resolve().parents[2]
MODULE = "semantic_transmission.qwen35_pairs_comparison"
ROOT = REPO / "outputs/qwen35_pairs_caption_comparison_20261007"
INTERNVL = REPO / "outputs/etri_hybrid_keys_tv_low_08_v1"
PAIRS = REPO / "outputs/qwen35_event_pairs_20261005"
REFERENCE = REPO / "outputs/qwen35_feedback_20261003/astra_reference/reference.json"
POLICY = "frozen_79keys_same_caption_repair_v1"
ARMS = ("internvl", "event_pairs")
LABELS = {"internvl": "기존 assistant 후보 + InternVL, 79키",
          "event_pairs": "Qwen 관찰 + 사건 전후 쌍 보존, 79키"}


def validate_keys(keys, frames, count=79):
    if (any(type(k) is not int for k in keys) or len(keys) != count
            or keys != sorted(set(keys)) or keys[0] != 0 or keys[-1] != frames - 1
            or max(b-a for a, b in zip(keys, keys[1:])) > 24):
        raise ValueError("selection must have 79 ordered keys, endpoints and maximum 24-frame gap")


def forbid_overlap(output, protected):
    output = output.resolve()
    for item in protected:
        item = Path(item).resolve()
        if output == item or item in output.parents or output in item.parents:
            raise ValueError(f"output overlaps protected input: {item}")


def build_protocol():
    # Verify frozen artifacts, without demanding that old live source code has
    # the old revision: this run consumes selections, not historic executables.
    old = ae.read(INTERNVL / "protocol.json")
    freeze = ae.read(INTERNVL / "selection_freeze.json")
    selection = ae.read(INTERNVL / "selection.json")
    for filename, field in (("protocol.json", "protocol_sha256"),
                            ("selection.json", "selection_sha256"), ("keyframes.json", "keyframes_sha256")):
        if sha256(INTERNVL / filename) != freeze[field]:
            raise ValueError(f"historical InternVL selection changed: {filename}")
    if selection["status"] != "COMPLETE" or selection["protocol_signature"] != old["signature"]:
        raise ValueError("historical InternVL selection is incomplete")
    for name, digest in freeze["score_files"].items():
        if sha256(INTERNVL / name) != digest:
            raise ValueError("historical SKEM score changed")
    plan = ae.read(PAIRS / "plan.json")
    signed = dict(plan)
    signature = signed.pop("signature")
    if ae.digest(signed) != signature:
        raise ValueError("event-pair plan signature changed")
    pairs = ae.unseal(PAIRS / "pairs_selection.json")
    if pairs["signature"] != signature or not pairs["solver"]["optimal"]:
        raise ValueError("event-pair selection is not a completed frozen optimum")
    # The completed evaluator also binds the selected keys to its source plan.
    pair_result = ae.read(PAIRS / "evaluation/RESULT.json")
    if (pair_result["signature"] != signature or pair_result["metrics"]["pairs"]["keyframes"] != len(pairs["keys"])
            or pair_result["reference_sha256"] != sha256(REFERENCE)):
        raise ValueError("event-pair evaluation provenance differs")
    completed_pairs = ae.unseal(PAIRS / "pairs.json")
    if completed_pairs["signature"] != signature or completed_pairs["keys"] != pairs["keys"]:
        raise ValueError("event-pair completed extraction differs from frozen selection")
    keys = dict(internvl=selection["indices"], event_pairs=pairs["keys"])
    if keys["internvl"] != ae.read(INTERNVL / "keyframes.json")["indices"]:
        raise ValueError("historical key lists differ")
    source = {k: plan[k] for k in ("source_frames", "source_frame_hashes", "normalized_video",
                                    "normalized_sha256", "frames", "fps", "video_id")}
    if (source["frames"], source["fps"]) != (1440, 24):
        raise ValueError("this comparison is the frozen tv_low_08 60-second experiment")
    if (old["normalized_sha256"] != source["normalized_sha256"]
            or old["source_frame_hashes"] != source["source_frame_hashes"]):
        raise ValueError("the selectors did not use identical source pixels")
    if sha256(Path(source["normalized_video"])) != source["normalized_sha256"]:
        raise ValueError("normalized video changed")
    for i in range(source["frames"]):
        if sha256(Path(source["source_frames"]) / f"{i}.png") != source["source_frame_hashes"][str(i)]:
            raise ValueError(f"source PNG changed: {i}")
    for arm in ARMS:
        validate_keys(keys[arm], source["frames"])
    for name, digest in freeze["frames"].items():
        if sha256(INTERNVL / "selected_frames" / name) != digest:
            raise ValueError("historical selected PNG changed")
    baseline = Path(old["baseline"])
    cfg = dict(ae.read(baseline / "run_config.json"))
    for key in ("selector_checkpoint", "caption_checkpoint", "caption_bundle", "hybrid_selection_root"):
        cfg.pop(key, None)
    cfg.update(profile=POLICY, input=source["normalized_video"], input_sha256=source["normalized_sha256"],
        semantic_clip_policy="frame_exact", official_semantic_clips=True, cpu_video_storage=True,
        concatenation_policy="endpoint_exact", visual_channel="source_frame_seeded_numpy_awgn_v1",
        caption_provider="local_qwen35_caption_repair", seed=2025, channel_seed=42)
    caption = dict(backend="qwen35", model_id=MODEL_ID, model_revision=REVISION,
        model_path=str(SNAPSHOT), quantization="nf4", max_pixels=262144, caption_max_words=79,
        caption_adapter=default_adapter(), prompts=dict(caption=ae.CAPTION_PROMPT, review=ae.REVIEW_PROMPT))
    code = set(tail.CODE) | set(text_cache.CODE) | {
        "src/semantic_transmission/" + name + ".py" for name in (
            "qwen35_pairs_comparison", "qwen35_pairs_review", "qwen35_backend", "qwen35_caption_adapter",
            "auto_extraction", "assisted_captions", "fc_lgvsc_qwen35", "hybrid_reconstruction",
            "condition_collision", "short_schedule", "decoder_memory", "generation_noise", "webvid_ablation", "artifacts",
            "codec_transport", "metadata_channel", "wire", "transmission_accounting", "workers",
            "etri_60s", "etri_60s_check", "research_quality", "official_quality", "semantic_clips",
            "temporal", "decoder_runner", "qwen35_feedback_eval")}
    code |= {"scripts/" + name for name in ("run_qwen35_pairs_caption_comparison.sh",
        "qwen35_caption.py", "qwen35_advanced_caption.py", "etri_schedule_fix_decoder.py",
        "etri_condition_fix_decoder.py", "etri_precision_decoder.py")}
    inputs = [INTERNVL / name for name in ("protocol.json", "selection.json", "selection_freeze.json")]
    inputs += [PAIRS / name for name in ("plan.json", "pairs_selection.json", "pairs.json", "evaluation/RESULT.json")]
    inputs += [baseline / "run_config.json", REFERENCE, Path(plan["model_manifest"])]
    manifest = ae.read(Path(plan["model_manifest"]))
    if (sha256(Path(plan["model_manifest"])) != plan["model_manifest_sha256"]
            or (manifest["model_id"], manifest["revision"]) != (MODEL_ID, REVISION)):
        raise ValueError("base model manifest changed")
    model_files = {}
    for item in manifest["files"]:
        path = SNAPSHOT / item["file"]
        stat = path.stat()
        if stat.st_size != item["size"]:
            raise ValueError(f"base model file is incomplete: {path}")
        model_files[str(path)] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=item["sha256"])
    protocol = dict(policy=POLICY, source=source, keys=keys, labels=LABELS, caption=caption, receiver=cfg,
        base_model_files=model_files,
        inputs={str(p): sha256(p) for p in inputs}, code={p: sha256(REPO / p) for p in sorted(code)},
        interpreters=dict(qwen=str(REPO / ".local/qwen35_env/bin/python"), **{
            k: settings(REPO)[k] for k in ("python", "channel_python")}),
        reference=str(REFERENCE), training_executed=False, selectors_reused=True,
        scope="One known development video; frozen historical assistant/InternVL vs Qwen event-pair selections. "
              "Same caption model/prompt/budget and decoder. Not a new current-proposal baseline or model-only ablation.",
        semantic_review="Local base Qwen, adapter disabled; NON-INDEPENDENT diagnostic, independent review PENDING",
        noise_scope="AWGN seeded per source frame; decoder noise scopes use same seed/segment ordinal. "
                    "Different boundaries/shapes are not identical diffusion tensors.")
    protocol["signature"] = ae.digest(protocol)
    return protocol


def preflight(root):
    forbid_overlap(root, [INTERNVL, PAIRS, REFERENCE.parent, SNAPSHOT,
                         REPO / "outputs/qwen35_evidence_repair_v3", REPO / ".local"])
    protocol = build_protocol()
    forbid_overlap(root, [protocol["source"]["source_frames"], protocol["source"]["normalized_video"]])
    for name in ("ffmpeg", "ffprobe"):
        if not shutil.which(name):
            raise FileNotFoundError(name)
    for path in protocol["interpreters"].values():
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    for key in ("stdit", "vae", "t5", "vae2d"):
        if not Path(protocol["receiver"]["models"][key]).is_dir():
            raise FileNotFoundError(protocol["receiver"]["models"][key])
    for path in (SNAPSHOT / "config.json", REPO / ".local/checkpoints/ntscc_hyperprior_quality_4_psnr.pth",
                 REPO / ".local/checkpoints/unimatch.pth", REPO / ".local/qwen35_training_probe_deps/peft/__init__.py"):
        if not path.is_file():
            raise FileNotFoundError(path)
    for arm in ARMS:
        collision.preview(protocol["keys"][arm], collision.upstream_functions(REPO))
    receipt = root / "protocol.json"
    if receipt.exists():
        if ae.read(receipt) != protocol:
            raise ValueError("frozen inputs/code changed; use a new --root, preserve the previous run")
    elif root.exists() and any(root.iterdir()):
        raise ValueError("unrecognized nonempty output; choose a new --root")
    return protocol


def load_protocol(root):
    value = ae.read(root / "protocol.json")
    identity = dict(value)
    signature = identity.pop("signature")
    if ae.digest(identity) != signature:
        raise ValueError("comparison protocol signature changed")
    for path, digest in value["inputs"].items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"frozen input changed: {path}")
    for path, digest in value["code"].items():
        if sha256(REPO / path) != digest:
            raise ValueError(f"frozen code changed: {path}")
    for path, entry in value["base_model_files"].items():
        stat = Path(path).stat()
        if (stat.st_size, stat.st_mtime_ns) != (entry["size"], entry["mtime_ns"]):
            raise ValueError(f"base model file changed: {path}")
    verify_adapter(value["caption"]["caption_adapter"])
    return value


def infer(vlm, path, identity, images, labels, prompt, validator, tokens=384):
    """Validate raw output/role even on resume; preserve every failed attempt."""
    identity = dict(identity, prompt_sha256=ae.digest(prompt), labels=labels, max_new_tokens=tokens)
    if path.exists():
        record = ae.unseal(path)
        if record["identity"] != identity:
            raise ValueError("inference cache identity changed")
        verify_record(vlm.config, record)
        runtime = record["model_runtime"]
        if (runtime.get("model_id"), runtime.get("revision"), runtime.get("quantization")) != (MODEL_ID, REVISION, "nf4"):
            raise ValueError("cached caption model provenance changed")
        if record["truncated"] or validator(ae.parse_json(record["raw_text"])) != record["result"]:
            raise ValueError("cached result differs from raw inference")
        return record["result"]
    for attempt in range(3):
        actual_prompt = prompt if attempt == 0 else prompt + "\nReturn only the exact JSON schema; be concise."
        response = vlm.generate(images, labels, actual_prompt, tokens + attempt*192, task=identity["task"])
        record = dict(identity=identity, actual_prompt=actual_prompt, created_utc=ae.now(), **response)
        try:
            if response["truncated"]:
                raise ValueError("generation reached token limit")
            verify_record(vlm.config, record)
            runtime = record["model_runtime"]
            if (runtime.get("model_id"), runtime.get("revision"), runtime.get("quantization")) != (MODEL_ID, REVISION, "nf4"):
                raise ValueError("inference model provenance differs")
            result = validator(ae.parse_json(response["raw_text"]))
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            record["error"] = str(exc)
            write_json(path.parent / "failed_attempts" / f"{path.stem}_{time.time_ns()}.json", ae.seal(record))
            continue
        write_json(path, ae.seal(dict(record, result=result)))
        return result
    raise RuntimeError(f"three invalid responses; successful calls are resumable: {path}")


def caption_worker(root):
    from PIL import Image
    from .qwen35_backend import LocalQwen35
    p = load_protocol(root)
    vlm = LocalQwen35(p["caption"])
    rows = {arm: [] for arm in ARMS}
    # Alternate arms at each segment; the model, prompts, budgets and roles match.
    for n in range(78):
        for arm in (ARMS if n % 2 == 0 else ARMS[::-1]):
            a, b = p["keys"][arm][n:n+2]
            ids = sample_indices(a, b)
            paths = [ae.check_frame(p["source"], i) for i in ids]
            identity = dict(protocol=p["signature"], arm=arm, segment=n, start=a, end_exclusive=b,
                source_indices=ids, frame_hashes=[sha256(path) for path in paths])
            labels = [f"Source frame {i}, time {i/24:.6f} seconds" for i in ids]
            images = [Image.open(path).convert("RGB") for path in paths]
            try:
                draft = infer(vlm, root / f"calls/{arm}/draft/{n:05d}.json", dict(identity, task="draft"),
                    images, labels, p["caption"]["prompts"]["caption"], ae.caption_validator(p["caption"]))
                review = infer(vlm, root / f"calls/{arm}/review/{n:05d}.json",
                    dict(identity, task="visual_review", draft_sha256=ae.digest(draft)), images, labels,
                    p["caption"]["prompts"]["review"] + draft["caption"], ae.caption_validator(p["caption"], True))
            finally:
                for im in images:
                    im.close()
            rows[arm].append(dict(segment=n, start=a, end_exclusive=b, source_indices=ids, **review))
            print(f"CAPTION {arm} {n+1}/78", flush=True)
    for arm in ARMS:
        write_json(root / f"captions/{arm}.json", ae.seal(dict(protocol=p["signature"], arm=arm,
            author=MODEL_ID, model_revision=REVISION, caption_adapter=p["caption"]["caption_adapter"],
            independent_verification=False, captions=rows[arm])))


def validate_bundle(root, arm, p):
    bundle = ae.unseal(root / f"captions/{arm}.json")
    if (bundle["protocol"] != p["signature"] or bundle["arm"] != arm or bundle["author"] != MODEL_ID
            or bundle["model_revision"] != REVISION or bundle["caption_adapter"] != p["caption"]["caption_adapter"]
            or len(bundle["captions"]) != len(p["keys"][arm])-1):
        raise ValueError("caption bundle identity/count differs")
    for n, (row, a, b) in enumerate(zip(bundle["captions"], p["keys"][arm], p["keys"][arm][1:])):
        ids = sample_indices(a, b)
        if (row["segment"], row["start"], row["end_exclusive"], row["source_indices"]) != (n, a, b, ids):
            raise ValueError("caption timeline differs")
        ae.caption_validator(p["caption"], True)(row)
        for role, task in (("draft", "draft"), ("review", "visual_review")):
            call = ae.unseal(root / f"calls/{arm}/{role}/{n:05d}.json")
            verify_record(p["caption"], call)
            runtime = call["model_runtime"]
            if (runtime.get("model_id"), runtime.get("revision"), runtime.get("quantization")) != (MODEL_ID, REVISION, "nf4"):
                raise ValueError("caption call model provenance differs")
            expected = dict(protocol=p["signature"], arm=arm, segment=n, start=a, end_exclusive=b,
                source_indices=ids, frame_hashes=[p["source"]["source_frame_hashes"][str(i)] for i in ids])
            if any(call["identity"].get(k) != v for k, v in expected.items()) or call["identity"]["task"] != task:
                raise ValueError("caption call source/role differs")
            if call["truncated"] or ae.parse_json(call["raw_text"]) != call["result"]:
                raise ValueError("caption call raw output differs")
            if role == "draft":
                draft = call["result"]
                prompt = p["caption"]["prompts"]["caption"]
            elif (call["identity"]["draft_sha256"] != ae.digest(draft)
                  or any(row[k] != call["result"][k] for k in ("caption", "uncertainties", "removed_claims"))):
                raise ValueError("caption review does not match bundle/draft")
            else:
                prompt = p["caption"]["prompts"]["review"] + draft["caption"]
            if (call["identity"]["prompt_sha256"] != ae.digest(prompt)
                    or call["identity"]["max_new_tokens"] != 384):
                raise ValueError("caption prompt or token budget differs")
        for i in ids:
            ae.check_frame(p["source"], i)
    return bundle


def import_captions(root, arm):
    p = load_protocol(root)
    bundle = validate_bundle(root, arm, p)
    run = root / f"reconstruction/{arm}/run"
    if ae.read(run / "keyframes.json")["indices"] != p["keys"][arm]:
        raise ValueError("receiver run keyframes differ")
    audit = ae.read(run / "semantic_clips_audit.json")
    if (audit["status"] != "PASSED" or audit["policy"] != "frame_exact"
            or audit["source"]["keyframes"] != p["keys"][arm]
            or sha256(run / "data/normalized.mp4") != p["source"]["normalized_sha256"]):
        raise ValueError("caption import requires matching source and exact clips")
    rows, samples = [], []
    for item in bundle["captions"]:
        clip = f"clips/sample/{item['segment']:05d}.mp4"
        rows.append(dict(path=clip, text=item["caption"], flow=0.0))
        samples.append(dict(clip=str(run / "data" / clip), decoded_frames=item["end_exclusive"]-item["start"],
            indices=[i-item["start"] for i in item["source_indices"]], source_indices=item["source_indices"],
            caption_provider="local_qwen35_caption_repair"))
    write_json(run / "captions.json", rows)
    write_json(run / "caption_sampling.json", samples)
    write_json(run / "caption_provenance.json", dict(provider="local_qwen35_caption_repair", model_id=MODEL_ID,
        model_revision=REVISION, quantization="nf4", caption_adapter=p["caption"]["caption_adapter"],
        bundle_sha256=sha256(root / f"captions/{arm}.json"), selector=LABELS[arm],
        assistant_authored=False, pllava_inference_executed=False, independent_verification=False))


def frame_noise(index, count, seed, snr_db):
    import numpy as np
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    sigma = math.sqrt(1 / (2 * 10 ** (snr_db / 10)))
    return (rng.normal(0, sigma, count) + 1j*rng.normal(0, sigma, count)).astype("<c8")


def paired_channel(run):
    import numpy as np
    from .metadata_channel import transmit
    from .transmission_accounting import channel_breakdown
    from .wire import unpack
    cfg = ae.read(run / "run_config.json")
    packet = (run / "transmitter/metadata.bin").read_bytes()
    restored, report = transmit(packet, cfg["snr_db"], cfg["channel_seed"])
    if restored != packet or report["bit_errors"]:
        raise ValueError("digital metadata did not survive AWGN exactly")
    header, _ = unpack(restored)
    sent = np.fromfile(run / "transmitter/visual.c64", dtype="<c8")
    if len(sent) != sum(k["complex_count"] for k in header["keyframes"]) or not np.isfinite(sent).all():
        raise ValueError("invalid visual transmission")
    values, cursor = sent.copy(), 0
    for item in header["keyframes"]:
        a, n = item["complex_offset"], item["complex_count"]
        if a != cursor:
            raise ValueError("noncontiguous visual transmission")
        values[a:a+n] += frame_noise(item["index"], n, cfg["channel_seed"], cfg["snr_db"])
        cursor += n
    output = run / "received"
    output.mkdir()
    (output / "metadata.bin").write_bytes(restored)
    values.tofile(output / "visual.c64")
    digital = report["complex_channel_uses"]
    report.update(status="PASSED", metadata_exact_match=True, visual_complex_channel_uses=len(sent),
        digital_complex_channel_uses=digital, total_complex_channel_uses=len(sent)+digital,
        cbr_complex_uses_per_source_scalar=(len(sent)+digital)/(3*cfg["width"]*cfg["height"]*cfg["frames"]),
        complete_sample_dependent_model_input_accounting=True, physical_link_overhead_included=False,
        shared_prior="installed models, code, fixed protocol", visual_awgn_rng=cfg["visual_channel"],
        channel_seed=cfg["channel_seed"], visual_tx_mean_power=float(np.mean(np.abs(sent)**2)),
        transmission_breakdown=channel_breakdown(packet, len(sent), header["video"], report),
        received_files={f.name: dict(bytes=f.stat().st_size, sha256=sha256(f)) for f in output.iterdir()})
    write_json(run / "channel_accounting.json", report)


def output_audit(root, arm):
    run = root / f"reconstruction/{arm}/run"
    # Use all existing receiver audits, then correct the historical selector label.
    fc.audit(run)
    report = ae.read(run / "output_audit.json")
    report.update(automatic_keyframe_insertion=False, automatic_keyframe_insertion_stage=None,
                  frozen_selection_reused=True, selector=LABELS[arm], caption_provider="local_qwen35_caption_repair")
    write_json(run / "output_audit.json", report)


def reconstruction_jobs():
    for stage, module, products in fc.job_plan():
        if stage == "comparison":
            continue  # One three-column comparison is made after both arms.
        yield stage, MODULE if module == fc.MODULE or stage == "channel" else module, products


def verify_model_bytes(root, p):
    for path, entry in p["base_model_files"].items():
        if sha256(Path(path)) != entry["sha256"]:
            raise ValueError(f"base model weight hash differs: {path}")
    write_json(root / "model_bytes.json", dict(status="PASS", files=p["base_model_files"]))


def reconstruct_arm(root, arm, p):
    dest = root / f"reconstruction/{arm}"
    cfg = dict(p["receiver"], selector=LABELS[arm], experiment_root=str(root), experiment_arm=arm)
    identity = dict(signature=p["signature"], arm=arm, config=cfg,
        condition_plan=collision.preview(p["keys"][arm], collision.upstream_functions(REPO)))
    receipt = dest / "execution_protocol.json"
    if receipt.exists() and ae.read(receipt) != identity:
        raise ValueError("reconstruction receipt changed")
    write_json(receipt, identity)
    stages = Stages(dest, p["signature"] + ":" + arm)
    stages.step("initialize", ["run"], initialization_products(p["source"]["frames"]),
        lambda _: initialize(dest, p["source"], dict(indices=p["keys"][arm], selector=LABELS[arm]), cfg))
    env = dict(environment(cfg["seed"]), PYTHONPATH=str(REPO / "src"), ETRI_RUN_SIGNATURE=p["signature"])
    for stage, module, products in reconstruction_jobs():
        python = p["interpreters"]["channel_python" if stage == "channel" else "python"]
        if module == MODULE:
            args = ["--worker", stage, "--root", str(root), "--arm", arm]
        elif module == fc.ETRI:
            args = ["--worker", stage, "--run-dir", str(dest / "run")]
        else:
            args = [stage, str(dest / "run")]
        required = [*[f"run/{v}" for v in products], f"resources/{stage}.json"]
        worker_env = dict(env, CUDA_VISIBLE_DEVICES="-1") if stage == "channel" else env
        stages.step(stage, required, required,
            lambda log, cmd=[python, "-m", module, *args], e=worker_env, s=stage:
                launch(cmd, log, e, dest / f"resources/{s}.json"))


def execute(root, p, stop_after):
    root.mkdir(parents=True, exist_ok=True)
    with lock(REPO / ".local/etri_60s_check.lock"), lock(root / ".execution.lock"):
        write_json(root / "protocol.json", p)
        status = dict(status="RUNNING", signature=p["signature"], training_executed=False)
        write_json(root / "status.json", status)
        try:
            stages = Stages(root, p["signature"])
            stages.step("model-bytes", ["model_bytes.json"], ["model_bytes.json"], lambda _: verify_model_bytes(root, p))
            env = dict(environment(2025), PYTHONPATH=str(REPO / "src"))
            def worker(stage, products, owned=None):
                required = [*products, f"resources/{stage}.json"]
                stages.step(stage, [*(owned if owned is not None else products), f"resources/{stage}.json"], required,
                    lambda log: launch([p["interpreters"]["qwen"], "-m", MODULE,
                        "--worker", stage, "--root", str(root)], log, env, root / f"resources/{stage}.json"))
            worker("captions", ["captions", "calls"], owned=["captions"])
            for arm in ARMS:
                validate_bundle(root, arm, p)
            if stop_after != "captions":
                for arm in ARMS:
                    reconstruct_arm(root, arm, p)
                if stop_after != "reconstruct":
                    from .qwen35_pairs_review import semantic_inputs
                    current = semantic_inputs(root, p)
                    saved = root / "semantic/inputs.json"
                    if saved.exists() and ae.read(saved) != current:
                        raise ValueError("semantic review inputs changed; do not reuse the previous judgments")
                    write_json(saved, current)
                    worker("semantic-review", ["semantic/RESULT.json", "semantic/inputs.json", "semantic/calls", "semantic/frames"],
                           owned=["semantic/RESULT.json"])
            # Report is cheap and reflects the actual completed scope on every resume.
            from .qwen35_pairs_review import report
            report(root, p)
            status.update(status="COMPLETE_INDEPENDENT_REVIEW_PENDING" if stop_after is None else "PARTIAL",
                          through=stop_after or "semantic-review", report=str(root / "report.html"))
        except BaseException as exc:
            status.update(status="STOPPED", error=str(exc))
            raise
        finally:
            write_json(root / "status.json", status)
        print(json.dumps(status, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check", action="store_true", help="read-only; no inference or output creation")
    parser.add_argument("--stop-after", choices=("captions", "reconstruct"))
    parser.add_argument("--worker", choices=("captions", "caption", "channel", "prepare-text", "reconstruct", "output-audit", "semantic-review"), help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=ARMS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))
    if args.worker:
        if args.check or args.stop_after:
            parser.error("worker cannot use --check/--stop-after")
        load_protocol(root)
        if args.worker == "captions":
            return caption_worker(root)
        if args.worker == "semantic-review":
            from .qwen35_pairs_review import review
            return review(root)
        if args.arm is None:
            parser.error("reconstruction worker requires --arm")
        run = root / f"reconstruction/{args.arm}/run"
        actions = {"caption": lambda: import_captions(root, args.arm), "channel": lambda: paired_channel(run),
                   "prepare-text": lambda: fc.prepare_text(run), "reconstruct": lambda: fc.reconstruct(run),
                   "output-audit": lambda: output_audit(root, args.arm)}
        return actions[args.worker]()
    p = preflight(root)
    if args.check:
        print(json.dumps(dict(status="READY_FOR_USER_EXECUTION", root=str(root), frames=p["source"]["frames"],
            keys={k: len(v) for k, v in p["keys"].items()}, adapter=p["caption"]["caption_adapter"]["id"],
            model_inference_started=False, training_executed=False, scope=p["scope"],
            semantic_review=p["semantic_review"]), ensure_ascii=False, indent=2))
        return
    execute(root, p, args.stop_after)


if __name__ == "__main__":
    main()
