"""One-command, resumable four-condition WebVid development comparison."""
import argparse
import contextlib
import datetime
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from .ablation_transport import densify, subdivide_metadata, preserve_transport
from .artifacts import sha256, write_json
from .cli import doctor, repository, settings
from .input_contract import video_config
from .video_io import probe
from .webvid5 import CATEGORIES, execution_identity, fingerprint, read_json

CASES = ("baseline", "aligned", "clean_keys", "dense_1s")
VERSION = 1


def load_selection(repo, category="single_subject"):
    cohort = read_json(repo / "configs/webvid5_manifest.json")
    matches = [v for v in cohort["videos"] if v["category"] == category]
    if len(matches) != 1:
        raise ValueError("select exactly one frozen WebVid category")
    selected = matches[0]
    name = selected["filename"]
    if Path(name).name != name or Path(name).suffix != ".mp4":
        raise ValueError("invalid video filename")
    profile = repo / cohort["profile"]
    if sha256(profile) != cohort["profile_sha256"]:
        raise ValueError("frozen WebVid profile hash changed")
    dataset = Path(read_json(repo / ".local/datasets.json")["webvid55"])
    for kind in ("raw", "processed"):
        if sha256(dataset / kind / name) != selected[f"{kind}_sha256"]:
            raise ValueError(f"{kind} source hash mismatch: {name}")
    source = (dataset / "processed" / name).resolve(strict=True)
    info = probe(source)
    if any(info[k] != selected[k] for k in ("frames", "fps", "width", "height")):
        raise ValueError("selected video dimensions/length changed")
    cfg = video_config(read_json(profile), info)
    cfg.update(input=str(source), models=read_json(repo / ".local/model_paths.json"))
    return selected, cfg


def environment(seed):
    env = os.environ.copy()
    for key in ("PYTHONPATH", "LD_LIBRARY_PATH"):
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONHASHSEED=str(seed), OMP_NUM_THREADS="8", TOKENIZERS_PARALLELISM="false",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return env


@contextlib.contextmanager
def exclusive_lock(path):
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("이미 WebVid 1편 비교 명령이 실행 중입니다") from None
        yield


def snapshot(root, paths):
    result = {}
    for relative in paths:
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(path)
        files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        if not files:
            raise ValueError(f"empty stage artifact: {path}")
        result.update({str(p.relative_to(root)): sha256(p) for p in files})
    return result


class Stages:
    """Archive partial output, reuse only hash-verified completed dependencies."""
    def __init__(self, root, identity):
        self.root, self.identity = root, identity
        self.completed = {}

    def step(self, name, owned, required, callback):
        receipt = self.root / "stages" / f"{name}.json"
        dependencies = {k: fingerprint(v) for k, v in self.completed.items()}
        if receipt.exists():
            saved = read_json(receipt)
            if saved.get("status") == "PASSED":
                if saved["identity"] != self.identity or saved["dependencies"] != dependencies:
                    raise ValueError(f"stage identity/dependencies changed: {name}")
                if snapshot(self.root, saved["required"]) != saved["artifacts"]:
                    raise ValueError(f"completed stage artifacts changed: {name}")
                self.completed[name] = saved
                print(f"REUSE {name} (완료 파일 검증됨)", flush=True)
                return
        # A failed/interrupted stage may have produced partial output. Move it
        # out of the worker's way; never accept it as completed or overwrite it.
        log = f"logs/{name}.log"
        leftovers = [p for p in [*owned, log, f"stages/{name}.json"]
                     if (self.root / p).exists() or (self.root / p).is_symlink()]
        if leftovers:
            archive = self.root / "failed_attempts" / f"{name}_{time.time_ns()}"
            for relative in leftovers:
                source = self.root / relative
                if source.exists() or source.is_symlink():
                    target = archive / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(source), str(target))
        entry = dict(status="RUNNING", identity=self.identity, dependencies=dependencies,
                     started=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write_json(receipt, entry)
        start = time.monotonic()
        print(f"START {name} | 로그: {self.root / log}", flush=True)
        try:
            callback(self.root / log)
            paths = required() if callable(required) else required
            entry.update(required=paths, artifacts=snapshot(self.root, paths), status="PASSED")
        except BaseException as error:
            entry.update(status="FAILED", error=str(error))
            raise
        finally:
            entry["seconds"] = time.monotonic() - start
            write_json(receipt, entry)
        self.completed[name] = entry
        print(f"DONE {name}: {entry['seconds'] / 60:.1f}분", flush=True)


def command(repo, args, log, env):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x") as stream:
        process = subprocess.Popen(list(map(str, args)), cwd=repo, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        start = time.monotonic()
        try:
            while True:
                try:
                    code = process.wait(timeout=45)
                    break
                except subprocess.TimeoutExpired:
                    print(f"  진행 중 {log.stem}: {(time.monotonic() - start) / 60:.1f}분 | {log}", flush=True)
            if code:
                raise RuntimeError(f"exit {code}; 로그 확인: {log}")
        except BaseException:
            # Also stop selector/decoder grandchildren, so resume cannot overlap
            # a model process left behind by Ctrl+C or a terminated worker.
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise


def initialize_variant(root, case):
    base, run = root / "baseline", root / case
    cfg = read_json(base / "run_config.json")
    cfg["concatenation_policy"] = "endpoint_exact"
    indices = read_json(base / "keyframes.json")["indices"]
    dense = densify(indices, round(cfg["fps"])) if case == "dense_1s" else indices
    run.mkdir()
    (run / "data").symlink_to(base / "data", target_is_directory=True)
    write_json(run / "run_config.json", cfg)
    write_json(run / "keyframes.json", {"indices": dense, "selector": "SKEM_preserved_max_gap_1s" if case == "dense_1s" else "reused_SKEM"})
    rows = subdivide_metadata(indices, dense, read_json(base / "metadata_tx.json"))
    write_json(run / "metadata_tx.json", rows)
    if case != "dense_1s":
        shutil.copytree(base / "receiver/frames", run / "receiver/frames")
        shutil.copyfile(base / "receiver/metadata.csv", run / "receiver/metadata.csv")
        inputs = read_json(base / "receiver/decoder_inputs.json")
        inputs["decoder"]["concatenation_policy"] = "endpoint_exact"
        write_json(run / "receiver/decoder_inputs.json", inputs)
        if case == "clean_keys":
            for i in indices:
                shutil.copyfile(base / f"data/frames/sample/{i}.png", run / f"receiver/frames/sample/key_frames_received/{i}.png")
    write_json(run / "experiment.json", {
        "case": case, "baseline": str(base), "training": False,
        "clean_keyframe_oracle": case == "clean_keys", "manual_caption": False,
        "conditioning_alignment": "endpoint_exact", "indices": dense,
        "metadata": "automatic baseline caption and flow repeated within each original interval",
        "channel_claim": "none: source-key oracle bypasses visual transmission" if case == "clean_keys" else "actual transmission; baseline visual blocks retained",
    })


def worker(stage, run):
    """Small child-process adapter; never retain GPU models in the driver."""
    repo, cfg = repository(), read_json(run / "run_config.json")
    base = run.parent / "baseline"
    if stage in {"send", "channel", "receive"}:
        from . import codec_transport
        getattr(codec_transport, stage)(cfg, repo, run)
        if stage in {"send", "channel"}:
            preserve_transport(run, base, received=stage == "channel")
        else:
            for i in read_json(base / "keyframes.json")["indices"]:
                shutil.copyfile(base / f"receiver/frames/sample/key_frames_received/{i}.png",
                                run / f"receiver/frames/sample/key_frames_received/{i}.png")
    elif stage == "reconstruct":
        from .codec_transport import decoder_config_text
        from .decoder_runner import run as decode
        inputs = read_json(run / "receiver/decoder_inputs.json")
        alignment = "official_release" if run.name == "baseline" else "endpoint_exact"
        config = decoder_config_text(cfg, repo, inputs)
        config += f"\nconditioning_alignment={alignment!r}\ncache_text_embeddings={run.name == 'dense_1s'!r}\n"
        path = run / "receiver/decoder_config.py"
        path.write_text(config)
        env = dict(os.environ, PYTHONPATH=str(repo / ".local/vendor/Open-Sora"))
        decode(run / "receiver/metadata.csv", run / "receiver/reconstruction", "key_frames_received", run / "receiver",
               decoder=repo / "04_semantic_decoder/scripts/mydemo_new_align_sh.py", config=path, environment=env)
    elif stage == "report":
        from .webvid_ablation_report import build
        build(run.parent)
    else:
        raise ValueError(stage)


def execute(repo, root, cfg, signature):
    local, env = settings(repo), environment(cfg["seed"])
    pipeline = Stages(root, signature)
    base = root / "baseline"
    def launch(module, stage, run, log):
        python = local["channel_python"] if stage == "channel" else local["python"]
        child_env = dict(env, CUDA_VISIBLE_DEVICES="-1") if stage == "channel" else env
        args = [python, "-m", module]
        args += ["--worker", stage, "--run-dir", str(run)] if module.endswith("webvid_ablation") else [stage, str(run)]
        command(repo, args, log, child_env)
    def step(case, stage, owned, required, module):
        pipeline.step(f"{case}.{stage}", [f"{case}/{p}" for p in owned],
                      lambda: [f"{case}/{p}" for p in (required() if callable(required) else required)],
                      lambda log: launch(module, stage, root / case, log))
    workers = "semantic_transmission.workers"
    transport = "semantic_transmission.codec_transport"
    adapter = "semantic_transmission.webvid_ablation"
    pipeline.step("baseline.config", ["baseline/run_config.json"], ["baseline/run_config.json"],
                  lambda log: write_json(base / "run_config.json", cfg))
    step("baseline", "prepare", ["data", "prepare.json"],
         ["prepare.json", "data/normalized.mp4", "data/16x24/videos.csv", "data/frames/sample/frames.csv",
          *[f"data/frames/sample/{i}.png" for i in range(cfg["frames"])]], workers)
    step("baseline", "select", [f"data/frames/sample/{cfg['method']}", "keyframes.json", "selector_runtime.json"],
         ["keyframes.json", f"data/frames/sample/{cfg['method']}", "selector_runtime.json"], workers)
    step("baseline", "caption", ["captions.json", "caption_sampling.json", "data/clips"],
         ["captions.json", "caption_sampling.json", "data/clips"], workers)
    step("baseline", "flow", ["metadata_tx.json", "flow_sampling.json"], ["metadata_tx.json", "flow_sampling.json"], workers)
    transport_outputs = {
        "send": ["transmitter", "sender_accounting.json"],
        "channel": ["received", "channel_accounting.json"],
        "receive": ["receiver/frames", "receiver/metadata.csv", "receiver/decoder_inputs.json", "receiver_accounting.json"],
    }
    for stage, paths in transport_outputs.items():
        step("baseline", stage, paths, paths, transport)
    for case in CASES:
        if case != "baseline":
            required = [f"{case}/{p}" for p in ("run_config.json", "keyframes.json", "metadata_tx.json", "experiment.json")]
            if case != "dense_1s":
                required += [f"{case}/receiver/{p}" for p in ("frames", "metadata.csv", "decoder_inputs.json")]
            pipeline.step(f"{case}.inputs", [case], required, lambda log, c=case: initialize_variant(root, c))
            if case == "dense_1s":
                for stage, paths in transport_outputs.items():
                    step(case, stage, paths, paths, adapter)
        paths = ["receiver/decoder_config.py", "receiver/reconstruction"]
        step(case, "reconstruct", paths, paths, adapter)
        dest = root / "evaluations" / case
        pipeline.step(f"{case}.evaluate", [f"evaluations/{case}"], [f"evaluations/{case}"],
                      lambda log, c=case, d=dest: command(repo, [local["python"], "-m", "semantic_transmission.research_quality",
                          "evaluate", root / c, "--output-dir", d, "--evaluation-profile", "lgvsc_official_metrics_v1",
                          "--compare-concatenation"], log, env))
    outputs = ["review_media", "comparison_four_panel.mp4", "comparison_oracle.mp4", "comparison.html",
               "REPORT.md", "comparison_summary.json", "AUDIT.json"]
    pipeline.step("report", outputs, outputs, lambda log: launch(adapter, "report", base, log))
    write_json(root / "COMPLETE.json", {"status": "PASSED", "identity": signature,
               "manual_semantic_review": "PENDING", "stages": list(pipeline.completed)})


def run(repo, *, category="single_subject", output=None, dry_run=False):
    selected, cfg = load_selection(repo, category)
    if doctor(repo)["status"] != "PASSED":
        raise RuntimeError("environment/model preflight failed")
    identity = {"version": VERSION, "selection": selected, "config": cfg, "cases": CASES,
                "execution": execution_identity(repo, cfg),
                "decoder_template_sha256": sha256(repo / "configs/official_opensora.py")}
    signature = fingerprint(identity)
    root = (output or repo / "outputs" / f"webvid1_ablation_{category}_{signature[:12]}").resolve()
    protocol = root / "protocol.json"
    if root.exists():
        if not protocol.exists() or read_json(protocol).get("signature") != signature:
            raise ValueError(f"output belongs to another protocol or is unrecognized; use a new --output: {root}")
    print(f"WebVid 1편: {selected['filename']}\n{cfg['frames']}프레임 / {cfg['frames']/cfg['fps']:.2f}초 / "
          f"{cfg['width']}×{cfg['height']} / {cfg['fps']}fps / seed={cfg['seed']} / {cfg['steps']} steps\n"
          "조건: 기존 설정 / 정렬 보정 / 정렬+원본 키프레임(진단) / 정렬+최대 1초 간격\n"
          f"첫 실행 예상 8~10시간(기본 영상 기준). 완료 단계 재사용. 결과: {root}", flush=True)
    if dry_run:
        if protocol.exists():
            for receipt in sorted((root / "stages").glob("*.json")):
                record = read_json(receipt)
                if record.get("status") == "PASSED" and snapshot(root, record["required"]) != record["artifacts"]:
                    raise ValueError(f"completed artifact changed: {receipt}")
        print("DRY_RUN_PASSED: 모델 추론·결과 파일 생성 없음", flush=True)
        return root
    root.mkdir(parents=True, exist_ok=True)
    if not protocol.exists():
        write_json(protocol, {"signature": signature, **identity, "scope": "DEVELOPMENT_PILOT",
                             "selection_rule": "preselected category, never selected by reconstruction quality"})
    state = dict(status="RUNNING", signature=signature, started=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write_json(root / "status.json", state)
    try:
        execute(repo, root, cfg, signature)
        state["status"] = "PASSED"
    except BaseException as error:
        state.update(status="FAILED", error=str(error))
        # Never leave an earlier success marker after a failed validation/resume.
        (root / "COMPLETE.json").unlink(missing_ok=True)
        raise
    finally:
        state["finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(root / "status.json", state)
    print(f"완료: {root / 'comparison.html'}\n보고서: {root / 'REPORT.md'}", flush=True)
    return root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", choices=sorted(CATEGORIES), default="single_subject")
    parser.add_argument("--output", type=Path, help="optional dedicated result folder; same command resumes it")
    parser.add_argument("--dry-run", action="store_true", help="validate selected source, models and existing artifacts without inference/writes")
    parser.add_argument("--worker", choices=["send", "channel", "receive", "reconstruct", "report"], help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.worker:
            if args.run_dir is None:
                raise ValueError("worker requires --run-dir")
            worker(args.worker, args.run_dir.resolve())
            return
        repo = repository()
        if args.dry_run:
            run(repo, category=args.category, output=args.output, dry_run=True)
        else:
            # SIGTERM follows the same cleanup path as Ctrl+C.
            def terminate(signum, frame):
                raise KeyboardInterrupt("terminated")
            signal.signal(signal.SIGTERM, terminate)
            with exclusive_lock(repo / ".local/webvid_ablation.lock"):
                run(repo, category=args.category, output=args.output)
    except KeyboardInterrupt:
        parser.exit(130, "중단됨. 같은 명령으로 완료 단계 이후부터 재실행할 수 있습니다.\n")
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
