"""One command for correction, budgeted prototypes, and frozen held-out evaluation."""
import argparse
import csv
import fcntl
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

from .artifacts import sha256, write_json
from .cli import repository, settings
from .webvid5 import fingerprint, read_json, execution_identity
from .webvid_ablation import environment

LABELS = ("키프레임 기준 오차 보정", "저해상도 제약·적응형 키프레임 비교", "별도 영상 5편·3개 시드")
VERSION = 1


class ProgressLine:
    """Exactly one active terminal line, one completed sentence per validation."""
    def __init__(self, index, label, stream=None):
        self.index, self.label = index, label
        self.stream = stream or sys.stdout
        self.last = -1
        self.tty = self.stream.isatty()

    def update(self, fraction):
        value = min(99.9, max(self.last, math.floor(fraction*1000)/10))
        if value == self.last:
            return
        self.last = value
        if self.tty:
            self.stream.write(f"\r\033[2K[{self.index}/3] {self.label} 진행률 {value:5.1f}%")
            self.stream.flush()

    def finish(self, failed=False):
        prefix = "\r\033[2K" if self.tty else ""
        sentence = (f"검증이 중단되었습니다 ({max(self.last, 0):.1f}%)." if failed
                    else "검증이 끝났습니다 (100.0%).")
        self.stream.write(f"{prefix}[{self.index}/3] {self.label} {sentence}\n")
        self.stream.flush()


def inventory(paths):
    """Hash explicitly owned products, including symlink targets when needed."""
    values = {}
    for item in paths:
        path = Path(item)
        if not path.exists():
            raise FileNotFoundError(path)
        files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        if not files:
            raise ValueError(f"empty required output: {path}")
        for file in files:
            values[str(file)] = sha256(file)
    return values


def task(name, group, spec, owned, required=None):
    return dict(name=name, group=group, spec=spec, owned=list(map(str, owned)),
                required=list(map(str, required if required is not None else owned)))


def pipeline_tasks(name, group, run, stages, config=None):
    products = {
        "prepare": ["run_config.json", "data", "prepare.json"],
        "select": ["keyframes.json", "selector_runtime.json", "data/frames/sample/key_framesinternvl_diff_0.35"],
        "caption": ["captions.json", "caption_sampling.json", "data/clips"],
        "flow": ["metadata_tx.json", "flow_sampling.json"],
        "send": ["transmitter", "sender_accounting.json"],
        "channel": ["received", "channel_accounting.json"],
        "receive": ["receiver/frames", "receiver/metadata.csv", "receiver/decoder_inputs.json", "receiver_accounting.json"],
        "reconstruct": ["receiver/decoder_config.py", "receiver/reconstruction"],
    }
    result = []
    for stage in stages:
        spec = dict(action="pipeline", stage=stage, run=str(run))
        if config is not None:
            spec["config"] = config
        required = products[stage]
        if stage == "prepare":
            # Subsequent caption/selector stages add files under data; hash only
            # immutable prepared frames, manifest and normalized video instead.
            required = ["run_config.json", "data/normalized.mp4", "data/16x24/videos.csv", "prepare.json"]
            if config is not None:
                required += [f"data/frames/sample/{i}.png" for i in range(config["frames"])]
                required += ["data/frames/sample/frames.csv"]
        result.append(task(f"{name}.{stage}", group, spec,
                           [run/p for p in products[stage]], [run/p for p in required]))
    return result


def build_plan(root, cfg, repo):
    result = []
    dev = cfg["development"]
    frozen = root / "correction/development/frozen.json"
    first = root / "correction/development"
    result.append(task("correction.development", 1, dict(action="correction", base=dev[0]["base"],
        dest=str(first), strengths=cfg["anchor_strengths"], freeze=True,
        lowres_strength=cfg["lowres_strength"]), [first]))
    second = root / "correction/validation"
    result.append(task("correction.validation", 1, dict(action="correction", base=dev[1]["base"],
        dest=str(second), frozen=str(frozen)), [second]))

    def prototypes(identifier, group, base, seeds, with_baseline):
        dest = root / ("prototype" if group == 2 else "heldout") / identifier
        bank, encoded, side, variants = (dest/p for p in ("bank", "side_encoded", "side_received", "variants"))
        result.append(task(f"{identifier}.bank", group, dict(action="make_bank", base=str(base), run=str(bank)),
            [bank], [bank/p for p in ("run_config.json", "keyframes.json", "metadata_tx.json", "rankings.json")]))
        result.extend(pipeline_tasks(f"{identifier}.bank", group, bank, ["send"]))
        result.append(task(f"{identifier}.side.encode", group, dict(action="encode_side", base=str(base),
            dest=str(encoded), lowres_size=cfg["lowres_size"], lowres_crf=cfg["lowres_crf"]), [encoded]))
        result.append(task(f"{identifier}.side.channel", group, dict(action="side_channel", encoded=str(encoded),
            dest=str(side), snr_db=cfg["snr_db"], channel_seed=cfg["channel_seed"]+90000), [side]))
        result.append(task(f"{identifier}.budget", group, dict(action="plan_wires", base=str(base), bank=str(bank),
            dest=str(variants), side=str(side), budget_multiplier=cfg["budget_multiplier"]), [variants],
            [variants/"budget.json"] + [variants/n/p for n in ("uniform", "adaptive", "adaptive_lowres")
             for p in ("run_config.json", "keyframes.json", "transmitter")]))
        for variant in ("uniform", "adaptive", "adaptive_lowres"):
            result.extend(pipeline_tasks(f"{identifier}.{variant}", group, variants/variant, ["channel", "receive"]))
        names = ["uniform", "adaptive", "adaptive_lowres"]
        if with_baseline:
            names = ["baseline", *names]
        for seed in seeds:
            for variant in names:
                run = dest / "seeds" / str(seed) / variant
                origin = base if variant == "baseline" else variants/variant
                result.append(task(f"{identifier}.{seed}.{variant}.inputs", group,
                    dict(action="seed_view", base=str(origin), run=str(run), seed=seed), [run],
                    [run/p for p in ("run_config.json", "receiver/decoder_inputs.json", "seed_protocol.json")]))
                result.extend(pipeline_tasks(f"{identifier}.{seed}.{variant}", group, run, ["reconstruct"]))
                assessment = dest / "evaluations" / str(seed) / variant
                spec = dict(action="assess", run=str(run), dest=str(assessment), frozen=str(frozen))
                if variant == "adaptive_lowres":
                    spec["side"] = str(side)
                result.append(task(f"{identifier}.{seed}.{variant}.evaluate", group, spec, [assessment]))
                if variant == "baseline":
                    corrected = assessment.with_name("corrected")
                    result.append(task(f"{identifier}.{seed}.corrected.evaluate", group,
                        dict(spec, dest=str(corrected), correction=True), [corrected]))
    for entry in dev:
        prototypes(entry["id"], 2, Path(entry["base"]), [cfg["seeds"][0]], False)
    for entry in cfg["heldout"]:
        identifier = entry["id"]
        base = root / "heldout" / identifier / "baseline"
        config = dict(input=entry["source"], seed=cfg["seeds"][0], channel_seed=cfg["channel_seed"],
                      snr_db=cfg["snr_db"], steps=cfg["steps"], frames=entry["frames"])
        result.extend(pipeline_tasks(identifier+".baseline", 3, base,
            ["prepare", "select", "caption", "flow", "send", "channel", "receive"], config))
        prototypes(identifier, 3, base, cfg["seeds"], True)
    return result


def resolve_config(repo, path):
    cfg = read_json(path)
    if (len(cfg["development"]) != 2 or len(cfg["heldout"]) != 5
            or len(cfg["seeds"]) != 3 or len(set(cfg["seeds"])) != 3):
        raise ValueError("protocol requires two development videos, five held-out videos and three distinct seeds")
    if not (0 < cfg["lowres_strength"] <= 1 and cfg["budget_multiplier"] > 1 and cfg["steps"] > 0):
        raise ValueError("invalid correction, budget or step settings")
    if (not cfg["anchor_strengths"] or sorted(set(cfg["anchor_strengths"])) != cfg["anchor_strengths"]
            or cfg["anchor_strengths"][0] != 0 or any(not 0 <= v <= 1 for v in cfg["anchor_strengths"])):
        raise ValueError("include no-op among sorted unique correction strengths")
    if (len(cfg["lowres_size"]) != 2 or any(type(v) is not int or v < 2 or v % 2 for v in cfg["lowres_size"])
            or any(type(v) is not int or v < 0 for v in [*cfg["seeds"], cfg["channel_seed"]])
            or not 0 <= cfg["lowres_crf"] <= 51):
        raise ValueError("invalid low-resolution encoding or random seed settings")
    for entry in cfg["development"] + cfg["heldout"]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", entry["id"]):
            raise ValueError("experiment identifiers must be plain names")
    if any(Path(e["filename"]).name != e["filename"] for e in cfg["heldout"]):
        raise ValueError("held-out filenames must be basenames within the dataset")
    cfg = copy_config(cfg)
    evidence = {}
    hashes = []
    for entry in cfg["development"]:
        base = (repo / entry["base"]).resolve()
        entry["base"] = str(base)
        files = [base/p for p in ("run_config.json", "keyframes.json", "metadata_tx.json", "channel_accounting.json",
                                  "receiver/decoder_inputs.json", "receiver/metadata.csv", "data/normalized.mp4")]
        files += sorted((base / "receiver/frames/sample/key_frames_received").glob("*.png"))
        files += sorted((base / "data/frames/sample").glob("*.png"))
        images = sorted((base / "receiver/reconstruction/sample_0000_frames").glob("*.png"))
        if not images:
            raise ValueError(f"missing completed development reconstruction: {base}")
        files += images
        evidence.update(inventory(files))
        hashes.append(sha256(base / "data/normalized.mp4"))
    datasets = read_json(repo / ".local/datasets.json")
    from .video_io import probe
    for entry in cfg["heldout"]:
        source = (Path(datasets["webvid55"]) / "processed" / entry["filename"]).resolve(strict=True)
        digest = sha256(source)
        if digest != entry["sha256"]:
            raise ValueError(f"held-out source hash changed: {source}")
        info = probe(source)
        if any(info[k] != entry[k] for k in ("frames", "width", "height", "fps")):
            raise ValueError(f"held-out video geometry changed: {source}")
        if digest in hashes:
            raise ValueError("development/held-out sources overlap")
        hashes.append(digest)
        entry["source"] = str(source)
        evidence[str(source)] = digest
    if len(set(e["id"] for e in cfg["development"]+cfg["heldout"])) != 7:
        raise ValueError("duplicate experiment identifiers")
    local = settings(repo)
    for key in ("python", "internvl_python", "channel_python"):
        if not Path(local[key]).is_file():
            raise FileNotFoundError(local[key])
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg/ffprobe is unavailable")
    models = read_json(repo / ".local/model_paths.json")
    for path in models.values():
        if not Path(path).is_dir():
            raise FileNotFoundError(path)
    return cfg, evidence


def copy_config(value):
    return json.loads(json.dumps(value))


def launch(spec, spec_path, log_path, ipc, repo, local, update):
    write_json(spec_path, spec)
    ipc.unlink(missing_ok=True)
    env = environment(42)
    env["QUALITY_PROGRESS_FILE"] = str(ipc)
    channel = spec["action"] == "side_channel" or (spec["action"] == "pipeline" and spec["stage"] == "channel")
    executable = local["channel_python"] if channel else local["python"]
    if channel:
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    args = [executable, "-m", "semantic_transmission.quality_validation_worker", str(spec_path)]
    with log_path.open("w") as stream:
        process = subprocess.Popen(args, cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            while process.poll() is None:
                if ipc.exists():
                    state = read_json(ipc)
                    if state["total"] > 0:
                        update(min(.999, max(0, state["done"]/state["total"])))
                time.sleep(.25)
            if process.returncode:
                raise RuntimeError(f"worker exit {process.returncode}; 상세 로그: {log_path}")
        except BaseException:
            # Grandchildren inherit the process group, including InternVL and Open-Sora.
            try:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            except ProcessLookupError:
                pass
            raise


class Runner:
    def __init__(self, root, identity, repo, local):
        self.root, self.identity, self.repo, self.local = root, identity, repo, local
        self.chain = fingerprint(identity)

    def execute(self, job, update):
        folder = self.root / "tasks" / job["name"]
        folder.mkdir(parents=True, exist_ok=True)
        receipt = folder / "receipt.json"
        signature = fingerprint({"job": job, "previous": self.chain})
        if receipt.exists():
            old = read_json(receipt)
            if old.get("status") == "PASSED":
                if old["signature"] != signature or inventory(job["required"]) != old["artifacts"]:
                    raise ValueError(f"완료 파일/의존성이 변경되었습니다: {job['name']}")
                self.chain = fingerprint(old)
                update(1)
                return
        leftovers = [Path(p) for p in job["owned"] if Path(p).exists() or Path(p).is_symlink()]
        if leftovers:
            archive = self.root / "failed_attempts" / f"{job['name']}_{time.time_ns()}"
            archive.mkdir(parents=True)
            for i, path in enumerate(leftovers):
                # An earlier prepare step can legitimately create the parent.
                # Ownership lists always point to only this task's outputs.
                shutil.move(str(path), str(archive / f"{i}_{path.name}"))
        entry = {"status": "RUNNING", "signature": signature, "started": time.time()}
        write_json(receipt, entry)
        start = time.monotonic()
        try:
            launch(job["spec"], folder/"spec.json", folder/"worker.log", folder/"progress.json",
                   self.repo, self.local, update)
            entry.update(status="PASSED", artifacts=inventory(job["required"]))
        except BaseException as error:
            entry.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED", error=str(error))
            raise
        finally:
            entry["seconds"] = time.monotonic()-start
            write_json(receipt, entry)
        self.chain = fingerprint(entry)
        update(1)


def build_report(root, cfg, completed):
    """Report execution separately from quality; never infer hallucination safety."""
    records = []
    for path in sorted(root.glob("**/metrics.json")):
        if "failed_attempts" in path.parts:
            continue
        metric = read_json(path)
        relative = path.relative_to(root)
        records.append({"case": str(relative.parent), **metric})
    comparisons = []
    for directory in sorted(root.glob("heldout/*/evaluations/*")):
        baseline = directory/"baseline/metrics.json"
        if not baseline.exists():
            continue
        reference = read_json(baseline)
        control = directory/"uniform/metrics.json"
        uniform = read_json(control) if control.exists() else None
        for variant in ("corrected", "uniform", "adaptive", "adaptive_lowres"):
            p = directory/variant/"metrics.json"
            if not p.exists():
                continue
            current = read_json(p)
            base = uniform if variant in {"adaptive", "adaptive_lowres"} else reference
            target = "uniform" if variant in {"adaptive", "adaptive_lowres"} else "baseline"
            if base is None:
                continue
            delta = {k: current["all"][k]-base["all"][k] for k in current["all"]}
            comparisons.append(dict(video=directory.parent.parent.name, seed=directory.name, variant=variant,
                reference=target, delta=delta, temporal_error_delta=current["temporal_error"]-base["temporal_error"],
                guard_pass=(delta.get("lpips_vgg", 1) < -.002 and delta["psnr_db"] >= -.1
                    and current["temporal_error"] <= base["temporal_error"]+.002)))
    aggregates = {}
    for name in ("corrected", "uniform", "adaptive", "adaptive_lowres"):
        values = [r for r in comparisons if r["variant"] == name]
        if values:
            videos = sorted({r["video"] for r in values})
            # Aggregate paired seeds within video before averaging videos.
            mean = {k: sum(sum(r["delta"][k] for r in values if r["video"] == v)/
                         sum(r["video"] == v for r in values) for v in videos)/len(videos)
                    for k in values[0]["delta"]}
            aggregates[name] = dict(videos=len(videos), paired_runs=len(values), mean_delta=mean,
                guards_passed=sum(r["guard_pass"] for r in values),
                finding="PRELIMINARY_GAIN" if len(values)==15 and all(r["guard_pass"] for r in values)
                    else "IMPROVEMENT_NOT_ESTABLISHED")
    summary = dict(execution="COMPLETED", completed_validations=completed, metrics=records,
                   paired_comparisons=comparisons, aggregates=aggregates,
                   hallucination_review="PENDING", significance_claimed=False,
                   seeds_are_not_independent_videos=True,
                   limitations=["five held-out videos are a small fixed cohort",
                     "identical budget cap does not imply identical actual channel uses",
                     "low-resolution constraint is postprocessing, not a trained diffusion conditioner"])
    write_json(root/"summary.json", summary)
    rows = ["# 복원 품질 검증 결과", "", f"완료한 검증: {completed}", "",
            "완료는 실험 실행 완료를 뜻합니다. 품질 개선 및 할루시네이션 없음의 보장은 아닙니다.", "",
            "| 조건 | PSNR | SSIM | LPIPS | 시간차 오차 |", "|---|---:|---:|---:|---:|"]
    for record in records:
        m = record["all"]
        rows.append(f"| {record['case']} | {m['psnr_db']:.4f} | {m['ssim']:.4f} | {m.get('lpips_vgg', float('nan')):.4f} | {record['temporal_error']:.5f} |")
    rows += ["", "## 같은 전송 예산 비교", ""]
    for path in sorted(root.glob("*/**/variants/budget.json")):
        if "failed_attempts" not in path.parts:
            data = read_json(path)
            rows.append(f"- {path.relative_to(root)}: 상한 {data['budget']} complex channel uses; " +
                        ", ".join(f"{n}={v['total_complex_channel_uses']}" for n,v in data['variants'].items()))
    rows += ["", "## 별도 영상 판정", "", "```json", json.dumps(aggregates, ensure_ascii=False, indent=2), "```", "",
             "원본/복원 동기 비교는 comparison.html을 사용합니다. 물체·행동 추가/누락 검수는 PENDING입니다.",
             "키프레임 시점과 중간 프레임 지표는 각 metrics.json과 per_frame.csv에서 확인합니다."]
    (root/"REPORT.md").write_text("\n".join(rows)+"\n")
    source_lookup = {e["id"]: str(Path(e["base"])/"data/normalized.mp4") for e in cfg["development"]}
    source_lookup.update({e["id"]: e["source"] for e in cfg["heldout"]})
    body = ['<!doctype html><html lang="ko"><meta charset="utf-8"><title>복원 품질 비교</title>',
            '<style>body{font:16px sans-serif;margin:2rem}video{width:45%}section{margin:2rem 0}</style>',
            '<h1>원본과 복원 비교</h1><p>자동 평가 완료와 별개로 의미적 오류 검수는 미완료입니다.</p>']
    review = []
    for record in records:
        parts = Path(record["case"]).parts
        if parts[0] == "correction":
            entry = cfg["development"][0 if parts[1] == "development" else 1]
            source = source_lookup[entry["id"]]
        else:
            source = source_lookup[parts[1]]
        video = root / record["case"] / "video.mp4"
        src = os.path.relpath(source, root)
        body.append(f'<section><h3>{html.escape(record["case"])}</h3><video controls preload="none" src="{html.escape(src)}"></video>'
                    f'<video controls preload="none" src="{html.escape(str(video.relative_to(root)))}"></video>'
                    '<button onclick="const v=this.parentElement.querySelectorAll(\'video\');v[1].currentTime=v[0].currentTime;v.forEach(x=>x.play())">동기 재생</button></section>')
        review.append(dict(case=record["case"], status="PENDING", added_objects="", missing_events="", timing_errors="", notes=""))
    (root/"comparison.html").write_text("\n".join(body)+"</html>")
    review_path = root/"manual_review.csv"
    existing = {}
    if review_path.exists():
        with review_path.open() as f:
            existing = {row["case"]: row for row in csv.DictReader(f)}
    for row in review:
        existing.setdefault(row["case"], row)
    with review_path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "status", "added_objects", "missing_events", "timing_errors", "notes"])
        writer.writeheader(); writer.writerows(existing.values())


def run(repo, args):
    cfg, evidence = resolve_config(repo, args.config)
    identity = {"version": VERSION, "config": cfg, "input_sha256": evidence,
                "execution": execution_identity(repo, {"models": read_json(repo/".local/model_paths.json")}),
                "profile_sha256": {name: sha256(repo/"configs"/name) for name in
                                   ("webvid5.json", "official_opensora.py", "rtx4080_opensora.py")},
                "config_sha256": sha256(args.config)}
    signature = fingerprint(identity)
    root = (args.output or repo/"outputs"/f"quality_validation_{signature[:12]}").resolve()
    plan = [t for t in build_plan(root, cfg, repo) if t["group"] <= args.through]
    if root.exists():
        if not (root/"protocol.json").exists() or read_json(root/"protocol.json")["signature"] != signature:
            raise ValueError(f"다른 실행 설정의 출력입니다. 새 --output을 지정하세요: {root}")
    if args.dry_run:
        print(f"실행 준비 확인 완료: 검증 {args.through}개, 세부 작업 {len(plan)}개, 별도 영상 5편·3개 시드; 결과 위치 {root}")
        return
    root.mkdir(parents=True, exist_ok=True)
    write_json(root/"protocol.json", dict(identity, signature=signature, selection_frozen_before_results=True))
    write_json(root/"plan.json", plan)
    (root/"COMPLETE.json").unlink(missing_ok=True)
    for group in range(1, args.through+1):
        (root/f"validation_{group}_complete.json").unlink(missing_ok=True)
    runner = Runner(root, signature, repo, settings(repo))
    done_groups = []
    line = None
    try:
        for group in range(1, args.through+1):
            jobs = [j for j in plan if j["group"] == group]
            line = ProgressLine(group, LABELS[group-1])
            line.update(0)
            for index, job in enumerate(jobs):
                def update(fraction, i=index):
                    value = (i + fraction)/len(jobs)
                    line.update(value)
                    write_json(root/"status.json", dict(status="RUNNING", validation=group,
                        task=job["name"], done_tasks=i, tasks=len(jobs), percent=round(value*100, 2),
                        progress_basis="completed tasks plus actual inner work fraction, not elapsed-time estimate"))
                runner.execute(job, update)
            done_groups.append(group)
            build_report(root, cfg, done_groups)
            write_json(root/f"validation_{group}_complete.json", dict(status="EXECUTION_COMPLETE", quality_claim=False))
            line.finish()
            line = None
        write_json(root/"status.json", {"status": "COMPLETED", "validations": done_groups})
        if args.through == 3:
            write_json(root/"COMPLETE.json", {"status": "EXECUTION_COMPLETE", "signature": signature,
                "hallucination_review": "PENDING", "quality_claim": False})
    except BaseException as error:
        if line is not None:
            line.finish(failed=True)
        write_json(root/"status.json", dict(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED",
                                           error=str(error), completed_validations=done_groups))
        raise
    print(f"결과: {root/'REPORT.md'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--through", type=int, choices=[1,2,3], default=3,
                        help="run through validation 1, 2, or 3 (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="read-only source/environment preflight")
    parser.add_argument("--demo", action="store_true", help="terminal rendering demo only; no validation")
    args = parser.parse_args(argv)
    if args.demo:
        print("출력 형식 시연입니다. 실제 검증은 실행하지 않습니다.")
        for i, label in enumerate(LABELS, 1):
            line = ProgressLine(i, label)
            for percent in (0, .125, .5, .875, .999):
                line.update(percent); time.sleep(.15)
            line.finish()
        return
    repo = repository()
    args.config = (args.config or repo/"configs/quality_validation.json").resolve()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        if args.dry_run:
            run(repo, args)
        else:
            lock_path = repo/".local/quality_validation.lock"
            lock_path.parent.mkdir(exist_ok=True)
            with lock_path.open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise RuntimeError("이미 복원 품질 검증 명령이 실행 중입니다") from None
                run(repo, args)
    except KeyboardInterrupt:
        parser.exit(130, "중단되었습니다. 같은 명령으로 완료 작업 이후부터 재개할 수 있습니다.\n")
    except Exception as error:
        parser.exit(1, f"검증 실패: {error}\n")


if __name__ == "__main__":
    main()
