"""Sequential, restartable FSO/EOI/UEP follow-up validation campaign.

This controller uses only the standard library. Workers select the existing
NumPy 2 visual and NumPy 1 pixel/LGVSC environments in separate processes.
Execution completion and scientific acceptance are separate states.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import time

from .artifacts import sha256

REPO = Path(__file__).resolve().parents[2]
PHASES = ("01_cross", "02_components", "03_natural")


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def resolve(path):
    return (REPO / path).resolve()


def python_path(path):
    # CPython finds pyvenv.cfg relative to the invoked executable path. Resolving
    # a venv's bin/python symlink instead launches its base interpreter and drops
    # the packages installed in the venv (including SAM2 in our evaluation env).
    return Path(os.path.abspath(REPO / Path(path).expanduser()))


def validate_config(config):
    if config.get("schema") != "metric-validation-campaign-v1":
        raise ValueError("Unsupported campaign schema")
    for key in ("synthetic_sources_per_family", "rx_sources_per_family", "natural_sources",
                "natural_frames", "natural_stride", "reconstruction_sources", "bootstrap_iterations",
                "minimum_positive_sources"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError("positive integer required: " + key)
    if not 16 <= config["natural_frames"] <= 128:
        raise ValueError("natural_frames must be 16..128 at the declared 8 Hz playback")
    if config["reconstruction_sources"] > config["natural_sources"]:
        raise ValueError("reconstruction_sources exceeds natural_sources")
    styles = config["styles"]
    if (len(styles) != len(set(styles)) or "identity" not in styles
            or not set(styles) <= {"identity", "brightness20", "texture", "camera", "palette"}):
        raise ValueError("invalid styles; include identity once")
    steps = config["reconstruction_steps"]
    if not steps or len(steps) != len(set(steps)) or any(type(x) is not int or x < 1 for x in steps):
        raise ValueError("reconstruction_steps must contain distinct positive integers")
    for key in ("synthetic_seed", "rx_seed", "natural_seed"):
        if type(config.get(key)) is not int or config[key] < 0:
            raise ValueError("invalid seed: " + key)
    if config["synthetic_sources_per_family"] > 100 or config["rx_sources_per_family"] > 100:
        raise ValueError("per-family count must not exceed the seed spacing of 100")
    a = {config["synthetic_seed"] + f * 100 + k for f in range(3)
         for k in range(config["synthetic_sources_per_family"])}
    b = {config["rx_seed"] + f * 100 + k for f in range(3)
         for k in range(config["rx_sources_per_family"])}
    if a & b:
        raise ValueError("cross and RX experiment seeds overlap")


def prior_sources():
    """Use recorded inventories, never candidate scores, to exclude seen videos."""
    used = set()
    for path in sorted((REPO / "outputs").glob("*/protocol.json")):
        data = read(path)
        for row in data.get("inventory", []):
            if row.get("source_id", "").startswith("davis/"):
                used.add(row["source_id"].split("/", 1)[1])
    for path in sorted((REPO / "outputs").glob("*/campaign.json")):
        for row in read(path).get("natural_inventory", []):
            used.add(row["source_id"].split("/", 1)[1])
    return used


def select_natural(config):
    root = resolve(config["davis_root"])
    used = prior_sources()
    candidates = []
    for directory in sorted((root / "JPEGImages/480p").iterdir()):
        if directory.name in used or not directory.is_dir():
            continue
        images = sorted(directory.glob("*.jpg"))[::config["natural_stride"]][:config["natural_frames"]]
        labels = [root / "Annotations/480p" / directory.name / (p.stem + ".png") for p in images]
        if len(images) == config["natural_frames"] and all(p.is_file() for p in labels):
            candidates.append({"source_id": "davis/" + directory.name,
                               "images": [str(p.resolve()) for p in images],
                               "labels": [str(p.resolve()) for p in labels]})
    candidates.sort(key=lambda r: digest([config["natural_seed"], r["source_id"]]))
    if len(candidates) < config["natural_sources"]:
        raise ValueError(f"Need {config['natural_sources']} unused annotated DAVIS sequences; found {len(candidates)}")
    return candidates[:config["natural_sources"]]


def estimate(config):
    base = resolve(config["base_run"])
    seconds_per_case = ((read(base / "completed_visual.json")["completed_unix"]
                         - read(base / "runtime_visual.json")["started_unix"]) / 768)
    evidence = read(REPO / "docs/validation/2026-09-11-etri01-official-full.json")
    model_seconds_per_second = evidence["wall_seconds"] / evidence["source"]["duration"]
    cross_cases = 3 * config["synthetic_sources_per_family"] * 15 * len(config["styles"])
    rx_cases = 3 * config["rx_sources_per_family"] * 3 * 5
    # Natural controls + 12 temporal transforms + two additions, each restyled.
    natural_cases = config["natural_sources"] * 15 * len(config["styles"])
    model_pairs = config["reconstruction_sources"] * len(config["reconstruction_steps"])
    # The official timing is a single video, not a universal throughput model.
    # Broader bands cover mask density, longer cache I/O and per-model loading.
    cross_h = cross_cases * seconds_per_case / 3600
    component_h = rx_cases * seconds_per_case / 3600 + .4
    model_h = model_pairs * config["natural_frames"] / 8 * model_seconds_per_second / 3600
    natural_h = natural_cases * seconds_per_case / 3600
    bands = {"01_cross": [round(cross_h * 1.3 + .3, 2), round(cross_h * 2.5 + .8, 2)],
             "02_components": [round(component_h, 2), round(component_h * 2.5 + .5, 2)],
             "03_natural": [round(model_h * .8 + natural_h * 1.5 + .5, 2),
                             round(model_h * 1.5 + natural_h * 3 + 1., 2)]}
    return {"hours": bands, "total_hours": [round(sum(v[i] for v in bands.values()), 2) for i in (0, 1)],
            "case_budget": {"cross": cross_cases, "rx": rx_cases, "natural_upper_bound": natural_cases},
            "LGVSC_runs": model_pairs, "seconds_per_LGVSC_input": config["natural_frames"] / 8,
            "observed_visual_seconds_per_case": seconds_per_case,
            "observed_LGVSC_seconds_per_input_second": model_seconds_per_second,
            "scope": "estimated command wall time on RTX 4080; excludes independent annotation creation, "
                     "new model development, novelty research, GPU contention and reruns after scientific failure"}


def tasks(config):
    result = []
    for phase in PHASES:
        operations = ["prepare", "visual", "pixel", "report"]
        if phase == "01_cross":
            operations.insert(1, "calibrate")
        if phase == "02_components":
            operations = ["prepare", "visual", "report"]
        if phase == "03_natural":
            operations = ["prepare", "visual", "pixel", "reconstruct", "real_visual", "real_pixel", "report"]
        result.extend((phase, op) for op in operations)
    return result


def file_inventory(paths):
    return {str(p): sha256(p) for p in sorted(set(Path(p).resolve() for p in paths))}


def fingerprint_files(config, natural):
    paths = list((REPO / "src/semantic_transmission").glob("*.py"))
    paths += [REPO / "scripts/run_metric_validation.sh", REPO / "docs/METRIC_VALIDATION_CAMPAIGN.md",
              resolve(config["reconstruction_profile"]), REPO / ".local/settings.json",
              REPO / ".local/model_paths.json"]
    base = resolve(config["base_run"])
    paths += [base / name for name in ("protocol.json", "calibration.json", "development_normals.json")]
    paths += [REPO / 'outputs/ere_sta_formal_20260914_v1/protocol.json',
              REPO / 'outputs/ere_sta_20260914_v2/protocol.json']
    for row in natural:
        paths.extend(Path(p) for p in row["images"] + row["labels"])
    if config.get("reconstruction_truth"):
        paths.append(resolve(config["reconstruction_truth"]))
    return file_inventory(paths)


def runtime_inventory(config):
    """Cheap restart guard for large local model files and environment metadata.

    Content hashes protect campaign inputs/code/results. Model checkpoints use
    resolved paths, size, mtime and ctime here; this is not a full weight audit.
    """
    local = read(REPO / '.local/settings.json')
    paths = []
    for key in ('python', 'channel_python', 'internvl_python'):
        python = Path(local[key])
        if not python.is_file():
            raise FileNotFoundError(python)
        paths += [python, *python.parent.parent.glob('lib/python*/site-packages/*.dist-info/METADATA')]
    visual = python_path(config['evaluation_python'])
    # Record both the venv overlay and its inherited base packages.
    for interpreter in (visual, visual.resolve()):
        paths += [interpreter, *interpreter.parent.parent.glob('lib/python*/site-packages/*.dist-info/METADATA')]
    venv_config = visual.parent.parent / 'pyvenv.cfg'
    if venv_config.is_file():
        paths.append(venv_config)
    for folder in read(REPO / '.local/model_paths.json').values():
        model = Path(folder)
        if not model.is_dir():
            raise FileNotFoundError(model)
        paths += [p for p in model.rglob('*') if p.is_file()]
    paths += [p for p in (REPO / '.local/checkpoints').iterdir() if p.is_file()]
    paths += [REPO / '.local/metric_v2_models/sam2.1_hiera_tiny.pt',
              REPO / '.local/metric_v2_models/dinov2_vits14_pretrain.pth',
              Path.home() / '.cache/torch/hub/checkpoints/raft_small_C_T_V2-01064c6d.pth',
              Path.home() / '.cache/clip/ViT-B-32.pt']
    for vendor in ('Open-Sora', 'NTSCC_JSAC22', 'InternVL', 'PLLaVA'):
        directory = REPO / '.local/vendor' / vendor
        paths += list(directory.rglob('*.py'))
    for vendor in ('sam2', 'dinov2'):
        paths += list((REPO / '.local/metric_v2_models' / vendor).rglob('*.py'))
    return {str(p): {'resolved': str(p.resolve()), 'size': p.stat().st_size,
                     'mtime_ns': p.stat().st_mtime_ns, 'ctime_ns': p.stat().st_ctime_ns}
            for p in sorted(set(paths))}


def probe_evaluation_python(config, runner=subprocess.run):
    command = [str(python_path(config['evaluation_python'])), '-c',
               'import json,sys,numpy,torch,cv2,scipy; '
               'from sam2.build_sam import build_sam2; '
               'from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator; '
               'print(json.dumps(dict(executable=sys.executable,prefix=sys.prefix,'
               'numpy=numpy.__version__,torch=torch.__version__)))']
    result = runner(command, cwd=REPO, env=worker_env(config), capture_output=True, text=True, timeout=45)
    if result.returncode:
        raise RuntimeError('Evaluation Python dependency check failed: ' + command[0] + '\n' + result.stderr[-2000:])
    return json.loads(result.stdout.strip().splitlines()[-1])


def preflight(config):
    if shutil.which('ffmpeg') is None or shutil.which('ffprobe') is None:
        raise FileNotFoundError('ffmpeg and ffprobe required')
    if not resolve(config['driver_library']).is_dir():
        raise FileNotFoundError('Configured driver library directory unavailable')
    profile = read(resolve(config['reconstruction_profile']))
    if (tuple(profile[k] for k in ('width', 'height', 'fps')) != (576, 320, 24)
            or not profile.get('variable_length') or not profile.get('preserve_input')
            or profile.get('official_preprocessing') or profile.get('decoder_policy') != 'official_release'):
        raise ValueError('Campaign source export requires the variable 576x320@24 official-release profile')
    from .cli import doctor
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        result = doctor(REPO)
    if result['status'] != 'PASSED':
        raise RuntimeError('LGVSC preflight failed: ' + str(result['checks']))
    probe_evaluation_python(config)
    return runtime_inventory(config)


def check_files(inventory):
    for path, expected in inventory.items():
        if not Path(path).is_file() or sha256(Path(path)) != expected:
            raise ValueError("Frozen input or code changed: " + path)


@contextmanager
def campaign_lock(root):
    with (root / ".campaign.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process is running this campaign") from exc
        yield


def worker_command(config, root, phase, operation):
    local = read(REPO / ".local/settings.json")
    visual = operation in ("visual", "calibrate", "real_visual")
    python = python_path(config["evaluation_python"]) if visual else Path(local["python"])
    return [str(python), "-m", "semantic_transmission.metric_campaign_worker", operation,
            "--output", str(root), "--phase", phase]


def worker_env(config):
    env = os.environ.copy()
    env.update(PYTHONPATH=str(REPO / "src"), PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1",
               PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
    env["LD_LIBRARY_PATH"] = str(resolve(config["driver_library"]))
    return env


def receipt(root, phase, operation):
    return root / phase / "receipts" / (operation + ".json")


def verify_receipt(path):
    data = read(path)
    if data.get("execution_status") != "COMPLETED" or not data.get("artifacts"):
        raise ValueError("Incomplete task receipt: " + str(path))
    check_files(data["artifacts"])
    return data


def execute(root, config, runner=subprocess.run):
    state_path = root / "state.json"
    state = read(state_path) if state_path.exists() else {"tasks": {}, "started_unix": time.time()}
    for phase, operation in tasks(config):
        key = phase + "/" + operation
        done = receipt(root, phase, operation)
        if done.exists():
            verify_receipt(done)
            state["tasks"].setdefault(key, {})["status"] = "COMPLETED"
            print("재사용:", key, flush=True)
            continue
        entry = state["tasks"].setdefault(key, {})
        entry.update(status="RUNNING", started_unix=time.time())
        state.update(execution_status="RUNNING", current_task=key)
        save(state_path, state)
        log = root / "logs" / (phase + "_" + operation + ".log")
        log.parent.mkdir(exist_ok=True)
        print("실행:", key, "—", log, flush=True)
        try:
            with log.open("a") as stream:
                result = runner(worker_command(config, root, phase, operation), cwd=REPO,
                                env=worker_env(config), stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"{key} exited {result.returncode}; see {log}")
            verify_receipt(done)
        except BaseException as error:
            entry.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED",
                         error=str(error), finished_unix=time.time())
            state["execution_status"] = entry["status"]
            save(state_path, state)
            raise
        entry.update(status="COMPLETED", finished_unix=time.time())
        entry["elapsed_s"] = entry["finished_unix"] - entry["started_unix"]
        save(state_path, state)
    reports = {phase: read(root / phase / "report.json") for phase in PHASES}
    result = {"execution_status": "COMPLETED", "scientific_status": "REVIEW_REQUIRED",
              "novelty_established": False, "phases": reports,
              "elapsed_hours": (time.time() - state["started_unix"]) / 3600,
              "scope": "all three evaluations executed; acceptance is per metric and evidence domain"}
    save(root / "summary.json", result)
    state.update(execution_status="COMPLETED", current_task=None, finished_unix=time.time())
    save(state_path, state)
    print("세 단계 실행 완료:", root / "summary.json", flush=True)
    print("최종 연구 판정은 report.json의 지표별 결과·정답 가용 범위를 확인하세요.", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/metric_validation.json")
    parser.add_argument("--output", type=Path, default=REPO / "outputs/metric_validation_20260916_v1")
    parser.add_argument("--dry-run", action="store_true", help="preflight and time estimate; no GPU work or output creation")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--evaluate-truth", type=Path,
                        help="assess completed generated pairs with separate independent annotations; preserve original reports")
    args = parser.parse_args(argv)
    root = args.output.resolve()
    if args.status:
        print(json.dumps(read(root / "state.json"), ensure_ascii=False, indent=2))
        return
    if args.evaluate_truth:
        protocol = read(root / 'campaign.json')
        check_files(protocol['files'])
        for phase, operation in tasks(protocol['config']):
            verify_receipt(receipt(root, phase, operation))
        local = read(REPO / '.local/settings.json')
        with campaign_lock(root):
            subprocess.run([local['python'], '-m', 'semantic_transmission.metric_campaign_worker',
                            'evaluate_truth', '--output', str(root), '--phase', '03_natural',
                            '--truth', str(args.evaluate_truth.resolve())],
                           cwd=REPO, env=worker_env(protocol['config']), check=True)
        return
    config = read(args.config)
    validate_config(config)
    runtime = preflight(config)
    frozen = root / "campaign.json"
    if frozen.exists():
        protocol = read(frozen)
        if protocol["config"] != config:
            raise ValueError("Campaign config changed; use a new output directory")
        check_files(protocol["files"])
        if protocol['runtime'] != runtime:
            raise ValueError('Model files or Python environment changed; use a new output directory')
        natural = protocol["natural_inventory"]
    else:
        natural = select_natural(config)
    for phase, op in tasks(config):
        if not Path(worker_command(config, root, phase, op)[0]).is_file():
            raise FileNotFoundError("worker Python unavailable")
    timing = estimate(config)
    print(json.dumps({"output": str(root), "estimate": timing,
                      "natural_sources": [r["source_id"] for r in natural],
                      "tasks": [p + "/" + o for p, o in tasks(config)]}, ensure_ascii=False, indent=2), flush=True)
    inventory = fingerprint_files(config, natural)
    if args.dry_run:
        return
    if not frozen.exists() and root.exists() and any(root.iterdir()):
        raise FileExistsError("Output exists without a campaign declaration: " + str(root))
    root.mkdir(parents=True, exist_ok=True)
    with campaign_lock(root):
        if not frozen.exists():
            save(frozen, {"schema": config["schema"], "created_unix": time.time(),
                          "config": config, "files": inventory, "natural_inventory": natural,
                          "estimate": timing, "runtime": runtime, "heldout_scores_observed": 0})
        elif read(frozen)['config'] != config or read(frozen)['files'] != inventory or read(frozen)['runtime'] != runtime:
            raise ValueError('Campaign declaration changed while acquiring lock')
        execute(root, config)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(str(exc))
