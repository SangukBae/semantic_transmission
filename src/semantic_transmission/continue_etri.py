"""One-command local continuation, with completed videos copied into a fresh batch."""
import argparse
import datetime
import json
from pathlib import Path
import uuid

from .artifacts import write_json
from .cli import repository
from .research import main as research


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    repo = repository()
    path = repo / ".local/etri_continue.json"
    if not path.is_file():
        raise FileNotFoundError("local continuation config missing: .local/etri_continue.json")
    config = json.loads(path.read_text())
    profile = repo / "configs/etri_official.json"
    profile_name = json.loads(profile.read_text())["profile"]
    history = config.setdefault("run_history_by_profile", {}).setdefault(profile_name, [])
    roots = [Path(p) for p in history]
    tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    output = repo / "outputs" / ("etri10_official_" + tag)
    command = ["--input-dir", config["input_dir"], "--output", str(output), "--profile", str(profile)]
    for previous in reversed(roots):
        command += ["--reuse-completed-from", str(previous)]
    if args.dry_run:
        command += ["--dry-run"]
    else:
        history.append(str(output))
        write_json(path, config)
    print(f"결과 폴더: {output}", flush=True)
    print(f"실행 설정: {profile_name} (576×320, 24 fps, SKEM+DSA)", flush=True)
    print("완료된 영상은 검증 후 재사용하며, 미완료 영상은 처음부터 생성합니다.", flush=True)
    research(command)
    if not args.dry_run:
        print(f"ETRI 영상 10개가 모두 준비됐습니다: {output}", flush=True)


if __name__ == "__main__":
    main()
