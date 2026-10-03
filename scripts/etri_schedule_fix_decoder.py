"""Apply short-segment scheduling before the frozen collision/noise decoder."""
import os
from pathlib import Path
import runpy

from semantic_transmission import short_schedule
from semantic_transmission.webvid5 import read_json


def main():
    run = Path(os.environ["ETRI_PRECISION_RUN"])
    policy = read_json(run / "receiver_policy.json")
    if policy.get("short_schedule_policy") != short_schedule.POLICY:
        raise ValueError("short-schedule policy missing")
    keys = read_json(run / "keyframes.json")["indices"]
    records = short_schedule.install(keys, run / short_schedule.TRACE)
    runpy.run_path(str(Path(__file__).with_name("etri_condition_fix_decoder.py")), run_name="__main__")
    short_schedule.validate_trace(records, keys, policy["noise_contract"]["steps"])


if __name__ == "__main__":
    main()
