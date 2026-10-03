"""Install the collision guard before existing scoped-noise/FP32/tail17 hooks."""
import os
from pathlib import Path
import runpy
from types import SimpleNamespace

from semantic_transmission import condition_collision as fix
from semantic_transmission.artifacts import write_json
from semantic_transmission.webvid5 import read_json


def main():
    from opensora.utils import inference_utils
    run = Path(os.environ["ETRI_PRECISION_RUN"])
    if read_json(run / "receiver_policy.json").get("condition_collision_policy") != fix.POLICY:
        raise ValueError("collision repair policy missing")
    count = len(read_json(run / "keyframes.json")["indices"])
    original = SimpleNamespace(**{n: getattr(inference_utils, n) for n in
        ("apply_mask_strategy", "parse_mask_strategy", "find_nearest_point")})
    records = []

    def apply(z, refs, strategies, loop_i, align=None):
        mask, record = fix.apply_guard(z, refs, strategies, loop_i, align, count, original)
        records.append(record)
        write_json(run / fix.TRACE, records)
        return mask

    inference_utils.apply_mask_strategy = apply
    runpy.run_path(str(Path(__file__).with_name("etri_precision_decoder.py")), run_name="__main__")
    fix.validate_trace(records, fix.preview(read_json(run / "keyframes.json")["indices"], original))


if __name__ == "__main__":
    main()
