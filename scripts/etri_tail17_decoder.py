"""Use one current 17-frame reference block; retain the frozen decoder otherwise."""
import os
from pathlib import Path
import runpy

from semantic_transmission.artifacts import write_json
from semantic_transmission.tail_reference import append_tail17


def main():
    from opensora.utils import inference_utils
    original = inference_utils.append_generated
    trace = Path(os.environ["ETRI_TAIL_REFERENCE_TRACE"])
    records = []

    def append(vae, video, refs, strategies, loop_i, length, edit):
        result, record = append_tail17(original, vae, video, refs, strategies, loop_i, length, edit)
        records.append(record)
        write_json(trace, records)
        return result

    inference_utils.append_generated = append
    # The existing probe records the normal cross-segment trace for output audits.
    runpy.run_path(str(Path(__file__).resolve().parent / "etri_decoder_probe.py"), run_name="__main__")


if __name__ == "__main__":
    main()
