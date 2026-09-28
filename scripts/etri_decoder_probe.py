"""Trace real cross-segment conditioning without changing decoder calculations."""
import os
from pathlib import Path
import runpy
import sys

from semantic_transmission.artifacts import write_json
from opensora.utils import inference_utils

original = inference_utils.append_generated
records = []


def traced(vae, generated_video, refs_x, mask_strategy, loop_i, length, edit):
    before = [len(refs or []) for refs in refs_x]
    result = original(vae, generated_video, refs_x, mask_strategy, loop_i, length, edit)
    records.append({"loop": loop_i, "generated_reference_shape": list(generated_video.shape),
                    "references_before": before, "references_after": [len(x) for x in result[0]],
                    "new_latent_shapes": [list(x[-1].shape) for x in result[0]],
                    "mask_strategy": list(result[1])})
    write_json(os.environ["ETRI_REFERENCE_TRACE"], records)
    return result


inference_utils.append_generated = traced
decoder = Path(__file__).resolve().parents[1] / "04_semantic_decoder/scripts/mydemo_new_align_sh.py"
sys.path.insert(0, str(decoder.parent))
sys.argv[0] = str(decoder)
runpy.run_path(str(decoder), run_name="__main__")
