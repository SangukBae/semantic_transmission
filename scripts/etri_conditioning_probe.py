"""Observe initial noise and applied mask positions without changing decoding."""
import hashlib
import os
from pathlib import Path
import runpy
import sys

import torch
from opensora.utils import inference_utils as utils
from semantic_transmission.artifacts import write_json


def tensor_hash(value):
    return hashlib.sha256(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


original_apply = utils.apply_mask_strategy
original_append = utils.append_generated
records = {"mask": [], "references": []}


def save():
    write_json(os.environ["ETRI_CONDITION_TRACE"], records)


def apply(z, refs, strategies, loop_i, align=None):
    noise_hash = tensor_hash(z)
    active = []
    for batch, strategy in enumerate(strategies):
        for loop, reference, ref_start, target_start, length, edit in utils.parse_mask_strategy(strategy):
            if loop != loop_i:
                continue
            ref = refs[batch][reference]
            requested_ref = ref.shape[1] + ref_start if ref_start < 0 else ref_start
            requested_target = z.shape[2] + target_start if target_start < 0 else target_start
            actual_ref = utils.find_nearest_point(requested_ref, align, ref.shape[1]) if align else requested_ref
            actual_target = utils.find_nearest_point(requested_target, align, z.shape[2]) if align else requested_target
            actual_length = min(length, z.shape[2] - actual_target, ref.shape[1] - actual_ref)
            active.append({"reference": reference, "reference_shape": list(ref.shape),
                "reference_sha256": tensor_hash(ref), "requested_ref_start": requested_ref,
                "actual_ref_start": actual_ref, "requested_target_start": requested_target,
                "actual_target_start": actual_target, "length": actual_length, "edit": edit})
    mask = original_apply(z, refs, strategies, loop_i, align=align)
    records["mask"].append({"loop": loop_i, "align": align, "noise_shape": list(z.shape),
        "noise_sha256": noise_hash, "active": active, "mask": mask.cpu().tolist() if mask is not None else None})
    save()
    return mask


def append(vae, video, refs, strategies, loop_i, length, edit):
    result = original_append(vae, video, refs, strategies, loop_i, length, edit)
    records["references"].append({"loop": loop_i, "pixel_shape": list(video.shape),
                                  "latent_shape": list(result[0][0][-1].shape)})
    save()
    return result


utils.apply_mask_strategy = apply
utils.append_generated = append
decoder = Path(__file__).resolve().parents[1] / "04_semantic_decoder/scripts/mydemo_new_align_sh.py"
sys.path.insert(0, str(decoder.parent))
sys.argv[0] = str(decoder)
runpy.run_path(str(decoder), run_name="__main__")
