"""Replay a saved baseline while independently controlling rounding and cropping."""
import os
from pathlib import Path
import runpy
import sys

from semantic_transmission.artifacts import write_json
from etri_tail_reference_probe import Replay, tensor_hash


class Hooks:
    def __init__(self, utils, replay, trace, crop_tail):
        self.utils, self.replay, self.trace = utils, replay, Path(trace)
        self.crop_tail = crop_tail
        self.original_apply, self.original_append = utils.apply_mask_strategy, utils.append_generated
        self.records = {"mode": "combined" if crop_tail else "no_rounding", "mask": [], "references": []}

    def save(self):
        write_json(self.trace, self.records)

    def apply(self, z, refs, strategies, loop_i, align=None):
        if align is not None:
            raise ValueError("both new factorial conditions must disable rounding")
        incoming = tensor_hash(z)
        rng = self.replay.synchronize("sampling", loop_i, z.device, z)
        used = tensor_hash(z)
        active = []
        for batch, strategy in enumerate(strategies):
            for loop, reference, ref_start, target_start, length, edit in self.utils.parse_mask_strategy(strategy):
                if loop != loop_i:
                    continue
                ref = refs[batch][reference]
                actual_ref = ref.shape[1] + ref_start if ref_start < 0 else ref_start
                actual_target = z.shape[2] + target_start if target_start < 0 else target_start
                active.append({"reference": reference, "reference_shape": list(ref.shape),
                    "reference_sha256": tensor_hash(ref), "requested_ref_start": actual_ref,
                    "actual_ref_start": actual_ref, "requested_target_start": actual_target,
                    "actual_target_start": actual_target,
                    "length": min(length, z.shape[2]-actual_target, ref.shape[1]-actual_ref), "edit": edit})
        mask = self.original_apply(z, refs, strategies, loop_i, align=None)
        self.records["mask"].append({"loop": loop_i, "align": None, "noise_shape": list(z.shape),
            "incoming_noise_sha256": incoming, "noise_sha256": used, "sampling_rng_sha256": rng,
            "active": active, "mask": mask.cpu().tolist() if mask is not None else None})
        self.save()
        return mask

    def append(self, vae, video, refs, strategies, loop_i, length, edit):
        if length != 5 or video.shape[2] < 17:
            raise ValueError("expected existing five-latent overlap and short-clip padding")
        before = list(video.shape)
        rng = self.replay.synchronize("reference", loop_i, video.device)
        if self.crop_tail:
            video = video[:, :, -17:]
        result = self.original_append(vae, video, refs, strategies, loop_i, length, edit)
        latent = result[0][0][-1]
        if self.crop_tail and latent.shape[1] != 5:
            raise ValueError("17-frame reference did not encode to five latent frames")
        self.records["references"].append({"loop": loop_i, "input_pixel_shape": before,
            "encoded_pixel_shape": list(video.shape), "latent_shape": list(latent.shape),
            "reference_rng_sha256": rng})
        self.save()
        return result


def main():
    from opensora.utils import inference_utils as utils
    case = os.environ["ETRI_FACTORIAL_CASE"]
    if case not in {"no_rounding", "combined"}:
        raise ValueError(f"unknown factorial case: {case}")
    hooks = Hooks(utils, Replay(os.environ["ETRI_REFERENCE_BANK"], "tail17"),
                  os.environ["ETRI_CONDITION_TRACE"], crop_tail=case == "combined")
    utils.apply_mask_strategy, utils.append_generated = hooks.apply, hooks.append
    decoder = Path(__file__).resolve().parents[1] / "04_semantic_decoder/scripts/mydemo_new_align_sh.py"
    sys.path.insert(0, str(decoder.parent))
    sys.argv[0] = str(decoder)
    runpy.run_path(str(decoder), run_name="__main__")


if __name__ == "__main__":
    main()
