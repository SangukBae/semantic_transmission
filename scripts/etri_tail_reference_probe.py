"""Record/replay diffusion noise while changing only the previous reference span."""
import hashlib
import os
from pathlib import Path
import runpy
import sys

import torch

from semantic_transmission.artifacts import write_json


def tensor_hash(value):
    data = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


class Replay:
    def __init__(self, directory, mode):
        if mode not in {"record", "tail17"}:
            raise ValueError(f"invalid reference mode: {mode}")
        self.directory, self.mode = Path(directory), mode
        if mode == "record":
            self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def state(device):
        state = {"cpu": torch.get_rng_state()}
        if device.type == "cuda":
            state["cuda"] = torch.cuda.get_rng_state(device)
        return state

    @staticmethod
    def restore(state, device):
        if (device.type == "cuda") != ("cuda" in state):
            raise ValueError("recorded and replay devices differ")
        torch.set_rng_state(state["cpu"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda"], device)

    def synchronize(self, kind, loop, device, noise=None):
        path = self.directory / f"{kind}_{loop:03d}.pt"
        if self.mode == "record":
            record = {"kind": kind, "loop": loop, "state": self.state(device)}
            if noise is not None:
                record["noise"] = noise.detach().cpu()
            temporary = path.with_suffix(".tmp")
            torch.save(record, temporary)
            os.replace(temporary, path)
        else:
            record = torch.load(path, map_location="cpu", weights_only=True)
            if record["kind"] != kind or record["loop"] != loop:
                raise ValueError("replay record does not match the requested loop")
            if noise is not None:
                expected = record["noise"]
                if noise.shape != expected.shape or noise.dtype != expected.dtype:
                    raise ValueError("replayed noise dimensions/dtype differ")
                noise.copy_(expected.to(device=device))
            self.restore(record["state"], device)
        return {name: tensor_hash(value) for name, value in record["state"].items()}


class Hooks:
    def __init__(self, utils, replay, trace):
        self.utils, self.replay, self.trace = utils, replay, Path(trace)
        self.original_apply, self.original_append = utils.apply_mask_strategy, utils.append_generated
        self.records = {"mode": replay.mode, "mask": [], "references": []}

    def save(self):
        write_json(self.trace, self.records)

    def apply(self, z, refs, strategies, loop_i, align=None):
        if align != 5:
            raise ValueError("tail-reference experiment must retain align=5")
        incoming = tensor_hash(z)
        rng = self.replay.synchronize("sampling", loop_i, z.device, z)
        used = tensor_hash(z)
        active = []
        for batch, strategy in enumerate(strategies):
            for loop, reference, ref_start, target_start, length, edit in self.utils.parse_mask_strategy(strategy):
                if loop != loop_i:
                    continue
                ref = refs[batch][reference]
                requested_ref = ref.shape[1] + ref_start if ref_start < 0 else ref_start
                requested_target = z.shape[2] + target_start if target_start < 0 else target_start
                actual_ref = self.utils.find_nearest_point(requested_ref, align, ref.shape[1])
                actual_target = self.utils.find_nearest_point(requested_target, align, z.shape[2])
                active.append({"reference": reference, "reference_shape": list(ref.shape),
                    "reference_sha256": tensor_hash(ref), "requested_ref_start": requested_ref,
                    "actual_ref_start": actual_ref, "requested_target_start": requested_target,
                    "actual_target_start": actual_target,
                    "length": min(length, z.shape[2]-actual_target, ref.shape[1]-actual_ref), "edit": edit})
        mask = self.original_apply(z, refs, strategies, loop_i, align=align)
        self.records["mask"].append({"loop": loop_i, "align": align, "noise_shape": list(z.shape),
            "incoming_noise_sha256": incoming, "noise_sha256": used,
            "sampling_rng_sha256": rng, "active": active,
            "mask": mask.cpu().tolist() if mask is not None else None})
        self.save()
        return mask

    def append(self, vae, video, refs, strategies, loop_i, length, edit):
        if length != 5 or video.shape[2] < 17:
            raise ValueError("expected the decoder's existing five-latent overlap and short-clip padding")
        before = list(video.shape)
        rng = self.replay.synchronize("reference", loop_i, video.device)
        if self.replay.mode == "tail17":
            video = video[:, :, -17:]
        result = self.original_append(vae, video, refs, strategies, loop_i, length, edit)
        latent = result[0][0][-1]
        if self.replay.mode == "tail17" and latent.shape[1] != 5:
            raise ValueError("17-frame reference did not encode to five latent frames")
        self.records["references"].append({"loop": loop_i, "input_pixel_shape": before,
            "encoded_pixel_shape": list(video.shape), "latent_shape": list(latent.shape),
            "reference_rng_sha256": rng})
        self.save()
        return result


def main():
    from opensora.utils import inference_utils as utils
    replay = Replay(os.environ["ETRI_REFERENCE_BANK"], os.environ["ETRI_REFERENCE_MODE"])
    hooks = Hooks(utils, replay, os.environ["ETRI_CONDITION_TRACE"])
    utils.apply_mask_strategy, utils.append_generated = hooks.apply, hooks.append
    decoder = Path(__file__).resolve().parents[1] / "04_semantic_decoder/scripts/mydemo_new_align_sh.py"
    sys.path.insert(0, str(decoder.parent))
    sys.argv[0] = str(decoder)
    runpy.run_path(str(decoder), run_name="__main__")


if __name__ == "__main__":
    main()
