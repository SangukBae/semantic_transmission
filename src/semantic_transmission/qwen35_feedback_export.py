"""Export verified feedback conditions and original keyframe PNGs for inspection.

The existing four-sample InternVL pipeline importer must not silently consume this
different selector/sampling policy. This export makes those differences explicit.
"""
import argparse
from pathlib import Path
import shutil

from . import auto_extraction as ae, qwen35_feedback as fb
from .artifacts import sha256, write_json


def export(root):
    fb.verify(root)
    from .qwen35_feedback_eval import audit_feedback_requests
    audit_feedback_requests(root)
    plan = fb.check_plan(root)
    result = {}
    for arm in plan["arms"]:
        bundle = ae.unseal(root / f"{arm}.json")
        dest = root / "conditions" / arm
        images = dest / "selected_frames"
        images.mkdir(parents=True, exist_ok=True)
        for i in bundle["keys"]:
            original = Path(plan["source_frames"]) / f"{i}.png"
            output = images / f"{i:05d}.png"
            if output.exists() and sha256(output) != plan["source_frame_hashes"][str(i)]:
                raise ValueError("existing exported frame differs from source")
            if not output.exists():
                shutil.copyfile(original, output)
        write_json(dest / "keyframes.json", dict(indices=bundle["keys"], fps=plan["fps"]))
        write_json(dest / "captions.json", dict(provider="local_qwen35_feedback",
            arm=arm, model_id=fb.MODEL_ID, revision=fb.REVISION,
            quantization="nf4", sampling="four original samples plus both target endpoints; bounded extra evidence on repair",
            captions=bundle["captions"]))
        manifest = dict(policy=plan["policy"], signature=plan["signature"],
            source_video=plan["normalized_video"], source_video_sha256=plan["normalized_sha256"],
            extraction=str(root / f"{arm}.json"), extraction_sha256=sha256(root / f"{arm}.json"),
            selector="Qwen direct select/skip, stateful latest accepted reference",
            keyframes_sha256=sha256(dest / "keyframes.json"), captions_sha256=sha256(dest / "captions.json"),
            source_png_sha256={str(i):sha256(images/f"{i:05d}.png") for i in bundle["keys"]},
            internvl_psss_executed=False, assistant_authored=False,
            astra_reference_used_in_extraction=False, reconstruction_executed=False,
            existing_four_sample_importer_compatible=False)
        write_json(dest / "provenance.json", manifest)
        result[arm] = dict(path=str(dest), keyframes=len(bundle["keys"]), captions=len(bundle["captions"]))
    return result


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=fb.DEFAULT_ROOT)
    args = parser.parse_args()
    print(json.dumps(export(args.root.resolve())))
