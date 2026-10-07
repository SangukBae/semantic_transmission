"""Import local Qwen conditions without relabeling assistant-authored bundles."""
from pathlib import Path

from . import auto_extraction as extraction
from .artifacts import sha256, write_json
from .assisted_captions import sample_indices
from .qwen35_backend import MODEL_ID, REVISION

PROVIDER = "local_qwen35"


def validated(root, video_id):
    root = Path(root).resolve()
    config = extraction.read(root / "run_config.json")
    if (config.get("backend"), config.get("model_id"), config.get("model_revision"),
            config.get("quantization")) != ("qwen35", MODEL_ID, REVISION, "nf4"):
        raise ValueError("expected pinned Qwen3.5-9B NF4 condition extraction")
    extraction.verify_config(config)
    source = extraction.source_for(config, video_id)
    extraction.verify_video(root, config, source, write_outputs=False)
    dest = root / source["dataset"].lower() / video_id
    return config, source, dest, extraction.unseal(dest / "captions.json")


def import_captions(run):
    run = Path(run)
    cfg = extraction.read(run / "run_config.json")
    if cfg.get("caption_provider") != PROVIDER:
        raise ValueError("not a Qwen3.5 caption import")
    config, source, dest, bundle = validated(cfg["condition_extraction_root"], cfg["condition_video_id"])
    if (cfg["condition_signature"] != config["signature"]
            or sha256(dest / "captions.json") != cfg["condition_captions_sha256"]):
        raise ValueError("frozen Qwen conditions changed")
    indices = extraction.read(run / "keyframes.json")["indices"]
    if indices != extraction.read(dest / "keyframes.json")["indices"]:
        raise ValueError("run keyframes differ from Qwen extraction")
    if sha256(run / "data/normalized.mp4") != source["normalized_sha256"]:
        raise ValueError("run video differs from Qwen source")
    audit = extraction.read(run / "semantic_clips_audit.json")
    if (audit["status"] != "PASSED" or audit["policy"] != "frame_exact"
            or audit["source"]["keyframes"] != indices):
        raise ValueError("Qwen import requires matching frame-exact clips")
    rows, sampling = [], []
    for item in bundle["captions"]:
        a, b = item["start"], item["end_exclusive"]
        ids = sample_indices(a, b)
        for i in ids:
            if sha256(run / f"data/frames/sample/{i}.png") != source["source_frame_hashes"][str(i)]:
                raise ValueError("run caption sample differs from Qwen source")
        relative = f"clips/sample/{item['segment']:05d}.mp4"
        rows.append(dict(path=relative, text=item["caption"], flow=0.0))
        sampling.append(dict(clip=str(run / "data" / relative), decoded_frames=b-a,
            indices=[i-a for i in ids], source_indices=ids, caption_provider=PROVIDER))
    write_json(run / "captions.json", rows)
    write_json(run / "caption_sampling.json", sampling)
    write_json(run / "caption_provenance.json", dict(provider=PROVIDER,
        model_id=config["model_id"], model_revision=config["model_revision"],
        quantization=config["quantization"], extraction_signature=config["signature"],
        caption_adapter=config.get("caption_adapter"),
        caption_review_model="base Qwen3.5-9B; adapter disabled",
        bundle=str(dest / "captions.json"), bundle_sha256=sha256(dest / "captions.json"),
        segments=len(rows), pllava_inference_executed=False, assistant_authored=False,
        independent_verification=False))
