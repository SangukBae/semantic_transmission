"""Resolve a frozen recipe for each already-normalized, variable-length video."""
import math


def video_config(cfg, source):
    result = dict(cfg)
    if cfg.get("variable_length", False):
        if cfg.get("official_preprocessing") or not cfg.get("preserve_input"):
            raise ValueError("variable_length requires preserve_input and preprocessed inputs")
        if "source_video" in cfg:
            raise ValueError("variable_length cannot override a frozen source_video contract")
        frames = source["frames"]
        if not isinstance(frames, int) or frames < 2 or frames > cfg["max_frames"]:
            raise ValueError(f"invalid variable video length: {frames}")
        result["frames"] = frames
    expected = cfg.get("source_video", result)
    for key in ("frames", "width", "height", "fps"):
        actual = source[key]
        if not math.isfinite(actual) or not math.isclose(actual, expected[key], rel_tol=0, abs_tol=1e-6):
            raise ValueError(f"video {key} differs from profile: {actual} != {expected[key]}")
    return result
