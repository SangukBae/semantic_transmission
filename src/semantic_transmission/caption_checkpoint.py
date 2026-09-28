"""Atomic per-segment captions bound to source, model, code and settings."""
import json
import os
from pathlib import Path

from .artifacts import sha256, write_json
from .webvid5 import fingerprint


def checkpoint_identity(cfg, repo, run, indices, clips, prompt):
    run, repo = Path(run), Path(repo)
    model = Path(cfg["models"]["pllava"])
    metadata = {}
    for path in sorted(model.rglob("*")):
        if path.is_file():
            stat = path.stat()
            metadata[str(path.relative_to(model))] = {
                "resolved": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                **({"sha256": sha256(path)} if path.suffix in (".json", ".py", ".txt", ".model") else {})}
    if not metadata:
        raise ValueError("caption checkpoint requires a local PLLaVA model identity")
    code_paths = [Path(__file__), Path(__file__).with_name("workers.py"),
                  Path(__file__).with_name("semantic_clips.py")]
    code_paths += list((repo / ".local/vendor/PLLaVA/models").rglob("*.py"))
    for relative in ("tools/caption/pllava_dir/caption_pllava.py", "opensora/datasets/read_video.py"):
        candidate = repo / ".local/vendor/Open-Sora" / relative
        if candidate.exists():
            code_paths.append(candidate)
    return {"version": 1, "run_signature": os.environ.get("ETRI_RUN_SIGNATURE"),
            "settings": {"seed": cfg["seed"], "paper_caption": cfg.get("paper_caption", False),
                "caption_attention": cfg.get("caption_attention", "sdpa"),
                "max_new_tokens": cfg.get("caption_max_new_tokens", cfg["max_new_tokens"]),
                "semantic_clip_policy": cfg.get("semantic_clip_policy", "official_release"),
                "official_semantic_clips": cfg.get("official_semantic_clips", False),
                "fps": cfg["fps"], "prompt": prompt, "do_sample": False,
                "num_frames": 4, "pooling_shape": [4, 12, 12]},
            "normalized_sha256": sha256(run / "data/normalized.mp4"), "keyframes": indices,
            "frames": [sha256(run / f"data/frames/sample/{i}.png") for i in range(cfg["frames"])],
            "clips": [sha256(p) for p in clips] if clips is not None else None,
            "model": {"path": str(model.resolve()), "files": metadata,
                      "weight_identity": "resolved_path_size_mtime_ns"},
            "code": {str(path): sha256(path) for path in sorted(set(code_paths))}}


class CaptionCheckpoint:
    def __init__(self, path, identity, total, require_sampling):
        self.path = Path(path)
        self.identity = fingerprint(identity)
        self.total = total
        self.require_sampling = require_sampling
        self.records = []
        if self.path.exists():
            saved = json.loads(self.path.read_text())
            checksum = saved.pop("checksum", None)
            if (checksum != fingerprint(saved) or saved.get("identity") != self.identity
                    or saved.get("version") != 1 or saved.get("total") != total
                    or saved.get("require_sampling") != require_sampling):
                raise ValueError("caption checkpoint identity/checksum mismatch")
            records = saved.get("records")
            if not isinstance(records, list) or len(records) > total:
                raise ValueError("invalid caption checkpoint records")
            for segment, record in enumerate(records):
                self.validate(segment, record)
            self.records = records

    def validate(self, segment, record):
        if not isinstance(record, dict) or set(record) != {"segment", "row", "sampling"}:
            raise ValueError("invalid caption checkpoint record")
        row, sampling = record["row"], record["sampling"]
        if (record["segment"] != segment or not isinstance(row, dict)
                or set(row) != {"path", "text", "flow"}
                or row["path"] != f"clips/sample/{segment:05d}.mp4"
                or not isinstance(row["text"], str) or not row["text"].strip() or row["flow"] != 0.0):
            raise ValueError("invalid caption checkpoint row")
        if self.require_sampling:
            if (not isinstance(sampling, dict) or not isinstance(sampling.get("clip"), str)
                    or type(sampling.get("decoded_frames")) is not int or sampling["decoded_frames"] < 1
                    or sampling.get("resize_short_edge") != 672
                    or not isinstance(sampling.get("indices"), list) or len(sampling["indices"]) != 4
                    or any(type(i) is not int or not 0 <= i < sampling["decoded_frames"]
                           for i in sampling["indices"])):
                raise ValueError("invalid caption checkpoint sampling")
        elif sampling is not None:
            raise ValueError("unexpected caption checkpoint sampling")

    def save(self, segment, row, sampling):
        if segment != len(self.records) or segment >= self.total:
            raise ValueError("caption checkpoint must advance exactly one segment")
        record = {"segment": segment, "row": row, "sampling": sampling}
        self.validate(segment, record)
        records = [*self.records, record]
        payload = {"version": 1, "identity": self.identity, "total": self.total,
                   "require_sampling": self.require_sampling, "records": records}
        write_json(self.path, dict(payload, checksum=fingerprint(payload)))
        self.records = records
