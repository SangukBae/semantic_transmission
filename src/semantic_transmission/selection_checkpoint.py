"""Atomic SKEM progress; resume only against the same arguments and source pixels."""
from pathlib import Path

from .artifacts import sha256, write_json
from .webvid5 import fingerprint, read_json


class SelectionCheckpoint:
    def __init__(self, path, arguments, frames):
        self.path = Path(path)
        self.frames = frames
        self.numbers = [Path(p).stem for p in frames]
        if not frames or len(set(self.numbers)) != len(frames):
            raise ValueError("checkpoint requires unique frame indices")
        self.identity = fingerprint({"arguments": arguments,
            "frames": [(p, sha256(p)) for p in frames]})
        self.done, self.selected = 0, []
        if self.path.exists():
            saved = read_json(self.path)
            digest = saved.pop("checksum")
            if fingerprint(saved) != digest or saved["identity"] != self.identity:
                raise ValueError("SKEM checkpoint identity/checksum mismatch")
            self.validate(saved["done"], saved["selected"])
            self.done, self.selected = saved["done"], saved["selected"]

    def validate(self, done, selected):
        if not isinstance(done, int) or not 1 <= done <= len(self.frames):
            raise ValueError("invalid SKEM completed-frame count")
        allowed = self.numbers[:done]
        if (not selected or selected[0] != allowed[0] or len(set(selected)) != len(selected)
                or any(n not in allowed for n in selected)
                or [n for n in allowed if n in selected] != selected):
            raise ValueError("invalid SKEM selected-frame state")

    def save(self, done, selected):
        self.validate(done, selected)
        if done < self.done:
            raise ValueError("SKEM checkpoint cannot move backwards")
        payload = {"identity": self.identity, "done": done, "selected": list(selected)}
        write_json(self.path, dict(payload, checksum=fingerprint(payload)))
        self.done, self.selected = done, list(selected)
