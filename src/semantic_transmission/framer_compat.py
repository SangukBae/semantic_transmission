"""One in-process diffusers 0.24 fix; keep vendor and installed files unchanged."""
import hashlib
import inspect
import subprocess
import sys
import textwrap
from pathlib import Path

FILE = "models_diffusers/attention.py"
OLD = "hidden_states = _chunked_feed_forward(self.ff, hidden_states, self._chunk_dim, self._chunk_size)"
NEW = "hidden_states = _chunked_feed_forward(self.ff_in, hidden_states, self._chunk_dim, self._chunk_size)"
DESCRIPTION = "TemporalBasicTransformerBlock chunked input uses ff_in, matching the unchunked path"


def check(vendor, apply=False):
    vendor = Path(vendor)
    original = subprocess.check_output(["git", "-C", str(vendor), "show", f"HEAD:{FILE}"])
    path = vendor / FILE
    actual = path.read_bytes()
    # Retire the diagnostic patch to the unused local copy of this class.
    if apply and actual == original.replace(OLD.encode(), NEW.encode()):
        path.write_bytes(original)
        actual = original
    if actual != original:
        raise ValueError("Unexpected Framer vendor modification")
    status = subprocess.check_output(["git", "-C", str(vendor), "status", "--porcelain"], text=True)
    if status.strip():
        raise ValueError(f"Unexpected Framer modifications: {status}")
    return dict(target="diffusers.models.attention.TemporalBasicTransformerBlock.forward",
                mode="in-process only; installed dependencies and vendor remain unchanged", fix=DESCRIPTION)


def patch_temporal_feed_forward():
    import diffusers.models.attention as attention
    cls = attention.TemporalBasicTransformerBlock
    if hasattr(cls.forward, "_sparc_fix"):
        return cls.forward._sparc_fix
    source = textwrap.dedent(inspect.getsource(cls.forward))
    if source.count(OLD) != 1:
        raise ValueError("Unexpected diffusers temporal block; refusing a speculative patch")
    fixed = source.replace(OLD, NEW)
    namespace = {}
    exec(compile(fixed, "<sparc-verified-temporal-feed-forward>", "exec"),
         attention.__dict__, namespace)
    metadata = dict(fix=DESCRIPTION, original_sha256=hashlib.sha256(source.encode()).hexdigest(),
                    corrected_sha256=hashlib.sha256(fixed.encode()).hexdigest())
    namespace["forward"]._sparc_fix = metadata
    cls.forward = namespace["forward"]
    return metadata


if __name__ == "__main__":
    print(check(sys.argv[1], apply="--apply" in sys.argv[2:]))
