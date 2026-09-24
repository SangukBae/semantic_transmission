"""Optional atomic progress IPC; silent unless a campaign supplies a file path."""
import os
from .artifacts import write_json


def emit(done, total):
    path = os.environ.get("QUALITY_PROGRESS_FILE")
    if path:
        write_json(path, {"done": done, "total": total})
