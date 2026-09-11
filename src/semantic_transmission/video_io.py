"""Read the temporal contract without changing the source video."""
from fractions import Fraction
import json
import subprocess


def probe(path):
    data = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames,duration",
        "-of", "json", str(path)], text=True))["streams"][0]
    return {"width": int(data["width"]), "height": int(data["height"]),
            "frames": int(data["nb_read_frames"]), "fps": float(Fraction(data["avg_frame_rate"])),
            "duration": float(data["duration"])}
