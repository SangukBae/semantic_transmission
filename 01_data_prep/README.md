# Stage 01 — Data preparation

**Env:** `llava`  ·  **Input:** a folder of raw `*.mp4`  ·  **Output:** normalized
videos + per-frame PNGs + `videos.csv`.

```bash
conda activate llava
./get_prepare.sh "$DATA_ROOT" 16x24
```

`16x24` means "≤16 s, 24 fps". The driver runs, in order:

| step | script | result |
|---|---|---|
| 1 | `get_csv.py --root $DATA_ROOT --subname ""` | `videos.csv` of raw clips |
| 2 | `edit_movie.py` | re-encode to 24 fps, ≤16 s into `$DATA_ROOT/16x24/` |
| 3 | `resize_videoframe.sh` | resize to **576×320** |
| 4 | `get_csv.py` (subname `16x24`) | csv repointed to processed clips |
| 5 | `get_frame.py` | dump frames to `$DATA_ROOT/16x24/frames/<video>/` |

> `get_csv.py` and `edit_movie.py` live in the upstream LLaVA working dir in the
> original setup; copies are provided here. They take `--root` / `--subname` and have
> no hardcoded paths.
