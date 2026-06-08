# Stage 02 — Semantic encoder (transmitter)

Splits each video into semantic segments and encodes them as `{I_text, I_frame, I_side}`.

## 2a. Keyframe selection (`I_frame`)

### SKIM — fixed interval (`skim/`)  ·  env `internvl`
```bash
python skim/baseline-keyframe.py --csv-path "$DATA_ROOT/16x24/videos.csv" --num-keyframes 4
```
3–4 keyframes per clip, evenly spaced (first and last frame always kept).

### SKEM — semantic-guided via PSSS (`skem/`)  ·  env `internvl`
```bash
# The scripts import InternVL's `internvl_chat` package — put the InternVL repo on
# PYTHONPATH (just `cd`-ing there is NOT enough when the script is given by abs path):
export PYTHONPATH="$INTERNVL_DIR:$PYTHONPATH"
#   --method: pass WITHOUT "key_frames" — the code prepends it (see note below)
#   --threshold: PSSS divergence eta_th (keep consistent with the value in --method)
#   --model_path: defaults to OpenGVLab/InternVL2-8B
python skem/MLM-keyframe-internvl.py \
    --csv-path "$DATA_ROOT/16x24/videos.csv" \
    --method    internvl_diff_0.35 \
    --threshold 0.35 \
    --model_path "$INTERNVL_MODEL"
```
> **Method naming:** the code writes keyframes to `frames/<v>/key_frames<method>/`, i.e.
> it **prepends `key_frames`**. So pass `--method internvl_diff_0.35` to get the folder
> `key_framesinternvl_diff_0.35`, which is the `--method` value the NTSCC and decoder
> stages then expect. (`eta_th` defaults to `0.35` and is set with `--threshold`; the value
embedded in `--method` is only a label and is **not** parsed — keep the two consistent.)
Uses the **relative** score `S_rel = P("No") − P("Yes")` (the paper's choice).
`skem/run.sh` is a small batch runner — set `--method` for your run.

The prompts (`--q1`, `--q2`) default to the PSSS templates: ask InternVL to describe
each frame, then "Determine whether they are similar from the perspective of
<Semantic Focus>, use yes or no to answer."

**Output:** keyframe PNGs in `frames/<video>/<method>/` and `<method>_video_paths.csv`.

## 2b. Caption (`I_text`) + optical flow (`I_side`) (`caption/`)

```bash
conda activate llava        # the driver switches envs internally (llava -> pllava)
./caption/after_extract.sh "$DATA_ROOT" 16x24 frames key_framesinternvl_diff_0.35
```
1. `get_key_video.py` cuts clips at keyframe boundaries → `<method>_video_paths.csv`
2. **PLLaVA** captions each clip (env `pllava`, model `pllava-7b`) →
   `<method>_video_paths_text.csv`
3. **UniMatch** optical flow via `OPENSORA_DIR/tools/scoring/optical_flow/inference.py`
   → `<method>_video_paths_text_flow.csv`

> `after_extract.sh` resolves the PLLaVA / UniMatch tools from `$OPENSORA_DIR` (set in
> `env.sh`) — no manual path edits needed; just `source env.sh` first.

## Patches
None — the InternVL library is used unmodified; only the scripts above are ours.
