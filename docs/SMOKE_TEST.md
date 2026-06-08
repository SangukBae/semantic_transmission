# Smoke test — end-to-end on the bundled sample

A minimal installation check using the single bundled clip
`assets/sample/stock-footage-aerial-drone-panoramic-video-of-famous-port-and-marina-of-zea-with-luxury-yachts-docked-as-seen.mp4`
(raw 596×336, 23.976 fps, 20.6 s). It exercises stages **01 → 02 → 03**. Open-Sora
generation (stage 04, needs the GPU decoder server) is intentionally **out of scope**
here. These are the actual outputs we observed on a single RTX 4090.

> Prereqs: `source env.sh` with `DATA_ROOT`, `INTERNVL_DIR`, `NTSCC_DIR` set; conda envs
> `llava`, `internvl`, `ntscc` created from `environment/`; InternVL2-8B + NTSCC
> `quality_4` checkpoint available.

```bash
source env.sh
export DATA_ROOT=/tmp/lgvsc_smoke/data
mkdir -p "$DATA_ROOT" && cp assets/sample/*.mp4 "$DATA_ROOT/"
```

## Stage 01 — data prep (env `llava`)
```bash
cd 01_data_prep && conda activate llava
./get_prepare.sh "$DATA_ROOT" 16x24
```
**Observed:** raw 596×336/20.6 s → normalized **576×320, 24 fps, 16 s**; 385 decoded
frames; `frames.csv` lists **96 sampled frames** (0.2 s interval); `16x24/videos.csv`
gains a `frame_save_dir` column. ✓

## Stage 02 — keyframe selection (env `internvl`)

### SKIM (fixed interval)
```bash
cd ../02_semantic_encoder/skim && conda activate internvl
python baseline-keyframe.py --csv-path "$DATA_ROOT/16x24/videos.csv" --num-keyframes 4
```
**Observed:** 96 frames → **4 evenly-spaced keyframes** `{0, 128, 256, 380}` in
`frames/<v>/key_frames-baseline_4frames/`. ✓

### SKEM (semantic-guided, PSSS + InternVL2-8B)
```bash
export PYTHONPATH="$INTERNVL_DIR:$PYTHONPATH"     # internvl_chat must be importable
cd ../skem
python MLM-keyframe-internvl.py --csv-path "$DATA_ROOT/16x24/videos.csv" --method internvl_diff_0.35
```
**Observed:** full 96-frame autoregressive pass in ~13 min (~8.3 s/frame-pair, matching
the complexity report). Example PSSS score for an adjacent pair:
`P("No")=0.120, P("Yes")=0.691 → S_rel = −0.571` (< η_th 0.35 → not a keyframe).
The aerial pan is semantically stable, so SKEM selects only **2 keyframes** `{0, 380}`
in `frames/<v>/key_framesinternvl_diff_0.35/`. ✓ (Demonstrates content-adaptive
selection: fewer keyframes than SKIM's fixed 4 on stable content.)

## Stage 03 — NTSCC keyframe transmission (env `ntscc`, SNR = 10 dB)
```bash
cp 03_jscc_transmission/ntscc/main_save.py "$NTSCC_DIR/" && cd "$NTSCC_DIR"
conda activate ntscc
# SKIM keyframes:
python main_save.py -p test --test_path "$DATA_ROOT/frames" --method key_frames-baseline_4frames --model 10 --save_frames
# SKEM keyframes:
python main_save.py -p test --test_path "$DATA_ROOT/frames" --method key_framesinternvl_diff_0.35  --model 10 --save_frames
```
**Observed (SNR = 10 dB):**

| keyframe set | #keyframes | JSCC PSNR | **cbr_of_the_video** |
|---|---|---|---|
| SKIM (`key_frames-baseline_4frames`) | 4 | ≈ 30.66 dB | **0.00080** |
| SKEM (`key_framesinternvl_diff_0.35`) | 2 | ≈ 30.56 dB | **0.00040** |

Reconstructed keyframes written to `frames/<v>/<method>_10/`; per-video CBR logged to
`frames/<method>.csv`. ✓

Both CBRs fall in the paper's reported `1e-4 ~ 1e-3` regime, and SKEM's content-adaptive
selection roughly **halves the CBR** vs fixed SKIM on this semantically-stable clip while
holding reconstruction PSNR — the core LGVSC behavior, reproduced end-to-end.

## Not covered here
- **Stage 02c** captioning (PLLaVA) + optical flow (UniMatch) — needs PLLaVA weights and
  the Open-Sora `tools/` (heavy; required only to feed stage 04).
- **Stage 04** Open-Sora generation — runs on the GPU decoder server (see its README).
- **Stages 06–07** quality metrics / downstream — need the stage-04 generated videos.
