# LGVSC pipeline — detailed walkthrough

This document explains what each stage does, the exact data that flows between
stages, and where the paper's modules live in the code. Read it alongside the
per-stage `README.md` files.

**ETRI follow-up scope (2026-09-21):** this walkthrough describes the existing
pipeline. Current research requirements and planned extensions are in the
[email summary](ETRI_FOLLOWUP_EMAIL_SUMMARY.md), [development plan](ETRI_DEVELOPMENT_PLAN.md)
and [evaluation protocol](ETRI_FOLLOWUP_PROTOCOL.md). In particular, the legacy
16-second preparation below must not be used unchanged for the 60+ second target.
SKEM boundaries and scalar flow are not evidence that scene-change recovery or
explicit object speed/action transmission has been validated.

## 0. Notation recap (from the paper)

- A video `X ∈ R^{F×H×W×C}` is split into `N` semantic segments `S_1..S_N` at
  keyframe boundaries `K_1..K_{N+1}`.
- Each segment is encoded into three modalities:
  - `I_text`  — a caption (compact spatiotemporal semantics), from a multimodal LLM.
  - `I_frame` — the keyframe itself (fine-grained semantics), transmitted via NTSCC.
  - `I_side`  — task-specific side information (optical flow).
- **PSSS** (probability-based semantic similarity score) is the continuous metric
  driving SKEM keyframe selection:
  - `S_abs = P("Yes" | Info.A, Info.B, Focus)`  — absolute confidence.
  - `S_rel = P("No") − P("Yes") ∈ (−1, 1)`      — relative (bias-cancelled), **used in the paper**.
  - A frame becomes a new keyframe when `S_rel > eta_th` (semantic-divergence threshold).

## 1. Data preparation — `01_data_prep/`

`get_prepare.sh <DATA_ROOT> 16x24` chains:
1. `get_csv.py`  — enumerate raw mp4s → `videos.csv`
2. `edit_movie.py` — normalize to 24 fps, ≤16 s
3. `resize_videoframe.sh` — resize to 576×320
4. `get_csv.py` (again) — repoint csv to processed videos
5. `get_frame.py` — dump every frame to `<DATA_ROOT>/16x24/frames/<video>/`

Output contract: `<DATA_ROOT>/16x24/videos.csv` and `<DATA_ROOT>/.../frames/`.

## 2. Semantic encoder (transmitter) — `02_semantic_encoder/`

### 2a. Keyframe selection → `I_frame`
- **SKIM** (`skim/baseline-keyframe.py`): fixed number of evenly spaced keyframes
  (`--num-keyframes`). Cheap, real-time, semantics-agnostic.
- **SKEM** (`skem/MLM-keyframe-internvl.py`): autoregressive, semantic-guided. For
  each frame it asks InternVL2 to caption the current frame and the latest keyframe,
  computes `S_rel`, and inserts a keyframe when `S_rel > eta_th` (`--threshold`, default 0.35).

Output: keyframe PNGs under `frames/<video>/<method>/`, and `<method>_video_paths.csv`.

### 2b. Caption (`I_text`) + optical flow (`I_side`) → `caption/after_extract.sh`
1. `get_key_video.py` — cut clips at keyframe boundaries → `<method>_video_paths.csv`
2. PLLaVA caption each clip → `<method>_video_paths_text.csv`
3. UniMatch optical flow (Open-Sora `tools/scoring/optical_flow`) →
   `<method>_video_paths_text_flow.csv`  ← **this is the decoder's input contract**

## 3. JSCC transmission — `03_jscc_transmission/`

- **Keyframes → NTSCC** (`ntscc/main_save.py`): nonlinear transform + JSCC. `--model`
  selects the SNR-matched checkpoint (0/2/4/6/8/10 dB). With `--save_frames` it
  writes the channel-corrupted reconstructed keyframes into
  `frames/<video>/<method>_<snr>/`, and logs the CBR per video.
- **Text + side info → LDPC** (`ldpc_sionna/SionnaPlus_sora.py`): binary stream →
  LDPC → modulation → AWGN, run inside the Sionna Docker image.

## 4. Semantic decoder (receiver) — `04_semantic_decoder/`

The world model is **Open-Sora v1.2** (pristine upstream). All customization lives in
standalone scripts that are dropped into `Open-Sora/scripts/`:

- `mydemo_new_align_sh.py` — the decoder. Reads `<method>_video_paths_text_flow.csv`,
  VAE-encodes the received keyframes as conditioning references, builds a per-segment
  mask strategy, and runs masked diffusion generation segment by segment.
  - **DSA (dynamic)**: per-segment frame count `num_frames_ls` is derived from the
    *actual* keyframe spacing, so each variable-length segment (SKEM) is regenerated at
    its true length — equivalently, the VAE latent temporal dimension adapts per segment.
  - **SFA (static)**: the same script consuming SKIM's uniform (equal-length) segments,
    i.e. the world model's inherent fixed-length behavior. SFA and DSA are **not**
    separate scripts — the distinction comes from the keyframe segmentation upstream.
- `run.bash` — splits the master CSV per-video and invokes the decoder; takes
  `<csv_name> <save_dir> <method> <root_dir>`.
- `run_text.bash` — the **Text-Only** baseline (no keyframe conditioning).

Output: reconstructed mp4s under `<DATA_ROOT>/<method>_save_dir_individual/`.

## 5. Baselines — `05_baselines/`

- `h264/h264compress.py`, `h265/h265compress.py` — source-code the video at the QP/GOP
  in the paper, then LDPC-transmit via `03_jscc_transmission/ldpc_sionna`.
- DVST is **not** included (collaborator's code; see top-level README).

## 6. Evaluation — `06_evaluation/`

- `final_score.py` — CLIP, PSNR, SSIM, LPIPS, DISTS between reference / NTSCC-recon /
  generated videos → `comparison_scores.csv`. (`final_score_calc_100.py` is the
  100-frame aggregation variant used for the paper's main table.)

## 7. Downstream zero-shot tasks — `07_downstream/`

Run the *same reconstructed videos* through off-the-shelf models and compare to the
originals, demonstrating that LGVSC preserves task-relevant semantics:
- **Action recognition**: TimeSformer on Kinetics-400.
- **Depth estimation**: Depth-Anything-V2 (`score.py` compares depth maps).
- **Video captioning**: PLLaVA captions scored with BERTScore / BLEU / ROUGE.
