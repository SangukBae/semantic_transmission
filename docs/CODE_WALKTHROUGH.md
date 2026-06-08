# LGVSC — code walkthrough in data-flow order

Every canonical script, its key functions, what it does, its parameters, and the exact
file it reads / writes. Read top-to-bottom; the **CSV chain** threads the whole pipeline:

```
videos.csv ──get_frame──▶ +frame_save_dir
        └─keyframe(SKIM/SKEM)──▶ frames/<v>/key_frames<method>/*.png
        └─get_key_video──▶ <method>_video_paths.csv (path)
        └─PLLaVA──▶ <method>_video_paths_text.csv (path,text)
        └─UniMatch──▶ <method>_video_paths_text_flow.csv (path,text,flow)   ◀── decoder input
NTSCC corrupts keyframes ──▶ frames/<v>/<method>_<snr>/*.png
decoder(run.bash) ──▶ <root>/<method>_<snr>_save_dir_individual/<v>.mp4
final_score ──▶ comparison_scores.csv
```

---

## Stage 01 — Data preparation  ·  env `llava`

### `01_data_prep/get_prepare.sh`  (driver, args: `<root_dir> <sub_name>`)
Orchestrates the four steps below. `sub_name` = `16x24` (≤16 s, 24 fps).

### `01_data_prep/get_csv.py`
- `generate_csv(root, target)` — lists `*.mp4` in `root/target`, writes `videos.csv`
  with a single `path` column.
- args: `--root`, `--subname`.
- in: folder of mp4 · out: `<root>/<subname>/videos.csv`.

### `01_data_prep/edit_movie.py`
- `process_videos(root, subname)` — reads `videos.csv`; per clip: clip to ≤16 s
  (`subclip(0,16)`), cap fps at 24, re-encode `libx264` into `<root>/<subname>/`.
- args: `--root`, `--subname`.

### `01_data_prep/resize_videoframe.sh`  (arg: `<path>`)
- ffmpeg `crop=iw:ih:((iw-576)/2):((ih-320)/2),scale=576:320`, `-crf 18 -preset medium`,
  overwrites each mp4 in `<path>`. → fixes resolution at **576×320**.

### `01_data_prep/get_frame.py`
- `extract_frames(video_path, output_dir, interval)` — dumps frames every
  `int(interval*fps)` to `<output_dir>/<frame_idx>.png` and writes `frames.csv`
  (`frame_path` column). **interval = 0.2 s** (hardcoded in `__main__`).
- args: `--root`, `--subname`. in: `videos.csv` · out: `frames/<video>/*.png` +
  `frames.csv`, and adds a `frame_save_dir` column back into `videos.csv`.

---

## Stage 02a — Keyframe selection (`I_frame`)

### SKIM — `02_semantic_encoder/skim/baseline-keyframe.py`  ·  env `internvl`
- `baseline_keyframe_extraction(frame_paths, num_keyframes)` — always keeps first &
  last frame, fills the rest at equal interval `total//(num_keyframes-1)`.
- `main(args)` — per video reads `frames.csv`, copies chosen frames to
  `frames/<v>/key_frames-baseline_<N>frames/`.
- args: `--csv-path`, `--num-keyframes`.

### SKEM — `02_semantic_encoder/skem/MLM-keyframe-internvl.py`  ·  env `internvl`
The paper's semantic-guided selector (uses **S_rel = P("No") − P("Yes")**).
- `build_transform / find_closest_aspect_ratio / dynamic_preprocess / load_image` —
  InternVL2 tile-based image preprocessing (≤12 tiles, 448px).
- `custom_chat(self, tokenizer, pixel_values, question, generation_config, …)` —
  patched `model.chat` that **returns generation `scores` (logits)** so token
  probabilities can be read. Bound via `types.MethodType`.
- `main(args)` — loads `InternVL2-8B` (bf16, flash-attn). Per video, walks frames in
  order keeping `cur_frame` = latest keyframe; for each frame:
  1. **round 1** (`q1`): caption both images `img1{…} img2{…}`;
  2. **round 2** (`q2`): "respond yes/no, same scene?";
  3. `no_prob = softmax(scores[0])[id("No")]`, `yes_prob = …[id("Yes")]`,
     `diff = no_prob − yes_prob`  ← **this is S_rel**;
  4. **if `diff > threshold`** → insert as new keyframe (copy png), update `cur_frame`.
  First frame and last frame are always keyframes.
- args: `--model_path` (def `OpenGVLab/InternVL2-8B`), `--csv-path`, `--method`,
  `--threshold` (eta_th, default 0.35), `--q1`/`--q2` (default PSSS prompts).
- the `--method` string is only a label; `eta_th` comes from `--threshold` — keep consistent.
- in: `videos.csv` (+`frame_save_dir`) · out: `frames/<v>/key_frames<method>/*.png`.

`02_semantic_encoder/skem/run.sh` — small batch runner; set `--method` for your run.

---

## Stage 02b — Caption (`I_text`) + optical flow (`I_side`)

### `02_semantic_encoder/caption/after_extract.sh`  (args: `<root_dir> <video_dir> <frame_dir> <method>`)
Three steps, switches conda env internally:
1. `get_key_video.py --root_directory --videos --frames --method` (env `llava`) — cuts
   each clip at keyframe boundaries → `<method>_video_paths.csv` (`path` per segment).
2. **PLLaVA** `caption_pllava.py` (env `pllava`, model `pllava-7b`, lora_alpha 4,
   num_frames 4, pooling 4-12-12) → `<method>_video_paths_text.csv` (`path,text`).
3. **UniMatch** `OPENSORA_DIR/tools/scoring/optical_flow/inference.py <csv>` →
   `<method>_video_paths_text_flow.csv` (`path,text,flow`)  ← **decoder input contract**.
- PLLaVA / UniMatch tools are resolved from `$OPENSORA_DIR` (set in `env.sh`); no manual
  path edits — `source env.sh` first.

### `02_semantic_encoder/caption/get_key_video.py`
- cuts video into per-segment clips between consecutive keyframe indices; writes the
  segment-level CSV. args: `--root_directory`, `--videos`, `--frames`, `--method`.

---

## Stage 03 — JSCC transmission

### NTSCC keyframes — `03_jscc_transmission/ntscc/main_save.py`  ·  env `ntscc`
- `test(net, test_path, logger, db, save=False)` — globs `*.png` in `test_path`,
  forwards each keyframe through NTSCC `net(input_image)` → `x_hat_ntscc`; accumulates
  PSNR/CBR; if `save` writes corrupted-reconstructed png to `test_path_<db>`. Adds the
  side-info CBR (`cbr_sideinfo`, capacity-achieving / "2/3-rate LDPC + 16QAM AWGN 10dB"
  noted in comments) and reports `cbr_of_the_video`.
- `parse_args` → `-p/--phase`, `--checkpoint`, `--test_path`, `--method`, `--model`
  (SNR), `--save_frames`, `--seed`(1024).
- `model_dict` (SNR→weights):
  - `'10' → checkpoints/ntscc_hyperprior_quality_4_psnr.pth`  (**released**)
  - `'8'/'6'/'4'/'2'/'0' → mayu/SNR=XdB_2_0.2_train/models/best_loss_XdB.model`
    (**NOT released — live in the 3.5 GB `mayu/` dir; see provenance note**)
- uses `config.py` (`lr=1e-4`, `train_lambda=64`, `eta`, `use_side_info`,
  `multiple_rate`) — modifications captured in `patches/ntscc_modifications.patch`.
- in: `frames/<v>/<method>/*.png` · out: `frames/<v>/<method>_<snr>/*.png` + CBR log.

### LDPC text/side-info — `03_jscc_transmission/ldpc_sionna/SionnaPlus_sora.py`  ·  Sionna 0.19.2
- `main()` — `LDPC5GEncoder(k, n)` + `LDPC5GDecoder(num_iter=20, return_infobits=True)`,
  `Constellation("qam", num_bits_per_symbol=order)`,
  `ebno = snr − 10·log10(order·k/n)`, AWGN via `ebnodb2no(...)`. Reads the text/side-info
  bitstream, transmits, writes `Results.txt`.
- args: `--snr` (def 10), `--seq_list` (defaults to `$DATA_ROOT/H264_test_10`).
  `snr_config` holds `(k, n)` per SNR; runs in the Sionna Docker.

---

## Stage 04 — Semantic decoder (Open-Sora v1.2 + SFA/DSA)  ·  env `opensora`, GPU server

### `04_semantic_decoder/scripts/run.bash`  (args: `<csv_name> <save_dir> <method> <root_dir>`)
Asserts env `opensora`; splits `<root>/16x24/<csv_name>.csv` into one temp CSV per
video; calls the decoder per video with
`configs/opensora-v1-2/inference/sample.py --num-frames 4s --resolution 576
--aspect-ratio 5:9 --aes 6.5`; logs per-video time.

### `04_semantic_decoder/scripts/mydemo_new_align_sh.py`  (the decoder)
- `encode_from_sender(key_frames, vae, image_size)` — VAE-encodes received keyframes as
  conditioning references (`refs_x[batch][ref]`).
- `get_mask_from_sender(key_frames)` — builds the per-segment mask strategy string
  (`"i-1,i,0,-1,1;…"`) and the loop count.
- `append_multi_score_to_prompts(prompts, aes, flow, camera_motion)` — appends
  `aesthetic score:` / `motion score:` (the optical-flow `flow` value) into the prompt.
- `fully_control_random_seed(seed=1024, deterministic=True)` — full reproducibility lock.
- **main flow**: read `<method>_video_paths_text_flow.csv` → group rows by video →
  derive **`num_frames_ls`** from real keyframe spacing (**DSA**: per-segment length) →
  build T5 + VAE → per video, per segment `loop_i`:
  `num_frames = num_frames_ls[loop_i]` → rebuild STDiT with `latent_size` from that
  `num_frames` (dynamic latent dim, params reused) → condition on keyframe refs (+ prior
  segment via `append_generated`) → `scheduler.sample(...)` → `vae.decode` → concat
  segments (trim `condition_frame_length` overlap) → `save_sample` mp4.
- in: `<method>_video_paths_text_flow.csv` + `frames/<v>/<method>(_snr)/` keyframes ·
  out: `<root>/<save_dir>/<v>.mp4`.

### `04_semantic_decoder/scripts/config_utils.py`
- `parse_configs(training=False)` → `parse_args` adds LGVSC args `--root_path`,
  `--csv_path`, `--save_dir`, `--method` on top of Open-Sora's inference config args
  (`--num-frames`, `--resolution`, `--aspect-ratio`, `--aes`, `--condition-frame-length`,
  `--mask-strategy`, …).

### `04_semantic_decoder/scripts/run_text.bash`  (**Text-Only** baseline)
Builds a caption-only CSV and calls stock `scripts/inference.py` (no keyframe refs) —
the "Text-Only" scheme in the paper.

**SFA vs DSA:** same script — SFA = SKIM uniform (equal-length) segments → constant
latent dim; DSA = SKEM variable-length segments → per-segment latent dim above.

---

## Stage 05 — Classical baselines  ·  env `llava`

### `05_baselines/h264/h264compress.py`
- `compress_video(input, output, snr, qp=51, gop_size=400, preset, bv)` — ffmpeg
  `libx264 -qp <qp> -g <gop> -profile:v baseline`; computes input/output bpp.
- per-SNR table (matches the paper): `10→qp51,gop20` · `8→qp51,gop80` ·
  `6/4/2/0→qp51,gop400`. `__main__` loops over `$VIDEO_FOLDER` (defaults to
  `$DATA_ROOT/16x24`) — no hardcoded path.
- `05_baselines/h265/h265compress.py` — same with `libx265`; `convert_yuv_to_mp4.py`,
  `preprocessing.py`, `video_bgr2rgb.py` = format helpers. Output then goes through
  `ldpc_sionna` (its `snr_config` has the H.264/H.265 `(k,n)` rates).

---

## Stage 06 — Evaluation  ·  env `llava`

### `06_evaluation/final_score.py`
- `calculate_clip_similarity / calculate_psnr / calculate_ssim / calculate_lpips /
  calculate_dists(frame1, frame2)` — per-frame metrics.
- `calculate_average_scores(folder1, folder2)` — averages over aligned frames.
- `extract_frames` / `clean_frame_dirs` — frame extraction & cleanup.
- `main(reference_video_folder, result_video_folder, generated_video_folder, output_csv,
  interval=1)` — compares **original vs NTSCC-recon vs generated** → `comparison_scores.csv`.
- args: `--reference_video_folder`, `--result_video_folder`, `--generated_video_folder`,
  `--output_csv`, `--interval`.
- `final_score_calc_100.py` — the 100-frame aggregation variant used for the paper's main
  table (same folder arguments).

---

## Stage 07 — Zero-shot downstream

### Action recognition — `07_downstream/timesformer/`  ·  env `timesformer`
Apply `patches/timesformer_modifications.patch` to upstream TimeSformer + add
`get_video_lable.py` (builds the label list). Run with the provided
`TimeSformer_divST_8x32_224_TEST.yaml` via `tools/run_net.py`. Kinetics-400 (14 clips).

### Depth estimation — `07_downstream/depth_anything_v2/score.py`  ·  env `depth`
- `calculate_metrics(ref_path, hyp_path, depth_anything, method1, method2)` — runs
  Depth-Anything-V2 (`encoder='vitl', features=256`) on both, computes depth metrics via
  `metric_depth.util.metric.eval_depth` (AbsRel / δ etc.) over valid masks.
- `main()` — `argv[1]` = origin CSV, `argv[2]` = method CSV → averaged metrics table.
- library unmodified — only `score.py` is ours.

### Video captioning — `07_downstream/video_summary/{video_summary.sh,video_depth.sh}`  ·  env `video_summary`
Caption originals + reconstructions with PLLaVA, then score with BERTScore / BLEU / ROUGE
(`video_depth.sh` wraps the depth pipeline). Both take `$DATA_ROOT`/method as args and
resolve tools from `$OPENSORA_DIR` / `$DEPTH_DIR` — `source env.sh` first, no manual edits.

---

## Reproducibility Notes

1. **SKEM threshold** `η_th` defaults to `0.35` (matches the paper) and is set with
   `--threshold`; the value embedded in `--method` is only a label, not parsed.
2. **Frame sampling interval = 0.2 s** in `get_frame.py`. This is the temporal
   granularity used for keyframe search.
3. **NTSCC checkpoints**: only **SNR=10** maps to a released weight
   (`quality_4_psnr.pth`). **SNR 0/2/4/6/8** map to `mayu/SNR=XdB…/best_loss_XdB.model`,
   which are not included in this lightweight code release. To reproduce sub-10 dB NTSCC
   experiments, provide those checkpoints separately.
4. **NTSCC configuration**: `config.py` records the released inference/training
   configuration, including `batch_size=10` and `train_lambda=64`.
5. **H.264/H.265 QP/GOP** in `h264compress.py`/`h265compress.py` match the paper's table
   and are used for the classical codec baselines.
6. **Decoder configuration** in `run.bash`: `--num-frames 4s --resolution 576
   --aspect-ratio 5:9 --aes 6.5`, `seed=1024`, `condition_frame_length=5`.
