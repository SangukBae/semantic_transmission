# Stage 06 — Evaluation

**Env:** `llava` (metrics need CLIP/LPIPS/DISTS).

## Video-quality metrics
```bash
python final_score.py \
    --reference_video_folder "$DATA_ROOT/16x24" \
    --result_video_folder    "$DATA_ROOT/<ntscc_recon_dir>" \
    --generated_video_folder "$DATA_ROOT/<method>_save_dir_individual" \
    --output_csv comparison_scores.csv
```
Computes **CLIP, PSNR, SSIM, LPIPS, DISTS** between the original, the NTSCC-reconstructed
keyframes, and the Open-Sora generated video. Writes `comparison_scores.csv` into the
generated folder.

`final_score_calc_100.py` is the 100-frame aggregation variant used for the paper's main
table; it takes the same folder arguments.
