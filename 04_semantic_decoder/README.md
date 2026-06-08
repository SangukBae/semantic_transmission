# Stage 04 — Semantic decoder (receiver)

The world model is **Open-Sora v1.2**, used **unmodified**. All LGVSC logic (SFA/DSA
adapters, the masked-generation driver) lives in the standalone scripts in `scripts/`,
which you drop into the Open-Sora `scripts/` directory.

> In our setup this stage ran on a separate GPU server (env `opensora`). The frames
> and the `<method>_video_paths_text_flow.csv` from stages 02–03 must be copied there
> first (e.g. with `rsync`/`scp`).

## Setup
```bash
git clone https://github.com/hpcaitech/Open-Sora.git "$OPENSORA_DIR"
cd "$OPENSORA_DIR"
git checkout bf4d6673
# install per Open-Sora instructions into the `opensora` env (environment/opensora.yml)
cp /path/to/LGVSC/04_semantic_decoder/scripts/* scripts/
```

## Run (SKEM + DSA — the main scheme)
```bash
conda activate opensora
cd "$OPENSORA_DIR/scripts"
./run.bash  <csv_name>  <save_dir>  <method>  <root_dir>
# e.g.
./run.bash key_framesinternvl_diff_0.35_10_video_paths_text_flow \
           key_framesinternvl_diff_0.35_10_save_dir_individual \
           key_framesinternvl_diff_0.35_10 \
           "$DATA_ROOT"
```
`run.bash` splits the master CSV into one file per video, then calls
`mydemo_new_align_sh.py` per video with the v1.2 sampling config
(`configs/opensora-v1-2/inference/sample.py`, `--num-frames 4s --resolution 576
--aspect-ratio 5:9 --aes 6.5`). Output mp4s land in `<root_dir>/<save_dir>/`.

## Scripts
| script | role |
|---|---|
| `mydemo_new_align_sh.py` | the decoder; **DSA** = per-segment `num_frames_ls` from real keyframe spacing |
| `config_utils.py` | config parsing for the above (adds `root_path`/`csv_path`/`method`) |
| `run.bash` | per-video driver (one scheme/SNR) |
| `run_text.bash` | **Text-Only** baseline (caption-conditioned, no keyframe refs) |

**SFA (static) vs DSA (dynamic):** DSA derives each segment's length from the keyframe
indices, so variable-length segments are regenerated at their true length and
concatenated. The static path (fixed segment length, the world model's inherent
behavior) corresponds to the SKIM+SFA configuration.
