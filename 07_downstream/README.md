# Stage 07 — Zero-shot downstream tasks

Run the **reconstructed** videos through off-the-shelf task models and compare against
the originals. Demonstrates that LGVSC preserves task-relevant semantics with no
task-specific fine-tuning.

## Action recognition — TimeSformer (`timesformer/`)  ·  env `timesformer`

Built on [facebookresearch/TimeSformer](https://github.com/facebookresearch/TimeSformer),
evaluated on Kinetics-400 (we used 14 clips).

```bash
git clone https://github.com/facebookresearch/TimeSformer.git "$TIMESFORMER_DIR"
cd "$TIMESFORMER_DIR"
git apply /path/to/LGVSC/07_downstream/timesformer/patches/timesformer_modifications.patch
cp /path/to/LGVSC/07_downstream/timesformer/get_video_lable.py .

python tools/run_net.py --cfg configs/Kinetics/TimeSformer_divST_8x32_224_TEST.yaml
```
The provided `TimeSformer_divST_8x32_224_TEST.yaml` is our test config; the patch
adapts the Kinetics dataloader / test loop. `get_video_lable.py` builds the label list.

## Depth estimation — Depth-Anything-V2 (`depth_anything_v2/`)  ·  env `depth`

Library is used **unmodified**; only `score.py` is ours.
```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2.git "$DEPTH_DIR"
cp /path/to/LGVSC/07_downstream/depth_anything_v2/score.py "$DEPTH_DIR"/
cd "$DEPTH_DIR"
python score.py <origin_csv> <method_csv>   # compares depth maps: origin vs reconstructed
```

## Video captioning  ·  env `video_summary`

Caption originals and reconstructions with PLLaVA, then score with BERTScore / BLEU /
ROUGE. Drivers: `video_summary.sh` (captioning) and `video_depth.sh` (depth pipeline
wrapper). Both take `$DATA_ROOT`/method as arguments and resolve tools from
`$OPENSORA_DIR` / `$DEPTH_DIR` — `source env.sh` first; no manual path edits needed.
