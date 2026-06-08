# Stage 03 — JSCC transmission

Two channels: keyframes go through the learned **NTSCC** codec; captions and
optical-flow side information go through classical **LDPC + modulation** (Sionna).

## NTSCC — keyframe transmission (`ntscc/`)  ·  env `ntscc`

Built on [wsxtyrdd/NTSCC_JSAC22](https://github.com/wsxtyrdd/NTSCC_JSAC22).

### Setup
```bash
git clone https://github.com/wsxtyrdd/NTSCC_JSAC22.git "$NTSCC_DIR"
cd "$NTSCC_DIR"
git apply /path/to/LGVSC/03_jscc_transmission/ntscc/patches/ntscc_modifications.patch
cp /path/to/LGVSC/03_jscc_transmission/ntscc/main_save.py .
```
The patch covers `config.py`, `main.py`, `net/NTSCC_Hyperior.py`, `requirements.txt`.
`main_save.py` is our inference/saving entry point (new file).

### Weights — what ships vs. what doesn't

> **`SNR=10` is the runnable default.** Only its weight is publicly available; the
> sub-10 dB points require **author-provided weights** that are *not* part of this release.

We trained on 100k frames (lr 1e-4, batch 64, 100 epochs). The checkpoints are **not** in
git (~132 MB each). The `--model` value selects the weight (`model_dict` in `main_save.py`):

| `--model` | Weight | Availability | Resolved from |
|---|---|---|---|
| `10` | `ntscc_hyperprior_quality_4_psnr.pth` | **Released** (NTSCC upstream quality-4) | `$NTSCC_CKPT` |
| `8` / `6` / `4` / `2` / `0` | `SNR=<n>dB_2_0.2_train/.../best_loss_<n>dB.model` | **Not released** (separately trained) | `$NTSCC_MAYU` |

To run a sub-10 dB point you must supply the matching weight yourself and point
`$NTSCC_MAYU` at it (or edit `model_dict`). Out of the box, run with `--model 10`.
See `environment/README.md` for where to place `$NTSCC_CKPT`, and `docs/CODE_WALKTHROUGH.md`
for the provenance note.

### Run
```bash
#   --model: SNR dB in {0,2,4,6,8,10} -> selects the matching checkpoint
#   --save_frames: bare flag; include to write frames, omit to only compute CBR
python main_save.py -p test \
    --test_path "$DATA_ROOT/frames" \
    --method    key_framesinternvl_diff_0.35 \
    --model     10 \
    --save_frames
```
Writes corrupted-then-reconstructed keyframes to `frames/<video>/<method>_<snr>/`
and logs per-video CBR to `<method>.csv`.

> `config.py` (via the patch) points `train_data_dir`/`test_data_dir` at absolute
> paths — edit them, or override, for your machine. Only needed for (re)training.

## LDPC — text / side-info transmission (`ldpc_sionna/`)  ·  Sionna 0.19.2

`SionnaPlus_sora.py` runs the bitstream of `I_text` + `I_side` through LDPC coding and
modulation over AWGN, inside the Sionna Docker image. No source patch to Sionna is required:

```bash
pip install sionna==0.19.2   # or use the official Sionna Docker image
source env.sh                # exports DATA_ROOT (the --seq_list default)
python SionnaPlus_sora.py --snr 10 --seq_list "$DATA_ROOT/H264_test_10"
```
