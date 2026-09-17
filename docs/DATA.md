# Data sources & how to obtain them

This computer's SSD layout, acquisition status reports, and verification commands
are documented in [LOCAL_DATASETS.md](LOCAL_DATASETS.md).

LGVSC uses **public** datasets only. We do not redistribute the full datasets; below is
exactly what we used and where to get it, so a third party can reproduce our results.

## 1. Main test set — WebVid

- **What we used:** 55 video clips randomly sampled from the **WebVid** dataset
  (Shutterstock stock-footage clips with captions).
- **Source:** WebVid (Bain et al., *Frozen in Time*, ICCV 2021) —
  https://github.com/m-bain/webvid . Clips are public stock footage; each has a numeric
  id and a `stock-footage-...` slug used as the filename throughout this repo.
- **Preprocessing:** every clip is normalized by `01_data_prep/get_prepare.sh` to
  **576×320, 24 fps, ≤16 s** (see `docs/CODE_WALKTHROUGH.md`, Stage 01).
- **Selection:** the 55 clips were drawn at random (no curation by content or by result).
- **Fixed manifest:** the exact 55 clips are listed in
  [`docs/eval_manifest_webvid.csv`](eval_manifest_webvid.csv) — one row per clip with its
  `stock-footage-...` slug filename and the SHA-256 of the source `.mp4` we used. After you
  obtain the clips from WebVid, verify with `sha256sum -c` (build a checklist from the CSV)
  to confirm you have byte-identical sources before reproducing the table numbers.
- **Caveat:** WebVid availability changes over time, so some clips may no longer be
  downloadable. The reported trends are not sensitive to the exact subset — any random
  WebVid subset of similar size reproduces them — but the manifest lets you reproduce the
  *exact* numbers when the sources are available.

## 2. Action-recognition set — Kinetics-400

- **What we used:** 14 clips randomly sampled from **Kinetics-400**.
- **Source:** https://github.com/cvdfoundation/kinetics-dataset (official mirror) or
  the DeepMind Kinetics page. Used only by `07_downstream/timesformer/`.
- **Fixed manifest:** the exact 14 clips are listed in
  [`docs/eval_manifest_kinetics.csv`](eval_manifest_kinetics.csv) — one row per clip with
  its Kinetics YouTube-id + time-span filename, the ground-truth `kinetics_label_id`, and
  the SHA-256 of the source `.mp4` we used.

## 3. NTSCC training data — OpenImages frames

- **What we used:** ~100k still frames to train the NTSCC keyframe codec
  (lr 1e-4, batch 64-ish, 100 epochs; see paper / `03_jscc_transmission/ntscc/`).
- **Source:** **OpenImages** (boxable subset) — https://storage.googleapis.com/openimages/web/index.html .
  The upstream NTSCC repo (https://github.com/wsxtyrdd/NTSCC_JSAC22) documents the
  training data layout; our `config.py` `train_data_dir`/`test_data_dir` point there.
- The released checkpoints `ntscc_hyperprior_quality_{1..4}_psnr.pth` are quality levels;
  the paper's main results use quality 4 (SNR=10). SNR 0–8 used separately-trained
  weights (not in this repo — see `docs/CODE_WALKTHROUGH.md` provenance note).

## 4. Bundled smoke-test sample

For a quick installation check we bundle **one** WebVid clip in the repo:

```
assets/sample/stock-footage-aerial-drone-panoramic-video-of-famous-port-and-marina-of-zea-with-luxury-yachts-docked-as-seen.mp4
```
(raw 596×336, 23.976 fps, 20.6 s — the un-normalized original.) See `docs/SMOKE_TEST.md`
for the exact commands and the outputs we observed. This single clip is sufficient to
exercise stages 01 → 02 → 03 end-to-end without downloading any dataset.

> **License note:** WebVid / Kinetics / OpenImages each carry their own terms. The
> bundled sample is included solely for reproducibility testing of this code; obtain the
> full datasets from their official sources under their respective licenses.
