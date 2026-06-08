# Third-party notices

LGVSC builds on several external models, libraries, and datasets. They are **not**
vendored into this repository — we ship only our own scripts and `.patch` files (see the
dependency table in `README.md`). When you obtain and run the upstream components, you do so
under **their** licenses and terms, summarized below.

The license names are recorded for convenience; the upstream `LICENSE` file is always the
authoritative source. Verify before any redistribution or commercial use.

## Software / models

| Component | Repository | License (as observed) | Notes |
|---|---|---|---|
| InternVL | https://github.com/OpenGVLab/InternVL | MIT | InternVL2-8B weights carry their own model-card terms. |
| Open-Sora | https://github.com/hpcaitech/Open-Sora | Apache-2.0 | Decoder world model (pinned commit `bf4d6673`). |
| NTSCC_JSAC22 | https://github.com/wsxtyrdd/NTSCC_JSAC22 | **No LICENSE file** | Research code with no explicit license — contact the authors before reuse/redistribution. |
| Sionna | https://github.com/NVlabs/sionna | Apache-2.0 | LDPC / channel simulation. |
| PLLaVA | https://github.com/magic-research/PLLaVA | See upstream repo | Captioning model; check the repo + model card. |
| UniMatch | https://github.com/autonomousvision/unimatch | See upstream repo | Optical flow; bundled inside Open-Sora `tools/scoring`. |
| TimeSformer | https://github.com/facebookresearch/TimeSformer | **CC BY-NC 4.0 (NonCommercial)** | ⚠ Non-commercial only. Used for downstream action-recognition evaluation. |
| Depth-Anything-V2 | https://github.com/DepthAnything/Depth-Anything-V2 | Apache-2.0 (code) | Pretrained depth checkpoints may carry separate terms — check the model card. |

## Datasets

| Dataset | Source | Terms (summary) |
|---|---|---|
| WebVid | https://github.com/m-bain/webvid | Research-only; clips link to third-party content under the dataset's stated terms. |
| Kinetics-400 | https://github.com/cvdfoundation/kinetics-dataset | Annotations under CC BY 4.0 (DeepMind); the underlying videos belong to their original uploaders. |
| OpenImages | https://storage.googleapis.com/openimages/web/index.html | Annotations under CC BY 4.0; images under their individual (mostly CC BY 2.0) licenses. |

> We do **not** redistribute any dataset media or model weights in this repository. The
> bundled smoke-test sample under `assets/` is our own minimal example (see `docs/DATA.md`).
