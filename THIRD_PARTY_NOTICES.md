# Third-party notices

LGVSC builds on several external models, libraries, and datasets. They are **not**
vendored into this repository — we ship only our own scripts and `.patch` files (see the
runtime requirements in `docs/RUN_GUIDE.md`). When you obtain and run the upstream components, you do so
under **their** licenses and terms, summarized below.

The license names are recorded for convenience; the upstream `LICENSE` file is always the
authoritative source. Verify before any redistribution or commercial use.

## Software / models

Current SPARC uses Qwen semantic selection, DCVC-RT and Practical-RIFE4.25. The separate learned-selector experiments use frozen ResNet50 V2 features and DCVC-UF/RIFE; completed training does not automatically replace the default. LTX/Open-Sora entries refer to retained generative experiments. Component roles and evidence limits are documented in [MODEL_ARCHITECTURE](docs/MODEL_ARCHITECTURE.md) and [RESULTS](docs/RESULTS.md).

| Component | Repository | License (as observed) | Notes |
|---|---|---|---|
| InternVL | https://github.com/OpenGVLab/InternVL | MIT | InternVL2-8B weights carry their own model-card terms. |
| Open-Sora | https://github.com/hpcaitech/Open-Sora | Apache-2.0 | Decoder world model (pinned commit `bf4d6673`). |
| LTX-2.5 22B Distilled | [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2/tree/be6d6d0a69d11a0a399c462ed92606692e5f8697) · [weights](https://huggingface.co/Lightricks/LTX-2.5) | [LTX-2.x Community License](https://github.com/Lightricks/LTX-2/blob/be6d6d0a69d11a0a399c462ed92606692e5f8697/LICENSE-2_x) | Experimental BSU receiver replacement; official gated weights are downloaded separately. |
| NTSCC_JSAC22 | https://github.com/wsxtyrdd/NTSCC_JSAC22 | **No LICENSE file** | Research code with no explicit license — contact the authors before reuse/redistribution. |
| Torchvision ResNet50 V2 | [pytorch/vision v0.17.2](https://github.com/pytorch/vision/tree/v0.17.2) | See upstream LICENSE and pretrained weight terms | Frozen ImageNet feature extractor for experimental SPARC selector v3; official `resnet50-11ad3fa6.pth` is downloaded separately and SHA256 recorded. |
| Sionna | https://github.com/NVlabs/sionna | Apache-2.0 | LDPC / channel simulation. |
| PLLaVA | https://github.com/magic-research/PLLaVA | See upstream repo | Captioning model; check the repo + model card. |
| UniMatch | https://github.com/autonomousvision/unimatch | See upstream repo | Optical flow; bundled inside Open-Sora `tools/scoring`. |
| TimeSformer | https://github.com/facebookresearch/TimeSformer | **CC BY-NC 4.0 (NonCommercial)** | ⚠ Non-commercial only. Used for downstream action-recognition evaluation. |
| Depth-Anything-V2 | https://github.com/DepthAnything/Depth-Anything-V2 | Apache-2.0 (code) | Pretrained depth checkpoints may carry separate terms — check the model card. |

<a id="bsu-predictive-dependencies"></a>
## Adopted BSU predictive components — 2026-10-16

| Component | Pinned upstream source | Observed code license | Role |
|---|---|---|---|
| DCVC-RT | [Microsoft DCVC at cbdae87](https://github.com/microsoft/DCVC/tree/cbdae87a5445114cdc7f48816da63ea80bdeac40) · [license](https://github.com/microsoft/DCVC/blob/cbdae87a5445114cdc7f48816da63ea80bdeac40/LICENSE.txt) | MIT | Initial image and predictive neural entropy coding |
| DCVC-UF HT-S (experimental) | [Microsoft DCVC at cbdae87](https://github.com/microsoft/DCVC/tree/cbdae87a5445114cdc7f48816da63ea80bdeac40) · [license](https://github.com/microsoft/DCVC/blob/cbdae87a5445114cdc7f48816da63ea80bdeac40/LICENSE.txt) | MIT | Training-free 8-frame chunk comparison; official image/HT-S weights are downloaded separately and SHA256-pinned in `sparc_uf.py` |
| Practical-RIFE 4.25 | [Practical-RIFE at bbfd2ea](https://github.com/hzwer/Practical-RIFE/tree/bbfd2ea90910789a860ea3e2b32a240cd577b75e) · [license](https://github.com/hzwer/Practical-RIFE/blob/bbfd2ea90910789a860ea3e2b32a240cd577b75e/LICENSE) | MIT | Interpolation between decoded anchors |

The pinned checkouts' license files were inspected. Source and pretrained assets are obtained separately; neither weights nor these checkouts are redistributed here. Exact inference-weight and RIFE implementation hashes are checked by [DCVC integration](src/semantic_transmission/bsu_dcvc_keys.py) and [RIFE integration](src/semantic_transmission/dense_temporal_receiver.py). Model download terms remain those of their upstream distribution. The historical mixed BSU model retains Open-Sora/NTSCC dependencies for its `combined_context` branch and the existing TimeSformer evaluation terms. The SPARC branch uses the Qwen3.5 selector, DCVC-RT, Practical-RIFE and Sionna; it does not invoke Open-Sora, NTSCC or a caption LoRA. All pretrained assets are obtained separately.

<a id="sparc-framer-dependencies"></a>
## Experimental Framer receiver — 2026-10-20

| Component | Pinned source / assets | Terms | Role |
|---|---|---|---|
| Framer | [aim-uofa/Framer at d0ba81f](https://github.com/aim-uofa/Framer/tree/d0ba81f1f23a639396d4c6a801c53a5b21412b06) · [weights at 023bd84](https://huggingface.co/wwen1997/framer_512x320/tree/023bd84da79faada5f27f3bafe7d8e5c8120a61f) | [Upstream LICENSE](https://github.com/aim-uofa/Framer/blob/d0ba81f1f23a639396d4c6a801c53a5b21412b06/LICENSE) and model distribution terms | Trajectory-conditioned interpolation using the public UNet and ControlNet |
| Stable Video Diffusion XT | [stabilityai model at 9e43909](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/tree/9e43909513c6714f1bc78bcb44d96e733cd242aa) | See upstream model card and license files | Image encoder, temporal VAE, preprocessing and scheduler |

The [setup script](scripts/setup_sparc_framer.sh) downloads these assets separately into ignored local directories and records their hashes. The repository does not redistribute their source checkouts or weights. The [compatibility helper](src/semantic_transmission/framer_compat.py) corrects the input feed-forward layer used by diffusers 0.24 temporal chunking in process; installed dependencies, upstream checkout files and public checkpoint tensors remain unchanged. The [numerical check](scripts/check_framer_chunking.py) compares chunked and unchunked outputs and input gradients.

## Datasets

| Dataset | Source | Terms (summary) |
|---|---|---|
| WebVid | https://github.com/m-bain/webvid | Research-only; clips link to third-party content under the dataset's stated terms. |
| LAVIB | [Official repository](https://github.com/alexandrosstergiou/LAVIB) · [official dataset](https://huggingface.co/datasets/astergiou/LAVIB) | CC BY-NC-SA 4.0. Downloaded separately; pinned upstream revision, original member names and media hashes retained in the training run. Custom source-grouped subset, not the official benchmark split. |
| Kinetics-400 | https://github.com/cvdfoundation/kinetics-dataset | Annotations under CC BY 4.0 (DeepMind); the underlying videos belong to their original uploaders. |
| OpenImages | https://storage.googleapis.com/openimages/web/index.html | Annotations under CC BY 4.0; images under their individual (mostly CC BY 2.0) licenses. |

> We do **not** redistribute any dataset media or model weights in this repository. The
> bundled smoke-test sample under `assets/` is our own minimal example (see `docs/DATA.md`).

<a id="lgvsc-upstream"></a>
## LGVSC upstream attribution

This repository derives from [TT2TER/LGVSC](https://github.com/TT2TER/LGVSC). Original copyright and the repository LICENSE are retained. FC-LGVSC additions, local experiments and limitations are identified separately in README, MODEL_ARCHITECTURE and RESULTS. Original LGVSC reported results are not FC-LGVSC results. The upstream citation is retained below.



```bibtex
@misc{ma2026lgvsclargemodeldrivengenerativevideo,
      title={LGVSC: A Large-Model-Driven Generative Video Semantic Communication Framework},
      author={Yu Ma and Hang Yin and Li Qiao and Shuo Sun and Zhen Gao and Yin Xu and Wenjun Zhang},
      year={2026},
      eprint={2606.12899},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2606.12899},
}
```


<a id="ntscc-upstream"></a>
## NTSCC upstream citation and acknowledgements

Original source: https://github.com/wsxtyrdd/NTSCC_JSAC22 . The previously bundled README attribution is retained below; usage requirements are in RUN_GUIDE.


If you find the code helpful in your research or work, please cite:
```
@ARTICLE{9791398,
  author={Dai, Jincheng and Wang, Sixian and Tan, Kailin and Si, Zhongwei and Qin, Xiaoqi and Niu, Kai and Zhang, Ping},
  journal={IEEE Journal on Selected Areas in Communications},
  title={Nonlinear Transform Source-Channel Coding for Semantic Communications},
  year={2022},
  volume={40},
  number={8},
  pages={2300-2316},
  doi={10.1109/JSAC.2022.3180802}
  }
```

## Acknowledgements
The NTSCC model is partially built upon the [Swin Transformer](https://github.com/microsoft/Swin-Transformer) and [CompressAI](https://github.com/InterDigitalInc/CompressAI/). We thank the authors for sharing their code.



## Additional FC-LGVSC dependencies

- Qwen3.5-9B: https://huggingface.co/Qwen/Qwen3.5-9B ; pinned revision and hashes are recorded in code/run manifests. The base model is obtained separately under its model terms.
- PEFT and bitsandbytes: used for local LoRA/NF4 inference and training; versions are pinned in environment requirements.
- TransNetV2: https://github.com/soCzech/TransNetV2 ; source/weight hashes are recorded in extraction manifests.
- TVSum and ClipShots inputs are obtained separately. Raw dataset media, training annotations and base model weights are not included in this code release. The LoRA handoff tool packages only the fixed adapter; it does not grant additional rights to source data or base weights.

The license names in the historical table above are provenance notes, not a fresh audit of every current model/weight distribution. Consult the actual upstream notices when obtaining artifacts.
