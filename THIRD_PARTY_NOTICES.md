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
