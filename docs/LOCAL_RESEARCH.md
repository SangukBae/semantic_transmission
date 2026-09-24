# Local LGVSC research environment

**ETRI follow-up scope (2026-09-21):** use this document for local setup and
smoke execution. The [email summary](ETRI_FOLLOWUP_EMAIL_SUMMARY.md) and
[follow-up protocol](ETRI_FOLLOWUP_PROTOCOL.md) govern current research:
AWGN hallucination detection/mitigation and longer continuous-video evaluation.
Short smoke runs below remain installation checks; they do not establish
60-second support, three-category coverage or mitigation success.

This repository preserves the history and staged implementation of
[TT2TER/LGVSC](https://github.com/TT2TER/LGVSC) at
`50c9ff98fb1aaa8136aef2e41e02bbb0d1a2d1c3` and adds an executable local research path.
The original license and paper citation remain applicable to the upstream work.

## Environment

- Tested machine: NVIDIA RTX 4080, 16 GB VRAM; 64 GB host RAM; Linux.
- `lgvsc`: Python 3.10, PyTorch 2.2.2/CUDA 12.1, Open-Sora v1.2, InternVL2-8B,
  PLLaVA-7B, UniMatch, NTSCC, and frame quality evaluation.
- `lgvsc-channel`: Python 3.10, TensorFlow 2.15.1 and Sionna 0.19.2, running on CPU.
- No existing research Conda environment or ROS container is modified.
- The launcher clears inherited `PYTHONPATH` and `LD_LIBRARY_PATH` in child processes:
  system ROS packages and CUDA 11.x libraries otherwise override the pinned environment.
- FlashAttention/Apex builds are unnecessary for this profile. Open-Sora uses its
  non-fused kernels and xFormers; PLLaVA uses PyTorch SDPA through a recorded patch.
- T5 uses CPU FP32 because the i7-14700F lacks hardware BF16 acceleration. Only its
  small conditioning tensors move to the GPU in BF16; VAE/STDiT remain on the GPU.
- Decord 0.6.0's published `py3-none` wheel has an inconsistent internal `cp36-cp36m`
  tag. `repair_decord_wheel.py` corrects WHEEL/RECORD only, preserving the original
  Python and native-library bytes, so modern pip can validate the installation.

```bash
bash scripts/bootstrap.sh
source scripts/activate.sh
semtx doctor
semtx smoke --output outputs/my_first_skem
```

The bootstrap script uses an existing Conda installation (`CONDA_BASE`, default
`~/anaconda3`) and ffmpeg. External repositories live in ignored `.local/vendor/`;
large weights use the shared Hugging Face cache and ignored `.local/checkpoints/`.
Model revisions and auxiliary checkpoint SHA-256 values are explicitly pinned.
The validated transitive package versions in `environment/locks/` constrain later
bootstrap runs; the editable project itself is installed from this checkout.

## Real-model execution

`semtx smoke` runs these stages in separate processes, releasing each model before
the next stage loads:

1. Normalize the bundled clip, decode the requested number of frames, retain both endpoints.
2. SKEM/PSSS through the original InternVL two-round comparison, or optional SKIM.
3. PLLaVA captions using all original checkpoint tensors, including LoRA (`r=128`, alpha 4).
4. UniMatch optical flow, reduced to the upstream mean absolute motion score.
5. Actual NTSCC keyframe encoding, AWGN transmission and reconstruction at 10 dB.
6. Serialize captions, motion scores and relative segment paths; transmit through
   LDPC(6144,9216), 16-QAM and AWGN; reject framing/CRC failures before decoding.
7. Open-Sora v1.2 with adjacent received keyframes, received captions/motion scores,
   and segment-length-dependent VAE dimensions. T5 runs on CPU.
8. Decode the final MP4 and require identical frame count and image dimensions;
   compute source-paired PSNR and SSIM.

Input example:

```bash
semtx smoke --input /path/to/video.mp4 --output outputs/custom_skem \
  --frames 33 --width 256 --height 256 --steps 10
semtx smoke --selector skim --skim-keyframes 3 \
  --frames 33 --output outputs/skim_multisegment
```

Output directories must be new. A failed attempt retains its logs and a `FAILED`
manifest; it is never silently resumed or promoted to a successful experiment.
`run_manifest.json`, `run_config.json`, per-stage logs, `metadata_channel.json`,
`ntscc.json`, `quality.json`, and `reconstruction/*.mp4` are the main artifacts.

## Scope and differences from the paper

The default is a **local execution smoke profile**, not a reproduction of paper metrics:
17 frames, 256x256, 10 sampling steps, 1 InternVL tile/image, 128 generated tokens,
int8 InternVL weights, PLLaVA CPU offload, a concise caption prompt, and four
uniformly sampled source frames for captioning/flow. These choices are recorded.
The original numbered scripts and [upstream instructions](../README_UPSTREAM.md)
remain available for developing a separately frozen paper-reproduction protocol.

Decoder corrections are explicit: the local profile disables block-of-five mask
alignment, which otherwise maps the last keyframe of a 5-latent clip onto its first
position. Adjacent segments retain both endpoints and discard the shared endpoint
when stitching, so the final frame count equals the source. Conditioning uses the
last 17 generated frames; a shorter prior segment repeats its first frame on the
left to fill the overlap. These changes need separate ablations for a paper claim.

The upstream README's NTSCC `411d3933` and InternVL `a20b7301` revisions could not be
resolved. This fork records accessible source revisions in `configs/upstreams.json`;
the NTSCC compatibility patch applies cleanly. These replacement revisions must
not be described as the authors' original environment.

Only the public NTSCC 10 dB quality-4 checkpoint is supported by this launcher.
The authors' 0–8 dB checkpoints and DVST implementation are not publicly included.

Metadata channel uses include the frame header, checksum and final LDPC padding.
NTSCC visual channel uses are measured from the actual channel call. However,
NTSCC's internal rate-index/mask metadata is still shared inside its upstream
encoder/decoder rather than serialized through an actual side channel. Accordingly,
`complete_wire_accounting=false`; do not claim complete system CBR from the visual
channel count alone. Keyframe indices, image dimensions and decoding settings are
also shared through this local simulation's files/configuration. Motion is a scalar prompt condition, not a spatial flow tensor
injected into the diffusion network.

A successful smoke run establishes installation and wiring only. It does not establish
research gains, faithful reproduction of the paper, training readiness, multi-video
generalization, robustness across SNR, or real-time performance.

## Development

- Reusable runtime and packet contracts: `src/semantic_transmission/`.
- Model boundaries and isolated stage calls: `workers.py`.
- Local decoder settings: `configs/rtx4080_opensora.py`.
- Pinned source/model manifests: `configs/upstreams.json`, `configs/models.json`.
- External source changes: small, reviewable files under `patches/`.
- CPU regressions: `python -m unittest discover -s tests -v`.
- Original baselines/downstream scripts remain under `05_baselines/` and
  `07_downstream/`; the local smoke profile does not install or validate all optional
  downstream models.

Keep new research modules behind explicit configurations, compare against a frozen
upstream-derived baseline, and record quality, semantic reliability, channel uses and
latency separately. Existing `sgdjscc_lab` results and code remain separate.
