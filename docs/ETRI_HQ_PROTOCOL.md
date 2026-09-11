# ETRI 10-video reconstruction and traffic comparison

This is an ETRI development-set application of the official LGVSC SKEM+DSA
architecture. It is not a reproduction of the paper's 55-video WebVid benchmark or
of the authors' unavailable NTSCC training checkpoint collection.

## Frozen inputs and model settings

- The existing ten processed ETRI videos: 512x256, 10 FPS, 100 frames, 10 seconds.
- Copy the original MP4 byte-for-byte; no resizing, resampling, clipping or H.264
  recompression before semantic extraction. Input hashes are frozen in the batch manifest.
- InternVL2-8B, paired-image two-round PSSS, every source frame, threshold 0.35,
  up to 12 tiles/image and 1,024 output tokens, mandatory first/final keyframes.
- InternVL transformer/vision weights retain BF16 precision. Vocabulary tables
  execute on CPU in FP32 using the original checkpoint values; only the final
  vocabulary projection is needed for autoregressive generation. Vision weights
  are staged per frame pair; image features are reused between its two rounds.
  Token-independent FFNs are evaluated in chunks; existing rotary-cache values
  are sliced to the needed initial capacity and grow on demand. No weight quantization.
- PLLaVA-7B, original Open-Sora caption prompt/examples, centered four-frame sampling,
  greedy generation, 256 output tokens, original LoRA tensors, SDPA and CPU offload.
- UniMatch, original offsets [0,10,20,30] clipped to each half-open semantic segment,
  original 320x576 preprocessing and mean-absolute flow score.
- Public quality-4 NTSCC checkpoint, lambda 64, eta 0.2, AWGN at 10 dB. Its optional
  hyperprior refinement is disabled, as in the pinned public configuration.
- Open-Sora v1.2 STDiT/VAE BF16, T5 CPU FP32, 50 sampling steps, CFG 7.0, seed 2025.
  The upstream inference example uses 30 steps; 50 is this quality-prioritized
  experiment's setting, not a number specified by the paper.
- The earlier endpoint-mask and temporal-stitching corrections remain enabled.

## Exact boundary accounting

`transmitter/metadata.bin` contains a CRC-framed JSON header plus packed 4-bit
NTSCC rate indices. The header carries video dimensions/FPS/frame count, keyframe
positions, per-frame channel lengths and normalization power, decoder seed/steps,
codec fingerprint, per-segment captions and scalar motion scores.

`transmitter/visual.c64` contains the actual normalized NTSCC channel symbols as
interleaved FP32 I/Q values. These are continuous-amplitude JSCC symbols. The file
size (8 bytes/complex symbol) is exact replay-storage size, **not an RF bitstream**.

The channel stage reloads both files, transmits metadata with Sionna
LDPC(6144,9216)/16QAM/AWGN and applies unit-power complex AWGN to the visual symbols.
It records padding, coded bits, both complex-channel-use counts and
`CBR = uses / (3 * height * width * frames)`. Visual AWGN uses a recorded PCG64 seed;
its law matches upstream NTSCC, while its individual noise realization differs.

The receiver reads only `received/{metadata.bin,visual.c64}` plus preinstalled
model weights. It rebuilds NTSCC rate masks from the received indices, restores
channel scale from the received normalization value, decodes keyframes, and passes
received captions/motion/positions to Open-Sora. Source pixels and sender-local
feature tensors are not receiver inputs. Keyframe PNGs produced by NTSCC at the
receiver are not counted again as transmitted data.

All sample-dependent model inputs are accounted for. Shared checkpoints, fixed
architectures, PHY preambles, pilots, MAC/IP headers and retransmission overhead
are excluded explicitly. Do not call this a complete physical-network measurement.

## Historical comparison

The preserved SGD-JSCC integrated experiment supplies three existing conditions:
`full50 + baseline`, `full50 + candidate_both_omit`, and
`few10 + candidate_both_omit`, all using fixed_int4 at the 10dB decoder reference.
The inventory verifies their actual `.sgbundle` byte totals and hashes, and proves
all ten historical source-frame sequences match the current MP4 inputs exactly.

Recompute PSNR, Gaussian-window SSIM and LPIPS-Alex with one evaluator on source
RGB uint8 and lossless reconstructed PNGs. Report a second set on decoded delivered
MP4s, preserving the distinction from video encoding loss. Do not compare new
metrics directly with differently implemented historical metric columns.

Report digital packet bytes, continuous complex symbols, and serialized model-input
bytes separately. A shared LDPC/16QAM conversion of SGD's video-packed bytes is an
explicit counterfactual channel-use estimate, not a measurement from its historical
reliable-digital run. This is not matched-rate testing. Historical RTX 4090 time and
new RTX 4080 time are not a hardware-controlled speed comparison.

## Commands

```bash
bash scripts/bootstrap_hq.sh
source scripts/activate.sh
python scripts/audit_sgd_baselines.py --sgd-repo ../sgdjscc_lab \
  --output .local/validation/etri_sgd_baseline_inventory.json
python -m semantic_transmission.research \
  --input-dir ../sgdjscc_lab/data/etri_video_eval/processed \
  --output outputs/etri10_hq_new
python -m semantic_transmission.compare_etri \
  --run-root outputs/etri10_hq_new \
  --baseline-inventory .local/validation/etri_sgd_baseline_inventory.json \
  --output outputs/etri10_hq_new/comparison
python scripts/render_etri_comparison.py \
  --run-root outputs/etri10_hq_new \
  --baseline-inventory .local/validation/etri_sgd_baseline_inventory.json \
  --comparison outputs/etri10_hq_new/comparison
```

Runs use new output directories and stop at a failing stage. `--reuse-completed-from`
can copy entirely successful videos into a new batch after checking the profile,
source identity, all stage return codes, reconstruction hash and transmitted-file
hashes. An interrupted parent batch may supply its completed videos; partial videos
are regenerated from their first stage. Each copied record retains `reused_from`
and `execution_code`, so copying is never presented as new model execution.
The one-command local entrypoint is `bash scripts/run_etri_remaining.sh`.
Focused probes and
failed attempts remain separate from the frozen full batch. Raw ETRI videos,
receiver videos and model weights remain local, outside Git.

## Preflight evidence (2026-09-11)

- FlashAttention 2.5.9.post1 passed an actual BF16 CUDA invocation in the pinned
  PyTorch 2.2.2 environment; dependency checks and 14 runtime tests passed.
- Optimized BF16 InternVL processed two selected frame pairs without OOM, around
  19 seconds/pair. This is a memory probe, not a complete 99-pair SKEM evaluation.
- NTSCC encode/packed-rate/decode without channel noise matched the original
  model forward with maximum pixel error **0.0**, on a 512x256 source frame.
- The 100-frame transport probe transmitted two keyframes and a metadata packet
  through 10dB AWGN. LDPC metadata had zero bit errors. Open-Sora generated
  100 frames at 512x256, 10 FPS with 50 steps in 117.6 seconds including process
  initialization. Both lossless-frame and delivered-video quality evaluation passed.
  Its keyframe selection used only two comparison pairs; it is not a full-study result.
- A fresh receiver directory containing only the two received files and an empty
  run configuration reproduced both decoded keyframe PNGs byte-for-byte. It had
  no source images, sender metadata, keyframe list or normalization side channel.
- The read-only historical audit verified 30 packet sets and their corresponding
  original frame sequences against all ten current ETRI source videos.

Evidence is retained locally under `.local/validation/` in
`ntscc_transport_parity.json`, `etri_bf16_flash_preflight_v5/`,
`receiver_only_v1/`, and `etri_sgd_baseline_inventory_v2.json`.

## Supplementary CLIP comparison and visual review

The reconstruction settings remain frozen. A separate post-processing script adds
the original evaluator's CLIP ViT-B/32 cosine and `(1 + cosine) / 2` scores on
lossless RGB frames. It uses the standard 224-pixel center crop. These scores
measure broad visual similarity and cannot establish correct motion direction,
small-object presence, readable signs or absence of hallucination.

The existing local `ptest` environment supplies CLIP, PyTorch 2.1.0 and its cached
ViT-B/32 checkpoint. All compared methods use the same CPU FP32 evaluator. The
checkpoint SHA-256 is verified against the original release. No package is added
to the running LGVSC inference environment. The baseline source PNGs are used
only after the audit proves they exactly match the current source-video pixels.

```bash
env -u LD_LIBRARY_PATH -u PYTHONPATH CUDA_VISIBLE_DEVICES=-1 \
  PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  ~/anaconda3/envs/ptest/bin/python scripts/evaluate_clip_reference.py \
  --baseline-inventory .local/validation/etri_sgd_baseline_inventory_v2.json \
  --output outputs/etri_baseline_clip_new
# After all ten LGVSC reconstructions pass:
env -u LD_LIBRARY_PATH -u PYTHONPATH CUDA_VISIBLE_DEVICES=-1 \
  PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  ~/anaconda3/envs/ptest/bin/python scripts/evaluate_clip_reference.py \
  --baseline-inventory .local/validation/etri_sgd_baseline_inventory_v2.json \
  --baseline-cache outputs/etri_baseline_clip_new \
  --run-root outputs/etri10_hq_new --output outputs/etri10_hq_new/comparison/clip
```

The final script verifies cached source and reconstruction pixel hashes before
reusing baseline scores. Regenerate the viewer after CLIP evaluation to show these
supplementary scores. `comparison/viewer.html` opens locally with video selection,
synchronized play/pause and common frame seeking. It references local videos, so
copying the HTML alone does not produce a self-contained shareable package.
