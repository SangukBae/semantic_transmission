# Local GPU validation — 2026-09-10

All three real-model runs completed successfully on the user's RTX 4080 (16 GB),
Intel i7-14700F and 64 GB RAM, using NVIDIA driver 580.173.02.
The executed source was clean commit `39a82f3dd01643e6d3fd34d6fe535f9a1b3ea1cd`.
Later documentation commits record these results without changing that runtime.

**These are installation and integration checks on one bundled clip, not paper
reproduction, a performance benchmark, or evidence of an ETRI research contribution.**

## Real pipeline results

All runs used 256x256, 24 FPS, 10 Open-Sora sampling steps, 10 dB AWGN and seed 1024.
Each completed preparation, keyframe selection, PLLaVA captioning, UniMatch flow,
NTSCC transmission, Sionna LDPC/QAM metadata transmission, Open-Sora reconstruction
and frame-aligned evaluation. SKEM used actual InternVL2-8B int8; SKIM used uniform
selection without an MLLM selector. PLLaVA and the downstream models remained real
in both cases.

| Run under `outputs/` | Keyframe indices | Output frames | PSNR dB | SSIM | Stage time sum | Metadata bytes / complex channel uses |
|---|---|---:|---:|---:|---:|---:|
| `rtx4080_skem_verified_v1` | [0, 16] | 17 | 19.948 | 0.6023 | 95.9 s | 431 / 2304 |
| `rtx4080_skim_33_verified_v1` | [0, 16, 32] | 33 | 19.470 | 0.5664 | 71.2 s | 886 / 4608 |
| `rtx4080_skim_dense_verified_v1` | [0, 8, 16] | 17 | 14.796 | 0.4425 | 69.8 s | 887 / 4608 |

Every output had exactly the source frame count and dimensions. The 33-frame run
validated two adjacent segments; the dense 17-frame run validated short-segment
conditioning and stitching. Metadata bit errors were zero in all three runs.
The lower quality of the dense case also shows why successful execution must not
be interpreted as a quality improvement.

Times include model loading with already-cached weights, CPU offload and per-stage
process startup. They are single observations, not latency distributions or a
real-time claim. PSNR/SSIM compare the decoded generated MP4 against each run's
normalized source MP4; do not use this table as a controlled model comparison.

[Machine-readable evidence](validation/2026-09-10-rtx4080.json) contains source/input
and output hashes, stage status/timing, exact metadata channel accounting, and
NTSCC visual channel counts. Raw logs, full manifests and MP4s remain locally under
the listed output directories; generated videos and model weights are not committed.
NTSCC rate indices and other shared configuration metadata are not fully transported,
so total system wire accounting remains incomplete.

## Other checks

- 11 CPU regression tests passed: Unicode/quoted multiline CSV, damaged packets,
  LDPC final-block padding, grouped decoder input, child failure propagation,
  missing outputs, stale outputs, exact stitched frame counts and short overlap.
- The bootstrap script was rerun successfully with locked dependency constraints;
  both isolated Conda environments passed `pip check`.
- Actual CUDA/cuDNN convolution, core imports and upstream keyframe masking passed.
  The last-keyframe mask now remains at the terminal latent position.
- All 36 tracked Python files parsed and all 11 `.sh`/`.bash` files passed `bash -n`.
- A real metadata transmission at -10 dB produced 1,532 packet-bit errors. The
  receiver rejected the damaged framing, returned failure and wrote no decoder CSV.
  This is an integrity-rejection check, not a statistical robustness measurement.

## Problems fixed during setup

- ROS/system CUDA library contamination of the dedicated model environment.
- PLLaVA's hard-coded FlashAttention requirement and 16 GB memory constraints.
- CPU BF16 T5 inference on a CPU without BF16 acceleration; CPU FP32 is used instead.
- Local Hugging Face snapshot directories being mistaken for ColossalAI checkpoints.
- Resolution-dependent NTSCC buffers mismatching the public checkpoint while
  retaining strict validation of learned parameter keys.
- Silent shell success after decoder failure and broken quoted/multiline CSV splitting.
- Block-aligned masking overwriting the first keyframe with the terminal keyframe.
- Duplicate adjacent endpoints and insufficient conditioning on short segments.

Earlier attempts and focused diagnostic probes are retained under ignored `outputs/`
and `.local/validation/`. Interrupted/failed attempts retain `FAILED` manifests and
were not reused as successful runs. The three runs above each started in a fresh
output directory from the committed runtime.

## Run it again

```bash
source scripts/activate.sh
semtx smoke --output outputs/check_skem_new
semtx smoke --selector skim --skim-keyframes 3 --frames 33 --output outputs/check_skim33_new
semtx smoke --selector skim --skim-keyframes 3 --frames 17 --output outputs/check_dense_new
```

Use a new output name on every invocation. See [LOCAL_RESEARCH.md](LOCAL_RESEARCH.md)
for setup, pinned source/model versions and the remaining paper-reproduction gaps.
