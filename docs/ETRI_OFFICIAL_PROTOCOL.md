# ETRI: published LGVSC execution profile

This profile follows the published **SKEM + DSA code path**, pinned to
[TT2TER/LGVSC 50c9ff98](https://github.com/TT2TER/LGVSC/tree/50c9ff98fb1aaa8136aef2e41e02bbb0d1a2d1c3).
It supersedes the previous HQ profile for `scripts/run_etri_remaining.sh`.
It is an ETRI application of the release, **not a reproduction of the paper's
WebVid/Kinetics aggregate results or a claim of identical generated pixels**.

## Settings

| Stage | Published path implemented here |
| --- | --- |
| Input | Original ETRI 512×256, 10 fps, 100 frames; preserve the original files |
| Normalize | `edit_movie.py`: MoviePy H.264 export at 24 fps, maximum 16 s; then `resize_videoframe.sh`: FFmpeg H.264 CRF 18, preset medium, 576×320. ETRI inputs become 240 frames |
| SKEM | InternVL2-8B BF16; all 239 successive candidates; original two prompts, max 1024 tokens, 12 image tiles, greedy generation, PSSS `P(No)-P(Yes)>0.35`, mandatory first/last keyframes |
| Semantic clips | MoviePy `ffmpeg_extract_subclip`, stream copy between selected keyframe timestamps, exactly as `get_key_video.py` |
| Caption | PLLaVA-7B checkpoint and LoRA (r=128, alpha=4); four centered frames from each actual decoded subclip; short-edge resize 672 before processor; original conversation and 256-token greedy generation; FlashAttention 2 |
| Flow | UniMatch; actual subclip frames 0/10/20/30 (upstream clamps short clips); 320×576 bilinear input; original three-pair batch and mean absolute scalar |
| Visual channel | Released NTSCC quality-4 PSNR checkpoint, lambda 64, eta .2, no side-info refinement, 10 dB AWGN; released CUDA noise function, seed 1024 |
| Digital channel | Sionna .19.2, LDPC (6144,9216), 16-QAM, AWGN at 10 dB; explicit framed metadata including all per-video decoder inputs |
| Generation | Pinned Open-Sora v1.2, STDiT3-XL/2, original VAE, T5 FP32; 30 sampling steps, seed 42, CFG 7, aesthetic score 6.5, timestep transform, BF16 diffusion/VAE, FlashAttention, Apex fused LayerNorm |
| DSA | Original variable segment lengths, align=5, condition-frame length=5, full previous generated clip as overlap reference, original 17-frame trimming |
| Output | Native 576×320 at 24 fps plus pre-MP4 uint8 PNG frames; no motion captions or intermediate keyframes added by hand |

The previous profile kept 512×256/10 fps, used seed 2025 and 50 steps, omitted
the caption's segment-MP4/672-resize preprocessing, used SDPA caption attention,
CPU FP32 SKEM vocabulary projection, NumPy visual noise, disabled fused LayerNorm,
and used locally corrected overlap/alignment rules. Its artifacts remain under
`outputs/etri10_hq_*` and cannot be reused by the new profile.

## Release inconsistencies and hardware adaptations

- `run.bash` specifies resolution `576`, which is absent from the pinned Open-Sora
  aspect lookup table. The explicit `(320,576)` image size supplies the published
  input geometry without inventing a different resolution preset.
- Original concatenation retains the shared endpoint of adjacent segments.
  With K selected keyframes, output length is `240 + K - 2`. We preserve that
  behavior and record `quality.json/output_source_indices`, repeating the reference
  endpoints for paired metrics. The original 10-second source is never overwritten.
- Official `align=5` can round a last-keyframe latent to an earlier position. This
  profile retains the released behavior rather than silently applying the old
  local correction. It is not evidence that the paper intended every edge case.
- A 16 GB RTX 4080 cannot hold InternVL's full GPU working set. Vision tiles and
  feed-forward token chunks run sequentially in BF16; embedding lookup stays in
  CPU BF16; the BF16 vocabulary projection runs on CUDA in independent 8192-row blocks.
  One transformer layer retains its weights on CPU and stages them to CUDA for each
  forward pass. This preserves GPU BF16 arithmetic while leaving room for long
  second-round prompts. An expandable CUDA allocator limits fragmentation.
  PLLaVA uses sequential CPU offload; T5 runs in its upstream default FP32 on CPU.
  No int8/int4 weight quantization is used. Kernel/library/device placement changes
  can affect numerical rounding and SKEM decisions.
  Streaming vocabulary rows changes the GEMM shape and can change BF16 rounding;
  the numerical probe artifacts explicitly do not claim bitwise equivalence.
- InternVL has its own environment matching the release's inference versions:
  Python 3.9.19, Torch 2.4.1+cu121, Transformers 4.37.2, FlashAttention 2.6.3,
  NumPy 2.0.2, Accelerate .34.2. Other stages retain the working core runtime
  (Python 3.10 / Torch 2.2.2 / Transformers 4.39.3 / FlashAttention 2.5.9.post1)
  and isolated Sionna environment. Exact original source revisions for
  InternVL `a20b7301` and NTSCC `411d3933` were unavailable; see pinned replacements
  in `configs/upstreams.json` and the prior reproducibility notes.
- The release lists Apex 0.1 without its source revision. We pin compatible Apex
  `4138d31ff0acf4071d1dc001ccb7cd6e00800324` (24.04.01), built for CUDA 12.1 / SM89.
  MoviePy 1.0.3 uses decorator 4.4.2, resolving the exported environment's conflicting
  decorator requirements. FFmpeg versions are recorded with the run environment.
- The released 10 dB NTSCC README explicitly uses the public quality-4 checkpoint.
  Other SNR-specific trained weights and all original experiment random states are
  not available. Metadata framing/CRC and explicit sender/receiver boundaries are
  local measurement infrastructure; recorded complex64 file bytes are not RF bits.

The release can still omit a turn or generate an inconsistent pose. Four caption
frames and one directionless flow scalar do not fully specify a trajectory. Full
pipeline completion and image-quality metrics do not establish action fidelity.

## Commands and evidence

From the parent `Semantic` directory:

```bash
bash semantic_transmission/scripts/run_etri_remaining.sh --dry-run
bash semantic_transmission/scripts/run_etri_remaining.sh
```

The wrapper uses only `run_history_by_profile/etri_lgvsc_official_release_v1` in
`.local/etri_continue.json`. Old HQ history remains intact. Completed videos are
reused only after source/config/payload/video/reference/timeline verification.
One selected video can be regenerated in a fresh root with:

```bash
python -m semantic_transmission.research \
  --profile configs/etri_official.json \
  --input-dir ../sgdjscc_lab/data/etri_video_eval/processed \
  --video 01_person_walk --output outputs/etri01_official_new
```

GPU preflight and the full first-video run are separate evidence. Full-run results
are recorded below only after generation and validation actually finish.
The existing HQ baseline comparator assumes 100 frames at the original geometry;
it must not be used unchanged to claim a matched comparison for this 240-frame
normalized protocol.

The [integration preflight](validation/2026-09-11-official-preflight.json) produced
a 240-frame/10-second video and verified transport/evaluation, using only two SKEM
candidate comparisons. A first full attempt then failed at candidate 4 due to CUDA
memory pressure. The [memory validation](validation/2026-09-11-internvl-memory.json)
records the repair: the failing real pair and two other pairs passed, with identical
PSSS probabilities before/after layer staging. A separate synthetic 1024-token
history plus 1024 generated-token stress also passed. Synthetic stress outputs are
not used as semantic information in any research reconstruction.
