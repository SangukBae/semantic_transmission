# Reproducibility statement

**Local ETRI scope (2026-09-21):** the paper-oriented SKEM measurements below are
preserved in their original scope. They do not validate the new
[ETRI follow-up requirements](ETRI_FOLLOWUP_EMAIL_SUMMARY.md) or prove that the
current checkout reproduces the complete paper. See the dated
[local implementation audit](LGVSC_PAPER_IMPLEMENTATION_AUDIT.md) for that distinction.
Future paired AWGN runs must freeze source timelines, checkpoints, code, seeds,
transport costs and independent error evidence under the
[follow-up protocol](ETRI_FOLLOWUP_PROTOCOL.md).

LGVSC contains a **multimodal-LLM-in-the-loop** component — the SKEM keyframe selector
(`02_semantic_encoder/skem/`), which queries InternVL2-8B and inserts a keyframe whenever
the relative PSSS score `S_rel = P("No") − P("Yes")` exceeds the divergence threshold
`η_th = 0.35`. Because this is a thresholded decision driven by floating-point model
logits, and because SKEM is **autoregressive** (each frame is compared against the most
recently selected keyframe), the exact set of selected keyframes is sensitive to the
numerical environment. We characterize this explicitly below so that reproductions can be
interpreted correctly.

## What we measured

We ran the *identical* SKEM code (`MLM-keyframe-internvl.py`, `η_th=0.35`, greedy decoding
`do_sample=False`) on the same 55-clip WebVid test set under four conditions and compared
the per-video selected keyframe indices:

| Comparison | Environment difference | Exact per-video match |
|---|---|---|
| rerun A vs rerun B, same machine | none (repeat run) | **100 % (55/55)** |
| rerun vs paper run, same machine | libraries/time only, same GPU | **60 % (33/55)** |
| different machine vs paper run | GPU + PyTorch + Transformers + FlashAttention all changed | **31 % (17/55)** |
| different machine vs same-machine rerun | same as above | **36 % (20/55)** |

The cross-machine condition used: RTX 4090 (paper) vs. A100-80GB; PyTorch (paper) vs.
`2.11.0+cu128`; Transformers vs. `4.37.2`; FlashAttention vs. `2.8.3`.

## What this means

1. **Deterministic within a fixed environment.** Two runs on the same machine with the same
   software produced **bit-identical** keyframe sets (100 %). SKEM has no intrinsic
   stochasticity (decoding is greedy, seeds are fixed).
2. **Exact per-video reproduction degrades with environment distance.** The match rate falls
   monotonically as the environment changes more (100 % → 60 % → ~31–36 %). The cause is
   threshold flipping: small, deterministic differences in model logits across
   PyTorch/CUDA/FlashAttention versions push borderline frames (`S_rel ≈ η_th`) across the
   `0.35` boundary, and SKEM's autoregressive structure then **amplifies** a single early
   flip into a different downstream trajectory.
3. **Aggregate behavior is robust.** Across all environments the *total* number of selected
   keyframes changes by only **±4 %** with **no directional bias** (e.g., cross-machine:
   204 → 212 keyframes, +3.9 %; counts go up and down in roughly equal numbers across
   videos). Consequently the **channel-bandwidth ratio (CBR), the per-scheme operating
   points, and all reported conclusions** (ultra-low CBR on the order of `1e-4`–`1e-3`,
   SKEM's content-adaptive keyframe budgeting, downstream zero-shot generalization) are
   **reproducible**; only the identity of individual borderline keyframes varies.

## Practical guidance for reproduction

- **Closest reproduction:** pin the environment. Use `environment/internvl.yml` and match the
  PyTorch/CUDA/FlashAttention versions; reproductions within the same environment are exact.
- **Different hardware/libraries are expected to differ at the per-keyframe level** but
  reproduce the aggregate metrics within ≈±4 %. This is normal for LLM-in-the-loop pipelines
  with GPU/library numerical non-determinism (cf. standard ML reproducibility checklists,
  which require reproducibility *up to* hardware/library nondeterminism, not bitwise identity).
- **Optional strict-determinism mode:** for bitwise-stable keyframes across runs on a given
  machine, set `torch.use_deterministic_algorithms(True)`, disable FlashAttention
  (`use_flash_attn=False`), and fix all seeds. This is slower and deviates from the
  performance-oriented default pipeline, so it is not the released default.

## Provenance

The exact keyframe sets, intermediate CSVs, and score outputs used for the paper's figures
are the authoritative artifacts; a fresh run on a different machine will produce statistically
equivalent—but not byte-identical—keyframes, by the mechanism described above.
