# Conda environments

The pipeline spans several large models with mutually incompatible dependencies, so
each stage runs in its own environment. Specs were exported with
`conda env export --no-builds` from the machines we ran on.

| File | Stage(s) | Notes |
|---|---|---|
| `llava.yml` | 01 data prep, 05 baselines, 06 eval | also hosts the captioning/eval helpers |
| `internvl.yml` | 02 SKEM / PSSS keyframes | needs InternVL2-8B weights |
| `pllava.yml` | 02 captioning | needs `pllava-7b` weights |
| `ntscc.yml` | 03 keyframe transmission | needs the `*_psnr.pth` checkpoints |
| `opensora.yml` | 04 decoder | exported from the GPU/decoder server |
| `timesformer.yml` | 07 action recognition | Kinetics-400 |
| `depth.yml` | 07 depth estimation | Depth-Anything-V2 |
| `video_summary.yml` | 07 video captioning eval | BERTScore/BLEU/ROUGE |

```bash
conda env create -f environment/<name>.yml
```

## Caveats for reproduction
- The exported `channels:` may include a Tsinghua (TUNA) mirror — replace with your
  preferred channels if needed.
- `--no-builds` keeps specs portable but not bit-exact; pin exact builds only if you
  hit version conflicts.
- CUDA: these envs target the CUDA toolkit available on our boxes (Open-Sora v1.2 +
  ColossalAI on the decoder server). Adjust `pytorch`/`cudatoolkit` to your driver.

## Large weights (not in git)
| Weight | Size | Where |
|---|---|---|
| `OpenGVLab/InternVL2-8B` | ~16 GB | HuggingFace (or local `$INTERNVL_MODEL`) |
| `pllava-7b` | ~15 GB | HuggingFace `ermu2001/pllava-7b` |
| `ntscc_hyperprior_quality_{1..4}_psnr.pth` | ~132 MB each | our trained checkpoints → `$NTSCC_CKPT` |
| Open-Sora v1.2 (STDiT3 + VAE + T5) | ~20 GB | HuggingFace `hpcai-tech/OpenSora-STDiT-v3` etc. |
| UniMatch `gmflow-scale2-regrefine6-mixdata...pth` | ~30 MB | auto-downloaded by `after_extract.sh` |
| Depth-Anything-V2 | ~1.3 GB | HuggingFace `depth-anything/Depth-Anything-V2-Large` |
