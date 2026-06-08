# Stage 05 — Classical baselines (H.264 / H.265 + LDPC)

Source-code the video with a standard codec, then transmit the bitstream through the
LDPC channel of `03_jscc_transmission/ldpc_sionna`. Compared against LGVSC at matched
SNR in the paper.

## H.264 (`h264/`)  ·  env `llava`
```bash
source env.sh                   # exports VIDEO_FOLDER (defaults to $DATA_ROOT/16x24)
python h264/h264compress.py     # reads $VIDEO_FOLDER; no manual path edits
```
`preprocessing.py` and `video_bgr2rgb.py` are color/format helpers.

## H.265 / HEVC (`h265/`)  ·  env `llava`
```bash
source env.sh
python h265/h265compress.py     # reads $VIDEO_FOLDER; no manual path edits
python h265/convert_yuv_to_mp4.py
```

## Codec parameters (from the paper)

| SNR (dB) | Codec | QP | GOP | LDPC rate | Modulation |
|---|---|---|---|---|---|
| 6  | H.264 / H.265 | 51 / 51 | 400 / 400 | 3/4 | 16-QAM / 4-QAM |
| 8  | H.264 / H.265 | 51 / 51 | 80 / 50  | 1/2 | 16-QAM |
| 10 | H.264 / H.265 | 51 / 50 | 20 / 13  | 2/3 | 16-QAM |

A cliff effect occurs below 6 dB. After compression, transmit with
`03_jscc_transmission/ldpc_sionna/SionnaPlus_sora.py`.

## DVST
The DVST baseline reported in the paper used a collaborator's implementation and is
**not** included or redistributed here. To reproduce it, request the code from the
**original DVST authors** — Wang et al., "Wireless Deep Video Semantic Transmission,"
*IEEE JSAC* 2023 ([paper](https://arxiv.org/abs/2205.13129)); we used `eta = 0.2`,
`lambda = 128` (minimum-CBR operating point).
