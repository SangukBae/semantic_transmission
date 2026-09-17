# LGVSC 실행·평가 정합성 수정 — 2026-09-17

이 변경은 공식 공개 코드의 10 dB SKEM+DSA 경로를 기준으로 평가·선택·전송량 기록을
맞추고, 프레임 연결 정책을 생성 설정과 분리한다. WebVid 55편 성능 재현 결과가 아니다.
기준 소스는 [TT2TER/LGVSC 50c9ff98](https://github.com/TT2TER/LGVSC/tree/50c9ff98fb1aaa8136aef2e41e02bbb0d1a2d1c3)이다.

## 실행 프로필

| 설정 | 선택 | 평가 | 생성 설정 | 최종 연결 |
|---|---|---|---|---|
| `configs/etri_official.json` | stride 1 | 공식 5개 지표 | official_release | 공식 경계 중복 유지 |
| `configs/lgvsc_variable.json` | stride 1 | 공식 5개 지표 | official_release | 공식 경계 중복 유지 |
| `configs/lgvsc_webvid_select.json` | stride 1로 수정 | 공식 5개 지표 | official_release | 공식 경계 중복 유지 |
| `configs/lgvsc_variable_endpoint_exact.json` | stride 1 | 공식 5개 지표 | official_release | 이후 구간의 중복 경계만 제거 |

프로필 이름은 v2로 구분한다. 이전 v1 출력·프로필·실행 기록은 수정하지 않는다.
ETRI 이어하기 명령도 v2 이력을 사용하므로 v1 결과를 새 지표의 평가 완료로 재사용하지 않는다.
기존 HQ 및 고정된 지표 개발 실험의 설정에 평가 버전이 없으면 이전 Alex/RGB 평가를 유지한다.
`research` 명령에서 `--profile`을 생략하면 기존 HQ 설정이 선택되므로 반드시 명시한다.

WebVid의 전처리 입력은 576×320, 24 fps, 최대 384프레임이다. 영상별 실제 길이는
기존 `variable_length` 입력 계약으로 결정한다. 전체 프레임 비교는 실행 시간을 늘린다.

```bash
python -m semantic_transmission.research \
  --profile configs/lgvsc_variable.json \
  --input-dir data/webvid55/processed --output outputs/webvid53_official_v2
```

현재 원본 확보 범위는 53/55편이다. 위 명령은 준비된 53편을 대상으로 하며,
두 편을 확보하기 전에는 결과를 55편 벤치마크라고 부르지 않는다.

## 공식 평가 지표

`evaluation_profile=lgvsc_official_metrics_v1`은 공개 `06_evaluation/final_score.py`와
같은 정의를 사용한다. 각 프레임 점수를 산술 평균한다.

- PSNR: OpenCV `PSNR`.
- SSIM: 추출 PNG의 OpenCV 흑백 디코딩 + skimage 기본 SSIM. RGB Gaussian SSIM과 다르다.
- LPIPS: VGG, v0.1, 입력 [-1,1]. 출력 키는 `lpips_vgg`이며 `lpips_alex`로 표시하지 않는다.
- CLIP: ViT-B/32, 공개 코드와 같은 `(cosine + 1) / 2` 정규화.
- DISTS: [공식 PyTorch 구현](https://github.com/dingkeyan93/DISTS), RGB [0,1], 추가 리사이즈 없음.

모델은 지표별로 하나씩 올리고 프레임 배치는 1로 제한한다. PNG와 최종 MP4를 각각
평가하며, 실제 GPU peak allocation을 결과에 기록한다. 공식 설치 스크립트는
`requirements-evaluation.txt`의 CLIP과 DISTS를 설치한다.

기존 영상은 생성 모델을 다시 실행하지 않고 새 폴더에 재평가할 수 있다.

```bash
python -m semantic_transmission.research_quality evaluate \
  outputs/etri01_official_20260911_v2/01_person_walk \
  --evaluation-profile lgvsc_official_metrics_v1 --compare-concatenation \
  --output-dir outputs/etri01_official_metrics_v2
```

`--compare-concatenation`은 같은 복원 픽셀에서 이후 구간의 중복 경계만 제외한
`endpoint_exact_view`와 별도 CSV를 기록한다. 이는 새 생성·MP4 재인코딩 결과가 아니다.
241프레임 원본 출력과 240프레임 비교 뷰는 각각 명시적인 source/generated index를 갖는다.
기존 `quality.json`은 덮어쓰지 않고, 참조·복원 영상 해시를 확인한다.
이동되어 찾을 수 없는 원본 입력은 이전 결과의 해시를 보존하고 현재 재확인 여부를 표시한다.

## 연결 정책의 독립성

`decoder_policy`는 기존 모델·align·오버랩 참조·VAE 설정을 선택한다.
새 `concatenation_policy`는 모든 구간의 생성이 끝난 뒤 수행하는 연결에만 적용한다.
공식 생성 설정의 `align=5`, VAE micro-batch=4, 이전 전체 구간 오버랩 참조,
30 steps, CFG 7, seed 42, FlashAttention과 fused LayerNorm은 두 연결 정책에서 같다.
`endpoint_exact` 디코더 프로필 전체로 전환하지 않는다.

연결 정책도 실제 메타데이터 패킷에 실어 수신기에서 적용한다. 옵션이 없는 기존
패킷은 기존 `decoder_policy`를 따라 이전 동작을 유지한다.

## 전송량 기록

송신 기록의 `metadata_breakdown`과 채널 기록의 `transmission_breakdown`에 다음을 기록한다.

- 캡션·광류·정규화 값·키프레임 위치·rate/stream 설명·영상/디코더 설정·기타 JSON.
- rate-index 이진 payload, 프레이밍, CRC32. 이들 바이트의 합은 실제 metadata.bin 크기와 같다.
- LDPC 정보 비트·패딩 비트·패리티 비트 및 전체 부호화 비트.
- 시각/디지털/전체 복소 채널 사용량과 각각의 CBR. 분모는 원본 `3×H×W×F`이다.

JSON 값의 바이트에는 따옴표·이스케이프도 포함한다. 개별 필드는 하나의 LDPC 패킷을
공유하므로 필드마다 패딩을 붙여 심벌을 중복 계산하지 않는다. 전체 CBR에는 모든
메타데이터·LDPC 패딩·패리티를 포함한다. 물리 링크 헤더는 포함하지 않으며,
complex64 직렬화 파일 크기를 RF 비트 수로 취급하지 않는다. 논문의 CBR와 동일한
집계 범위라는 주장은 하지 않는다.

기존 패킷은 별도 파일로 재집계할 수 있다.

```bash
python -m semantic_transmission.transmission_accounting \
  --run-dir outputs/etri01_official_20260911_v2/01_person_walk \
  --output outputs/etri01_transmission_breakdown_v2.json
```

## 검증 범위

검증 기록은 [2026-09-17 실행 기록](validation/2026-09-17-lgvsc-alignment.json)에 둔다.
공식 함수와의 지표 수치 대조, 연결 정책 간 생성 설정 동일성, 프레임 수·픽셀 보존,
패킷 바이트·부호 비트 합산, 평가 버전이 다른 결과의 재사용 차단을 검사한다.
GPU 지표 대조는 `LGVSC_GPU_TESTS=1`로 별도 활성화한다.

2026-09-17 검사 결과:

- 전체 회귀 검사 159개 통과, 별도 활성화하는 GPU 검사 1개 제외.
- GPU 지표 대조를 포함한 관련 검사 31개 통과. GPU 지표는 공식 함수와 절대 오차
  `1e-6` 이내, PSNR·SSIM은 테스트 입력에서 정확히 일치했다.
- 기존 ETRI 241프레임을 PNG·MP4 모두 재평가하고, 같은 픽셀의 240프레임 연결 비교 뷰를 기록했다.
  평가기의 최대 PyTorch 할당 메모리는 545,537,024 bytes, 약 0.51 GiB였다.
- 기존 WebVid 9프레임의 선택·캡션·광류를 재사용해 수정된 송신·채널·수신·복원·평가 5단계를
  실제 GPU/Sionna에서 실행했다. 메타데이터가 일치하고 공식 생성 설정과 별도 연결 정책이
  수신기에 전달되었다. 이 GPU 복원은 단일 구간이며, 여러 구간 연결은 텐서 회귀 검사와
  기존 ETRI 비교 뷰로 확인했다.
- 확인 대상 기존 결과 파일 593개의 SHA-256이 변경되지 않았다.

재평가·실행 결과는 로컬 `outputs/lgvsc_alignment_20260917_v1/`에 있다.
표시된 0.51 GiB는 평가기 할당량이며 생성 모델의 전체 VRAM 사용량이 아니다.

SNR별 미공개 NTSCC 가중치, CPU 오프로딩 및 의존성 차이, 논문 55편 전체 성능 검증은
이번 수정으로 해결되었다고 주장하지 않는다.
