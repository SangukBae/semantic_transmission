# ETRI01 디코더 정렬·입력 조건 비교

2026-09-17~18, 현재 WSL/RTX 4080에서 실행했다. 모델을 추가 학습하지 않고
기존 체크포인트, 576×320/24fps/240프레임, seed 42, 30 steps를 유지했다.

- [비교 영상: 원본·이전·12장·통합](../outputs/diagnostics/etri01_ablation_20260917/comparison_four_panel.mp4)
- [조건별 동기 재생 페이지](../outputs/diagnostics/etri01_ablation_20260917/comparison.html)
- [전체 측정값과 해석](../outputs/diagnostics/etri01_ablation_20260917/REPORT.md)
- [입력·시간축·전송량 검증](../outputs/diagnostics/etri01_ablation_20260917/AUDIT.json)

큰 출력 파일은 로컬 `outputs/diagnostics/etri01_ablation_20260917/`에 있으며 Git에 포함하지 않는다.

| 조건 | PSNR | SSIM | LPIPS VGG | 총 채널 사용량 배수 |
|---|---:|---:|---:|---:|
| 이전 공식 복원 | 19.7081 | 0.7683 | 0.3553 | 1.000 |
| 정렬 보정 | 19.4705 | 0.7553 | 0.3683 | 1.000 |
| 정렬 + 원본 키프레임 | 19.9303 | 0.7633 | 0.3213 | 진단 전용 |
| 정렬 + 수동 행동 캡션 | 19.9484 | 0.7560 | 0.4083 | 1.000 |
| 정렬 + 키프레임 7장 | 21.4710 | 0.8083 | 0.3322 | 2.387 |
| 정렬 + 키프레임 12장 | 23.8337 | 0.8439 | 0.2957 | 4.091 |
| 정렬 + 12장 + 수동 행동 캡션 | 23.7729 | 0.8282 | 0.3089 | 3.986 |

동일한 240개 원본 시점, MP4 인코딩 전 PNG 기준이다. 과거 영상의 중복 경계
1프레임을 제외했다. PSNR/SSIM은 높을수록, LPIPS는 낮을수록 좋다.
현재 PC의 기존 설정 재실행은 과거 PNG 241장과 MP4 모두 바이트 단위로 일치했다.

정렬 보정은 끝점 PSNR을 개선했지만 전체 평균 화질 개선으로 이어지지 않았다.
이번 한 영상/seed에서는 키프레임 간격 축소의 효과가 컸다. 수동 행동 캡션의
추가 효과는 일관되지 않았다. 포즈/외형/배경 변화는 여전히 남아 있다.
원본 키프레임과 수동 캡션은 원인 분리용이며 자동 시스템 개선이나 새 모델 학습의
성과로 보고하지 않는다. 키프레임 추가 조건은 동일 전송량 비교가 아니다.

## 구현

`04_semantic_decoder/scripts/mydemo_new_align_sh.py`의 선택적 설정:

```python
decoder_policy = "official_release"  # 기존 모델 recipe 유지
conditioning_alignment = "endpoint_exact"
concatenation_policy = "endpoint_exact"
cache_text_embeddings = True        # 반복되는 CPU T5 입력만 재사용
```

이는 생성된 `receiver/decoder_config.py`의 설정이다. 일반 `run_config.json`에
동일 키를 추가하면 자동 전달된다고 가정하지 않는다. 실험 runner가 명시적으로 쓴다.
새 정렬 옵션은 마지막 잠재 위치의 반올림을 없애고, 이전 출력의 실제 마지막
17프레임을 overlap으로 인코딩한다. VAE 손실이 있어 픽셀 단위 고정을 뜻하지 않는다.
기본값은 기존 동작을 유지한다. 코드 변경으로 사전학습 가중치를 수정하지 않았다.

7장/12장 조건은 기존 0/179/239 프레임을 유지한 채 최대 간격을 48/24프레임으로
제한했다. 기존 수신 시각 데이터는 그대로 재사용했다. 추가 키프레임의 NTSCC와
10dB AWGN, 전체 메타데이터의 Sionna LDPC/16QAM/AWGN 전송 및 사용량을 기록했다.
12장 조건과 통합 조건의 수신 시각 심벌/PNG는 동일하다.

VAE는 posterior를 샘플링한다. 분할/overlap 길이가 달라지면 seed가 같아도
난수 소비 순서가 달라질 수 있다. 여러 seed나 별도 평가 영상에서의 일반화는 검증하지 않았다.

## 실행 기록과 도구

- `scripts/run_etri01_ablation.py`: 기존 입력을 보존한 새 조건 생성. `--resume`은 완료된 조건을 건너뛴다.
- `scripts/evaluate_etri01_ablation.py`: PNG/MP4의 공식 5개 지표, 동일 원본 ROI, 위치 proxy.
- `scripts/audit_etri01_ablation.py`: 원본 보존, 시간축, wire hash, 동일 시각 입력 검증.
- `scripts/report_etri01_ablation.py`: 비교 MP4, 오프라인 HTML, 그래프, 보고서.
- `tests/test_decoder_conditioning.py`: 실제 공개 masking 함수로 끝점/overlap 동작 검사.

관련 회귀는 24 passed, 1 skipped이며 실제 7개 복원과 최종 5개 지표 계산을 완료했다.
입력과 전송량 audit도 통과했다. 메모리를 위해 추가한 실험 전용 swap 8GiB는
작업 후 해제하고 삭제했다. 기존 환경 복원용 swap은 변경하지 않았다.
