# 이 PC의 WSL 복원 기록

작업일: 2026-09-17. 코드 위치는 `/home/sangukbae/semantic_transmission`이며,
Ubuntu 22.04.5 / WSL2 / RTX 4080 16GB 환경이다.
T9 원본은 `E:\semantic_transmission_backup_20260917_195148`에 보존한다.

## 파일 배치

| 자료 | 현재 위치 |
|---|---|
| 코드 | `~/semantic_transmission` |
| 데이터 전체 | `~/datasets/semantic_transmission` → 저장소의 `data` 링크 |
| 과거 실험 출력 | 저장소의 `outputs/` |
| 핵심 모델 snapshot 및 blob | `~/.cache/huggingface/hub/` |
| 평가 모델 캐시 | `~/.cache/torch/`, `~/.cache/clip/` |
| NTSCC/UniMatch 가중치 | `.local/checkpoints/` |
| SAM2/DINOv2 소스와 가중치 | `.local/metric_v2_models/` |
| 이전 PC의 `.local` 원본 | `~/migration_saved_state/.local/` |
| 복원 체크섬·환경 기록·로그 | `.local/migration/` |

이전 절대경로로 파일을 열 수 있도록 다음 호환 링크를 추가했다.

- `/home/sangukbae/ETRI/Semantic/semantic_transmission` → 현재 저장소
- `/legend/semantic_transmission/datasets` → 현재 데이터 디렉터리

과거 결과 안의 선언·경로·해시는 수정하지 않는다. 위 링크가 있어도 새 환경의
실행 식별자는 달라지므로 과거 동결 실험의 자동 재개를 보장하지 않는다.
새 `.local/etri_continue.json`은 새 데이터 경로와 빈 실행 이력으로 시작한다.
원본 ETRI/WebVid 이력 및 진단은 `~/migration_saved_state/.local/`에 보관한다.

## 실행 환경

Linux Miniforge는 기존 활성화 스크립트의 기본 경로인 `~/anaconda3`에 설치했다.
기존 PHM 등 다른 Python 환경은 변경하지 않는다.

| 환경 | 용도 |
|---|---|
| `~/anaconda3/envs/lgvsc` | 생성·광류·NTSCC·픽셀 평가, Python 3.10 / PyTorch 2.2.2+cu121 |
| `~/anaconda3/envs/lgvsc-channel` | TensorFlow 2.15.1 / Sionna 0.19.2 LDPC 전송 |
| `~/anaconda3/envs/lgvsc-internvl` | 공식 selector, Python 3.9.19 / PyTorch 2.4.1+cu121 |
| `.local/metric_v2_env` | 독립 지표 환경, Python 3.10 / PyTorch 2.12.0+cu130 / torchvision 0.27.0 |

생성 환경의 잠금 파일에서 `decorator`, `fabric` 두 항목이 requirements 및
백업의 실제 설치 기록과 달라 설치가 실패했다. 각각 `4.4.2`, `3.0.0`으로 일치시켰다.
전송 환경은 최신 Conda SQLite/ICU와 TensorFlow의 C++ 런타임 로딩 순서가 충돌했다.
`lgvsc-channel`의 `libsqlite=3.50.4` / Python 3.10.19 조합으로 해결하고,
환경의 `conda-meta/pinned`에 SQLite 버전을 고정했다. LDPC 실제 송수신에서 비트 오류 0건을 확인했다.
SAM2는 고정 소스로 설치하되 선택적인 CUDA 후처리 확장은 빌드하지 않았다.
현재 `ObjectExtractor`는 `apply_postprocessing=False`, `min_mask_region_area=0`을 사용한다.

```bash
cd ~/semantic_transmission
source scripts/activate.sh
semtx doctor
bash scripts/run_webvid5.sh --dry-run
bash scripts/run_etri_remaining.sh --dry-run
```

새 GPU 축소 검증은 매번 새 출력 폴더를 지정한다.

```bash
semtx smoke --selector skim --skim-keyframes 2 --frames 9 \
  --output outputs/wsl_smoke_new
```

## 메모리 설정

물리 RAM은 32GB이며 복원 당시 WSL의 RAM은 약 16GB, swap은 4GB였다.
CPU에 올라가는 FP32 T5 가중치가 약 19GB이므로 메모리 보완이 필요하다.
`C:\Users\halmo\.wslconfig`에 `memory=24GB`, `swap=16GB`를 설정했다.
이는 다음 WSL VM 재시작부터 적용되며, 모든 WSL2 배포판에 공통으로 적용된다.
[Microsoft 설정 설명](https://learn.microsoft.com/en-us/windows/wsl/wsl-config).

이번 세션의 검증에는 `/var/tmp/semantic-transmission-migration.swap` 16GiB를 추가했다.
영구 부팅 설정에는 등록하지 않았다. WSL 재시작으로 RAM/swap 설정을 적용한 후
이 임시 파일이 `swapon --show`에 없는 것을 확인하면 삭제해 공간을 회수할 수 있다.
WSL 재시작은 실행 중인 WSL 및 Docker 작업을 종료하므로 작업을 저장한 뒤 Windows에서 수행한다.

## 검증 상태

- T9 TAR **10개 / 196,746,485,760 bytes (183.23GiB)** SHA-256 일치 및 복원 완료.
- 일반 파일 **253,662개**의 존재·크기·종류 및 심볼릭 링크 대상 대조 통과.
  백업 당시 변경 중이던 WebVid 로그 2개도 백업 기록의 크기 범위와 일치한다.
  근거: `.local/migration/restoration_verification.json`, `restore_results.json`.
- WebVid **53/55편**, Kinetics **14/14편** 원본 SHA-256 재검증 통과.
  데이터 파일 113,779개와 이전 `.local` 파일 9,645개도 누락 없이 복원했다.
- CPU 회귀: `178 passed, 1 skipped` (`.local/migration/logs/pytest.log`).
- SAM2/DINOv2: RTX 4080에서 실제 프레임 추론 통과 (`.local/migration/metric_gpu_probe.json`).
- RAFT-small: 실제 광류 추론 통과 (`.local/migration/metric_raft_probe.json`).
- InternVL: 공식 Python/BF16/FlashAttention/CPU layer 1개 설정에서 실제 두 프레임 비교 통과.
  축소 입력 256×256이며 공식 전체 영상 성능 검증은 아니다.
  결과: `outputs/wsl_internvl_probe_20260917_2059/`.
- 전체 축소 실행: **9프레임 / 256×256 / 10 steps**, SKIM → PLLaVA → UniMatch →
  NTSCC → Sionna LDPC → Open-Sora → 평가 모두 통과.
  [복원 영상](../outputs/wsl_smoke_20260917_2055/reconstruction/sample_0000.mp4),
  [실행 기록](../outputs/wsl_smoke_20260917_2055/run_manifest.json).
  PSNR 13.1759dB, SSIM 0.352923이며 환경 동작 확인용 결과다.
- ETRI 10편, WebVid5 5편, WSL 지표 campaign의 `--dry-run` 모두 통과.
  본 실험은 새로 시작하지 않았다.
- Apex 고정 커밋 빌드, BF16 CUDA LayerNorm 및 FlashAttention CUDA 검사 통과.
  `.local/migration/official_gpu_extensions.json`과 `logs/finish_environment.log`에 기록했다.
- 생성·전송·InternVL 환경의 `pip check`, 독립 지표 환경의 `uv pip check` 통과.
  최종 설치 패키지 목록은 `.local/migration/*-packages-current.json`,
  `*-conda-current.txt`, `metric-requirements-current.txt`에 보관한다.
- 공식 5개 지표(PSNR·SSIM·LPIPS·CLIP·DISTS)의 실제 9프레임 계산 통과.
  `.local/migration/official_metrics_probe.json`에 별도 기록했다.
  공식 평가의 프레임별 PSNR 평균·회색조 SSIM은 위 smoke의 전체 MSE PSNR·RGB SSIM과 정의가 다르다.

백업은 WebVid5 실행 도중의 스냅샷이다. 이후 결과를 담은 최종 overlay는 T9에서
발견되지 않았다. 원래 확보되지 않은 WebVid 2편과 별도 `sgdjscc_lab` 프로젝트도
이 백업에 포함되지 않는다.
