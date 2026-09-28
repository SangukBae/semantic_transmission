# 실행 명령 모음

갱신: **2026-09-28**. 연구 목적과 현재 판정은 [과제 현황](README.md)을 먼저 확인한다.
아래 명령은 저장소 루트에서 실행한다. `data/`·모델 가중치·기존 `outputs/`는 별도 로컬 자료다.

## 설치와 짧은 점검

Linux/WSL2, NVIDIA 드라이버, Conda, FFmpeg가 필요하다. 기본 Conda 경로는 `~/anaconda3`이며
다르면 `CONDA_BASE`로 지정한다. Windows 이전은 [이전 안내](MIGRATION_WINDOWS.md),
이 PC의 복원 경로는 [WSL 기록](WSL_LOCAL_SETUP.md)을 따른다.

```bash
bash scripts/bootstrap.sh
source scripts/activate.sh
semtx doctor
semtx smoke --output outputs/my_first_skem

# 자신의 영상으로 짧은 실행
semtx smoke --input /path/to/video.mp4 --frames 33 --output outputs/custom_skem
```

`lgvsc`는 GPU 모델 실행 환경, `lgvsc-channel`은 Sionna/LDPC 전송 환경이다.
기본 smoke는 17프레임·256×256·10 sampling steps의 설치 점검으로, 논문 성능·ETRI 완화 효과와 구분한다.
smoke 출력은 새 폴더를 사용한다. 프로필별 차이는 [로컬 연구 환경](LOCAL_RESEARCH.md),
공식 설정 환경 구성은 [공식 실행 기록](ETRI_OFFICIAL_PROTOCOL.md)을 참고한다.

## 현재 ETRI 장시간 연구

| 목적 | 명령 | 안내 |
|---|---|---|
| 60초 입력·짧은 모델 점검 | `bash scripts/check_etri_60s.sh` | [준비 점검](ETRI_60S_CHECK.md); 전체 복원 아님 |
| 60초 전체 기준선 | `bash scripts/run_etri_60s.sh` | [전체 실행](ETRI_60S_RUN.md) |
| 완료 SKEM을 가져오는 기존 복구 경로 | `bash scripts/recover_etri_60s.sh` | 같은 환경의 완료 실행은 검증 후 재사용 |
| 복구 경로 상태 확인 | `bash scripts/recover_etri_60s.sh --status` | 현재 `tv_low_08`은 전체 완료 |
| 다른 개발 영상 | `bash scripts/run_etri_60s.sh --video tv_medium_02` | 완료·미완료를 새 실행 결과로 판단 |

점검·전체 실행의 `--dry-run`은 추론 없이 입력·환경을 확인한다. `--cpu-only`는 준비 점검에만 사용한다.
새 환경에서 기존 SKEM·완료 결과가 없으면 복구 명령 대신 입력·모델 준비 후 새 실행이 필요하다.
첫 영상의 SKEM은 23시간 19분, 재사용 후 나머지는 112.44분이었다. 다른 영상의 시간은 달라진다.

## 완료된 짧은 조건 비교

기존 60초 수신 산출물과 실행 환경이 있어야 한다. 해시가 같은 완료 단계는 재사용한다.
세 창의 진단이며 60초 전체 완화 실행과 구분한다.

| 비교 | 명령·안내 |
|---|---|
| 조건 위치 반올림 해제 | [진단 문서의 실행 명령](ETRI_CONDITIONING_DIAGNOSIS.md) |
| 마지막 17프레임 참조 | `bash scripts/check_etri_tail_reference.sh` · [결과](ETRI_TAIL_REFERENCE_DIAGNOSIS.md) |
| 네 조건 결합 비교 | `bash scripts/check_etri_combined_reference.sh` · [결과](ETRI_COMBINED_REFERENCE_DIAGNOSIS.md) |

## 이전 실험 재현

아래는 기존 짧은 영상·지표 실험이다. 현재 장시간 정식 평가를 실행하는 명령이 아니다.
조건과 재개·출력 경로는 각 기록을 먼저 확인한다.

| 실험 | 명령 | 상세 기록 |
|---|---|---|
| 기존 ETRI 10초 배치 | `bash scripts/run_etri_remaining.sh` | [공식 프로필](ETRI_OFFICIAL_PROTOCOL.md), [정렬·전송량](LGVSC_ALIGNMENT.md) |
| WebVid 한 편·네 조건 | `bash scripts/run_webvid_ablation.sh` | [한 편 비교](WEBVID_ONE_ABLATION.md) |
| WebVid 다섯 편 | `bash scripts/run_webvid5.sh` | [다섯 편 검증](WEBVID5_VALIDATION.md) |
| 품질 후보 일괄 실행 | `bash scripts/run_quality_validation.sh` | [실행 범위](QUALITY_VALIDATION.md); 1·2차 결과와 전체 완료를 구분 |
| 지표 후보 검증 | `bash scripts/run_metric_validation.sh` | [동결 캠페인](METRIC_VALIDATION_CAMPAIGN.md) |
| 지표 v6 재평가 | `bash scripts/run_metric_revision.sh` | [최신 지표 결과](METRIC_REVISION_RESULTS.md) |

기존 HQ 프로필(100프레임·512×256·10fps)은 [HQ 기록](ETRI_HQ_PROTOCOL.md)을 따른다.

```bash
bash scripts/bootstrap_hq.sh
source scripts/activate.sh
python -m semantic_transmission.research \
  --input-dir data/etri_video_eval/processed --output outputs/etri10_hq_new
```

## 개발 점검

```bash
source scripts/activate.sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q --disable-warnings
python scripts/probe_environment.py
```

테스트는 구현 검사다. 모델 실행 완료는 해당 `RESULT.json`·완료 영수증·출력 영상으로,
의미 오류·완화 효과는 독립 근거와 대응 평가로 확인한다. [모든 상세 문서](EXPERIMENT_INDEX.md)
