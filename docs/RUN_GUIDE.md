# 설치·실행·복구 안내

갱신: **2026-09-28**. [과제 현황](README.md) · [실험 결과](ETRI_COMBINED_REFERENCE_DIAGNOSIS.md)
명령은 저장소 루트에서 실행한다. 데이터·가중치·기존 결과는 Git에 포함되지 않는다.

<a id="setup"></a>
## 환경 준비

검증 환경은 WSL2 Ubuntu 22.04 / RTX 4080 16GB다. Linux Conda·NVIDIA 드라이버·FFmpeg가 필요하다.
현재 코드 위치는 `~/semantic_transmission`, 데이터는 `~/datasets/semantic_transmission`이며 저장소의 `data/`가 이를 가리킨다.

| 환경·경로 | 용도 |
|---|---|
| `~/anaconda3/envs/lgvsc` | 생성·광류·NTSCC·화질 평가 |
| `~/anaconda3/envs/lgvsc-channel` | Sionna/LDPC 디지털 전송 |
| `~/anaconda3/envs/lgvsc-internvl` | 공식 SKEM 선택기 |
| `.local/metric_v2_env/` | SAM2·DINOv2·RAFT 등 독립 지표 평가 |
| `.local/checkpoints/`, `~/.cache/huggingface/hub/` | 보조 가중치·모델 snapshot과 blob |
| `outputs/`, `.local/migration/` | 전체 실험 산출물·이전 검증 기록 |

```bash
# 새 환경: 실제 Linux Conda 경로가 다르면 CONDA_BASE를 지정
bash scripts/bootstrap_official.sh
source scripts/activate.sh
semtx doctor
python scripts/probe_environment.py

# 짧은 설치 점검: 매번 새 출력 경로 사용
semtx smoke --selector skim --skim-keyframes 2 --frames 9 --output outputs/setup_smoke_new
```

가벼운 기본 smoke 환경은 `bash scripts/bootstrap.sh`로 구성한다. 기본 smoke는 17프레임·256×256·10 steps다.
실제 GPU 생성 성공, 논문 재현, ETRI 완화 성능을 각각 구분한다. 기존 환경에서는 재설치 없이 활성화부터 시작한다.

<a id="migration"></a>
## 다른 PC로 이전

- Windows에는 NVIDIA 드라이버를 설치하고 WSL2 Ubuntu를 사용한다. WSL에 Linux 디스플레이 드라이버를 설치하지 않는다.
- 코드·캐시·대량 프레임은 WSL Linux 파일시스템에 둔다. 데이터·`outputs/`·필요 모델 캐시를 별도로 복사한다.
- Hugging Face의 `snapshots` 링크와 실제 `blobs`를 함께 보존한다. 가상환경·CUDA 드라이버 폴더를 통째로 덮어쓰지 않는다.
- Conda 환경·`.local/settings.json`·모델 경로는 새 PC에서 재구축한다. 과거 선언의 경로·해시를 일괄 치환하지 않는다.
- T5 CPU 가중치와 오프로딩을 위해 RAM·swap 여유를 확인한다. WSL 메모리 변경은 실행 중인 작업 종료와 재시작을 고려한다.
- 지표 캠페인은 모델만이 아니라 이전 결과·교정값·관측 캐시도 요구한다. `.local/metric_v2_env`만 복사해 재현된다고 보지 않는다.

```bash
# 복사한 실제 데이터 경로에 맞춰 새 clone에서 한 번 연결; 기존 data가 있으면 먼저 확인
ln -s "$HOME/datasets/semantic_transmission" data
python scripts/prepare_lgvsc_datasets.py audit --root "$HOME/datasets/semantic_transmission"
bash scripts/run_etri_remaining.sh --dry-run
bash scripts/run_webvid5.sh --dry-run
```

기존 호환 경로 `/legend/semantic_transmission/datasets`와 `/home/sangukbae/ETRI/Semantic/semantic_transmission`는
옛 선언을 읽기 위한 링크다. 환경 변경 뒤 자동 재개를 보장하지 않는다.
상세 [Windows 이전 절차](https://github.com/SangukBae/semantic_transmission/blob/f41bc8885e8d8409e59b162fe7be63c0d11ee737/docs/MIGRATION_WINDOWS.md)와
[이 PC 복원·패키지 문제 해결 기록](https://github.com/SangukBae/semantic_transmission/blob/f41bc8885e8d8409e59b162fe7be63c0d11ee737/docs/WSL_LOCAL_SETUP.md)은 통합 전 버전에 보존한다.

<a id="long-video"></a>
## 60초 기준선과 재개

`tv_low_08`은 `outputs/etri_60s_tv_low_08_42057b2ee8ed/`에서 **전체 복원 완료** 상태다.
`PASS_60S_RECONSTRUCTION`은 실행·시간축 통과이며 의미 오류 검수와 완화 효과는 미완료다.

```bash
# 전체 입력 검사 + 짧은 실제 모델 점검; 60초 전체 생성은 아님
bash scripts/check_etri_60s.sh

# 실제 60초 전체 기준선: 새 실행 또는 검증된 완료 단계 재사용
bash scripts/run_etri_60s.sh

# 이전 tv_low_08 SKEM을 검증해 가져오는 경로; 같은 조건의 완료 실행은 재사용
bash scripts/recover_etri_60s.sh
bash scripts/recover_etri_60s.sh --status
bash scripts/recover_etri_60s.sh --dry-run

# 다른 개발용 영상
bash scripts/run_etri_60s.sh --video tv_medium_02
```

복구 명령의 SKEM 원본은 `outputs/etri_60s_tv_low_08_263507d45874`다. 다른 완료 선택 결과는
`--reuse-selection-from outputs/PREVIOUS_RUN`으로 지정한다. 검증 실패 시 처음부터 몰래 재실행하지 않는다.
`--stop-after input-audit`는 입력 검사 뒤 정지, `--output outputs/my_etri_60s`는 출력 경로 지정이다.
`--dry-run`은 추론 없는 사전 검사, `--cpu-only`는 준비 점검 명령에만 사용한다.

| 실행 조건 | 값·범위 |
|---|---|
| 입력 | 연속 60초·1,440프레임·24fps·576×320, 개발 분할 |
| SKEM | threshold 0.35, 후보 간격 1프레임, BF16·CPU 보관 계층 3·KV 압축 |
| 송수신·생성 | PLLaVA·UniMatch, NTSCC quality-4, AWGN 10 dB, 채널 seed 42, 생성 seed 2025, 30 steps |
| 생성·연결 | `official_release` 조건 정렬, `endpoint_exact` 경계 중복 제거 |
| 클립·메모리 | 프레임 기준 무손실 클립과 전수 PTS/픽셀 검사, 순차 광류 표집, 생성 구간 CPU 보관 |

첫 실행의 SKEM은 23시간 19분, 가져온 뒤 나머지 실행은 112.44분이었다. 다른 영상의 시간은 달라진다.
한 구간이 매우 길면 GPU 메모리 부족이 남는다. 실패 시 로그를 남기며 키프레임·생성 단계를 자동 변경하지 않는다.
WSL·터미널을 유지하거나 tmux를 사용한다. 서로 다른 연구 실행기를 같은 GPU에서 동시에 돌리지 않는다.

| 단계 | 재개 범위 |
|---|---|
| SKEM | 후보 비교마다 저장. 최근 키프레임 상태를 복원해 마지막 완료 프레임 다음부터; 현재 탐욕적 선택 경로 |
| 캡션 | 구간별 저장. 검증된 캡션 재사용, 중단 당시 구간은 다시 처리 |
| 광류·전송·생성·평가 | 완료 단계 해시 검사 후 재사용, 미완료 단계는 처음부터 실행 |

**생성 구간 내부 체크포인트는 없다.** 실패 출력은 `failed_attempts/`에 보존한다.
코드·설정·환경·모델이 다르면 새 실행 식별자·폴더를 사용하며 이전 결과와 섞지 않는다.

| 결과 경로 — 실행 폴더 기준 | 내용 |
|---|---|
| `RESULT.json`, `REPORT.md`, `status.json` | 완료 상태·지표·전송량·의미 검증 보류 |
| `baseline/receiver/reconstruction/sample_0000.mp4`, `comparison.mp4` | 복원과 동기 비교 영상 |
| `baseline/output_audit.json`, `baseline/quality*` | PNG·MP4 프레임·PTS·참조 전달과 화질 |
| `baseline/run_config.json`, `baseline/selection_*.json` | 실행 조건·구간 길이·SKEM 가져오기 근거 |
| `baseline/semantic_clips_audit.json`, `checkpoints/`, `stages/` | 클립 검사·진행 상태·완료 파일 해시 |
| `logs/`, `progress/`, `resources/` | 로그·진행률·시간·메모리 |

GPU 주기 표집은 장치 전체 메모리이며 순간 최고값이나 다른 작업을 포함할 수 있다.
프로세스별 PyTorch 할당/예약량과 구분한다. [전체 실행·복구 원문](https://github.com/SangukBae/semantic_transmission/blob/f41bc8885e8d8409e59b162fe7be63c0d11ee737/docs/ETRI_60S_RUN.md)

<a id="experiments"></a>
## 조건 비교와 이전 실험

아래 명령은 해당 로컬 자료·환경이 필요하다. 완료 수와 효과는 [실험 기록](ETRI_COMBINED_REFERENCE_DIAGNOSIS.md)에서 확인한다.

| 목적 | 명령 |
|---|---|
| 17프레임 참조 진단 | `bash scripts/check_etri_tail_reference.sh` |
| 네 조건 결합 진단 | `bash scripts/check_etri_combined_reference.sh` |
| 기존 ETRI 짧은 영상 배치 | `bash scripts/run_etri_remaining.sh` |
| WebVid 한 편 / 다섯 편 | `bash scripts/run_webvid_ablation.sh` / `bash scripts/run_webvid5.sh` |
| 품질 후보 비교 | `bash scripts/run_quality_validation.sh` |
| 지표 캠페인 / v6 | `bash scripts/run_metric_validation.sh` / `bash scripts/run_metric_revision.sh` |

반올림 해제 진단은 `scripts/diagnose_etri_conditioning.py`, 비교자료 생성은 `scripts/etri_conditioning_evidence.py`다.
지표 환경은 `LGVSC_METRIC_PYTHON`, 생성 환경은 `LGVSC_PYTHON` 또는 `.local/settings.json`으로 지정한다.
WSL 지표 캠페인은 `--config configs/metric_validation_wsl.json --output outputs/metric_validation_wsl_new --dry-run`으로 먼저 점검한다.
기존 HQ 프로필·추가 옵션은 [통합 전 실행 모음](https://github.com/SangukBae/semantic_transmission/blob/f41bc8885e8d8409e59b162fe7be63c0d11ee737/docs/RUN_GUIDE.md)에 있다.

<a id="verification"></a>
## 구현 검사

```bash
source scripts/activate.sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q --disable-warnings
python scripts/probe_environment.py
```

테스트는 구현 검사다. 모델 실행은 산출물과 영상으로, 완화 효과는 독립 오류 근거로 확인한다.
동결 선언·검사 입력과 과거 로그는 보존한다. 세부 이전·진단 문서는 [고정 Git 버전](https://github.com/SangukBae/semantic_transmission/tree/f41bc8885e8d8409e59b162fe7be63c0d11ee737/docs)에서 열 수 있다.
