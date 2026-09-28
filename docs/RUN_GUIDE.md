# 설치·실행·복구

[과제 현황](README.md) · [실험 결과](ETRI_COMBINED_REFERENCE_DIAGNOSIS.md)

- 실행 위치: 저장소 루트.
- 별도 자료: 데이터·가중치·기존 결과는 Git에 미포함.

<a id="setup"></a>
## 환경 준비

- 검증 장비: WSL2 Ubuntu 22.04 · RTX 4080 16GB.
- 필수 도구: Linux Conda · NVIDIA 드라이버 · FFmpeg.
- `lgvsc`: 생성·움직임 추출·키프레임 전송·화질 평가 환경.
- `lgvsc-channel`: 디지털 통신 환경.
- `lgvsc-internvl`: 공식 키프레임 선택 환경.
- `.local/metric_v2_env`: 새 지표 평가 환경.

```bash
# 새 환경 설치; 기존 환경은 activate부터 실행
bash scripts/bootstrap_official.sh
source scripts/activate.sh
semtx doctor
python scripts/probe_environment.py

# 짧은 설치 점검; 새 출력 폴더 사용
semtx smoke --selector skim --skim-keyframes 2 --frames 9 --output outputs/setup_smoke_new
```

<a id="long-video"></a>
## 60초 실행

- 현재 결과: `tv_low_08` 전체 복원 완료; 의미 오류 검수·완화 효과는 미완료.
- 기본 조건: 576×320 · 24fps · AWGN 10 dB · 생성 30 steps.
- 실행 시간: 기존 사례의 키프레임 선택 약 23시간, 이후 처리 약 112분.

```bash
# 입력 검사 + 짧은 모델 점검
bash scripts/check_etri_60s.sh

# 실제 60초 전체 복원
bash scripts/run_etri_60s.sh

# 기존 tv_low_08 선택 결과를 재사용해 복구
bash scripts/recover_etri_60s.sh
bash scripts/recover_etri_60s.sh --status

# 다른 개발용 영상
bash scripts/run_etri_60s.sh --video tv_medium_02
```

- 사전 확인: `run`·`recover` 명령에 `--dry-run`을 붙이면 추론 없이 검사.
- 재개 범위: SKEM은 후보별, 캡션은 구간별 저장; 나머지는 완료 단계만 재사용.
- 생성 중단: 생성 구간 내부 체크포인트가 없어 미완료 생성 단계는 재실행.
- 실행 관리: WSL·터미널 유지 또는 tmux 사용; GPU 실험은 하나씩 실행.
- 결과 폴더: `outputs/etri_60s_tv_low_08_42057b2ee8ed/`.
- 결과 파일: `RESULT.json`은 판정·수치, `comparison.mp4`는 원본·복원 비교.

<a id="experiments"></a>
## 다른 실험 명령

- 17프레임 참조: `bash scripts/check_etri_tail_reference.sh`
- 두 개선안 결합: `bash scripts/check_etri_combined_reference.sh`
- 기존 ETRI 배치: `bash scripts/run_etri_remaining.sh`
- WebVid 비교: `bash scripts/run_webvid_ablation.sh` / `bash scripts/run_webvid5.sh`
- 품질 비교: `bash scripts/run_quality_validation.sh`
- 지표 평가: `bash scripts/run_metric_validation.sh` / `bash scripts/run_metric_revision.sh`

<a id="migration"></a>
## 다른 PC로 이전

- 복사: 데이터·실험 결과·모델 캐시를 별도 보존.
- 캐시: Hugging Face의 `snapshots` 링크와 실제 `blobs`를 함께 복사.
- 환경: 새 PC에서 Conda·모델 경로 재설정; 가상환경·드라이버 폴더 덮어쓰기 금지.
- 경로: 코드·캐시는 WSL Linux 파일시스템에 배치.
- 지표: 이전 결과·교정값·관측 캐시도 함께 준비.
- 기록: 과거 실험의 선언·경로·해시를 일괄 치환하지 않음.

<a id="verification"></a>
## 구현 검사

```bash
source scripts/activate.sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q --disable-warnings
python scripts/probe_environment.py
```

- 판정: 구현 검사 통과와 모델 복원·완화 성능은 각각 확인.
- 상세 안내: [설치·이전·재개 조건·결과 경로](https://github.com/SangukBae/semantic_transmission/blob/4f566feabe213b870e5c1e441ef9b369b348d589/docs/RUN_GUIDE.md).
