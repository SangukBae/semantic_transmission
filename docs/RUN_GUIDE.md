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

<a id="local-extraction"></a>
## 로컬 캡션·전체 원본 추출

- Qwen3-VL 자동 배치의 설치·새 실행·상태 확인: [전용 안내](FC_LGVSC_LOCAL_EXTRACTION.md#status).
- Qwen3.5 NF4/INT8 환경·단일 캡션·고정 비교·문맥 실험: [전용 안내](QWEN35_LOCAL_SETUP.md).
- 전체 원본 보정 상태와 미완료 범위: [데이터 기록](DATA.md#full-extraction). 아래 명령은 중단한 **기본 보정본** 재개용이며 현재 실행하지 않은 명령이다.

```bash
PYTHONPATH=src /home/sangukbae/anaconda3/envs/lgvsc-internvl/bin/python \
  scripts/correct_fc_dataset_extraction.py run
```

- 원자적으로 저장한 쌍별 점수를 재사용하며 진행 중이던 한 쌍은 재계산할 수 있다. Qwen 비교를 위해 중단한 기록과 명령은 `outputs/fc_lgvsc_webvid_tvsum_faithful_20261002/paused_for_qwen_caption_comparison.json`이 기준이다.
- `_v2`는 기본 보정본 102편을 연결하고 WebVid 031만 별도 수정한 경로다. 위 명령만으로 `_v2`의 재선택·변경 캡션 검수·전체 완료가 보장되지 않는다. 인계 파일의 남은 작업을 확인한다.
- 문맥 추출 `RESULT.json`의 `full_78_accuracy_evaluated=false`는 추출 종료 시점 기록이다. 후속 AI 검토는 별도 `_accuracy/RESULT.json`과 [통합 결과](ETRI_COMBINED_REFERENCE_DIAGNOSIS.md#qwen-context)에서 확인한다. 기존 기록을 덮어쓰지 않는다.

<a id="default-method"></a>
<a id="caption-v2"></a>
## 기본 복원 — FC-LGVSC + 두 가지 구현 오류 수정

```bash
# 기본: tv_low_08 60초
bash scripts/reconstruct_fc_lgvsc.sh

# 보행 10초
bash scripts/reconstruct_fc_lgvsc.sh --video person_walk

# 준비 검사만; 모델 실행 없음
bash scripts/reconstruct_fc_lgvsc.sh --check
```

- **기본 구성:** 혼합 키프레임·캡션 v2·마지막 17프레임 참조·T5 FP32 저장값 재사용.
- **추가 수정:** 시작·끝 키프레임 조건 충돌 방지 + 2~16프레임 구간의 생성 시간표 정상화.
- **다른 영상:** `--video single_subject`(밀밭 인물), `--video candle_flowers`(등갓). 준비된 4편 지원.
- **실행:** 기존 FP32 수신 자료로 한 번 복원 → 검사 → 화질 평가 → 원본·수정 전·기본 방식 비교. 선택·캡션·전송·T5 계산 재실행 없음.
- **통제:** 기존 VAE·초기·반복 생성 잡음 해시 일치 검사. 생성기 BF16·30단계 유지.
- **표시:** 진행률·GPU 전체 사용률을 한 줄로 갱신. 상세 로그는 `.local/reconstruction_console/fc-lgvsc-*/console.log`.
- **출력:** 기존 FP32 결과와 나란히 `*_fc_lgvsc_default_v1/review.html`·`RESULT.json` 저장. `--output 새경로` 지정 가능.
- **재개:** 같은 명령 사용. 완료 단계는 검증 후 재사용; 중단된 생성 단계는 처음부터 실행.
- **채택 근거:** 보행 10초에서 두 수정 결합 검증 완료. 초반 모자이크 해소·평균 개선, 후반 일부 왜곡 잔존. [결과](ETRI_COMBINED_REFERENCE_DIAGNOSIS.md#schedule-fix-person)
- **검증 범위:** 새 명령으로 4편 전체 복원·화질 평가 완료. 평균 개선과 새 왜곡 공존; 독립 할루시네이션 완화는 미검증. [결과](ETRI_COMBINED_REFERENCE_DIAGNOSIS.md#fc-lgvsc-four)
- **이전 명령:** `reconstruct.sh` 등은 완료 실험의 해시 검증 때문에 수정 전 비교용으로 보존. 앞으로는 위 명령 사용.

<a id="fc-lgvsc-all"></a>
### 현재 FC-LGVSC로 4편 일괄 복원

```bash
bash scripts/reconstruct_fc_lgvsc_all.sh

# 4편 준비 검사만; 모델 실행 없음
bash scripts/reconstruct_fc_lgvsc_all.sh --check
```

- **순서:** `tv_low_08`(60초) → `single_subject`(약 14초) → `candle_flowers`(약 7.5초) → `person_walk`(10초·기존 이름 `01_person_walk`).
- **방식:** 위 기본 실행기로 한 편씩 복원·검사·평가. 캡션 v2·T5 FP32·마지막 17프레임 참조·두 구현 오류 수정 모두 적용.
- **재사용:** 준비된 키프레임·캡션·수신 자료·T5 저장값 사용. 네 편의 입력을 모두 검사한 뒤 첫 복원 시작.
- **화면:** 전체 진행률·GPU 사용률·현재 영상 한 줄 표시. 진행률은 생성 반복 수 기준이며 남은 시간 비율은 아님.
- **출력:** 개별 기본 명령과 같은 `*_fc_lgvsc_default_v1/` 폴더. 완료 시 4개의 `review.html` 경로 표시.
- **재개:** 같은 명령 사용. 완료 단계는 해시 검증 후 재사용; 실패 시 다음 영상으로 넘어가지 않음. 상세 로그는 `.local/reconstruction_console/fc-lgvsc-all-*/<영상>.log`.
- **별도 저장:** `--output-root outputs/fc_lgvsc_four_new`를 붙이면 그 아래 영상별 새 폴더 사용. 기존 결과는 보존.
- **실행 완료:** 사용자 일괄 실행으로 4편 복원 완료. 생성 합계 56분 53초, 첫 단계부터 마지막 비교 완료까지 약 63분 51초; T5 저장값 재사용. [기존 LGVSC 비교](../outputs/fc_lgvsc_four_comparison_20261001/review.html)

<a id="legacy-bf16"></a>
## 이전 BF16 비교용 — 마지막 17프레임 참조 + T5 재사용

```bash
bash scripts/reconstruct.sh

# 준비 검사만; 모델 실행 없음
bash scripts/reconstruct.sh --check

# 캡션의 T5 변환·저장까지만; 영상 생성 없음
bash scripts/reconstruct.sh --prepare-text-only
```

- 당시 구성: 혼합 키프레임 + AI 캡션 v2 → 매 구간 마지막 17프레임을 참조하는 LGVSC 복원. 두 구현 오류 수정은 미포함.
- 터미널: `복원 진행률:  12.34% | GPU 사용률:  87%` 한 줄 갱신. 생성 78구간×30단계 기준; 준비 중 0%·평가/비교 중 99.99%·정상 종료 시 100%.
- GPU: 사용 중인 GPU 전체 사용률을 1초마다 조회. 다른 프로그램 부하도 포함하며 조회 불가 시 `--%`; CPU에서 T5 준비 중에는 GPU 사용률이 낮을 수 있음.
- 상세 로그: 단계별 `logs/`와 `.local/reconstruction_console/run-*/console.log`에 보관. 실패·Ctrl+C 때 로그 위치 표시; `--check`·`--prepare-text-only`는 기존 안내 출력 유지.
- 현재 대상: `tv_low_08` 60초·79장·78캡션. 다른 영상은 같은 기준으로 선택·캡션을 새로 준비해야 함.
- 사전 자료: 기존 원본·모델·선택·v1/v2 완료 결과 필요; 현재 WSL에 준비됨.
- 실행 내용: 기존 수신 자료 → T5 GPU 사전 계산·저장 → T5 메모리 해제 → 복원 → 평가·비교. 선택·캡션 생성·전송 재실행 없음.
- T5 재사용: 같은 모델·문장·설정이면 저장한 조건 사용; 반복 실행에서 T5 모델 로딩 생략. 저장 위치 `.local/cache/t5_embeddings/`.
- 정밀도: T5만 기존 CPU FP32 → GPU BF16. 문장·움직임 점수·생성 30단계는 유지; 수치·복원 화질의 동일성은 미검증.
- 재개: 같은 명령 입력; 완료 단계는 검증 후 재사용, 중단된 생성 단계는 재실행.
- 시간: 이번 복원 42분 28초, T5 캐시 준비·검사 28초, 평가·비교 포함 단계 합계 46분 8초. 최초 T5 사전 계산 48.7초는 이전에 수행한 별도 비용.
- 상태: 60초 복원 완료, T5 저장값 78개 재사용. 이전 v2 대비 PSNR 개선·다른 지표 4개 소폭 악화; 할루시네이션 완화 입증은 미완료. [결과](ETRI_COMBINED_REFERENCE_DIAGNOSIS.md#tail17-t5-result)
- 결과 폴더: `outputs/etri_hybrid_keys_tv_low_08_v1/reconstruction_assistant_captions_v2_tail17_t5gpu_console/`; 복원 완료 후 `review.html`·`RESULT.json` 생성.
- 이전 기록 보존: 표시 실행기가 바뀌어 새 결과 폴더 사용. 기존 `_t5gpu/`의 준비 기록·검증 자료는 보존하며 T5 저장값은 공유.
- 새 폴더 지정: `bash scripts/reconstruct.sh --output outputs/etri_hybrid_keys_tv_low_08_v1/reconstruction_tail17_new`; 기존 결과와 같은 상위 폴더의 새 경로 사용.
- T5 변경 전 비교용: `bash scripts/reconstruct_tail17.sh` → `_v2_tail17/`. 참조 변경 전: `bash scripts/reconstruct_hybrid_v2.sh`.
- 처리 기록: `run/receiver/text_embeddings.json`에 계산·재사용 수, 소요 시간, 모델·조건 해시 저장. 복원 시 `text_embedding_trace.json`으로 전체 구간 사용 검사.
- 이전 결과: [기존 LGVSC / 변경 전 v2](../outputs/etri_hybrid_keys_tv_low_08_v1/reconstruction_assistant_captions_v2/analysis_lgvsc/review.html) · [v1/v2 비교](../outputs/etri_hybrid_keys_tv_low_08_v1/reconstruction_assistant_captions_v2/caption_comparison.html).
- 비교 한계: 기존 v2 대비 참조 방식과 T5 정밀도가 함께 변경됨. 초기 생성 잡음 완전 일치는 미보장; 첫 구간 키프레임 조건 충돌도 별도 문제.
- 기준·한계: [선택·캡션 작성 규칙](ETRI_DEVELOPMENT_PLAN.md#default-method). 한 개발 영상의 결과이며 캡션 단독 효과·일반화는 미확정.

<a id="short3"></a>
## 수정 전 세 영상 FP32 비교용 복원

```bash
# 세 영상을 순서대로 복원
bash scripts/reconstruct_short3.sh

# 준비 검사만; 모델 실행 없음
bash scripts/reconstruct_short3.sh --check

# 한 영상만 복원
bash scripts/reconstruct_short3.sh --video single_subject
```

- 대상: `single_subject` 14.04초 / `candle_flowers` 7.54초 / `person_walk` 10초. 기존 복원과 같은 입력 구간.
- 영상 내용: 밀밭·스카프·인물 / 노란 등갓의 초점 변화 / 벽 앞 보행·방향 전환. `candle-flowers`는 실제 장면과 다른 기존 파일명.
- 복원 완료: 순서대로 키프레임 **20 / 9 / 16장**, 캡션 v2 **19 / 8 / 15개**. [키프레임·캡션 기록](../outputs/etri_latest_short3_20261001/review.html) · [기존 LGVSC 비교](../outputs/etri_latest_short3_20261001/analysis_lgvsc/review.html).
- 시간: T5+생성은 **12분 / 6분 21초 / 9분 31초**, 세 영상의 명령 전체는 약 **33분 2초**. 미리 수행한 선택·AI 캡션 작성 제외.
- 결과: 세 영상의 PSNR·SSIM·LPIPS·DISTS 평균 개선, 밀밭·보행 CLIP 소폭 악화. 전송량 +39.6% / +192.0% / +443.4%; 보행 0.33초 화면 붕괴와 등갓 4.13초 새 무늬 등은 미해결.
- 구성: 미리 선택한 혼합 키프레임·AI 캡션 v2 → 움직임 추출·AWGN 10dB 전송 → 마지막 17프레임 참조·T5 CPU FP32 저장·생성 30단계.
- 화면: 전체 진행률·GPU 사용률·현재 영상 한 줄 표시. 자세한 로그는 `.local/reconstruction_console/short3-*/`에 저장.
- 재개: 같은 명령 사용. 완료 단계를 해시 검증 후 재사용하며, 실패한 단계만 다시 실행. 선택·캡션 작성은 반복하지 않음.
- 자료: `outputs/etri_latest_short3_20261001/<영상>/selected_frames/`, `assistant_captions/captions_bundle.json`.
- 결과: 각 영상의 `reconstruction_fp32/review.html`. 원본·기존 LGVSC·수정 전 FP32 결과를 같은 시각으로 비교하며 기존 출력의 중복 경계 프레임은 비교용 사본에서 제거.
- 비교 범위: 공통 키프레임의 수신 심벌 재사용, 새 키프레임에는 고정 AWGN. 과거 생성 잡음과 동일한 비교는 아님.

<a id="t5-fp32"></a>
## T5 FP32 복원 — 잡음 고정·기록

```bash
bash scripts/reconstruct_fp32.sh

# 준비 검사만; 모델 실행 없음
bash scripts/reconstruct_fp32.sh --check
```

- 대상: 같은 `tv_low_08` 60초·79장·78캡션; 마지막 17프레임 참조·영상 생성 BF16·30단계 유지.
- T5: 기존처럼 **CPU FP32로 새 계산** 후 저장·재사용. 4.7B 모델의 FP32 가중치는 16GB GPU보다 큼; 최초 준비는 GPU BF16보다 느릴 수 있음.
- 잡음: 수신 키프레임·이전 영상의 VAE 표본, 초기 생성 잡음, 생성 30단계의 추가 잡음을 각각 고정. 시드·모든 잡음 해시를 `run/receiver/generation_noise.json`에 기록.
- 화면: 진행률·GPU 사용률을 한 줄로 표시; 상세 로그는 기존 위치에 보관.
- 결과: `outputs/etri_hybrid_keys_tv_low_08_v1/reconstruction_assistant_captions_v2_tail17_t5fp32_fixednoise/review.html`.
- 재개: 같은 명령 사용. 완료 단계·캡션 캐시는 검증 후 재사용; 중단된 영상 생성은 처음부터 재실행.
- 비교 한계: 과거 실행에는 잡음 기록이 없어 과거 BF16과 동일 잡음 비교는 불가. 새 잡음 기준의 FP32 결과만으로 T5 원인을 확정하지 않음.
- 상태: FP32 준비 7분 22초·영상 생성 37분 54초·평가 완료. 비교 페이지의 영상 크기 목록 표시 오류를 2026-10-01 수정해 전체 완료; 모델·평가 재실행 없이 기존 결과 재사용.
- 분석: 이전 BF16 대비 최종 평균 지표 5개 소폭 개선, 일부 큰 오류 개선과 새 오류 공존. 준비·평가 포함 48분 13초; T5만의 효과는 미확정. [상세 비교](ETRI_COMBINED_REFERENCE_DIAGNOSIS.md#t5-fp32-result)

FP32 완료 후 **같은 잡음의 BF16 비교**가 필요할 때만 아래 명령 실행. 두 새 실행의 잡음 해시가 다르면 중단하며, FP32는 CPU·BF16은 GPU여서 계산 장치 차이도 포함됨.

```bash
bash scripts/reconstruct_fp32.sh --t5-precision bf16 \
  --noise-reference outputs/etri_hybrid_keys_tv_low_08_v1/reconstruction_assistant_captions_v2_tail17_t5fp32_fixednoise/run/receiver/generation_noise.json
```

- BF16 결과는 별도 `_tail17_t5bf16_fixednoise/`에 저장. 기존 결과·기존 캐시는 보존.
- 새 FP32/BF16 캐시: `.local/cache/t5_precision_embeddings/`; 정밀도별 별도 저장, 기존 BF16 값을 FP32로 형변환해 대체하지 않음.

<a id="condition-fix"></a>
## 조건 충돌 수정 단독 비교 — 과거 실험 재현용

```bash
# 보행 영상 10초: 같은 입력·잡음으로 충돌 수정안 복원
bash scripts/reconstruct_condition_fix.sh --video person_walk

# tv_low_08 60초
bash scripts/reconstruct_condition_fix.sh --video tv_low_08

# 준비 검사만; 모델 실행 없음
bash scripts/reconstruct_condition_fix.sh --video person_walk --check
```

- **수정:** 조건이 겹치는 구간만 끝 키프레임 위치 변경. 각 영상의 첫 구간 1곳이 대상.
- **재사용:** 기존 수신 키프레임·캡션·T5 FP32 저장값. 키프레임 선택·설명 생성·전송 재실행 없음.
- **비교:** 이전 FP32의 VAE·초기·반복 생성 잡음 해시 일치 검사; 조건 위치 변경 효과 비교용.
- **출력:** 기존 FP32 결과의 옆 폴더 `*_condition_fix/review.html`; 원본·수정 전·수정 후 영상 및 지표 비교.
- **진행:** 진행률·GPU 사용률 한 줄 표시. 같은 명령으로 완료 단계 재사용; 중단된 생성은 처음부터 실행.
- **다른 영상:** `--video single_subject` 또는 `--video candle_flowers`. 현재 두 영상은 조건 충돌이 없어 변경 대상 0개.
- **지원 범위:** 준비된 4편 검사 완료. 압축 칸이 하나뿐인 첫 구간(4프레임 이하)은 실행 전 중단.
- **상태:** 보행 복원·비교 완료. 0.33초 개선과 잔여 깨짐·후반 악화 공존. 현재 기본에는 아래 시간표 수정도 함께 포함. [결과](ETRI_COMBINED_REFERENCE_DIAGNOSIS.md#condition-fix-person) · [처리](MODEL_ARCHITECTURE.md#condition-fix)
- **후속 진단:** 남은 초반 모자이크는 생성 시간표 오류로 확인. 아래 별도 명령으로 시간표까지 수정해 비교. [검사](ETRI_COMBINED_REFERENCE_DIAGNOSIS.md#vae-short-schedule)

<a id="schedule-fix"></a>
### 시간표 수정 추가 비교 — 과거 실험 재현용

```bash
bash scripts/reconstruct_schedule_fix.sh --video person_walk
```

- **대상:** 완료된 보행 조건 충돌 수정 결과의 수신 자료·캡션·T5·난수 재사용.
- **수정:** 생성기에 들어가는 2~16프레임 구간만 시간표 수정; 현재 보행에서는 첫 9프레임 1곳.
- **검사:** 모든 구간의 생성 시각·고정 조건 보존·잡음 갱신 검사 후 전체 화질 평가.
- **표시:** 기존과 같은 한 줄 진행률·GPU 사용률. `--check`는 모델 실행 없이 준비 검사.
- **출력:** `outputs/etri_latest_short3_20261001/person_walk/reconstruction_fp32_condition_fix_schedule_fix/`.
- **결과:** 보행 10초 완료, 생성 7분 40초. 초반 개선·후반 일부 악화; [분석·비교 영상](../outputs/etri_latest_short3_20261001/person_walk/reconstruction_fp32_condition_fix_schedule_fix/analysis/review.html).
- **재개:** 같은 명령 재실행 시 검증된 완료 단계 재사용. 새 복원은 [기본 명령](#default-method)에서 두 수정을 한 번에 적용.

<a id="long-video"></a>
## 비교용 기존 SKEM·PLLaVA — 60초 실행

- 현재 결과: `tv_low_08` 전체 복원 완료; 의미 오류 검수·완화 효과는 미완료.
- 비교 조건: 576×320 · 24fps · AWGN 10 dB · 생성 30 steps.
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

- 선택 속도 비교: `bash scripts/run_skem_speed_benchmark.sh`
- 17프레임 참조: `bash scripts/check_etri_tail_reference.sh`
- 두 개선안 결합: `bash scripts/check_etri_combined_reference.sh`
- 기존 ETRI 배치: `bash scripts/run_etri_remaining.sh`
- WebVid 비교: `bash scripts/run_webvid_ablation.sh` / `bash scripts/run_webvid5.sh`
- 품질 비교: `bash scripts/run_quality_validation.sh`
- 지표 평가: `bash scripts/run_metric_validation.sh` / `bash scripts/run_metric_revision.sh`

### 키프레임 선택 속도 비교

- 조건: 기존 BF16 / 8비트 / 설명 간략화 / 초당 4장 후보 검사.
- 설명 간략화: 키프레임 선택용 설명만 축약; 전송용 캡션 설정은 동일.
- 범위: `tv_low_08`의 사람·차량·문, 각 25프레임; 출력은 24fps 유지.
- 검사: 선택 시간·실제 복원·전송량; 완료 결과는 재사용.
- 장면전환: 개발 영상 3편에서 후보 포함 여부만 별도 검사; 복원 평가는 별개.
- 결과: `outputs/skem_speed_20260929/review.html`과 `SUMMARY.json`.
- 시간 기준: 세 구간 기준선은 검증한 과거 선택 로그; 부하 변동 때문에 전체 가속 배수는 미확정.
- 별도 재측정: 게임 종료 후 같은 프레임 쌍을 반복; `timing_recheck/`에 측정·실패 이력 저장.
- 8비트 준비: 기존 환경을 바꾸지 않고 의존성을 별도 폴더에 설치.

```bash
"$(python3 -c 'import json; print(json.load(open(".local/settings.json"))["internvl_python"])')" \
  -m pip install --no-deps --target .local/selector_speed_deps bitsandbytes==0.43.3
bash scripts/run_skem_speed_benchmark.sh
```

- 주의: 짧은 개발 실험이며, 60초 검증·기본 선택기 교체를 뜻하지 않음.
- 판정: 12개 복원 완료; 새 오류로 세 방법 모두 채택 보류. [결과](ETRI_COMBINED_REFERENCE_DIAGNOSIS.md#selector-speed)

### AI 직접 선택 결과 재사용

- 선택 목록·이유: [assistant_selection.json](../outputs/etri_visual_keys_20260929/assistant_selection.json).
- 원본 키프레임 85장: `outputs/etri_visual_keys_20260929/selected_frames/`.
- 60초 비교·판정: [review.html](../outputs/etri_visual_keys_20260929/review.html); 새 오류로 기본값 대체 보류.
- 선택 목록은 같은 원본·프레임 번호에 한해 재사용; 선택 이후 설명 생성·복원 비용은 별도.

<a id="hybrid-selection"></a>
### 비교용 혼합 키프레임 + PLLaVA / 캡션 v1

- 대상: `tv_low_08` 60초; AI 후보·필수 사건 + SKEM 중복 판단 + 최대 1초 간격.
- 현재 상태: 79장 + AI 캡션 v1/v2로 각각 60초 복원 완료; 같은 79장의 PLLaVA 조건은 미실행.
- 소요 시간: AI 캡션 조건의 후속 처리 약 53분(영상 생성 약 48분); 사전 선택·설명 작성 시간 제외.
- 복원 명령: 저장된 선택을 재사용해 설명·움직임 추출 → 전송 → 복원 → 평가·비교 영상 생성.

```bash
bash scripts/reconstruct_hybrid.sh

# AI가 미리 작성한 78개 설명 사용; PLLaVA 실행 생략
bash scripts/reconstruct_hybrid.sh --captions
```

- 준비 검사만: `bash scripts/reconstruct_hybrid.sh --check`; 모델 실행 없이 입력·경로 검사.
- AI 캡션 준비 검사: `bash scripts/reconstruct_hybrid.sh --captions --check`.
- AI 캡션: [구간별 설명·표본](../outputs/etri_hybrid_keys_tv_low_08_v1/assistant_captions/review.html); 기존과 같은 4개 프레임 위치를 보고 작성, 복원 성능 우위는 미검증.
- 선택만 추출: `bash scripts/extract_hybrid_keyframes.sh`; 완료된 선택은 검증 후 재사용.
- 재개: 같은 복원 명령 입력; 완료 단계는 재사용, 캡션은 구간별 재개, 미완료 생성 단계는 다시 실행.
- 진행 표시: 단계 시작·완료만 출력; 상세 로그는 아래 결과 폴더의 `logs/`.
- 키프레임: [목록·미리보기](../outputs/etri_hybrid_keys_tv_low_08_v1/selection.html), `outputs/etri_hybrid_keys_tv_low_08_v1/selected_frames/`.
- 복원 결과: `outputs/etri_hybrid_keys_tv_low_08_v1/reconstruction/` 아래 `review.html`·`RESULT.json`.
- AI 캡션 결과: `--captions` 사용 시 같은 상위 폴더의 `reconstruction_assistant_captions/`에 별도 저장.
- 최신 분석: [수치·오류 표본](../outputs/etri_hybrid_keys_tv_low_08_v1/reconstruction_assistant_captions/analysis/review.html); 전송량 감소, 새 왜곡으로 대체 보류.
- 복원 영상: 같은 폴더 아래 `run/receiver/reconstruction/sample_0000.mp4`.
- 실행 환경: 기존 WSL·모델·원본·기준선 결과 필요; 별도 환경 활성화 없이 명령 실행.
- 비교 한계: 키프레임이 달라지면 구간·캡션·생성 잡음도 달라짐; 화질·새 오류는 복원 후 별도 검수.
- PLLaVA 비교: 같은 79장으로 위 명령의 옵션 없음·`--captions`를 각각 실행해야 PLLaVA/v1 차이를 비교할 수 있음. 현재 기본 v2 명령은 [위 안내](#default-method) 참고.

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
