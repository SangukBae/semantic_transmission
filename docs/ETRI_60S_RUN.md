# ETRI 60초 복원 실행과 결과

갱신: **2026-09-28**. [과제 현황](README.md) · [실행 명령 모음](RUN_GUIDE.md)

## 현재 완료 상태

**2026-09-27 개발용 `tv_low_08`의 전체 복원이 완료됐다.** 실행 경로는
`outputs/etri_60s_tv_low_08_42057b2ee8ed/`이며 `PASS_60S_RECONSTRUCTION`을 기록했다.

| 항목 | 실제 결과 |
|---|---|
| 입력·출력 | 저전환 개발 영상 한 편, 60초·1,440프레임·24fps·576×320 |
| 키프레임·구간 | 110개 키프레임, 109개 생성 구간, 구간 간 참조 전달 108회 |
| 채널·생성 | AWGN 10 dB, 채널 seed 42, 생성 seed 2025, Open-Sora 30단계 |
| MP4 화질 | PSNR 14.85 dB, SSIM 0.648, LPIPS-VGG 0.388, CLIP 0.936, DISTS 0.181 |
| 실행 시간 | 기존 SKEM 23시간 19분. 이를 가져온 뒤 재개 실행 112.44분(캡션 47.8분, 생성 54.4분) |
| 의미 오류 검증 | `hallucination_review=PENDING`, `hallucination_mitigation_verified=false` |

전 프레임·시간축·산출물 무결성과 참조 전달을 검사했다. 이 성공은 한 편의 실행 가능성 근거이며,
화질 합격·할루시네이션 완화·장면전환 3유형 평가 완료를 의미하지 않는다.
[완료 결과](../outputs/etri_60s_tv_low_08_42057b2ee8ed/RESULT.json) ·
[결과 분석](../outputs/etri_60s_result_analysis_20260928/ANALYSIS.md) ·
[원본/복원 비교 영상](../outputs/etri_60s_tv_low_08_42057b2ee8ed/comparison.mp4)

## 실행·확인 명령

로컬 데이터·가중치·Conda 환경이 준비된 저장소 루트에서 실행한다.

```bash
# 새 60초 기준선 실행 또는 같은 실행의 검증된 단계 재사용
bash scripts/run_etri_60s.sh

# 기존 tv_low_08의 완료 SKEM을 검증해 가져오는 복구 경로
bash scripts/recover_etri_60s.sh

# 복구 경로의 상태 / 실행 없는 사전 검사
bash scripts/recover_etri_60s.sh --status
bash scripts/recover_etri_60s.sh --dry-run

# 다른 개발용 영상
bash scripts/run_etri_60s.sh --video tv_medium_02
```

복구 명령은 현재 완료된 실행의 재사용 경로다. 같은 입력·코드·환경이면 완료 단계의 해시를 검사해
재사용한다. 기존 SKEM 원본은 `outputs/etri_60s_tv_low_08_263507d45874`다.
다른 완료 SKEM은 `--reuse-selection-from outputs/PREVIOUS_RUN`으로 지정한다.
검증이 실패하면 중단하며 몰래 SKEM을 처음부터 실행하지 않는다.

`--stop-after input-audit`는 입력 검증 뒤 정지한다. `--output outputs/my_etri_60s`는 결과 경로를 지정한다.
코드·환경·설정·모델 파일 정보가 바뀌면 새 실행 식별자·폴더를 사용하고, 다른 실행의 출력 폴더와 혼합하지 않는다.
새 영상의 소요시간은 내용에 따라 달라진다. 첫 실행의 기존 추정 15~35시간보다 위 실측을 우선 참고한다.
WSL과 터미널을 유지하거나 사용자가 관리하는 tmux에서 실행한다.

## 고정 조건과 로컬 수정

- 실제 연속 입력 전체를 처리한다. SKEM threshold 0.35, 후보 간격 1프레임, BF16,
  CPU 보관 계층 3개, KV 캐시 압축을 사용한다.
- PLLaVA·UniMatch, NTSCC quality-4 경로를 사용한다.
  `semantic_clip_policy=frame_exact`로 `[시작 키프레임, 다음 키프레임)` 클립을 무손실 추출한다.
  캡션 전에 모든 클립의 프레임 수·PTS·원본 픽셀을 검사한다.
- 광류는 기존 `[0,10,20,30]` 표집 위치를 구간 끝으로 제한하고 순차 디코딩한다.
  매우 짧은 클립에서 반복 탐색 실패로 검은 프레임이 들어가던 문제를 피한다.
- `decoder_policy=official_release`, `conditioning_alignment=official_release`다.
  최종 연결의 `endpoint_exact`는 공유 경계를 한 번만 남기는 정책이며 생성 조건 정렬 변경과 별개다.
- 생성된 구간은 CPU에 보관하고 다음 참조에 필요할 때 같은 dtype으로 GPU에 복사한다.
  추가 키프레임·수동 캡션·프레임 축소를 적용하지 않는다.

한 구간 자체가 길면 GPU 메모리 부족이 남을 수 있다. 실패 시 로그를 남기며 기준선 설정을 자동으로 완화하지 않는다.
[모델 구조](MODEL_ARCHITECTURE.md)와 [논문·구현 감사](LGVSC_PAPER_IMPLEMENTATION_AUDIT.md)를 함께 읽는다.

## 중단과 재개

| 단계 | 재개 범위 |
|---|---|
| SKEM | 매 후보 비교 완료 시 저장. 마지막 완료 프레임 다음부터 최근 키프레임 상태를 복구; 현재 `do_sample=false` 경로 |
| 캡션 | 구간별 저장. 검증된 구간은 재사용하고 중단 당시의 한 구간은 다시 처리 |
| 광류·전송·생성·평가 | 완료 단계는 해시 검증 후 재사용. 미완료 단계는 처음부터 다시 실행 |

**생성 구간 내부 상태 체크포인트는 없다.** 실패 산출물은 `failed_attempts/`에 보존한다.
준비 점검과 전체 복원 명령은 GPU 잠금을 공유한다. 다른 연구 실행기는 별도 잠금을 쓰므로 같은 GPU에서 동시에 실행하지 않는다.

## 산출물과 복구 이력

아래 경로는 각 실행 폴더 기준이며 `outputs/`는 Git에 포함되지 않는다.

| 경로 | 내용 |
|---|---|
| `REPORT.md`, `RESULT.json` | 완료 상태·화질·전송량·미완료 의미 검증 |
| `baseline/receiver/reconstruction/sample_0000.mp4`, `comparison.mp4` | 복원 원본과 동기 비교 영상 |
| `baseline/output_audit.json`, `baseline/quality*.json`, `baseline/quality_*.csv` | PNG·MP4 프레임/PTS/참조 전달 검사와 화질 |
| `baseline/run_config.json`, `baseline/selection_audit.json`, `baseline/selection_reuse.json` | 설정·구간 길이·SKEM 가져오기 근거 |
| `baseline/semantic_clips_audit.json`, `checkpoints/` | 클립 검사, SKEM·캡션 진행 상태 |
| `status.json`, `stages/`, `logs/`, `progress/`, `resources/` | 진행·단계 영수증·로그·시간·메모리 |

2026-09-26 실행은 SKEM 완료 후 87번째 캡션 클립의 빈 디코딩 때문에 실패했다.
프레임 기준 추출과 구간별 캡션 저장을 추가하고, 완료 SKEM을 검증해 새 실행으로 가져왔다.
109개 클립 검사와 짧은 GPU 캡션·광류 검증 후 9월 27일 전체 복원을 완료했다.

- [실패 진단](../outputs/etri_60s_caption_diagnosis_20260927/diagnosis.json)
- [복구 구현 검증](../outputs/etri_60s_recovery_validation_20260927/validation.json): 당시 관련 테스트 86개, 이전 산출물 보존 확인
- [CPU 보관 경로의 GPU 출력 동일성](../outputs/etri_60s_runner_validation_20260926/decoder_parity.json): 25프레임 PNG 일치
- [후속 오류·조건 진단](ETRI_CONDITIONING_DIAGNOSIS.md): 전체 복원 이후 짧은 창의 개선 비교

구현 검사 재현 명령은 다음과 같다. 이 검사는 실제 60초 모델 실행을 대신하지 않는다.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/anaconda3/envs/lgvsc/bin/python -m pytest -q \
  tests/test_etri_60s.py tests/test_etri_60s_check.py \
  tests/test_caption_recovery.py tests/test_etri_selection_reuse.py \
  tests/test_decoder_conditioning.py tests/test_webvid_ablation.py tests/test_exact_reuse.py
```
