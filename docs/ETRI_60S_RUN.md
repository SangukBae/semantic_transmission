# ETRI 60초 전체 기준선 복원

저장소 폴더에서 다음 명령을 실행한다. 기본 대상은 개발용 `tv_low_08`이다.

```bash
bash scripts/run_etri_60s.sh
```

이 명령은 **60초 전체 1,440프레임**을 처리한다. SKEM의 1,439회 비교부터 캡션·광류,
NTSCC·AWGN 송수신, Open-Sora 생성, 시간축 검사, 품질 지표와 원본/복원 비교 영상까지 실행한다.
완료 전에는 `PASS_60S_RECONSTRUCTION`을 기록하지 않는다.

## 캡션 실패 실행 복구

2026-09-26 `tv_low_08`의 캡션 실패 후에는 다음 **한 명령**을 실행한다.

```bash
bash scripts/recover_etri_60s.sh
```

기본 원본은 `outputs/etri_60s_tv_low_08_263507d45874`이다. 완료 기록·입력·키프레임 파일·
SKEM 설정과 코드·모델·환경을 검사한 뒤, **110개 키프레임을 새 실행 폴더에 가져온다.**
약 23시간 19분이 걸렸던 SKEM 추론은 반복하지 않는다. 원래 실패 폴더는 수정하지 않는다.
검증에 실패하면 중단하며, SKEM을 몰래 처음부터 실행하는 대체 경로는 없다.

이어서 프레임 번호 기준으로 109개 클립을 만들고, 캡션 모델을 올리기 전에 모든 클립의
프레임 수·시간축·원본 픽셀을 확인한다. 캡션을 구간별로 저장하면서 광류·송수신·전체 복원·
품질 평가·비교 영상 생성까지 순서대로 진행한다. 기존 실행의 86개 캡션은 저장되지 않아
캡션은 처음부터 생성하지만, 이번부터 중단하면 마지막 저장 구간 다음부터 재개한다.

SKEM을 재사용한 뒤 남은 자동 실행은 **약 3~6시간 추정**이다. 캡션 실행 기록과 짧은 생성
시험을 바탕으로 한 값이며, 전체 복원 완료 실측값은 아니다.

```bash
# 실행 없이 기존 SKEM 재사용 가능 여부 검사
bash scripts/recover_etri_60s.sh --dry-run

# 다른 터미널에서 현재 복구 실행의 상태 확인
bash scripts/recover_etri_60s.sh --status

# 다른 완료 SKEM 실행을 명시적으로 선택
bash scripts/run_etri_60s.sh --video tv_medium_02 --reuse-selection-from outputs/PREVIOUS_RUN
```

복구 후 중단되면 **동일한 `recover_etri_60s.sh` 명령**을 다시 실행한다.
기존 실패 폴더를 `--output`으로 지정하지 않는다. 코드가 바뀐 실행은 별도 식별자와 폴더를 쓴다.

## 실행 조건

- 개발 분할에 속한 실제 연속 60초 입력, 576×320, 24fps, 1,440프레임.
- SKEM threshold 0.35, 후보 간격 1프레임, BF16, CPU 보관 계층 3개, KV 캐시 압축.
- 공식 PLLaVA·UniMatch, NTSCC quality-4, AWGN 10dB.
- v2는 `semantic_clip_policy=frame_exact`로 `[시작 키프레임, 다음 키프레임)` 구간을
  무손실 추출한다. 공식 공개 코드의 시간 반올림·stream-copy 클립 방식에서 발생한 빈 클립을
  수정한 로컬 변경이며, 기존 동결 프로필과 짧은 실행의 기본 동작은 바꾸지 않는다.
- v2 광류 표집은 같은 `[0,10,20,30]` 위치를 구간 끝으로 제한하되 순차 디코딩으로 읽는다.
  1프레임 클립의 반복 탐색 실패 시 검은 프레임을 넣던 공개 유틸리티 동작을 피한다.
- Open-Sora 30단계, 생성 seed 2025, 채널 seed 42.
- `decoder_policy=official_release`, `conditioning_alignment=official_release`.
- 최종 연결은 `endpoint_exact`로 공유 경계를 한 번만 남겨 1,440프레임을 유지한다.
  이 설정은 생성 끝점 조건의 정렬 변경과 별개이며, 완화 기법으로 집계하지 않는다.
- 생성한 구간의 텐서는 CPU 메모리에 보관한다. 다음 구간의 참조가 필요할 때 같은 dtype으로
  GPU에 복사한다. 프레임 축소·간격 변경·추가 키프레임·수동 캡션은 적용하지 않는다.

기존 동결 입력·주석·프로필은 수정하지 않으며 출력 폴더의 `baseline/run_config.json`에
해당 실행의 설정을 기록한다. 전체 평가 세트의 방법 동결이나 완화 효과 검증을 뜻하지 않는다.

## 시간과 메모리

현재 단편 실행 기록을 바탕으로 **첫 실행은 약 15~35시간**을 예상한다. 대부분은 SKEM이며
영상 내용에 따라 달라진다. 이는 60초 전체 완료를 실측한 시간이 아니다.
WSL·터미널을 유지하거나 사용자가 관리하는 tmux 세션에서 실행한다.

CPU 보관은 이미 생성한 구간들이 GPU 메모리에 누적되는 것을 줄인다. 하나의 SKEM 구간이
매우 길면 해당 구간 자체의 생성 메모리가 부족할 수 있다. 이때 명령은 실패 단계와 로그를
남기며, 임의로 키프레임을 늘리거나 생성 단계를 줄여 기준선 조건을 바꾸지 않는다.
`baseline/selection_audit.json`에 실제 구간별 생성 길이와 최장 길이가 기록된다.

## 결과 확인

터미널에 `outputs/etri_60s_tv_low_08_<실행식별자>/`가 표시된다.

| 경로 | 내용 |
|---|---|
| `REPORT.md`, `RESULT.json` | 전체 복원 완료 상태, 지표·전송량·영상 링크 |
| `baseline/receiver/reconstruction/sample_0000.mp4` | 실제 60초 복원 |
| `comparison.mp4` | 왼쪽 원본 / 오른쪽 복원, 같은 시간축 |
| `baseline/output_audit.json` | MP4·PNG 1,440프레임, 모든 PTS, 구간 참조 전달 검사 |
| `baseline/quality.json`, `baseline/quality_*.csv` | PNG·MP4 각각 PSNR·SSIM·LPIPS·CLIP·DISTS |
| `checkpoints/selector.json` | 완료된 SKEM 프레임 수와 최근 선택 상태 |
| `baseline/selection_reuse.json` | 가져온 SKEM의 원래 실행과 검증 근거 |
| `baseline/semantic_clips_audit.json` | 모든 구간의 프레임 수·시간축·픽셀 검사 |
| `checkpoints/caption.json` | 구간별 캡션·표집 정보와 재개 검증 정보 |
| `status.json`, `stages/`, `logs/`, `progress/`, `resources/` | 진행률·완료 파일 해시·로그·실행 시간·메모리 |

완료 검사는 파일·시간축·실행 무결성 검사다. Added/Missing/Distorted 검수와 완화 성능은
별도로 수행하며 결과에 `PENDING`으로 남긴다.

## 중단과 재개

같은 명령을 다시 실행하면 완료 파일의 해시를 검사한 뒤 재사용한다.
SKEM은 매 후보 프레임의 비교가 끝날 때 상태를 원자적으로 저장하므로 **마지막으로 완료한
프레임 다음부터** 이어 간다. 최근 키프레임을 복구해 원래의 순차 비교를 계속하며,
처리 도중 중단된 한 프레임의 비교는 다시 실행한다. 입력 픽셀·설정·실행 식별자가 달라진
체크포인트는 거부한다. 이 재개 경로는 `do_sample=false`인 현재의 탐욕적 선택에 적용된다.

캡션은 각 구간의 결과와 표집 정보를 원자적으로 저장한다. 재개 시 입력·클립·설정·모델·코드와
체크포인트 무결성을 확인하고 저장된 구간을 건너뛴다. 처리 도중 중단된 한 구간은 다시 실행한다.
광류·전송·생성·평가 등 다른 단계는 미완료 단계를 처음부터 재실행한다.
특히 **생성 구간 내부 상태를 체크포인트로 저장하는 기능은 없다.** 실패한 산출물과 로그는
`failed_attempts/`로 이동한다. 코드·환경·설정·모델 파일 정보가 달라지면 별도 결과 폴더를
사용하고, 지정한 `--output`이 다른 실행에 속하면 거부한다.

```bash
# 읽기 전용 입력·환경·기존 완료 파일 검사
bash scripts/run_etri_60s.sh --dry-run

# 실행 중 다른 터미널에서 진행 상태 확인
bash scripts/run_etri_60s.sh --status

# 다른 개발용 영상
bash scripts/run_etri_60s.sh --video tv_medium_02

# 입력 검증까지 실행하고 정지. 같은 기본 명령으로 다음 단계 진행 가능
bash scripts/run_etri_60s.sh --stop-after input-audit
```

`--stop-after`는 점검용이며 기본 명령은 모든 단계를 실행한다.
`--output outputs/my_etri_60s`로 결과 폴더를 고정할 수도 있다.
기존 `check_etri_60s.sh`와 GPU 잠금을 공유한다. 다른 연구 실행기는 별도 잠금을 쓰므로
같은 GPU에서 동시에 실행하지 않는 것이 좋다.

## 검증 범위

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/sangukbae/anaconda3/envs/lgvsc/bin/python -m pytest -q \
  tests/test_etri_60s.py tests/test_etri_60s_check.py \
  tests/test_caption_recovery.py tests/test_etri_selection_reuse.py \
  tests/test_decoder_conditioning.py tests/test_webvid_ablation.py tests/test_exact_reuse.py
```

실제 SKEM 반복문의 모델 호출을 테스트용 응답으로 대체해 중단 전후의 비교 순서와
키프레임 결과가 같은지 확인한다. 체크포인트 변조·입력 변경·짧은 결과의 성공 처리도 검사한다.
긴 전체 GPU 실행 완료 여부는 실행 폴더의 `RESULT.json`으로 판단한다.

2026-09-26 구현 검증에서는 관련 테스트 **52개**가 통과했다. 여기에는 모델 호출을 대체한
1,440프레임 전체 실행 순서·실제 FFmpeg 시간축·완료 단계 재사용 검사가 포함된다.
실제 GPU 비교에서는 CPU 보관 경로의 **25프레임 PNG가 기존 출력과 모두 바이트 단위로
일치**했다([동일성 결과](../outputs/etri_60s_runner_validation_20260926/decoder_parity.json)).
당시 실제 `tv_low_08`은 전체 입력 전처리·검증까지 수행하고 정지했다.
이 검증 기록은 60초 전체 SKEM·GPU 복원 완료 기록이 아니다.

이후 2026-09-26 전체 SKEM은 완료됐으나 캡션 87번째 클립의 디코딩 가능한 프레임 수가 0이어서
실패했다. [진단 기록](../outputs/etri_60s_caption_diagnosis_20260927/diagnosis.json)에 원인과
109개 클립 검사가 남아 있다. 복구 명령 구현·부분 검증과 60초 최종 복원 완료는 구분한다.

2026-09-27 복구 구현 검증에서는 관련 테스트 **86개**가 통과했다. 실제 원본의 키프레임
110개를 검증해 가져왔고, 109개 클립 모두 프레임 수·PTS·원본 픽셀 검사를 통과했다.
문제가 발생한 원본 1259번 프레임의 1프레임 구간과 다음 6프레임 구간으로 실제 GPU 캡션·
광류를 실행했다. 저장된 캡션은 모델을 다시 로딩하지 않고 같은 해시로 복구됐다.
기존 실패 실행 1,697개 파일과 동결 입력·문서 1,329개 항목의 해시가 유지됐다.
[검증 기록](../outputs/etri_60s_recovery_validation_20260927/validation.json)에 근거를 저장했다.

실제 복구 실행은 `outputs/etri_60s_tv_low_08_42057b2ee8ed`에서 `semantic-clips`까지 완료하고
정지했다. 같은 복구 명령으로 이 여섯 단계를 재사용하는 것까지 확인했다. 현재 코드·환경에서
`bash scripts/recover_etri_60s.sh`를 실행하면 캡션부터 진행한다. 60초 전체 복원은 아직 미완료다.
