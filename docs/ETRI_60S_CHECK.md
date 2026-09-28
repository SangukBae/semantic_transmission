# ETRI 60초 실행 준비 점검

저장소에서 다음 한 명령을 실행한다. Conda 환경을 직접 활성화할 필요는 없다.

```bash
bash scripts/check_etri_60s.sh
```

기본 입력은 개발용 `tv_low_08`이다. 명령은 **60초 전체 입력 검증과 짧은 실제 모델 실행**을
수행한다. 60초 전체 SKEM·영상 생성이나 성능평가 캠페인을 시작하는 명령은 아니다.

## 수행하는 작업

1. 동결 manifest·영상 해시와 개발 분할을 검사한다. 576×320, 24fps, 1,440프레임,
   60초 입력만 허용하며 평가용·교정용 영상은 거부한다.
2. 결과 폴더의 `full/run_config.json`에 별도 개발 설정을 만든다.
   `frames=max_frames=1440`, `preserve_input=true`, `official_preprocessing=false`다.
   기존 WebVid 실행에서 사용한 BF16·CPU 보관 계층 3개·KV 캐시 압축 설정을 이어받는다.
   전체 실행 설정의 SKEM 후보 간격은 1프레임이다. AWGN 10dB, 생성 seed 2025,
   채널 seed 42를 사용한다.
3. 실제 `workers.prepare`로 1,440개 PNG를 만든다. 정규화 영상의 바이트가 원본과 같은지,
   모든 PNG 픽셀이 해당 원본 프레임과 같은지, 모든 PTS가 `i/24`인지 검사한다.
4. 실제 구간 연결 함수를 합성 프레임 번호 텐서로 검사한다. 단일 구간, 짧은 첫 구간을
   포함한 가변 구간, 매 프레임 경계의 경우 모두 1,440개 시점을 보존해야 한다.
5. 재개 자체 검사에서 중간 실패를 의도적으로 발생시킨다. 완료 단계 재사용,
   미완료 산출물 보존·재실행, 완료 파일 변조 거부를 확인한다.
6. 원본의 첫 25프레임으로 **SKEM 2회 비교**를 실행한다. 이 단계만 후보 간격 12를
   사용한다. 이는 환경 점검이며 전체 60초 키프레임 선택 결과가 아니다.
7. 별도 진단 경로에서 키프레임을 `[0, 8, 24]`로 고정한다. 실제 PLLaVA·UniMatch·NTSCC·
   AWGN·Open-Sora 30단계를 실행한다. 첫 구간 9프레임과 두 번째 구간을 통해
   짧은 참조 패딩, 이전 생성 참조의 전달, 최종 25프레임 시간축을 확인한다.
   고정 키프레임은 경계 경로를 반드시 실행하기 위한 것이며 SKEM 기준선으로 집계하지 않는다.

생성 모델 설정은 `decoder_policy=official_release`를 유지한다. 최종 연결은
`concatenation_policy=endpoint_exact`로 공유 경계 프레임을 한 번만 남긴다.
이는 조건부 생성의 끝점 정렬 방식과 별개다. `conditioning_alignment`는 기존 공식 경로를
유지하며, 위 설정 자체를 할루시네이션 완화로 주장하지 않는다.

## 결과와 재개

터미널에 `outputs/etri_60s_check_tv_low_08_<설정식별자>/` 경로가 표시된다.

| 산출물 | 내용 |
|---|---|
| `RESULT.json`, `REPORT.md` | 점검 결과와 완료하지 않은 범위 |
| `full/run_config.json`, `full/input_audit.json` | 60초 설정과 전체 입력 검사 |
| `temporal.json`, `resume_probe/result.json` | 연결·재개 자체 검사 |
| `selector/selector_resources.json` | 실제 SKEM의 PyTorch GPU 메모리 최고값 |
| `smoke/audit.json`, `smoke/receiver/reference_trace.json` | 짧은 생성·시간축·참조 전달 검사 |
| `smoke/receiver/reconstruction/` | 실제 짧은 복원 MP4와 PNG |
| `logs/`, `resources/`, `progress/`, `stages/` | 단계 로그·자원·진행률·해시 검증 영수증 |

같은 명령을 다시 실행하면 입력·코드·설정·환경·모델 파일 정보가 같은 결과 폴더를 찾고,
**완료 파일의 해시를 확인한 단계만** 재사용한다. Ctrl+C 또는 SIGTERM 시 자식 모델
프로세스도 종료한다. 실패한 단계는 처음부터 다시 실행하며, 그 산출물과 로그는
`failed_attempts/`에 보존한다. 모델 추론 도중의 내부 상태를 저장·복원하는 기능은 아니다.
코드·설정이 바뀌면 새 식별자를 사용한다. 명시한 `--output`이 다른 실행에 속하면 거부한다.

단계별 wall time과 GNU time의 최대 RSS를 기록한다. GPU 사용량은 약 2초마다 수집한
**장치 전체 메모리** 최고값이므로 순간 최고값을 놓칠 수 있고 다른 GPU 작업도 포함된다.
SKEM의 `selector_resources.json`에는 해당 프로세스의 PyTorch 할당·예약 최고값도 남는다.

```bash
# 입력·환경·기존 결과 검증만 수행. 추론·결과 파일 생성 없음
bash scripts/check_etri_60s.sh --dry-run

# CPU 검사만 수행. 실제 모델 실행 통과로 처리하지 않음
bash scripts/check_etri_60s.sh --cpu-only

# 다른 개발용 60초 영상 / 명시적인 결과 폴더
bash scripts/check_etri_60s.sh --video tv_medium_02 --output outputs/etri_60s_medium_check
```

## 통과의 의미

기본 명령의 성공 상태는 `PASS_60S_INPUT_AND_SHORT_MODEL_CHECK`이며 CPU 전용은
`PASS_CPU_CHECKS`다. 둘 다 **60초 전체 복원, 장시간 생성 구간의 GPU 메모리 수용량,
할루시네이션 완화 성능은 검증하지 않는다.** 기존 동결 입력·검수 주석·프로필은 수정하지 않는다.
이 점검 다음에는 `bash scripts/run_etri_60s.sh`로 실제 60초 기준선 복원을 실행한다.
[전체 복원 안내](ETRI_60S_RUN.md)에 실행 범위·재개·메모리 제약을 정리했다.

## 구현 검증

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/sangukbae/anaconda3/envs/lgvsc/bin/python -m pytest -q \
  tests/test_etri_60s_check.py tests/test_decoder_conditioning.py tests/test_webvid_ablation.py
```

단위·회귀 검사는 시간축 손상, 입력 변조·잘못된 분할, 자식 프로세스 실패,
참조 전달 기록 누락, CPU 검사의 상태 구분, 완료 단계 재사용을 검사한다.
실제 모델 완료 여부는 해당 실행 폴더의 `RESULT.json`을 기준으로 확인한다.

이번 RTX 4080 실행은 [결과 JSON](../outputs/etri_60s_check_tv_low_08_82644229e672/RESULT.json)에
`PASS_60S_INPUT_AND_SHORT_MODEL_CHECK`를 기록했다. 최초 점검의 실행 구간은 약 452초
(7.5분)이었으며 환경 사전 검사·구현 시간은 제외한다. 재실행에서는 19개 완료 단계를 모두
재사용했다. 관련 테스트 34개가 통과했고, 기존 동결 패키지의 1,329개 파일이 보존됐음을
[해시 검사](../outputs/etri_60s_check_tv_low_08_82644229e672/frozen_input_preservation.json)로 확인했다.
이는 이 입력·환경에서의 측정이며 다른 영상의 실행 시간을 보장하지 않는다.
