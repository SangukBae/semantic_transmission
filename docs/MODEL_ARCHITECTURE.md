# 모델 구조

[과제 현황](README.md) · [개선 실험](ETRI_COMBINED_REFERENCE_DIAGNOSIS.md#decoder)

## 전체 흐름

**영상 → 대표 프레임·설명·움직임 추출 → 전송 → AI 영상 생성 → 시간순 연결**

<a id="fc-lgvsc"></a>
## 현재 기본 방식 — FC-LGVSC

**FC-LGVSC (Faithful Caption-guided LGVSC): 원본에 충실한 캡션 v2와 혼합 키프레임으로 영상을 복원하는 LGVSC 파생 방식.**

- **선택:** AI가 변화 후보·필수 사건을 고르고 SKEM이 중복 판단; 최대 1초 간격으로 키프레임 보강.
- **설명:** 원본 표본 4장으로 AI가 작성한 수정 캡션 v2; 위치·가림·움직임·흐림을 명시하고 추측 배제.
- **전송:** 키프레임·캡션·UniMatch의 움직임 정보를 NTSCC·디지털 전송 경로로 보내며 AWGN 잡음 적용.
- **복원:** LGVSC의 DSA/Open-Sora가 수신 정보와 직전 구간의 마지막 17프레임을 참고해 영상 생성.
- **T5 처리:** 검증에 사용한 CPU FP32 문장 조건을 저장·재사용. 영상 생성기는 BF16·30단계 유지.
- **조건 충돌 방지:** 시작·끝 키프레임이 같은 칸에 겹치면 끝 조건을 마지막 칸으로 이동.
- **시간표 수정:** 2~16프레임 구간의 길이 계산을 고쳐 초기 잡음이 실제로 제거되도록 처리.
- **준비 방식:** AI 후보·캡션은 원본을 확인해 미리 작성·저장한 자료를 불러옴.
- **실행:** `bash scripts/reconstruct_fc_lgvsc.sh`; 다른 영상은 `--video person_walk|single_subject|candle_flowers` 중 하나 지정. [선택·작성 기준](ETRI_DEVELOPMENT_PLAN.md#default-method) · [실행 안내](RUN_GUIDE.md#default-method)
- **채택·검증:** 2026-10-01 두 수정을 개발 기본에 추가하고 4편 전체 복원 완료. 기존 LGVSC 대비 평균 화질 개선, 일부 새 왜곡·보행 시간 증가. 독립 완화 효과는 미검증. [비교](ETRI_COMBINED_REFERENCE_DIAGNOSIS.md#fc-lgvsc-four)

<a id="caption-providers"></a>
## 캡션·선택 경로와 출처

| 경로 | 실제 처리 | 연결 상태 |
|---|---|---|
| 기존 LGVSC | InternVL 순차 SKEM + PLLaVA 캡션 | 비교 기준선 |
| 현재 FC-LGVSC | assistant가 원본을 보고 후보·v2 캡션 작성, 선택 후보는 SKEM 검사 | 준비된 4편 복원 완료 |
| 로컬 자동 전처리 | Qwen3-VL-8B INT8 후보·캡션 + TransNetV2·픽셀 컷 후보 + 기존 SKEM | 별도 배치 중단, 복원 연결 미검증 |
| Qwen3.5 캡션 실험 | Qwen3.5-9B NF4로 고정 구간의 캡션 생성 | 문맥 유무 비교까지 완료, 기본 복원 미적용 |
| Qwen3.5 선택 실험 | 직전 선택 프레임과 후보의 짧은 단일 호출 판정 | 30쌍 시간 측정; SKEM 점수 동등성·전체 선택 품질 미검증 |

- **기존 캡션 import:** 구간별 `[시작, 끝)`의 PLLaVA와 같은 4개 샘플 인덱스, 원본·선택·샘플 해시와 캡션 bundle을 검증. `assistant_provided`로 출처를 남기고 PLLaVA 추론은 생략. 샘플 위치가 같아도 작성 모델·프롬프트·정확도가 같다는 뜻은 아님.
- **문맥 입력:** 현재 구간 표본 일부를 직전 최대 2초·최대 4장으로 교체. 두 조건의 이미지 수·시각 토큰 예산을 맞추고 원래 4개 샘플은 유지. CONTEXT/TARGET와 시각을 표시하고, 검출한 컷을 넘는 문맥은 사용하지 않음.
- **출력 검증:** 새 일반 실행은 최대 79단어·빈 출력·잘림을 검사. 형식 검사는 사실 검증이 아니며, 같은 VLM의 재검토도 독립 정답이 아님.
- **미연결 범위:** 자동 캡션 스키마를 기본 복원에 가져올 때 모델·샘플 출처를 보존하는 importer와 동일 조건 복원 검증 필요. [자동 추출](FC_LGVSC_LOCAL_EXTRACTION.md) · [Qwen 실행](QWEN35_LOCAL_SETUP.md) · [결과](ETRI_COMBINED_REFERENCE_DIAGNOSIS.md#qwen-caption)

## 모델별 역할

- **SKEM / InternVL:** 기본 방식에서는 AI가 고른 후보의 중복 여부 판단; 기존 비교 방식에서는 전체 프레임 검사.
- **AI 캡션 v2:** 기본 방식의 구간 설명; 원본 관찰 후 작성·저장한 문장을 불러옴.
- **PLLaVA:** 기존 비교 방식의 설명 모델; 기본 v2 실행에서는 생략.
- **T5:** 수신한 문장을 영상 생성기가 사용하는 숫자 정보로 변환.
- **UniMatch:** 프레임 사이 움직임을 추정해 구간당 숫자 하나로 요약.
- **NTSCC:** 키프레임을 통신 신호로 바꿔 보내고 수신 이미지로 복원.
- **LDPC / 16-QAM:** 설명·위치 등 보조정보의 오류 정정과 신호 변환.
- **AWGN:** 잡음이 섞이는 통신 채널; 현재 주 조건 10 dB.
- **DSA:** 구간 길이에 맞춰 생성 길이와 조건을 구성.
- **Open-Sora:** 수신 키프레임·설명·움직임·이전 구간을 이용해 영상 생성.
- **VAE:** 영상과 압축 표현을 변환; 키프레임의 시간 위치를 현재 점검 중.

## 현재 한계

- 선택 속도: 기존 전체 SKEM은 60초 사례에서 약 23시간; 혼합 선택 기록은 12분 30초, 사전 AI 관찰 시간 제외.
- 자동화 범위: 기본 복원은 사전 작성 v2를 사용. 로컬 자동 전처리와 Qwen 비교는 구현했으나 임의 영상의 자동 추출→전송→복원 전체 검증은 미완료.
- 변경 전 v2: 기존 SKEM·PLLaVA 대비 전송량 −30.0%·평균 화질 소폭 개선. 이후 참조·T5 변경은 시간 −16.6%, 화질 지표 혼합. [중간 BF16 실험](ETRI_COMBINED_REFERENCE_DIAGNOSIS.md#tail17-t5-result)
- 동적 정보: 객체별 속도·방향·등장 시각을 충분히 전달하는지 미검증.
- 생성 품질: 중간 프레임의 형태·동작 오류와 이전 구간 오류 전파 가능.
- 장면전환: 검출·패킷 갱신·검출 실패 대안의 효과 검증이 남음.

<a id="static-audit"></a>
## 코드 분석 — 할루시네이션·화질에 영향을 주는 부분

2026-09-30, **참조 변경 전 v2의 코드·설정·기존 로그만 분석한 기록**. 당시 모델 실행·비교 실험·모델 코드 변경 없음. 이후 적용한 참조 변경은 아래 별도 기록. 영상 오류의 직접 원인·기여도는 미확정.

| 위치 | 코드에서 확인한 동작 | 품질·의미 오류에 미칠 수 있는 영향 |
|---|---|---|
| 수신: 키프레임 조건 위치 | 첫 구간에서 시작·끝 키프레임의 압축 표현이 같은 위치에 배치되어 시작 조건을 덮어씀 | 시작 모습 소실·끝 장면 조기 반영; 최우선 확인 대상 |
| 수신: 이전 구간 참조 | 이전 생성 영상을 재인코딩하고 참조 시작 위치를 5칸 단위로 조정. 다음 구간의 시작 수신 키프레임은 따로 재주입하지 않음 | 이전 오류 전파·과거 동작 반복·경계 불연속 |
| 수신: VAE | 영상을 압축 표현으로 바꾸고 복원. 시간축은 17프레임→5칸, 한 장의 키프레임은 1칸 | 압축 표현 고정이 최종 픽셀·객체 형태 보존을 보장하지 않음 |
| 수신: 텍스트 조건 생성 | 문장·움직임 점수·미적 점수를 조건으로 생성; 텍스트 유도 강도 `cfg_scale=7`, 미적 점수 `aes=6.5` | 설명 해석에 따른 새 구도·객체·질감 가능; 특정 설정이 악화 원인인지는 미확정 |
| 송신: 움직임 요약 | 광류의 절댓값 평균을 숫자 하나로 축약하고 소수점 한 자리의 문장으로 전달 | 객체별 방향·속도·위치 소실; 카메라 이동과 객체 이동 구분 부족 |
| 송신: 선택·캡션 | AI 후보·필수 사건 + SKEM 비교, 최대 1초 간격. 캡션은 구간별 표본 4장 | 후보·표본 사이의 짧은 사건 누락 가능; 기본 v2에서는 PLLaVA 미사용 |
| 양단: 시각 정보 전송 | NTSCC 손실 부호화·AWGN을 거친 수신 이미지를 생성 조건으로 사용 | 작은 객체·경계의 손실이 생성에 영향을 줄 가능성 |
| 수신: 장면 갱신·오류 대응 | 실행 경로에 컷별 참조 초기화나 생성 후 의미 오류에 따른 재처리 분기 없음 | 이전 장면 혼입·잘못 생성한 구간을 그대로 출력할 위험 |

- **첫 구간의 확정된 충돌:** 원본 0~12번(13프레임) → 압축 표현 4칸. 끝 조건 위치 `3`이 `align=5`로 `0`이 되어 시작 조건을 덮어씀. 저장된 마스크도 `[0, 1, 1, 1]`이며, 마지막 칸은 생성 대상. [조건 생성](../04_semantic_decoder/scripts/mydemo_new_align_sh.py#L162) · [위치 조정·대입](../.local/vendor/Open-Sora/opensora/utils/inference_utils.py#L168)
- **프레임 수와 내용 대응은 별개:** `concatenation_policy='endpoint_exact'`는 출력 중복 제거용. 현재 `conditioning_alignment`는 기본 `official_release`여서 조건 위치 조정은 계속 적용됨. [분기](../04_semantic_decoder/scripts/mydemo_new_align_sh.py#L390) · [연결](../src/semantic_transmission/temporal.py#L36)
- **위치 조정 규모:** 끝 조건 70/78구간, 이전 참조 시작 68/77회. 기존 로그의 마스크 78개와 계산 결과 일치; 이 비율은 오류 발생률이 아님. [시점별 계산·코드 해시](../outputs/etri_v2_code_audit_20260930/analysis.json)
- **동작 표본 반복:** 짧은 구간의 `[0,10,20,30]` 요청을 끝 프레임으로 제한해 36/78구간에서 표본이 중복. 전체 234쌍 중 53쌍은 같은 프레임 비교; 간격으로 나눠 속도로 변환하지 않음. [광류](../src/semantic_transmission/workers.py#L290)
- **생성의 자유도:** 키프레임은 VAE 압축 표현에 주입하며, 복원 후 수신 PNG로 되돌려 붙이는 과정은 없음. 텍스트는 객체 수·정체성·궤적을 강제로 고정하는 규칙이 아님. [생성·디코딩](../04_semantic_decoder/scripts/mydemo_new_align_sh.py#L568)
- **난수:** VAE도 확률 표본을 뽑고 생성기는 별도 잡음을 사용. 같은 시드라도 참조 길이·호출 순서를 바꾸면 같은 잡음이라는 보장이 없어, 후속 비교에서는 각각 고정 필요. [VAE](../.local/vendor/Open-Sora/opensora/models/vae/vae.py#L182) · [생성기](../.local/vendor/Open-Sora/opensora/schedulers/rf/__init__.py#L73)
- **구분할 사항:** 현재 v2의 메타데이터는 송수신 일치·비트 오류 0. 캡션의 통신 중 변조 근거는 없으며, 시각 채널 손실은 별개. T5는 300토큰에서 자르지만 실제 v2 문장이 잘렸는지는 이번 분석에서 확인하지 않음.

세 오류 시점과의 연결·미확정 사항은 [결과 문서](ETRI_COMBINED_REFERENCE_DIAGNOSIS.md#static-code-audit) 참고. 조건 위치 조정 해제는 과거 다른 구간에서 새 왜곡을 만들어 일괄 해결책으로 확정하지 않음.

<a id="fc-lgvsc-cause-audit"></a>
## 최신 오류 분석 — 2026-10-01

기존 FP32 영상 4편·로그 확인과 CPU 조건 재현. **새 복원·모델 수정 없이 처리 오류와 원인 후보를 구분.** [이미지·상세 근거](../outputs/fc_lgvsc_condition_audit_20261001/review.html)

| 오류 | 확인한 처리 | 판단 |
|---|---|---|
| 보행 0.33초·`tv_low_08` 0.5초 화면 붕괴 | 짧은 첫 구간의 끝 조건이 시작 조건을 덮어씀; 수신 키프레임은 정상 | 충돌 확정, 화면 붕괴의 최우선 원인 후보 |
| `tv_low_08` 54.5초 큰 얼굴 생성 | 끝 조건 위치·마지막 17프레임 참조 정상; 중간 생성 프레임에서 새 얼굴 발생 | 첫 구간 충돌과 별도 문제; 생성기·VAE의 세부 기여도 미확정 |
| `tv_low_08` 59.21초 문 위치·형태 왜곡 | 끝 압축 조건 11→10칸 이동, 움직임은 숫자 하나로 전달 | 시간·동작 정보 후보; 각각의 영향 미검증 |

- **재현:** 실제 코드로 조건 마스크 120개 재현·로그 일치, 마지막 17프레임 참조 116회 검사 통과. 위치 이동 횟수는 오류 발생률이 아님.
- **전달:** 캡션 120개 순서·저장값 정상, 최대 109/300토큰으로 길이 제한에 걸리지 않음. 보조정보 비트 오류 0.
- **발생 단계:** 검사한 오류는 MP4 변환 전 PNG에 이미 존재; 주변 수신 키프레임에는 같은 대형 왜곡이 없음.
- **후속:** 충돌 해소 뒤 남은 초반 모자이크는 짧은 구간의 생성 시간표 오류로 확인. [VAE 대조 검사](ETRI_COMBINED_REFERENCE_DIAGNOSIS.md#vae-short-schedule)

<a id="condition-fix"></a>
## 기본에 적용한 수정 — 키프레임 조건 충돌 방지

- **처리:** 모든 구간의 조건 겹침 검사 → 충돌할 때만 끝 키프레임을 원래 마지막 압축 위치에 배치.
- **범위:** 현재 보행·60초 영상의 첫 구간 각 1곳 수정; 충돌 없는 구간의 위치 보정·이전 참조 처리 유지.
- **검사:** 4편·120구간 CPU 재현 통과. 충돌 2곳의 양 끝 조건 보존, 나머지 118구간의 텐서·마스크 동일. [기록](../outputs/fc_lgvsc_condition_fix_validation_20261001/validation.json)
- **비교 통제:** 수신 자료·T5 저장값 재사용, 기존 VAE·생성 잡음 해시와 일치 검사. 구간 길이·전송량 동일.
- **한계:** 압축 칸이 하나뿐인 첫 구간(4프레임 이하)은 두 조건을 담을 수 없어 실행 전에 중단. 중간 얼굴 생성·VAE 시간축 대응은 별도 문제.
- **상태:** 단독 비교에서 0.33초 개선·초반 깨짐 잔존. 아래 시간표 수정과 결합해 보행 전체 검증 후 기본 채택. [단독 결과](ETRI_COMBINED_REFERENCE_DIAGNOSIS.md#condition-fix-person)
- **연결:** [충돌 처리](../src/semantic_transmission/condition_collision.py) · [기본 실행기](../src/semantic_transmission/fc_lgvsc.py) · [복원 명령](RUN_GUIDE.md#default-method). 기존 비교 실행기·완료 결과는 보존.

<a id="short-schedule"></a>
## 기본에 적용한 수정 — 짧은 구간의 생성 시간표

- **위치:** Open-Sora `timestep_transform`의 `num_frames // 17 * 5` 계산.
- **동작:** 2~16프레임을 0칸으로 계산 → 첫 생성 시각 NaN, 나머지 0. 보행 9프레임에서는 30번 계산해도 초기 잡음이 그대로 남음.
- **시험:** 실제 VAE 압축 길이 3칸으로 시간표만 바꾸자 모자이크 해소. 같은 VAE의 정상 영상 압축·복원은 통과.
- **구현:** 2~16프레임만 실제 VAE 압축 길이로 계산. 1프레임·17프레임 이상은 기존 시간표 유지; 유효 시각·고정 조건 보존·생성 정보 갱신을 매 구간 검사.
- **검사:** CPU 회귀 검사 88개 통과, GPU 전용 1개 제외. 생성 잡음 소비량·긴 구간 계산 동일성 확인.
- **상태:** 보행 10초·15구간 완료. 첫 9장은 짧은 시험과 픽셀 일치, 난수 581개 일치. 초반·전체 평균 개선, 후반 일부 악화 잔존. 구현 오류 수정으로 기본 채택. [전체 결과](ETRI_COMBINED_REFERENCE_DIAGNOSIS.md#schedule-fix-person)
- **코드:** [시간표 수정·검사](../src/semantic_transmission/short_schedule.py) → [두 수정 연결](../scripts/etri_schedule_fix_decoder.py) → [기본 실행기](../src/semantic_transmission/fc_lgvsc.py). 기존 완료 실행은 유지.
- **기본 연결 검사:** 관련 테스트 107개·4편 입력 검사 통과 후 전체 복원 완료. TVSum·보행 첫 구간에 두 수정 적용; 밀밭·등갓은 적용 대상이 없어 이전 FP32와 영상 해시 동일.

<a id="tail17"></a>
## 적용한 변경 — 마지막 17프레임 참조

- 처리: 직전 생성 구간의 마지막 17프레임 추출 → VAE 압축 5칸 → 다음 구간의 참조로 사용.
- 짧은 구간: 첫 프레임을 앞에 반복해 17장으로 맞춤. 시간 위치 보정 `align=5`는 유지.
- 연결: [기본 명령](../scripts/reconstruct_fc_lgvsc.sh) → 두 수정·FP32 조건 연결 → [수신단 연결 코드](../scripts/etri_tail17_decoder.py)에서 매 구간 적용.
- 기록: 실행 시 `receiver/tail_reference_trace.json`에 선택한 프레임 번호·압축 크기 저장; 전체 구간 적용 여부 검사.
- 입력·출력: 기존 v2 수신 자료 재사용, 추가 전송량 0. 과거 BF16은 `_v2_tail17_t5gpu_console/`, 현재 기본은 기존 FP32 경로에 `_fc_lgvsc_default_v1/`을 붙여 저장.
- 검증 상태: T5 변경과 함께 60초 복원·화질 측정 완료. 77회 참조 전달 확인; 참조만의 효과·할루시네이션 완화 입증은 미완료.
- 남은 문제: 잘못 생성된 이전 내용의 전파, 캡션·동작 정보 부족. 첫 구간 조건 충돌은 기본 경로에서 수정.

<a id="t5-cache"></a>
## 이전 가속 실험 — T5 GPU 사전 계산·재사용

현재 기본은 보행 검증에 사용한 CPU FP32 저장값을 재사용합니다. 아래 GPU BF16은 과거 비교 기록입니다.

- 순서: 실제 생성용 문장 준비 → T5 GPU 변환 → 디스크 저장 → T5 프로세스 종료 → 영상 생성.
- 입력 보존: 캡션·미적 점수·움직임 점수·문장 정리 순서는 기존 코드와 동일.
- 메모리: 한 문장씩 BF16으로 계산; T5와 영상 생성 모델을 동시에 GPU에 올리지 않음.
- 재사용 조건: 모델·토크나이저·문장·정밀도·코드·환경 일치. 해시가 다르거나 저장값이 손상되면 재사용하지 않음.
- 복원 연결: 저장된 문장 조건을 읽고, CFG의 무조건 조건은 기존 생성 모델에서 그대로 사용.
- 확인 결과: 실제 78개 GPU 사전처리 48.7초(모델 검사·로딩 포함), 저장값 78개 재사용·기존 입력 문장 일치·관련 테스트 77개 통과. [검증 기록](../outputs/etri_hybrid_keys_tv_low_08_v1/reconstruction_assistant_captions_v2_tail17_t5gpu/validation/text_cache_validation.json)
- 후속 결과: 78개 저장값을 사용해 복원 42분 28초 완료. 참조 변경도 함께 적용돼 T5만의 가속·화질 영향은 미분리; 독립 의미 오류 검수는 미완료.
- 구현: [저장·검증 코드](../src/semantic_transmission/text_embedding_cache.py) · [복원 연결](../scripts/etri_t5_cached_decoder.py) · [과거 실행](RUN_GUIDE.md#legacy-bf16).
- 터미널 표시: [표시 실행기](../src/semantic_transmission/reconstruction_console.py)가 기존 로그의 진행률과 GPU 전체 사용률(1초 간격)을 한 줄에 표시. 표시 기능은 생성 계산에 관여하지 않음.

### T5 FP32 확인용 별도 경로 — 60초 복원 완료

- T5: CPU FP32로 새 계산·저장. 영상 생성기의 BF16 변환·30단계·마지막 17프레임 참조 유지.
- 난수: VAE 호출별·구간별 초기 잡음·RFLOW 반복 잡음을 분리된 시드로 생성하고 해시 기록. 텍스트 준비 과정의 난수 소비와 분리.
- 비교: 새 FP32/BF16 실행 사이에는 잡음 해시 검증 가능. 과거 잡음은 복구 불가; CPU/GPU 장치 차이도 있어 순수 정밀도 효과로 단정하지 않음.
- 구현: [복원 실행기](../src/semantic_transmission/precision_reconstruction.py) · [FP32 캐시](../src/semantic_transmission/precision_text_cache.py) · [난수 제어](../src/semantic_transmission/generation_noise.py) · [명령](RUN_GUIDE.md#t5-fp32).
- 상태: FP32 준비·60초 복원·화질 측정 완료. 비교 페이지의 목록 표시 오류 수정 후 기존 결과로 완료; 같은 잡음의 BF16 비교·의미 오류 검수는 미완료.
- 출력 수정: [비교 페이지 실행기](../src/semantic_transmission/precision_review.py)는 지표 5개만 숫자로 표시. 동결 추론 코드·난수 계약은 유지하고 출력 코드 해시를 결과에 별도 기록.

<a id="implementation"></a>
## 주요 코드

- [workers.py](../src/semantic_transmission/workers.py): 전처리·선택·설명·움직임 추출 연결.
- [SKEM](../02_semantic_encoder/skem/MLM-keyframe-internvl.py): 키프레임 비교·선택.
- [혼합 선택](../src/semantic_transmission/hybrid_selection.py): AI 후보·필수 사건·SKEM 비교·1초 간격 보강.
- [혼합 선택 복원](../src/semantic_transmission/hybrid_reconstruction.py): 저장된 키프레임부터 복원·평가까지 실행·재개.
- [AI 캡션](../src/semantic_transmission/assisted_captions.py): 원본 표본 준비·작성된 설명 검증·PLLaVA 대신 불러오기.
- [수정 캡션 v2](../src/semantic_transmission/caption_revision.py): 새 설명을 별도 동결하고 기존 설명과 복원·전송량 비교.
- [codec_transport.py](../src/semantic_transmission/codec_transport.py): 부호화·채널·전송량 계산.
- [생성기](../04_semantic_decoder/scripts/mydemo_new_align_sh.py): 조건 주입·이전 구간 참조·영상 생성.
- [temporal.py](../src/semantic_transmission/temporal.py): 구간 경계의 중복 프레임 제거.
- [etri_60s.py](../src/semantic_transmission/etri_60s.py): 장시간 실행·재개·전체 복원 검사.
- [official_quality.py](../src/semantic_transmission/official_quality.py): 화질 평가.
- [models.json](../configs/models.json): 모델·가중치 설정.

<a id="reproduction"></a>
## 논문 재현·학습 범위

- 알고리즘: LGVSC의 주요 구성 사용; 논문 전체 실험 재현은 미완료.
- 가중치: 공개 사전학습 모델 사용; NTSCC는 quality-4, 논문 가중치와 동일성 미확인.
- 로컬 변경: 메모리 절약·결과 재사용·시간축 연결·실패 재개 지원.
- 경계 연결: `endpoint_exact`로 중복 제거; 생성 조건의 시간 위치 정렬은 별도 문제.
- 전송량: 시각·메타데이터를 합산하며 모든 물리 통신 비용을 포함하지는 않음.
- 추가 학습: 최근 실험은 학습 없이 선택 방식·생성 조건을 변경.
- 남은 확인: 모듈별 이미지·동영상 학습 자료와 사용 체크포인트의 이력.

[상세 구조·학습 범위](https://github.com/SangukBae/semantic_transmission/blob/4f566feabe213b870e5c1e441ef9b369b348d589/docs/MODEL_ARCHITECTURE.md) · [논문 대응 감사](https://github.com/SangukBae/semantic_transmission/blob/f41bc8885e8d8409e59b162fe7be63c0d11ee737/docs/LGVSC_PAPER_IMPLEMENTATION_AUDIT.md)
