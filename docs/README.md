# ETRI 과제 한눈에 보기

문서·실험 현황 기준: **2026-10-03**. 완료·중단 상태는 실행 산출물로 확인합니다.

## 어떤 연구인가

**영상의 핵심 정보만 전송하고, 수신단 AI가 나머지 영상을 복원하는 연구.**

- 키프레임: 복원의 기준으로 보내는 대표 프레임.
- 캡션: 영상 구간의 내용을 설명하는 문장.
- 의미 패킷: 캡션·움직임·프레임 위치를 담은 전송 정보.
- 할루시네이션: 원본에 없는 내용의 추가, 있던 내용의 누락·왜곡.
- 처리 흐름: **영상 → 핵심 정보 추출 → 잡음 채널 전송 → AI 복원 → 평가**.

## 현재 모델: FC-LGVSC

**Faithful Caption-guided LGVSC:** 원본에 충실한 캡션 v2로 영상 복원을 유도하는 LGVSC 파생 방식.

- **키프레임 선택:** AI가 중요한 변화 후보를 고르고, SKEM이 중복을 판단하며 최대 1초 간격으로 보강.
- **캡션 v2 작성:** 원본에서 확인한 객체·위치·움직임·가림·흐림을 구체적으로 설명.
- **핵심 정보 전송:** 키프레임·캡션·움직임 정보를 잡음 채널로 전달.
- **영상 복원:** LGVSC 생성기가 수신 정보와 직전 구간의 마지막 17프레임을 참고해 영상 생성.
- **수신단 수정:** 시작·끝 키프레임 충돌 방지, 짧은 구간의 잡음 제거 시간표 정상화. T5 FP32 저장값 재사용.

현재 후보·캡션은 AI가 미리 작성해 저장하고 재사용합니다. [모델 구조](MODEL_ARCHITECTURE.md#fc-lgvsc)

별도 개발 경로는 Qwen3-VL 자동 전처리와 Qwen3.5 캡션·선택 실험입니다. 기존 assistant 작성 v2와 모델·작성 출처가 다르며, 기본 복원 경로 교체와 품질 동등성은 미검증입니다. [구조·출처 구분](MODEL_ARCHITECTURE.md#caption-providers)

준비된 **4편 일괄 복원:** `bash scripts/reconstruct_fc_lgvsc_all.sh`. [대상·실행 안내](RUN_GUIDE.md#fc-lgvsc-all)

## 기존 목표 4개

- **E1 시간축 신뢰성:** 동작·사건 순서를 보존하고 깜빡임을 줄이기.
- **E2 할루시네이션 완화:** 객체·내용의 추가·누락·왜곡을 찾아 줄이기.
- **E3 평가 신뢰성:** 실제 의미 오류를 측정하는 새 지표 1~2개 개발.
- **E4 전송량 절감:** 의미를 정확히 전달하면서 보내는 정보 줄이기.

## ETRI가 요청한 방향

- 중점: 기존 목표를 유지하며 **E2 검출·완화에 집중**.
- 채널: AWGN, 즉 가우시안 잡음을 더하는 기본 통신 조건.
- 영상: 실제 연속 60초 이상, 장면전환 저·중·고 세 유형.
- 비교: 같은 입력·채널의 원본 / 완화 전 / 완화 후.
- 보고: 성공·실패·새 왜곡, 장면전환 대응, 학습·동적 정보의 충분성.

<a id="status"></a>
## 현재 상태

| 연구 항목 | 확인한 결과·남은 범위 |
|---|---|
| 60초 평가 입력 | TVSum·ClipShots 원본 60개 + 120초 확장 6개 준비. 전체 복원·독립 주석은 미완료. [데이터](DATA.md#inputs) |
| FC-LGVSC 복원 | 60초 TVSum·짧은 영상 3편 완료. 4편 모두 주요 화질 지표 4개 개선; 새 얼굴·무늬·위치 오류 잔존. [결과](ETRI_COMBINED_REFERENCE_DIAGNOSIS.md#fc-lgvsc-four) |
| 전송·시간 | TVSum 전송량 −30.0%, 다른 3편은 증가. 복원 시간은 T5 사전 계산 제외·저장값 재사용 조건. [측정 범위](ETRI_COMBINED_REFERENCE_DIAGNOSIS.md#fc-lgvsc-four) |
| 전체 원본 추출 | WebVid 53편·TVSum 50편 준비. 직접 선택 산출물과 순차 SKEM 보정본을 구분; 보정은 중단·미완료. [상태](DATA.md#full-extraction) |
| 로컬 자동 전처리 | Qwen3-VL+TransNetV2+SKEM 구현·표본 검사 완료. 103편 배치는 7편 완료·1편 실패 후 중단. [실행 기록](FC_LGVSC_LOCAL_EXTRACTION.md#status) |
| Qwen3.5 캡션 | 78개 생성 및 PLLaVA·FC와 동일 65구간 AI 비교 완료. 48개 상향 설정 시험도 완료; 복원 경로에는 미적용. [결과](ETRI_COMBINED_REFERENCE_DIAGNOSIS.md#qwen-caption) |
| 문맥 캡션 | 같은 예산의 78구간 비교에서 오류 포함 구간 27→20, 누락 21→14. 새 오류 7구간 발생; 독립 평가·복원 검증은 미완료. [결과](ETRI_COMBINED_REFERENCE_DIAGNOSIS.md#qwen-context) |
| Qwen 선택 속도 | 30쌍 평균 1.138초. 전체 선택·품질은 미검증; 전체 시간은 추정값만 존재. [결과](ETRI_COMBINED_REFERENCE_DIAGNOSIS.md#qwen-selector-timing) |
| 의미 오류·새 지표 | 독립 할루시네이션 완화 검증 미완료. 지표 후보는 오경보·포화 때문에 최종 채택 보류. [지표](METRICS.md) |

과거 선택 가속·AI 직접 선택·캡션 v1/v2·참조/T5·수신단 수정 실험은 [실험 결과](ETRI_COMBINED_REFERENCE_DIAGNOSIS.md)에 성공·실패·미검증 범위를 함께 보존합니다. 다음 비교의 채택 기준은 [개발 계획](ETRI_DEVELOPMENT_PLAN.md)에 있습니다.

<a id="documents"></a>
## 핵심 문서 8개

- [현황](README.md): 과제 소개·목표·진행 상태.
- [ETRI 메일](ETRI_FOLLOWUP_EMAIL_SUMMARY.md): 요청사항과 연구 범위.
- [개발 계획](ETRI_DEVELOPMENT_PLAN.md): 다음 실험과 평가 기준.
- [실행 안내](RUN_GUIDE.md): 설치·실행·복구 명령.
- [모델 구조](MODEL_ARCHITECTURE.md): 모델별 역할과 한계.
- [데이터셋](DATA.md): 사용 데이터와 검수 상태.
- [모델 실험](ETRI_COMBINED_REFERENCE_DIAGNOSIS.md): 복원·개선 결과.
- [평가 지표](METRICS.md): 지표의 의미와 개발 결과.

## 상세 기록

- 비교 영상: [60초 원본·복원](../outputs/etri_60s_tv_low_08_42057b2ee8ed/comparison.mp4) · [개선안 비교](../outputs/etri_combined_reference_20260928/review.html).
- 로컬 자료: `data/`·`outputs/` 링크는 해당 파일이 있는 PC에서 확인.
- 문서 역할: 핵심 안내 8개 + 재현용 동결 원문 8개. 기존 전용 안내 2개는 [자동 추출](FC_LGVSC_LOCAL_EXTRACTION.md)·[Qwen 실행](QWEN35_LOCAL_SETUP.md)을 담당하며 실험 수치는 결과 문서에 모읍니다.
- 동결 원문: 실행기가 읽거나 해시를 검사하는 과거 기록; 당시 상태 유지.
- 원문 찾기: [통합 경로·해시](documentation_map.json) · [통합 전 문서](https://github.com/SangukBae/semantic_transmission/tree/f41bc8885e8d8409e59b162fe7be63c0d11ee737/docs).
- 갱신 원칙: 현황은 요약, 계획은 다음 검증, 구조는 처리 방식, 실행은 명령, 데이터는 입력·완료 수, 결과는 수치·한계를 기록합니다. 동결 원문·과거 산출물은 보존합니다.
- Git 기록: [10월 3일 근거 요약·원본 해시](validation/2026-10-03-research-status.json). 대용량 영상·모델·원본 로그는 로컬 `outputs/`에 있으며 Git에는 포함하지 않습니다.
