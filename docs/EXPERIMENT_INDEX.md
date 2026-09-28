# 상세 문서와 실험 기록

갱신: **2026-09-28**. 처음 읽을 때는 [과제 현황](README.md), 다음 행동은 [개발 계획](ETRI_DEVELOPMENT_PLAN.md)을 본다.
이 목록은 개별 실험의 조건·결과·재현 근거를 찾기 위한 색인이다.
과거 프로토콜·결과는 해당 날짜의 기록이며, 현재 상태와 최종 채택을 뜻하지 않는다.

## 요구사항·설계·실행

| 문서 | 역할 |
|---|---|
| [ETRI 후속 메일](ETRI_FOLLOWUP_EMAIL_SUMMARY.md) | 요구사항 원문 요약; 구현·평가 상태와 구분 |
| [평가·보고 기준](ETRI_FOLLOWUP_PROTOCOL.md) | AWGN 대응 비교, 오류·시간축·비용·독립 정답 |
| [새 지표 개발 지침](METRIC_DEVELOPMENT_STRATEGY.md) | E3의 목적과 유효성·신규성 판단 |
| [실행 명령 모음](RUN_GUIDE.md) | 설치·장시간 실행·과거 실험 재현 |
| [모델 구조](MODEL_ARCHITECTURE.md) | 송신·채널·생성 모듈, 동적 정보·학습 이력의 한계 |

## 장시간 입력과 60초 실행

| 기록 | 날짜·범위 |
|---|---|
| [연속 입력 파일럿](ETRI_LONG_VIDEO_INPUTS.md) | 9월 24일 초기 6개 준비 기록 |
| [평가 입력 v1](ETRI_BENCHMARK_V1.md) | 9월 24일 60초 60원본·120초 확장 6개, 분할·예정 작업; **해시 동결 문서** |
| [원본 AI 검수](ETRI_AI_SOURCE_REVIEW.md) | 9월 25일 66개 입력 표본 검토, 독립 정답 미완료 |
| [실행 준비 점검](ETRI_60S_CHECK.md) | 60초 입력과 짧은 실제 모델 실행의 검사 |
| [60초 전체 실행·복구](ETRI_60S_RUN.md) | `tv_low_08` 전체 복원 완료와 현재 재개 명령 |
| [60초 오류·조건 진단](ETRI_CONDITIONING_DIAGNOSIS.md) | 9월 28일 오류 표본·캡션·마스크 감사, 짧은 세 창의 반올림 해제 비교 |
| [17프레임 참조](ETRI_TAIL_REFERENCE_DIAGNOSIS.md) | 9월 28일 이전 참조 길이만 변경한 대응 실험 |
| [네 조건 결합 비교](ETRI_COMBINED_REFERENCE_DIAGNOSIS.md) | 9월 28일 최신 결과; 새 흐림 발생, 기본값·60초 확대 보류 |

입력 준비 문서의 ‘복원 미실행’은 **작성 당시 상태**다. 이후 한 편의 60초 실행 완료는
[현재 현황](README.md)과 [실행 결과 안내](ETRI_60S_RUN.md)를 따른다. 동결 파일을 갱신해 과거 해시를 바꾸지 않는다.
`data/`·`outputs/` 산출물은 Git 제외 자료이며 해당 로컬 파일이 있어야 링크가 열린다.

## 기존 짧은 영상·품질·가속 실험

| 문서 | 범위 |
|---|---|
| [공식 프로필](ETRI_OFFICIAL_PROTOCOL.md) · [HQ 프로필](ETRI_HQ_PROTOCOL.md) | 기존 짧은 ETRI 기준선·환경·당시 실행 |
| [ETRI 첫 영상 ablation](ETRI01_DECODER_ABLATION.md) | 정렬·키프레임 간격 등 조건별 개발 비교 |
| [WebVid 한 편](WEBVID_ONE_ABLATION.md) · [다섯 편](WEBVID5_VALIDATION.md) | 짧은 영상 비교; 실행 계획과 실제 완료 수를 구분 |
| [품질 후보 검증](QUALITY_VALIDATION.md) | 보정·적응형·저해상도 후보의 실행 절차; 독립 의미 검수와 구분 |
| [중복 처리 제거](EXACT_REUSE_VALIDATION.md) | 전처리·모델 재사용의 제한된 GPU A/B와 출력 동일성 |
| [정렬·전송량 정합성](LGVSC_ALIGNMENT.md) | 시간축·평가·시각/메타데이터 비용 처리 |
| [논문·구현 감사](LGVSC_PAPER_IMPLEMENTATION_AUDIT.md) | 체크포인트·알고리즘·전송량·실제 재현 범위 |

## 지표 개발 이력

최신 판단은 **FSO·EOI·UEP 최종 채택 보류**다. 통제 실험 통과와 자연 영상·실제 복원 유효성을 구분한다.

| 실험 | 조건·실행 | 결과 |
|---|---|---|
| 자동 지표 1차 | [프로토콜](AUTOMATIC_METRIC_PROTOCOL.md) | [결과](AUTOMATIC_METRIC_RESULTS.md) |
| MTE·OTF | [프로토콜](MTE_OTF_PROTOCOL.md) | [결과](MTE_OTF_RESULTS.md) |
| ERE·STA 초기·재감사 | [정의](ERE_STA_PROTOCOL.md) · [실행 메모](ERE_STA_EXECUTION_NOTES.md) | [초기 실패](ERE_STA_RESULTS.md) · [재감사](ERE_STA_REAUDIT_RESULTS.md) |
| ERE·STA 본 평가 | [실행](ERE_STA_FORMAL_EXECUTION.md) | [결과](ERE_STA_FORMAL_RESULTS.md) · [신규성 검토](ERE_STA_NOVELTY_REVIEW.md) |
| 잔존·지연·STA | [프로토콜](EVENT_DURATION_STA_PROTOCOL.md) | [결과](EVENT_DURATION_STA_RESULTS.md) |
| FSO·EOI·UEP v5 | [프로토콜](FSO_UEP_PROTOCOL.md) | [결과](FSO_UEP_RESULTS.md) |
| 합성·자연·실제 복원 캠페인 | [동결 실행 안내](METRIC_VALIDATION_CAMPAIGN.md) | [후속 재평가에 비교 포함](METRIC_REVISION_RESULTS.md) |
| 관측·정답 정렬 v6 | [프로토콜](METRIC_REVISION_PROTOCOL.md) | [최신 결과](METRIC_REVISION_RESULTS.md) |
| SGD-JSCC 이식 검토 | [후보 검토](SGDJSCC_METRIC_TRANSFER_REVIEW.md) | 제안이며 실행 완료 근거가 아님 |

일부 프로토콜·실행 문서는 산출물 해시 검사에 포함된다. 문서의 과거 통과·실패를 소급 변경하지 않는다.

## 조사·후속 아이디어

아래는 각 작성일의 조사·제안이다. 채택·구현·학습·효과 검증 여부는 현재 현황과 실행 결과로 확인한다.

- [할루시네이션 문헌 검토](research/ETRI_HALLUCINATION_LITERATURE_REVIEW_2026-09-21.md)
- [관련 코드 감사](research/ETRI_HALLUCINATION_CODE_AUDIT_2026-09-21.md)
- [9월 21일 단기 실험 제안](research/ETRI_LGVSC_HALLUCINATION_EXPERIMENT_PLAN_2026-09-21.md)
- [학습형 키프레임 선택기 제안](research/ETRI_LGVSC_LEARNED_KEYFRAME_SELECTOR_PROPOSAL_2026-09-21.md)
- [장시간 데이터셋 검토](research/ETRI_LONG_VIDEO_DATASET_REVIEW_2026-09-24.md)
- [우선순위 개정 전 조사 기록](research/archive/2026-09-21-before-prioritization/)

## 환경·데이터·코드 참고

| 주제 | 문서 |
|---|---|
| 데이터 | [출처](DATA.md) · [로컬 위치·확보](LOCAL_DATASETS.md) |
| 환경 설치·이전 | [Windows/WSL 이전](MIGRATION_WINDOWS.md) · [현재 WSL 기록](WSL_LOCAL_SETUP.md) · [로컬 연구 환경](LOCAL_RESEARCH.md) · [환경 파일](../environment/README.md) |
| 실행 검증 | [smoke](SMOKE_TEST.md) · [GPU 검증](VALIDATION.md) |
| 코드 설명 | [파이프라인](pipeline.md) · [코드 해설](CODE_WALKTHROUGH.md) · [재현성](REPRODUCIBILITY.md) |
| 원저작물 | [LGVSC 원본 README](../README_UPSTREAM.md) · [인용](../CITATION.cff) |
