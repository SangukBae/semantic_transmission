# 연구 문서 안내

[Windows 11 / 같은 GPU로 컴퓨터 이전](MIGRATION_WINDOWS.md): GitHub 포함 범위,
별도 이전할 데이터·실험 결과, WSL2 환경 구성 및 검증 절차.

## 현재 개발 방향

2026-09-21부터 [ETRI 후속 메일 요약](ETRI_FOLLOWUP_EMAIL_SUMMARY.md)을 연구개발의 요구사항
기준으로 삼는다. **AWGN에서 할루시네이션을 검출·완화하고, 실제 연속 영상의 60초 이상 확장을
검토하며, 원본/완화 전/완화 후와 성공·실패·새 왜곡 사례를 제시**하는 데 우선순위를 둔다.
장면전환 3개 조건, 씬 체인지 누락 대안, 학습 이력과 동적 의미정보 검토를 포함한다.
채널 종류별 비교는 현재 범위에서 제외한다.

기존 모델·지표 연구는 이 목적에 연결한다. **개선할 특성 선정 → 모델과 지표 개발 → 독립 정답으로
실제 개선 효과 검증**의 원칙과 새 지표 1~2개 목표를 유지하되, 지표의 최종 채택을 기다리느라
완화 개발·장시간 평가를 미루지 않는다. 문서 반영을 해당 기능의 구현·평가 완료로 해석하지 않는다.

| 문서 | 용도 |
|---|---|
| [ETRI 후속 메일 요약](ETRI_FOLLOWUP_EMAIL_SUMMARY.md) | 상위 요구사항: 메일의 요청과 검토 범위 |
| [ETRI 연구 개발 계획](ETRI_DEVELOPMENT_PLAN.md) | E1~E4와 새 요구의 연결, 우선순위·개발 순서·남은 작업 |
| [ETRI 후속 연구 프로토콜](ETRI_FOLLOWUP_PROTOCOL.md) | AWGN 대응 비교, 60초 이상·3유형 설계, 오류 표시·실패 사례·누락 대안·학습 검토 |
| [새 지표 개발 지침](METRIC_DEVELOPMENT_STRATEGY.md) | 지표의 목적, 모델 비교·독립 정답 원칙, 기존 후보와 과거 기준의 적용 범위 |
| [현재 모델 구조](MODEL_ARCHITECTURE.md) | 구현된 LGVSC SKEM+DSA 경로, 동적 정보의 한계와 학습 이력 확인 항목 |
| [데이터 안내](DATA.md), [로컬 데이터셋](LOCAL_DATASETS.md) | 기존 데이터·학습 자료의 출처, 장시간 연속 영상의 별도 확보·관리 기준 |
| [60초 연속 입력 준비 결과](ETRI_LONG_VIDEO_INPUTS.md) | 2026-09-24 실제 3유형·6개 입력, 출처·해시·구간·검수 상태; 복원은 미실행 |
| [ETRI 평가 입력 v1](ETRI_BENCHMARK_V1.md) | 60초 60원본·120초 확장 6개, 개발/교정/평가 18/12/30, 출처 교차·그룹 분할·검수 도구·396개 대응 작업 계획; 독립 정답·복원 평가 미완료 |
| [중복 처리 제거 비교](EXACT_REUSE_VALIDATION.md) | 전처리·디코더 모델 재사용의 최소 GPU A/B, 출력 동일성과 시간 비교 |
| [복원 품질 개선안 일괄 검증](QUALITY_VALIDATION.md) | 기존 보정·저해상도/적응형 후보 실행, 후속 요구에 필요한 확장과 미검증 범위 |
| [공식 실행 기록](ETRI_OFFICIAL_PROTOCOL.md), [HQ 기록](ETRI_HQ_PROTOCOL.md) | 기존 짧은 영상 기준선·설정·당시 실행 근거 |
| [WebVid5](WEBVID5_VALIDATION.md), [한 편 비교](WEBVID_ONE_ABLATION.md) | 기존 단편 영상 개발·진단 실행; 장시간 3유형 평가와 구분 |
| [논문·구현 감사](LGVSC_PAPER_IMPLEMENTATION_AUDIT.md) | 체크포인트·시간축·전송량·재현 범위에 관한 2026-09-18 감사 근거 |
| [파이프라인](pipeline.md), [코드 해설](CODE_WALKTHROUGH.md), [재현성](REPRODUCIBILITY.md) | 기존 구조·전처리·상위 공개 기록 설명; 신규 기능 완료 근거와 구분 |
| [로컬 실행](LOCAL_RESEARCH.md), [WSL 복원](WSL_LOCAL_SETUP.md), [smoke](SMOKE_TEST.md), [GPU 검증](VALIDATION.md) | 환경 구성·실행 점검·당시 검증 기록; 장시간 완화 성능 평가와 구분 |

적용 순서는 **메일 요약 → 개발 계획·후속 프로토콜 → 지표 지침·실행 안내 → 실험별 결과**다.
환경·이전·smoke·코드 해설은 각자의 실행·구조 설명 역할을 유지한다. 아래 과거 프로토콜과
결과는 당시 기준으로 보존하며, 새 요구와 다른 길이·자료·기준은 해당 실험의 범위로 읽는다.

## 가장 최근 지표 실험 결과

[2026-09-16 FSO·EOI·UEP 수정·재평가](METRIC_REVISION_RESULTS.md)를 확인한다.
추적 대응·정답 정의 정렬은 구현했지만 자연 영상 오경보, 실제 복원 점수 포화·관측 부족이 남았다.
FSO·UEP·EOI는 개발 후보이며 최종 지표 채택이나 개선 모델의 우위가 입증된 상태는 아니다.

## 실험별 재현 기록

아래 문서는 당시 정의·사전 등록 기준·결과를 보존한 기록이다. 문서 속 통과·실패는 해당 실험의
범위에서 읽는다. 현재 지침의 개정으로 과거 기준·판정을 변경하지 않는다.

| 실험 | 정의·실행 | 결과·검토 |
|---|---|---|
| 자동 지표 1차 | [프로토콜](AUTOMATIC_METRIC_PROTOCOL.md) | [결과](AUTOMATIC_METRIC_RESULTS.md) |
| MTE·OTF | [프로토콜](MTE_OTF_PROTOCOL.md) | [결과](MTE_OTF_RESULTS.md) |
| ERE·STA | [프로토콜](ERE_STA_PROTOCOL.md), [실행 메모](ERE_STA_EXECUTION_NOTES.md), [본 평가 실행](ERE_STA_FORMAL_EXECUTION.md) | [초기 결과](ERE_STA_RESULTS.md), [재감사](ERE_STA_REAUDIT_RESULTS.md), [본 평가](ERE_STA_FORMAL_RESULTS.md), [신규성 검토](ERE_STA_NOVELTY_REVIEW.md) |
| 잔존·지연·STA 후속 | [프로토콜](EVENT_DURATION_STA_PROTOCOL.md) | [결과](EVENT_DURATION_STA_RESULTS.md) |
| FSO·EOI·UEP v5 | [프로토콜](FSO_UEP_PROTOCOL.md) | [결과](FSO_UEP_RESULTS.md) |
| 오류×외형·구성·자연/실제 복원 | [일괄 실행](METRIC_VALIDATION_CAMPAIGN.md) | [후속 재평가의 비교 결과](METRIC_REVISION_RESULTS.md) |
| 관측·정답 정렬 v6 | [개발 프로토콜](METRIC_REVISION_PROTOCOL.md) | [수정·재평가 결과](METRIC_REVISION_RESULTS.md) |
| SGD-JSCC 지표 이식 검토 | [당시 후보·설계 검토](SGDJSCC_METRIC_TRANSFER_REVIEW.md) | 실행 결과가 아닌 제안·검토 기록 |

일괄 실행 문서 말미의 미실행 상태는 2026-09-15 구현 확인 시점의 기록이며 이후 실행은 완료됐다.
일괄 실행 문서와 v6 개발 프로토콜 등은 실행 선언의 해시 검사에도 쓰이므로, 현재 지침은 별도로 관리한다.
새 연구 방향에 맞춘 모델 개선·비교 실험은 별도 구현과 사전 등록이 필요하다.
