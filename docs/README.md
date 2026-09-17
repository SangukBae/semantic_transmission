# 연구 문서 안내

## 현재 개발 방향

2026-09-16부터 새 지표 개발은 **개선할 의미적 특성 선정 → 모델과 지표 개발 → 독립 정답으로
실제 개선 효과 검증**의 흐름으로 진행한다. 목표는 과제에 맞는 의미적 개선을 측정할 지표 1~2개다.
기존 모델이 목표 특성을 더 잘 보존하면 같은 기준으로 그 모델에도 더 좋은 평가를 준다.

| 문서 | 용도 |
|---|---|
| [ETRI 연구 개발 계획](ETRI_DEVELOPMENT_PLAN.md) | E1~E4 목표, 모델·지표 개발 순서, 완료 판단과 다음 작업 |
| [새 지표 개발 지침](METRIC_DEVELOPMENT_STRATEGY.md) | 지표의 목적, 모델 비교·독립 정답 원칙, 기존 후보와 과거 기준의 적용 범위 |
| [현재 모델 구조](MODEL_ARCHITECTURE.md) | LGVSC SKEM+DSA 경로와 구현 구조 |
| [공식 실행 기록](ETRI_OFFICIAL_PROTOCOL.md) | 기존 모델 실행 설정과 기준선 근거 |

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
