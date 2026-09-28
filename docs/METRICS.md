# 지표 연구 결과와 재현 기록

갱신: **2026-09-28**. [E3 개발 원칙](ETRI_DEVELOPMENT_PLAN.md#metrics) · [과제 현황](README.md)
현재 FSO·EOI·UEP는 개발 후보이며 **최종 채택·자연 영상 유효성·신규성 입증을 보류**한다.

## 현재 후보와 한계

| 후보 | 측정 대상 | 실제 복원에서 남은 문제 |
|---|---|---|
| FSO: 금지 상태 점유량 | 소멸 후 잔존·조기 등장·잘못 지속되는 움직임/방향 | 관측 부족, 정확한 수치 정답 부족 |
| UEP: 근거 없는 존재 점유량 | 수신 근거로 뒷받침되지 않는 객체 존재 | 정상 외형 변화 오경보, 실제 복원 점수 포화 |
| EOI: 사건 순서 역전율 | 원본 사건 순서의 보존 | 사건 대응률 부족으로 보류 |

세 후보를 모두 채택할 필요는 없다. 의미 객체와 배경/부분 영역의 구분·동일 객체 대응을 개선하고
목표 오류와 연결되는 1~2개를 독립 자료에서 검증한다.

<a id="latest"></a>
## 최신 실행: v6 관측·정답 정렬 — 2026-09-16

추적 조각 대응·화면 좌표 사건 정의·EOI 관측 부족 보류를 구현하고
합성 1,800 + 수신 근거(RX) 180 + 자연 영상 600 + 실제 복원 12 = **2,592건**을 재평가했다.
캐시 재사용 평가 시간은 277.9초이며 새 GPU 복원·관측 모델 재추론 시간은 포함하지 않는다.

| 대상 | 결과 | 해석 |
|---|---|---|
| 합성 FSO 움직임·방향 | 같은 정답 정의 AUC 0.981 / 0.989 | 구분력 개선, 움직임 관측률 약 94%는 부족 |
| 합성 EOI | 보류 제외 AUC 0.979 | 양성/음성 관측률 82.6% / 85.2%로 전체 성능 주장 불가 |
| RX | AUC 0.910, 검출 81.9%, 정상 오경보 0/36 | 원본·복원 고정 후 수신 근거 변경 효과 유지 |
| 자연 영상 UEP | AUC 0.683, 정상 오경보 **32/40** | 외형 변화 안정성 미달 |
| 실제 복원 12개 | UEP **12/12가 1.0**, EOI 전부 보류 | 모델 차이를 신뢰성 있게 판별하지 못함 |

AI 검토에서는 원본 2개에서 파생한 4개 복원에 UEP 음성 정답을 기록했지만 모두 1.0이었다.
나머지 8개는 모호해 보류했다. 사람·완전 맹검 검수나 충분한 양성 정답은 없다.
FSO의 움직임/방향은 각각 4/12, 3/12에서만 계산됐다. 추출기의 부분/배경 분할을 의미 객체로 세는 문제가 남는다.

[원자료 폴더](../outputs/metric_revision_20260916_v6/) ·
[같은 정답 정의 비교](../outputs/metric_revision_20260916_v6/aligned_synthetic_truth.json) ·
[원문 결과](https://github.com/SangukBae/semantic_transmission/blob/f41bc8885e8d8409e59b162fe7be63c0d11ee737/docs/METRIC_REVISION_RESULTS.md)

<a id="history"></a>
## 개발 이력과 채택 판정

| 실험 | 핵심 결과 | 당시 판정·원문 |
|---|---|---|
| 자동 지표 RTE·LSSD, 9/12 | 2,432개 파생 사례. 짧은 부분 오류를 평균이 희석 | 두 후보 미달 · [결과](https://github.com/SangukBae/semantic_transmission/blob/f41bc8885e8d8409e59b162fe7be63c0d11ee737/docs/AUTOMATIC_METRIC_RESULTS.md) |
| MTE·OTF, 9/14 | 1,176사례. MTE 평균 검출 85.9%이나 순서 오류 취약; OTF 추가 검출 0% | 두 후보 미달 · [결과](https://github.com/SangukBae/semantic_transmission/blob/f41bc8885e8d8409e59b162fe7be63c0d11ee737/docs/MTE_OTF_RESULTS.md) |
| ERE·STA 초기·재감사, 9/14 | 초기 추출 관문 실패 → 재감사 합성 사건 재현율 95.8%·정밀도 92.0% | 개발 관문 통과와 본 평가를 구분 · [재감사](https://github.com/SangukBae/semantic_transmission/blob/f41bc8885e8d8409e59b162fe7be63c0d11ee737/docs/ERE_STA_REAUDIT_RESULTS.md) |
| ERE·STA 본 평가, 9/14 | ERE 시간 오류 검출 11.1%, STA 주 원인 판정 52.6% | 주 기준 미달 · [결과](https://github.com/SangukBae/semantic_transmission/blob/f41bc8885e8d8409e59b162fe7be63c0d11ee737/docs/ERE_STA_FORMAL_RESULTS.md) |
| 잔존·지연·STA, 9/14 | 새 합성 720건. 잔존 오류 96/96 검출, 정상 오탐 2.5%, AUC 0.981 | 잔존 후보만 통제 기준 통과 · [결과](https://github.com/SangukBae/semantic_transmission/blob/f41bc8885e8d8409e59b162fe7be63c0d11ee737/docs/EVENT_DURATION_STA_RESULTS.md) |
| FSO·EOI·UEP v5, 9/15 | 새 합성 768건의 여섯 후보 통제 기준 통과; 네 성분의 등록 기준선 우위 | 자연 영상·실제 복원 유효성은 별도 · [결과](https://github.com/SangukBae/semantic_transmission/blob/f41bc8885e8d8409e59b162fe7be63c0d11ee737/docs/FSO_UEP_RESULTS.md) |
| 후속 캠페인·v6, 9/16 | 2,592건 실행·재평가, 자연 오경보·실제 포화 잔존 | 최종 채택 보류; 위 최신 결과 |

FSO 존재 성분은 기존 `ghost_max`와 수치가 같아 새로운 기여가 아니다. v5의 `eoi` 차별성은 미입증이고 `fso_max`는 탈락했다.
STA 동일 분류기 보조 비교 99.2%를 주 규칙 52.6%의 성공으로 바꾸지 않는다. 통제 원본의 변형·시드는 독립 자연 영상 수가 아니다.
[신규성 검토 원문](https://github.com/SangukBae/semantic_transmission/blob/f41bc8885e8d8409e59b162fe7be63c0d11ee737/docs/ERE_STA_NOVELTY_REVIEW.md) ·
[SGD-JSCC 이식 후보 검토](https://github.com/SangukBae/semantic_transmission/blob/f41bc8885e8d8409e59b162fe7be63c0d11ee737/docs/SGDJSCC_METRIC_TRANSFER_REVIEW.md)

## 실행과 재현용 원문

```bash
bash scripts/run_metric_validation.sh
bash scripts/run_metric_revision.sh
PYTHONPATH=src .local/metric_v2_env/bin/python scripts/verify_metric_revision.py outputs/metric_revision_20260916_v6
```

입력·이전 결과·관측 캐시와 환경이 필요하다. [환경·명령](RUN_GUIDE.md)
최신 지표 실행을 다시 수행했다는 뜻이 아니며, 위는 저장된 실험의 재현 명령이다.

일곱 지표 원문은 실행기가 직접 읽거나 해시를 검사하므로 **경로와 바이트를 유지**한다.
아래 이름은 실행 입력이며 일반 독자는 이 통합 문서를 사용한다.

- `ERE_STA_PROTOCOL.md`, `ERE_STA_EXECUTION_NOTES.md`, `ERE_STA_FORMAL_EXECUTION.md`
- `EVENT_DURATION_STA_PROTOCOL.md`, `FSO_UEP_PROTOCOL.md`
- `METRIC_VALIDATION_CAMPAIGN.md`, `METRIC_REVISION_PROTOCOL.md`

동결 원문 안의 상대 링크·옛 상태는 해당 버전의 맥락이다.
[통합 전 Git 문서](https://github.com/SangukBae/semantic_transmission/tree/f41bc8885e8d8409e59b162fe7be63c0d11ee737/docs)에서 원문과 연결 문서를 함께 읽는다.
`docs/validation/`의 JSON, 기존 `outputs/`의 선언·점수·로그도 수정하지 않는다.
