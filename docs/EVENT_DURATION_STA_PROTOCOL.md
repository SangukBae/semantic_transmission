# 사건 지속량·반영 지연·STA 정상/보류 검증

2026-09-14. 실행 전 고정 문서. 사람의 추가 검수 없이 자동 정답으로 검증한다.
[기존 결과](ERE_STA_FORMAL_RESULTS.md)와 [이식 검토](SGDJSCC_METRIC_TRANSFER_REVIEW.md)를
참고하되 사건별 집계·source-pairing·SFR만으로 기존 문제가 해결된다고 가정하지 않는다.

## 범위와 자료 분리

- 개발: 기존 `ere_sta_formal_20260914_v1`의 합성 원본 48개와 저장된 RGB 관측·점수.
  이전 평가 원본 32개도 이번에는 개발 자료다. 새 평가 점수로 정의·문턱을 바꾸지 않는다.
- 새 평가: `cross_axis_exit`, `entrance_and_stop`, `reentry_and_restart` 3계열,
  각 8개 원본(총 24개). 이전과 객체 수·모양·배치·궤적·사건 순서가 다르다.
  32프레임, 192×320, 8 Hz의 절차적 그래픽 장면이다. 자연 영상 일반화로 해석하지 않는다.
- 사건 시험: 정상 5종, 잔존 0.25/0.5/1초, 퇴장 후 재등장, 사건 지연 0.25/0.5/1초,
  끝까지 상태 미반영. 정상군은 identity/밝기/질감/카메라/색상 변화다.
- STA: 같은 원본의 정상 5종, 송신 누락·패킷 손실·비트 손상·생성 정지·객체 추가의
  12.5/25% 구간 개입. 독립 사건 정답이 바뀌지 않은 개입은 제외 수와 함께 보고한다.
  수신 표현 전체 부재·원본 미제공은 별도 기술적 보류 시험이며 원인 5분류에 섞지 않는다.
- 원본 RGB 해시 교집합, 원자료·모델·코드·문턱 해시를 확인하고 기존 출력은 유지한다.

## 후보 정의

1. **잔존 오류 지속량**: RGB에서 읽은 원본 퇴장 사건 이후 4프레임(0.5초)의 동일 객체 존재
   판정 비율과 존재 시간, 재등장 횟수. ROC-AUC와 다른 시간축 적분이며 Kaplan–Meier
   생존함수로 부르지 않는다. 전체 창을 관측하지 못하면 적분 점수는 null이다.
2. **반영 지연·미반영률**: 원본 사건과 같은 객체·종류·방향의 복원 사건을 최대 1초 이내에서
   일대일로 대응. 1프레임의 선행 오차는 허용하고 지연은 0으로 절단한다. 대응된 지연/1초,
   전체 창을 관측했지만 대응이 없으면 1을 사건 벌점으로 쓴다. 관측 종료는 right-censored,
   원본 사건/복원 관측 불가는 unobservable로 분리한다.
3. 후보별 영상 경고는 측정 가능한 사건 벌점의 최댓값을 사용한다. 사건별 결과도 모두 남기며,
   원본 정답 사건을 읽지 못한 경우를 외부 감사의 측정률 분모에서 제외하지 않는다.

객체 대응은 기존 SAM2/DINO의 전체 트랙 대응과, 겹치지 않는 복원 트랙 조각의 제한된 재연결이다.
추가 연결은 DINO cosine≥0.80, 두 후보 사이 margin≥0.03, 정규화 중심 거리≤0.35를 요구한다.
추출기·기존 사건 설정은 v3.1과 동일하다. RGB와 복호 상태만 후보에 입력하고, 렌더러 ID·마스크·
조작 이름은 독립 정답 감사에만 쓴다. 검출기 누락을 실제 객체 부재의 정답이라고 주장하지 않는다.

위 정의를 수식으로 쓰면, 원본 퇴장 프레임이 `t_e`이고 대응 객체의 복원 존재 판정이
`p_B(t) ∈ {0, 1}`일 때 `G(e) = Σ[j=0..3] p_B(t_e+j) / 4`이고,
관측 존재 시간은 `Σ p_B / 8`초다. 대응 사건의 지연 벌점은
`D(e) = max(0, t_B − t_A) / 1초`이며, 충분한 관측 창 안에 대응이 없으면 1이다.
영상 점수는 각 후보의 측정 가능한 `G(e)` 또는 `D(e)`의 최댓값이다. 측정 가능한 사건이
없으면 null이다. 0.5초 잔존 점유량은 0.5초와 1초 지속을 구분하지 못하므로 장기 생존 시간의
대체 지표로 해석하지 않는다. 이 식은 고정 구현을 풀어 쓴 설명이며 추가 변경이 아니다.

## STA 정상·보류 판정

기존 STA의 EDR/CLR/GFR/GHR 네 성분은 바꾸지 않는다. 이전 개발 원본 16개에서
정상+4오류의 표준화된 최근접 클래스 중심을 학습한다. 클래스 안에서 원본별 가중치를 같게 둔다.
이전 평가 원본 32개로 각 정답 클래스 중심 거리의 95백분위(higher)와, 맞게 구분된 사례의
중심 간 거리 차이 5백분위(lower)를 정한다. 거리가 너무 멀거나 구분이 모호하면 보류한다.
성분 결측·필수 입력 전체 부재 역시 보류한다. 원인 라벨은 추론 함수에 주지 않는다.

같은 원본/TX/RX/복원 RGB의 세 구간 MSE·MAE 6개에도 동일한 학습·보류 규칙을 적용한다.
보류를 틀린 판정으로 포함한 정확도, 보류 제외 정확도, 측정률, 정상 오탐과 정상 보류를 따로
보고한다. 정상까지 포함한 평균과 오류 네 종류 평균 모두 보고한다.

## 비교와 판정

- 사건 비교: 기존 ERE/SER, tLP, MTE, IDF1, LPIPS, tOF-Farneback와 동일 RGB를 비교한다.
  SFR는 기존 의미 라벨 패킷이 없어 **RGB 추론 객체 ID에 적용한 변형**으로 별도 표시한다.
- source-paired 추가율은 원본에 등장한 적 있는 알려진 객체 어휘의 정답 부재 프레임만 사용한다.
  GT-absent이면서 원본 검출도 음성인 경우가 조건부 분모다. 원본 관측 불가·양성 제외 수와
  원본 검출 재현율을 함께 보고한다. 열린 어휘 할루시네이션이나 독립 검증기로 주장하지 않는다.
- 주 경고 문턱: 이전 정상 480건의 95백분위(higher), 엄격한 `score > threshold`.
- 오류 종류별 AUC≥0.9, 검출률≥0.8, 정상 오탐률≤0.1, 측정률≥0.95를 모두 요구한다.
  추출된 평가 대상 사건의 비율≥0.8도 필요하다. 잔존 적분 MAE≤0.25, 관측된 지연의
  초 단위 MAE≤0.25, 미반영 F1≥0.8을 해당 대상에서 요구한다. 결측은 0오류로 대치하지 않는다.
- STA는 보류를 오답으로 포함한 정상+4오류 및 오류4종 평균 정확도≥0.8,
  정상 오탐률≤0.1, 측정률≥0.95를 요구한다. 기술적 입력 부재는 100% 보류해야 한다.
- 통계: 원본 24개를 단위로 bootstrap 1,000회(seed 20260914), 95% 구간. 사건을 독립 원본처럼
  세지 않는다. 후보와 기준선은 같은 원본에서 대응 비교한다. 계열별 결과도 별도로 보고한다.
- 어떤 결과도 논문 신규성 입증을 자동으로 뜻하지 않는다. 기존 논문의 평가 정의와 비교해야 한다.

## 재현

```bash
metric_run=outputs/event_duration_sta_replay
bash scripts/run_metric_v4.sh prepare "$metric_run"
bash scripts/run_metric_v4.sh declare "$metric_run"
bash scripts/run_metric_v4.sh development "$metric_run"
bash scripts/run_metric_v4.sh visual "$metric_run"
bash scripts/run_metric_v4.sh pixel "$metric_run"
bash scripts/run_metric_v4.sh report "$metric_run"
PYTHONNOUSERSITE=1 PYTHONPATH=src /home/sangukbae/anaconda3/envs/lgvsc/bin/python scripts/audit_metric_v4_inputs.py --output "$metric_run"
PYTHONNOUSERSITE=1 PYTHONPATH=src /home/sangukbae/anaconda3/envs/lgvsc/bin/python scripts/diagnose_metric_v4.py --output "$metric_run"
PYTHONNOUSERSITE=1 PYTHONPATH=src /home/sangukbae/anaconda3/envs/lgvsc/bin/python scripts/plot_metric_v4.py --output "$metric_run"
PYTHONNOUSERSITE=1 PYTHONPATH=src /home/sangukbae/anaconda3/envs/lgvsc/bin/python scripts/verify_metric_v4.py --output "$metric_run"
```

자료와 정의를 동결한 뒤 시각·픽셀 평가를 실행한다. 시각·픽셀 실행은 같은 고정된 개발 문턱을
읽는 독립 작업이다. 정의를 바꿀 필요가 생기면 변경 사유를 남기고 새 평가 계획으로 구분한다.

위 명령은 같은 장면·시드의 **재현**이며 새로운 독립 평가가 아니다. 기존 출력이 없는 폴더를
사용한다. 이전 개발 실행의 점수·RGB 관측 캐시 및 모델 가중치가 필요하다.

실행 당시의 원문은 `PROTOCOL.frozen.md`에 보존했다. 통계 선언·보고서·그림·복호 감사 명령은
재현을 위해 보충한 설명이며 지표 정의와 판정 기준을 바꾸지 않았다. `report`는 고정 통계 코드의
NumPy 불리언을 JSON 표준 타입으로만 변환하는 래퍼를 사용하고 수정 사유·해시를 기록한다.
`diagnose`는 완료된 점수의 사후 설명용 분해이며 채택 기준에 사용하지 않는다.
[결과 문서](EVENT_DURATION_STA_RESULTS.md).
