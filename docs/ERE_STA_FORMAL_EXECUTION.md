# ERE·STA 본 평가 실행 전 고정 사항

2026-09-14. 사용자가 요청한 독립 자료의 오류 검출·귀착 검증을 실행하기 위해,
개발 재감사 이후 비어 있던 실행·통계 경로를 추가했다. 아래는 **평가용 후보 점수를 보기 전**
고정한 구현 세부사항이다. 기존 ERE/STA 수식·추출기·선정값·채택 기준은 수정하지 않았다.
본 평가와 보조 비교를 완료했으며 [최종 결과·분석](ERE_STA_FORMAL_RESULTS.md)에 판정을 기록했다.
실행 전 원문은 출력 폴더의 고정 문서 사본으로 보존하고 아래 재현·진단 설명은 완료 후 보충했다.

## 자료와 단계

- 기반 실행은 `outputs/ere_sta_20260914_v2`이며 코드·정의·원자료·관문을 그대로 검증한다.
- 새 실행은 `outputs/ere_sta_formal_20260914_v1`이다. 이전 출력 폴더에 쓰지 않는다.
- 기존 적격 사례 2,211개를 모두 계산한다. 개발 원본은 합성 16·공개 8개,
  평가 원본은 합성 32·공개 14개다. 공개 결과는 사전 선언대로 진단 전용이다.
- 개발 시각/픽셀 지표 계산 → 개발 문턱 동결 → 평가 시각/픽셀 지표 계산 → 통계 순서다.
  평가 점수가 있으면 개발 문턱을 다시 계산할 수 없다.
- 새 실행 모듈·보고서·스크립트는 자료 준비에서 해시와 사본을 고정한다.
  추출 결과 캐시는 RGB 해시와 코드 서명으로 구분하며 추출기에 변형 종류·정답을 넣지 않는다.

## 지표·비교·통계

- ERE, SER, 기존 OOR/HOR/ODR를 그대로 계산한다. 비교는 PSNR, SSIM, LPIPS-Alex,
  CLIP cosine, RTE, LSSD, tLP-Alex, tOF-Farneback, tOF-RAFT, MTE, OTF, IDF1이다.
- 추가 시간 구간 기준선은 세 가지 tLP/tOF 전이 곡선의 개발 정상 표본 95백분위(higher)를
  넘는 연속 구간 수를 전체 전이 수로 나눈다. 각 기준선도 영상 평균과 별도로 비교한다.
- 모든 주 문턱은 도메인별 **전체 개발 정상군 95백분위(higher)**다. 엄격한 `score > threshold`를
  검출로 사용한다. PSNR·SSIM·CLIP의 부호는 오류가 클수록 점수가 높도록 반전한다.
- 측정 불가 `null`은 0으로 바꾸지 않는다. 검출률에서는 미검출로 포함하고 측정률을 따로
  보고한다. AUC는 유효한 점수에서 계산하며 그 범위를 명시한다.
- 오류 종류별 AUC≥0.9, 검출률≥0.8, 정상 오탐률≤0.1, 측정률≥0.95를 모두 요구한다.
- 동일 원본의 함께 측정 가능한 사례에서 후보·기준선 AUC 차이를 구한 뒤, 원본을 재추출하는
  bootstrap 500회(seed 20260914)의 95% CI를 보고한다. 모든 비교 대상에서 하한>0이어야 우위다.
- 정상군은 기존 픽셀·재렌더링·공개 외형 근사 변형으로 나눠 오탐과 AUC도 보고한다.
- SER는 적격 시간 변형 중 독립 정답의 미대응 추가 사건이 있는 사례를 보조 분석한다.
  정답 대응에는 원래 객체 ID·종류·시각·방향을 쓰며 RGB 추출기의 트랙 ID는 쓰지 않는다.
  별도로 생성한 새로운 가짜 사건 벤치마크라고 주장하지 않는다.

## STA 영상·수신 표현 검증

1. 합성 원본 RGB 32장을 PNG 키프레임 패킷으로 부호화한다. 원본 사건 정답은 부호화하지 않는다.
2. 같은 중앙 구간(12.5/25/50%)에 송신 패킷 누락, 수신 패킷 손실, 실제 바이트의 비트 반전,
   정상 수신 후 생성 영상 정지, 표현에 없는 객체 추가를 각각 적용한다. 무훼손 정상군도 둔다.
3. 수신기는 패킷·검사합만 읽어 실제 PNG를 복호한다. 손실 구간은 직전 정상 키프레임을
   유지하는 raster 복호기로 영상을 만든다. 캡션은 쓰지 않아 사건 근거가 읽힌 픽셀에서만 나온다.
4. **평가 추출기는 복호된 RGB에만** 고정 ERE/SAM2/DINOv2를 적용한다. 사건 상태는 해당
   사건 시각의 정상 수신 키프레임에 붙이며, 원본 사건과의 허용 시간은 기존처럼 0.25초다.
5. STA의 의미 단위는 RGB에서 추출한 사건과 전체 객체 트랙의 합집합이다. 단위별 가중치는 같다.
   객체 토큰 대응도 기존 트랙 대응 규칙만 사용하고 렌더러 ID를 전달하지 않는다.
6. 개입 정답은 별도 경로에서 생성한다. 독립 렌더러 궤적에 실제 복호 인덱스 대응을 적용해
   사건이 변하지 않은 개입은 제외·집계한다. 객체 추가의 정답은 삽입 마스크이며 후보에 넣지 않는다.

STA 문턱은 성분별 개발 무훼손 정상군 95백분위다. 자기 문턱을 넘은 성분 중 값이 가장 큰 것을
원인으로 예측하며 동률 순서는 EDR→CLR→GFR→GHR이다. 활성 성분이 없으면 정상으로 판정한다.
4종 원인 정확도는 클래스별 재현율의 평균이며 채널 두 변형의 수 때문에 CLR을 더 가중하지 않는다.
성분별 대상 조작 대 정상 AUC, 다른 오류 종류와의 구분 AUC, 정상 오탐, 측정률을 함께 보고한다.

같은 원본/복원 쌍을 받는 모든 기준선에도 4분류 과제를 부여한다. 개발 자료의 결측값 평균 대치,
표준화, 원본별 평균을 적용한 **최근접 클래스 중심** 분류기를 사용한다. 각 지표 단독과 모든
지표 결합을 모두 보고한다. 학습·분류 규칙 선택에 평가 자료를 쓰지 않는다. 이 분류 과제는 오류가
있는 사례의 원인 구분이므로 우연 수준은 25%이며, 이 4분류기에서 정상 오탐률을 추정하지 않는다.

STA는 정확도≥0.8, 각 성분의 대상 조작 대 정상 AUC≥0.9, 정상 오탐률≤0.1을 요구한다.
단위 검사에만 기대값을 주입하고, 영상 평가의 근거 상태는 RGB 추출 결과로 계산한다.

## 해석 범위와 재현

이 STA 실험은 **실제 바이트 복호를 포함하는 조밀한 키프레임·영상 모의 실험**이다.
LGVSC의 희소 키프레임 조건 생성이나 실제 무선 디코더의 인과적 고장 위치를 입증하지 않는다.
단일 이미지에서 알 수 없는 사건은 복호 영상의 시간 문맥으로 읽지만, 손실 보간 자체가 상태를
왜곡할 수 있다. 이 오차도 그대로 평가한다. 실제 복원 쌍은 의미 정답 없이 별도 진단만 가능하다.

```bash
bash scripts/run_metric_v3_formal.sh prepare outputs/ere_sta_formal_20260914_v1
bash scripts/run_metric_v3_formal.sh visual-development outputs/ere_sta_formal_20260914_v1
bash scripts/run_metric_v3_formal.sh pixel-development outputs/ere_sta_formal_20260914_v1
bash scripts/run_metric_v3_formal.sh calibrate outputs/ere_sta_formal_20260914_v1
bash scripts/run_metric_v3_formal.sh visual-heldout outputs/ere_sta_formal_20260914_v1
bash scripts/run_metric_v3_formal.sh pixel-heldout outputs/ere_sta_formal_20260914_v1
bash scripts/run_metric_v3_formal.sh summarize outputs/ere_sta_formal_20260914_v1
```

같은 seed의 재실행은 새 독립 검증이 아니다. 실행 후 수식·문턱을 고쳐 같은 평가 자료를 재채점하지 않는다.
위 명령은 주 실험만 계산한다. 보조 비교·실제 복원 진단·최종 분석까지 재현하는 순서는 아래에 있다.

## 평가 자료 관측 전 추가한 동일 입력 기준선

원본/복원만 보는 기준선과 STA의 비교에는 TX/RX 관측 정보량의 차이가 있다. 이 효과를
구분하기 위해 `evaluate_sta_same_input_baselines.py`와 선언 파일의 해시를 평가 자료 점수
관측 전에 별도로 고정했다. 주 실험의 수식·문턱·채택 기준에는 영향을 주지 않는 보조 비교다.

이 기준선은 STA와 같은 원본·TX 복호 영상·RX 복호 영상·최종 복원 네 영상을 받는다.
원본/TX, TX/RX, RX/복원 사이의 MSE·MAE를 구하고 세 MSE, 세 MAE, 여섯 값 결합 각각으로
동일한 개발 자료 최근접 클래스 중심 분류기를 학습한다. 주 실험의 개발 문턱이 고정된 뒤
이 보조 실험도 개발 → 평가 순서로 실행한다. 원본/복원만의 기준선에 대한 우위와, 같은
송수신 정보를 보는 간단한 기준선에 대한 이익을 구분해 해석한다.

## 실행 중 적용한 계산 재사용

개발 영상 78건을 계산한 뒤, 객체 마스크의 중심 좌표를 동일한 마스크에서 반복 계산하는
CPU 비용을 확인했다. `run_metric_v3_memoized.py`는 원래 `center(mask)`의 반환값만
재사용한다. 마스크를 강한 참조로 유지해 객체 ID 재사용을 막고 비교가 끝나면 캐시를 비운다.
지표 코드·대응 규칙·매개변수·자료·결과 계산식은 바꾸지 않았다.

이미 계산된 개발 자료 두 쌍에서 모든 JSON 결과가 정확히 같은지 확인한 뒤 작업자를
체크포인트에서 재개했다. 완료된 78건은 유지했다. 이 검사 당시 평가 자료 점수는 0건이었다.
비교 계산 시간은 각각 3.565→1.537초, 6.243→2.641초였으며 이는 CPU 비교 단계의
진단 수치다. 전체 지표의 지연 성능이나 별도 반복 실험으로 사용하지 않는다.

시각 지표와 픽셀 지표의 개발 계산은 서로 독립이라 동시에 실행했다. 개발 문턱은 양쪽의
모든 개발 점수가 있어야 고정할 수 있다. 평가 계산에도 고정된 문턱의 해시를 요구한다.
새 출력 폴더의 `memoization_probe.json`, `memoization_activation.json`,
`concurrent_development_operation.json`에 적용 시점과 근거를 남겼다.

## 자료 독립성의 정확한 범위

개발·평가 원본 ID와 원본 RGB 해시의 교집합은 없다. 다만 합성 원본은 객체별 사건 종류의
순서가 모두 같은 한 장면 계열이며, 원본마다 정답 사건이 6개다. 개발 16개와 평가 32개에서
시간까지 포함한 서로 다른 사건 서명은 각각 14개, 24개다. 새 seed의 분리는 새 장면 종류나
자연 영상 분포에 대한 일반화 검증과 구분한다.

공개 자료는 DAVIS 개발 8개·평가 14개이며, 기존 객체 마스크의 중심 궤적으로부터 정답
사건을 자동 정의한다. 사람의 행동 의미를 새로 검수한 자료는 아니다. 객체 주석은 영상의
모든 객체를 망라하지 않으므로 공개 사건 대응률을 전체 사건 정밀도로 해석하지 않는다.
세부 집계는 새 출력 폴더의 `data_diversity_audit.json`에 기록한다.

주 STA 정상군은 패킷 훼손도 후처리도 없는 PNG 복원이며 원본과 최종 RGB가 같다. 이 정상군의
오탐률은 일반적인 생성 모델의 의미 보존 재생성에 대한 오탐률을 대변하지 않는다. 별도 ERE
검사에는 재렌더링 정상군을 포함한다. 아래 보조 검사에서 그 정상군의 STA 오탐을 따로 측정한다.

같은 입력에 같은 출력을 내는 분류기에 대해, 원본·최종 복원만 관측할 때 이 모의 실험의
4종 원인 정확도 상한은 정확히 **50%**다. 송신 누락·채널 손실·생성 정지가 같은 최종 RGB를
만들기 때문이다. 네 단계 영상을 모두 볼 때는 서로 다른 원인의 관측 충돌이 없으며 그 상한은
100%다. 이는 실측 지표 성능이 아닌 입력 해시와 클래스별 가중치에서 계산한 정보 제약이다.
개발·평가 분할 모두 확인했으며 `input_information_audit.json`에 정확한 유리수 계산을 남겼다.

## 평가 자료 관측 전 추가한 STA 정상 외형 변화 검사

주 STA 정상군의 한계를 확인하고, 평가 자료 점수가 아직 0건일 때
`evaluate_sta_semantic_controls.py`의 해시와 검사 정의를
`sta_semantic_controls_declaration.json`에 별도로 등록했다. 개발 원본의 identity 한 건으로
캐시 전용 입력 경로가 네 성분 모두 0을 내는지 먼저 확인했다.

TX와 RX는 원본 PNG 영상을 온전히 복호하고, 최종 복원만 ERE의 정답 관문을 통과한 합성
정상 변형으로 바꾼다. 원본당 10종(개발 160건·평가 320건)을 모두 쓴다. 원본의 사건과 존재를
보존하는 외형 변화가 STA에서 생성 실패·근거 없는 생성으로 잘못 해석되는지 검사한다.
원본 RGB·TX/RX 복호 RGB·변형 RGB만 추출기에 주며 변형 종류·정답은 판정에 넣지 않는다.

개발 → 평가 순서로 계산하며 문턱은 주 STA의 동결된 정상 문턱을 그대로 쓴다. 별도로
문턱을 학습하거나 주 채택 기준을 바꾸지 않는다. 정상 변형별 오탐률·원본 bootstrap CI와
각 성분의 대상 오류 대 의미 보존 정상군 AUC를 보조 결과로 보고한다. 추출 결과는 주 실험에서
이미 계산한 같은 RGB·코드 해시의 캐시만 읽으며 GPU 모델을 추가로 실행하지 않는다.

## 평가 자료 관측 전 추가한 동일 분류기 비교

주 STA는 활성 성분 중 최댓값으로 원인을 판정하지만 비교 지표에는 개발 자료에서 학습한
최근접 중심 분류기를 쓴다. 특징의 품질과 판정 규칙의 영향을 나누기 위해
`evaluate_sta_classifier_head.py`와 정의의 해시를 평가 점수 0건일 때 별도로 등록했다.
STA 네 성분에도 비교 지표와 **동일한 개발 자료·표준화·최근접 클래스 중심 분류기**를 적용한다.
개발 자료로 한 번 고정하고 평가 결과를 본 뒤 다시 학습하지 않는다. 오류 4분류의 보조 비교이며
정상 판정기는 아니다. 주 STA의 최댓값 규칙·성분 값·채택 기준은 그대로 둔다.

## 실제 복원 진단의 시간 대응

실제 LGVSC·SGD 복원 6쌍은 공통 물리 시간 구간에서 별도 진단했다. LGVSC의 저장된 희소
키프레임과 캡션만 해석하며 송수신 광류는 해석하지 않는다. 송신·수신 키프레임을 마지막 정상
화면으로 유지하는 근사 자체의 정확도는 입증하지 않았다. SGD는 호환 경계 기록이 없어 STA를
측정하지 않는다. 입력·키프레임 시각·복호 상태·보류 이유를 `real_pair_diagnostics.json`에 남겼다.

실제 진단 시작 전에 어댑터의 시각 반올림을 고쳤다. 키프레임의 실제 시각 이후 첫 8 Hz 샘플에만
읽기 상태를 붙여 이전 프레임에 새 패킷의 상태를 붙이지 않는다. 예를 들어 24 fps의 16번
키프레임은 5/8초가 아닌 6/8초 샘플에서 반영한다. 격자 지연도 기록하고 공통 시간창 밖의
마지막 키프레임은 제외한다. 주 PNG 모의 실험의 정의·점수는 바뀌지 않았다.
이 사례의 회귀 검사를 포함해 최종 92개 테스트가 통과했다.

## 보조 평가까지 포함한 재현 순서

현재 컴퓨터의 두 Python 환경과 기존 v3.1 기반 실행·모델·원자료가 필요하다. 다음은 **새 출력
폴더**에서 순차 실행하는 명령이다. 모든 보조 정의와 개발 문턱을 평가 점수 관측 전에 고정한다.
원 실행의 중간 ERE 집계는 병렬 실행 중 먼저 기록한 파일이므로 새 순차 실행에서 필수로 만들지
않는다. 파일이 있는 경우 최종 무결성 검사가 최종 집계와의 동일성을 확인한다.

```bash
metric_result_dir=outputs/ere_sta_replay_20260914_v1
metric_cpu_python=/home/sangukbae/anaconda3/envs/lgvsc/bin/python
metric_object_python="$PWD/.local/metric_v2_env/bin/python"
metric_driver_lib="$PWD/.local/metric_v2_driver/extracted/usr/lib/x86_64-linux-gnu"
export PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/src"

bash scripts/run_metric_v3_formal.sh prepare "$metric_result_dir"
env -u LD_LIBRARY_PATH "$metric_cpu_python" scripts/audit_metric_v3_targets.py --output "$metric_result_dir"
env -u LD_LIBRARY_PATH "$metric_cpu_python" scripts/evaluate_sta_same_input_baselines.py --output "$metric_result_dir" --phase declare
env -u LD_LIBRARY_PATH "$metric_object_python" scripts/evaluate_sta_semantic_controls.py --output "$metric_result_dir" --phase declare
env -u LD_LIBRARY_PATH "$metric_cpu_python" scripts/evaluate_sta_classifier_head.py --output "$metric_result_dir" --phase declare

bash scripts/run_metric_v3_formal.sh visual-development "$metric_result_dir"
bash scripts/run_metric_v3_formal.sh pixel-development "$metric_result_dir"
env -u LD_LIBRARY_PATH "$metric_object_python" scripts/run_metric_v3_memoized.py --output "$metric_result_dir" --benchmark
bash scripts/run_metric_v3_formal.sh calibrate "$metric_result_dir"
env -u LD_LIBRARY_PATH "$metric_cpu_python" scripts/evaluate_sta_classifier_head.py --output "$metric_result_dir" --phase fit
env -u LD_LIBRARY_PATH "$metric_object_python" scripts/evaluate_sta_semantic_controls.py --output "$metric_result_dir" --phase development
env -u LD_LIBRARY_PATH "$metric_cpu_python" scripts/evaluate_sta_same_input_baselines.py --output "$metric_result_dir" --phase run

env LD_LIBRARY_PATH="$metric_driver_lib" "$metric_object_python" scripts/run_metric_v3_memoized.py --output "$metric_result_dir" --split heldout
bash scripts/run_metric_v3_formal.sh pixel-heldout "$metric_result_dir"
bash scripts/run_metric_v3_formal.sh summarize "$metric_result_dir"
env -u LD_LIBRARY_PATH "$metric_cpu_python" scripts/evaluate_sta_classifier_head.py --output "$metric_result_dir" --phase evaluate
env -u LD_LIBRARY_PATH "$metric_object_python" scripts/evaluate_sta_semantic_controls.py --output "$metric_result_dir" --phase heldout
env LD_LIBRARY_PATH="$metric_driver_lib" "$metric_object_python" scripts/diagnose_metric_v3_pairs.py --output "$metric_result_dir"
env -u LD_LIBRARY_PATH "$metric_cpu_python" scripts/analyze_metric_v3_formal.py --output "$metric_result_dir"
env -u LD_LIBRARY_PATH "$metric_cpu_python" scripts/audit_metric_v3_targets.py --output "$metric_result_dir" --verify
env -u LD_LIBRARY_PATH "$metric_cpu_python" scripts/verify_metric_v3_formal.py --output "$metric_result_dir"
```

로컬 CUDA 드라이버 라이브러리 경로는 해당 GPU 프로세스에만 적용하며 시스템 설치를 변경하지
않는다. 개발 영상 두 쌍의 중심 좌표 재사용 동일성 검사가 없으면 최적화 작업자는 실행하지 않는다.
주 RGB 점수와 자료·모델 해시는 고정하고 분석용 집계·그림만 다시 생성할 수 있다.
공개 사건 정의 차이, 입력 해시의 정보 상한, 패킷 재복호 같은 추가 감사는 원 실행의 별도 JSON에
보존했다. 위 명령을 실행했다고 이 부가 파일 모두가 새로 생성되는 것은 아니다.

사후 정답 감사는 렌더러 사건과 실제 복호 인덱스만으로 누락·추가 관계를 재구성한다.
주 실험 라벨·점수·채택 기준을 변경하지 않는다. 현재 실행에서 기존 사후 집계와 스크립트
재생성 값이 일치함을 `target_audit_reproduction.json`으로 확인했다.
