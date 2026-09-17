# 사람 검수 없는 지표 개발·검증

작성일: 2026-09-12. 사용자의 요청에 따라 사람 검수 없이 1단계를 진행한다.
**자동 평가 구현과 첫 통제 실험은 완료했으나, 두 지표 후보의 채택 기준은 미달이다.**
[쉽게 읽는 결과](AUTOMATIC_METRIC_RESULTS.md)에 검출률과 실제 실행 범위를 정리했다.
신규 지표의 유효성이 입증되었거나 1단계 연구가 최종 완료되었다는 뜻은 아니다.

## 정답을 어떻게 만드는가

- 실제 영상: 원본의 표본 프레임에 역순·정지·지연·순서 교환을 적용한다. 프레임 대응표와
  실제 바뀐 시점을 코드로 기록한다. 이것은 시간 조작 정답이며 사건 의미를 해석한 정답이 아니다.
- 렌더링 장면: 이동 객체와 등장·소멸하는 객체를 코드로 그린다. 객체를 추가·삭제·변형한
  시간 구간과 원본 장면 상태를 저장한다. 삭제는 검은 사각형 덮기 대신 해당 객체 없이
  같은 배경을 다시 그리는 방식이다. 객체 오류의 지속 시간을 25%, 50%, 100%로 조절한다.
- 대조군: 원본 그대로, 밝기 +5/255, Gaussian blur σ=0.5, JPEG 품질 90.
  원본의 정상 등장·소멸도 대조군에 남아 있다. 이 설정에서 의미를 보존한다는 가정이다.
- 평가기에는 원본·변형 RGB 배열만 전달한다. 변형 종류, 위치, 정답 객체 정보는 전달하지 않는다.
  기존 ETRI 사람 주석이나 VLM 판정을 정답으로 읽지 않는다.

## 지표 후보 정의

CLIP ViT-B/32를 고정한다. 전체 화면을 포함하도록 중간 회색 여백으로 224×224 정사각형을
만들고 CLIP 정규화를 적용한다. 기존 기본 center crop과 수치가 같다고 주장하지 않는다.
CLIP은 시간축 모델이나 객체 검출기가 아니므로 아래 점수는 의미 오류의 대리 측정치다.

### RTE: Reference Transition Error

원본·복원의 정규화된 프레임 CLIP 특징을 각각 $a_t,b_t$라 하고,
$\Delta a_t=a_{t+1}-a_t$, $\Delta b_t=b_{t+1}-b_t$로 둔다.

$$RTE=\frac{\sum_{t\in V}\|\Delta a_t-\Delta b_t\|_2}
{\sum_{t\in V}(\|\Delta a_t\|_2+\|\Delta b_t\|_2)}.$$

$V$는 분모 항이 $10^{-4}$보다 큰 전이들의 집합이다. 점수는 0~1, 작을수록 좋다.
분모가 모두 퇴화하면 `null`로 반환하고 전이 측정 비율을 함께 기록한다. 이 수치적 기준은
실제 동작이 관측 가능하다는 보장이 아니다. 특징 변화의 크기와 방향을 비교하므로 단순한
프레임 CLIP 평균과 다르지만, CLIP 차분이 실제 이동 방향이나 동작 의미를 나타낸다고 보장하지 않는다.

### LSSD: Local Semantic Support Discrepancy

CLIP visual transformer의 마지막 `ln_post`와 projection을 거친 7×7 패치 토큰을 각각
L2 정규화한다. 원본 패치 $p_i$, 복원 패치 $q_j$ 사이 거리는
$d_{ij}=(1-p_i^Tq_j)/2$다. 격자 Chebyshev 거리 1 이내에서만 대응을 허용한다.

원본에서 복원으로의 지원 부족은 $u_i=\min_j d_{ij}$,
반대 방향은 $v_j=\min_i d_{ij}$로 정의한다.

$$LSSD=\operatorname{mean}_t\max\left(
\operatorname{TopMean}_{25\%}(u_t),\operatorname{TopMean}_{25\%}(v_t)\right).$$

상위 25%는 49개 중 올림한 13개다. 점수는 0~1, 작을수록 좋으며 두 방향의 점수도 저장한다.
대응은 객체 ID가 아니라 격자 토큰의 최근접 대응이고 다대일을 허용한다. 따라서 중복 객체의
개수, 객체 동일성, 실제 가려짐을 판정하지 못한다. 방향별 지원 부족을 곧바로 객체 누락률·환각률로
해석하지 않는다. 마스크나 객체 검출 정답을 사용한 점수가 아니다.

이 두 이름은 로컬 연구 후보 이름이다. 선행 지표의 단순 명칭 변경이나 임의 가중합으로
신규성을 주장하지 않는다. 아래 선행 연구 검토와 첫 결과만으로 논문 신규성이 확정되지 않는다.
두 점수 모두 오류 발생 확률이나 틀린 객체의 비율을 뜻하지 않는다.

## 선행 지표와 비교 범위

| 방법 | 이번 비교에서의 역할과 차이 |
|---|---|
| PSNR·SSIM | 원본과 같은 시점의 픽셀·구조 차이. 기존 프레임 평균을 유지한다. 일부 프레임이 정확히 같을 때 평균 PSNR 상한의 영향을 명시하고 전체 MSE의 PSNR도 별도로 추가한다. |
| [LPIPS](https://richzhang.github.io/PerceptualSimilarity/) | LPIPS-Alex v0.1의 프레임 평균. 깊은 특징 거리 기준선이다. |
| [CLIP](https://github.com/openai/CLIP) | 원본·복원의 프레임 특징 cosine 평균. 텍스트 의미 정답이나 객체 존재 판정으로 쓰지 않는다. |
| [TecoGAN tLP·tOF](https://github.com/thunil/TecoGAN/blob/master/metrics.py) | 시간축 기준선. 인접 표본 프레임의 LPIPS 변화량 차이와 Farneback flow 차이를 측정한다. |
| [Ge et al., CVPR 2024](https://arxiv.org/abs/2404.12391) | FVD의 시간 왜곡 민감도·내용 편향 분석. 통제된 시간 오류를 별도로 평가하는 근거다. 이번에는 FVD를 계산하지 않는다. |
| [ARGUS, ICCV 2025](https://arxiv.org/abs/2506.07371) | Video-LLM의 자유 서술과 사람의 정답 캡션을 비교하는 과제다. 픽셀 복원과 과제가 다르고 사람 정답이 필요하므로 자동 정답으로 전용하지 않는다. |

tLP는 $\operatorname{mean}_t|LPIPS(A_t,A_{t+1})-LPIPS(B_t,B_{t+1})|$이다.
tOF는 같은 구간에서 계산한 flow 차이의 픽셀별 L2 norm 평균이다. 공개 TecoGAN 코드와 같은
Farneback 파라미터 `(0.5,3,15,3,5,1.2,0)`를 사용한다. 원래 코드의 공간 가장자리 crop,
앞뒤 2프레임 제외, tLP×100은 적용하지 않는다. 기존 공개 수치의 재현이 아닌 명시적 변형이다.
표본 간격이 길면 optical flow가 부정확할 수 있으므로 실제 속도·방향 정답으로 간주하지 않는다.

## 분할·통계·채택 기준

- 실제 영상: ETRI 10개로 임계값 조정, WebVid 53개·Kinetics 14개로 평가.
  렌더링: seed 1000~1015 조정용, 9000~9031 평가용. 같은 원본에서 파생한 사례는 같은 분할에 둔다.
- 총 원본 125개, 사례 2,432개. 각 영상 전체 길이에서 끝점을 포함해 균등한 12개 시점을 추출한다.
  실제 영상은 시간 조작 12개와 대조군 4개, 렌더링은 여기에 객체 조작 9개를 추가한다.
- 원본 파일 SHA256의 중복을 거부한다. 동일 내용의 다른 인코딩까지 탐지한 분할은 아니며
  사전학습 CLIP의 학습 데이터와 중복되지 않는다고 보장하지 않는다.
- 경보는 조정용 대조군 점수의 95분위수(`higher`)를 엄격히 초과할 때 발생한다.
  기준은 높을수록 오류가 크도록 방향을 맞춘 점수에 적용한다.
- AUC, 검출률, 오탐률, 판단 가능 비율, 강도–점수 Spearman을 보고한다.
  500회 bootstrap은 프레임·변형 사례 대신 **원본 전체**를 재표집한다. `null`은 0점으로 바꾸지
  않으며 판단 보류를 미검출로 센 보수적 검출률도 기록한다.
- 실험 전 채택 기준: 대상 오류별 AUC≥0.9, 검출률≥0.8, 대조군 오탐률≤0.1,
  측정 비율≥0.95, 원본별 강도 반응 Spearman 중앙값≥0.5.
  RTE 대상은 네 시간 오류, LSSD 대상은 세 객체 조작이다. **두 후보 모두 NOT_PASSED**다.
- tLP·tOF·전체 MSE의 PSNR은 첫 결과를 본 뒤 비교 보강용으로 추가했다. 원래 후보·임계값을
  다시 맞추지 않고 저장된 모든 사례의 픽셀 SHA256을 확인해 재생한다. 최초 결과와 추가 결과를 분리한다.

## 실제 복원 및 가변 길이 실행

[가변 길이 설정](../configs/lgvsc_variable.json)은 이미 전처리한 576×320, 24fps 영상의
프레임 수를 2~384 범위에서 입력별로 정한다. 전처리를 중복 실행하지 않는다.
기존 ETRI 고정 설정은 유지하고, 재개 검증에도 입력별 실제 길이를 적용한다.

공통 평가 입력은 `source-reconstruction-pairs-v1` JSON이다. 각 행에 모델 이름, 원본·복원 경로,
각 SHA256과 실행 근거를 저장한다. 원본의 물리 시간과 가장 가까운 복원 시점을 비교한다.
영상 전체 길이를 맞춰 늘이거나 특징으로 시간축을 정렬하지 않는다. 공식 LGVSC의 중복 경계 프레임도
실제 재생 시간에 남긴다. 끝부분이 없으면 표본 누락 수와 커버리지를 기록하며 2개 미만이면 실패한다.

[SGD 연결 코드](../src/semantic_transmission/sgd_bridge.py)는 새 입력의 **모든 프레임**을 기존
`sgdjscc_lab`의 실제 pretrained JSCC·ControlNet·diffusion으로 복원한다. 기본 50 step, text 사용,
AWGN 10dB이며 실패·누락·비정상 값·길이 불일치를 검사한 뒤 공통 입력을 등록한다.
기존 ETRI의 int4+프레임 재사용 실험과는 별도 framewise AWGN 실행이다.
두 모델의 전송량·실제 채널 조건이 일치하는 성능 비교를 구현한 것은 아니다.

## 실행 방법

저장소 루트에서 실행한다. 아래 `$SEMTX_PY`는 로컬 평가 환경이며, SGD 작업은 내부에서
기존 `ptest` 환경의 별도 프로세스를 호출한다. 큰 모델 실행은 순차 진행한다.

```bash
export SEMTX_PY=/home/sangukbae/anaconda3/envs/lgvsc/bin/python
env -u PYTHONPATH -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 "$SEMTX_PY" -m pip install --no-deps -r requirements-evaluation.txt
env -u PYTHONPATH -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 "$SEMTX_PY" -m semantic_transmission.automatic_validation --data-root data --output outputs/automatic_validation_new
env -u PYTHONPATH -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 "$SEMTX_PY" scripts/add_temporal_baselines.py outputs/automatic_validation_new
env -u PYTHONPATH -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 "$SEMTX_PY" scripts/report_automatic_validation.py outputs/automatic_validation_new
env -u PYTHONPATH -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 "$SEMTX_PY" -m semantic_transmission.research --input-dir data/webvid55/processed --profile configs/lgvsc_variable.json --output outputs/webvid_new --dry-run
# --dry-run을 빼면 전체 SKEM+DSA 복원이 실행되므로 긴 시간이 필요하다.
env -u PYTHONPATH -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 "$SEMTX_PY" -m semantic_transmission.sgd_bridge --input path/to/source.mp4 --output outputs/sgd_new
env -u PYTHONPATH -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 "$SEMTX_PY" scripts/register_lgvsc_pairs.py --batch outputs/lgvsc_completed --output outputs/lgvsc_pairs.json
env -u PYTHONPATH -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 "$SEMTX_PY" -m semantic_transmission.evaluate_pairs --pairs outputs/lgvsc_pairs.json --pairs outputs/sgd_new/pairs.json --output outputs/common_scores.json
```

다음 후보 설계에 이번 평가 자료를 사용하면 그 자료는 더 이상 미사용 최종 평가셋이 아니다.
신규 후보의 최종 채택에는 다른 원본·다른 오류 생성 방식의 독립 자동 정답 자료를 사용해야 한다.
