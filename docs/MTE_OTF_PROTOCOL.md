# MTE·OTF: 사람의 추가 검수 없는 2차 지표 실험

2026-09-14. 기존 RTE·LSSD 결과를 참고한 **새 후보의 개발·검증**이다.
이 문서의 설계와 채택 기준을 점수 계산 전에 고정한다. 성능 및 신규성은
실험과 선행 연구 비교로 판단하며, 구현 자체를 연구 성공으로 간주하지 않는다.

## 데이터와 정답

- 이전 2,432개 사례는 재평가의 독립 시험 자료로 쓰지 않는다.
- [DAVIS 공식 TrainVal 480p](https://davischallenge.org/davis2017/code.html)의
  train에서 8개 개발 원본, val에서 16개 평가 원본을 이름 해시 순으로 고정 선택한다.
  원본 JPEG의 매 3번째 프레임을 최대 32개 사용한다. **JPEG 묶음에 촬영 FPS가
  없으므로 8 Hz를 부여한 시험용 재생 시간축**이며 원 촬영 시간이라고 주장하지 않는다.
  최대 4초의 일부 구간에 대한 시험이다. 픽셀 해시와 원 프레임 번호를 저장한다.
- 새 렌더러의 개발 seed 260914100–107, 평가 seed 260914900–915를 사용한다.
  합성 원본은 24개, 각 32프레임, 320×192, 8 fps이다. 독립 원본 수와
  같은 원본에서 파생한 오류 사례 수를 구분한다.
  합성 개발/평가는 같은 렌더러·궤적 형식의 다른 seed다. 미지의 장면 유형으로의
  일반화나 독립적인 영상 생성 모델 반복 실험이라고 해석하지 않는다.
- 공개 영상에는 역순·정지·지연·구간 순서 교환을 코드로 삽입한다.
  합성 영상에는 배경을 유지한 객체 움직임 역전·정지, 객체 추가·누락·모양 변경도 넣는다.
  오류 구간 비율은 12.5%, 25%, 50%이다. 지연은 비율에 따라 별도 지연량을 적용한다.
- 정상 대조군은 원본, 밝기 +20, gamma 0.75, JPEG 75이다. 합성에는 **마스크·궤적·
  평균 객체 색은 유지하면서 배경/표면 질감만 바꾸는** 두 대조군을 추가한다.
  색·질감 자체가 의미인 작업까지 정상이라고 가정하지 않는다. SDEdit은 정답 생성에 쓰지 않는다.
- 코드의 변형 기록과 렌더러의 객체 마스크가 정답이다. 지표에는 RGB만 입력한다.
  DAVIS의 기존 객체 마스크는 추출기 검출률 감사에만 사용할 수 있다. 기존 주석은 사람이
  만든 자료이며, **이번 작업에서 사람이 추가 검수하지 않는 것**과 구분한다.
- 독립성은 이전 지표 실험과 원본·seed를 분리했다는 뜻이다. SAM2 등 사전학습 모델이
  DAVIS를 전혀 보지 않았다는 보장은 아니다. 실제 LGVSC·SGD 복원에는 의미 오류 정답이 없다.

## MTE 정의

`motion_metric.py`: torchvision RAFT-small C_T_V2의 FP32, 12회 갱신 광류를 사용한다.
Farneback은 이름을 구분한 대체 백엔드이며 이번 주 실험은 RAFT이다.

1. 두 영상을 동일한 시험 재생 시점에서 비교한다. 실제 복원 쌍은 원본·복원 FPS로
   대응하며, `output_source_indices`에 따른 중복 제거·시간 재정렬을 하지 않는다.
2. 8픽셀 간격 광류에 RANSAC similarity affine을 맞춘다. inlier ≥50%이면 카메라
   변위장을 빼고, 미만이면 0 카메라로 처리해 추정 실패율을 별도 기록한다.
3. 잔여 움직임이 0.25픽셀 이상인 곳에 `ω=|R|/(|R|+1)` 가중치를 적용한다.
4. 3×3 공간 격자에서 **8방향 비음수 광류 히스토그램**을 만들고 0.5초 구간별로 집계한다.
   반대 방향 객체의 벡터 평균이 0이 되는 문제를 줄인다. 장거리 객체 궤적 자체를 복원하는
   지표는 아니며, 히스토그램만으로 객체 동일성·복잡한 사건을 모두 표현하지 못한다.
5. 구간 방향 오류 `e=||hA-hB||/(||hA||+||hB||+ε)`와, 양쪽에 움직임이 있을 때
   전경 중심 오류 `p=min(1,||cA-cB||/0.15)`의 최댓값을 구한다.
   평균 잔여 활동 0.08픽셀 미만인 구간은 측정 불가다.
6. 측정 구간 중 상위 20% 평균이 `mte_tail`이다. 범위 [0,1], 클수록 차이가 크다.
   전체 평균, 최대값, 원 제안의 방향 벡터 평균을 제거 비교(ablation)로 함께 기록한다.
   짧은 오류에 대한 민감도는 실험 대상이며 **보장되지 않는다**.
7. 정지/시작/방향 전환은 연속 두 광류 전이로 확인하고, 같은 사건 종류끼리
   ±0.25초 내 일대일 대응한다. 실제 미대응 사건으로 누락·추가 비율을 계산한다.
   사건이 없으면 해당 비율은 null이다. 사건 정의가 못 잡는 복잡한 동작도 있다.

카메라 변위 차이는 별도 점수다. MTE가 null이거나 낮다고 카메라 오류까지 없다고
해석하지 않는다. 자동 카메라 추정은 전경이 화면 대부분을 차지할 때 실패할 수 있다.

## OTF 정의

`object_metric.py`: SAM2.1 Hiera-tiny 자동 마스크 생성 + DINOv2 ViT-S/14.
원본·복원 각각의 모든 평가 프레임에서 독립적으로 객체를 발견한다. 사람 클릭,
객체 이름, 원본 정답 마스크를 추출기에 주지 않는다. SAM2 영상 전파의 초기 객체
지정 문제를 피하기 위해 **프레임별 자동 분할 뒤 별도 추적**을 적용한다.

- SAM2 자동 그리드 12×12, predicted IoU ≥0.8, stability ≥0.9. 마스크 면적은 화면의
  0.2–65%, 프레임당 최대 20개다. 중첩된 부분 마스크를 억제한다. 이 크기·분할 범위
  밖의 객체를 측정한다고 주장하지 않는다.
- DINOv2는 392×224 입력의 28×16 패치를 마스크별 평균하고 L2 정규화한다.
- 영상 내부에서는 IoU/특징과 제한된 위치 차이로 연결하고, 최대 두 누락 프레임을
  넘어가면 새 트랙으로 둔다. 연결에 성공해도 실제 관측 누락을 메우지는 않는다.
- 영상 간에는 동시 관측 프레임의 DINO cosine·마스크 IoU로 **트랙 전체를 일대일
  Hungarian 대응**한다. 매 프레임 다른 객체로 갈아타는 방식은 쓰지 않는다.
- 모양 비교에서만 최대 화면 크기의 3% 평행 이동을 허용한다. 시간 이동은 허용하지 않는다.
- `N_A`, `N_B`는 양쪽 관측 객체-프레임 수, `M`은 대응 트랙이 함께 관측된
  객체-프레임 수다. `OTF-F1=2M/(N_A+N_B)`, `OOR=1-M/N_A`, `HOR=1-M/N_B`.
  서로 다른 면적 가중치로 만든 비율을 precision/recall로 오인하지 않도록 수정했다.
- 대응 관측 중 cosine <0.65 또는 모양 IoU <0.5인 비율은 `ODR`이다.
  주 객체 오류 후보는 `otf_distortion_error=max(1-OTF-F1, ODR)`이며 [0,1]이다.
  `1-OTF-F1`, OOR/HOR/ODR을 함께 제시해 왜 점수가 나왔는지 구분한다.
- **0.5초보다 짧은 오류도 객체-프레임 수에 반영**한다. 연속 0.5초 이상 사건 수는
  보조 결과이며 짧은 사건을 정상으로 없애지 않는다. 원본 트랙이 없으면 null이다.
  복원에서 객체가 하나도 검출되지 않으면 예측 누락으로 기록하되 검출기 실패와 혼동될 수 있다.

마스크 추출/트랙 분할 오류가 OTF를 오염시킬 수 있다. 렌더러 정답과 추출 마스크를
독립 비교해 도구 검출률을 함께 보고한다. 예측 객체 미대응을 실제 환각의 확정 정답으로 쓰지 않는다.

## 비교·채택

- 같은 픽셀·시점에서 PSNR, SSIM, LPIPS-Alex, CLIP cosine, RTE, LSSD,
  tLP-Alex, tOF-Farneback, 그리고 **MTE와 같은 RAFT 기반 tOF**를 계산한다.
- 개발 정상 대조군의 95백분위(higher)를 문턱으로 고정한다. 평가 자료를 본 뒤
  수식이나 문턱을 바꾸지 않는다. 공개/합성 도메인을 나눠 보고한다.
- 평가 원본별 bootstrap 500회, 95% CI를 사용한다. null은 오탐 계산에서는
  측정 비율과 함께 별도 표시하고, 검출률에서는 미검출로 센다.
- 후보 채택 기준: 오류 종류마다 AUC ≥0.9, 검출률 ≥0.8, 오탐률 ≤0.1,
  측정 비율 ≥0.95. 짧은 오류 강도별 결과도 별도 제시한다.
- 기존 지표보다 낫다는 주장은 동일 원본 쌍 bootstrap AUC 차이의 95% CI가
  모든 사전 비교 지표에 대해 0보다 클 때에만 허용한다. 작은 평가 표본의 불확실성을 남긴다.
- 기준 통과와 신규성 입증은 별개다. 기존 기술 조합이 주된 구성이라면
  실용적 진단 도구로만 보고하고, 새 논문 지표로 확정하지 않는다.

## 선행 연구와 겹치는 부분

| 연구 | 이미 있는 요소 | 이번 후보가 입증해야 할 부분 |
|---|---|---|
| [TecoGAN](https://github.com/thunil/TecoGAN/blob/master/metrics.py) | 광류·시간 지각 차이 | 같은 광류에서 구간/방향 표현이 검출과 오탐을 개선하는지 |
| [HOF / Laptev et al., CVPR 2008](https://www.di.ens.fr/~laptev/actions/) | 공간·시간 구간의 광류 방향 히스토그램 | 방향 히스토그램 자체는 기존 표현이며 새 표현이라고 주장할 수 없음 |
| [InterDyn](https://interdyn.is.tue.mpg.de/media/upload/interdyn_video.pdf) | SAM2/CoTracker3 객체 움직임 궤적의 원본-생성 비교 | 움직임 비교 자체는 신규성이 아님; 짧은 오류·무오류 외형 변화에 대한 추가 이점 |
| [HOTA](https://arxiv.org/abs/2009.07736) | 검출·객체 연결·위치 정확도 분리 | 자동 기준 트랙을 쓰는 것만으로 새 평가 원리가 되지는 않음 |
| [IDF1 / Ristani et al.](https://arxiv.org/abs/1609.01775) | 전역 객체 대응 뒤 공유 관측 수로 계산하는 identity F1 | OTF-F1과 대수적 형태가 같음; 특징 기반 대응·왜곡 분해의 추가 이점을 따로 입증해야 함 |
| [VBench](https://arxiv.org/abs/2311.17982) | 객체 정체성 등 영상 품질 분해 | 원본과 복원 간 비교의 추가 효용과 객체 추출 오류 분석 |
| [PLACID, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Tarres_PLACID_Identity-Preserving_Multi-Object_Compositing_via_Video_Diffusion_with_Synthetic_Trajectories_CVPR_2026_paper.pdf) | 객체 미검출률과 객체별 CLIP/DINO 비교 | 객체 누락 및 특징 비교 자체를 신규성으로 주장할 수 없음 |

InterDyn의 방법은 point-trajectory correlation을 사용하므로 현재 MTE와 동일 수식은
아니지만 목적과 구성 요소가 겹친다. 본 파일은 포괄적인 신규성 증명이 아닌 근접 선행 연구 감사다.

OTF 정식 계산 전에 `tracking_baseline_protocol.json`을 추가 고정했다. 같은 SAM2/DINO
예측 트랙에 마스크 IoU ≥0.5의 IDF1 및 프레임별 검출 F1을 적용한다. 사람 주석을
기준 트랙으로 쓰는 정식 추적 벤치마크 점수와 구분한다. IDF1은 전역 대응에서
IoU 조건을 만족하는 관측 수를 최대로 하고, OTF는 특징·모양 평균으로 트랙을 대응한 뒤
존재·왜곡을 나누는 차이가 있다. 수식의 F1 형태 자체는 새롭지 않다.
이 추가 비교를 고정할 때 MTE 공개 영상 결과는 이미 보았으며 OTF 정식 결과는 보지 않았다.

## 재현

`metric_v2_validation`의 `prepare → motion → object → summarize` 순서로 실행한다.
`prepare`가 코드·가중치·원본·픽셀 해시와 프로토콜을 고정한다. 출력별 JSON, CSV,
원본별 bootstrap, 측정 불가, 실패 사례를 보존한다. 진행 도중 결과를 바탕으로 수식을
바꾸면 새 실험·새 평가 원본이 필요하다.

GPU 라이브러리 불일치가 발견되어 커널과 같은 NVIDIA 580.173.02 라이브러리를
`.local/metric_v2_driver/`에만 풀고 실행 프로세스의 `LD_LIBRARY_PATH`로 연결했다.
시스템 드라이버를 설치·교체하거나 재부팅하지 않았다. OTF 의존성은 기존 환경을
수정하지 않도록 `.local/metric_v2_env`에 분리했다.

이 컴퓨터의 재현 명령은 다음과 같다. 새 실험의 출력 경로는 기존 결과와 구분한다.

OTF에는 PyTorch ≥2.5.1이 필요하므로 `lgvsc`의 PyTorch 2.2.2를 업그레이드하지 않는다.
이번에는 기존 `semantic-diffusers`의 PyTorch 2.12.0/CUDA 13.0을 바탕으로 다음 별도
환경을 만들었다. 실제 패키지 목록은 출력의 `object_packages.txt`와 `motion_packages.txt`에 있다.

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 \
  /home/sangukbae/anaconda3/envs/semantic-diffusers/bin/python \
  -m venv --system-site-packages .local/metric_v2_env
.local/metric_v2_env/bin/python -m pip install hydra-core==1.3.2 iopath==0.1.10
SAM2_BUILD_CUDA=0 .local/metric_v2_env/bin/python -m pip install \
  --no-deps --no-build-isolation -e .local/metric_v2_models/sam2
```

아래 자료 준비 명령을 먼저 실행해 모델 저장소·가중치를 확보한 뒤 환경을 구성한다.

```bash
python3 scripts/prepare_metric_v2_assets.py
bash scripts/run_metric_v2.sh prepare outputs/mte_otf_new
# OTF 결과를 계산하기 전에 가까운 추적 지표의 비교 정의도 고정한다.
PYTHONPATH=src /home/sangukbae/anaconda3/envs/lgvsc/bin/python \
  scripts/add_metric_v2_tracking_baselines.py --declare --output outputs/mte_otf_new
bash scripts/run_metric_v2.sh motion outputs/mte_otf_new
bash scripts/run_metric_v2.sh object outputs/mte_otf_new
bash scripts/run_metric_v2.sh summarize outputs/mte_otf_new
PYTHONPATH=src /home/sangukbae/anaconda3/envs/lgvsc/bin/python \
  scripts/report_metric_v2.py --output outputs/mte_otf_new
```

`prepare`의 같은 원본·seed로 다시 실행한 결과는 반복 검증이지 새 독립 시험이 아니다.
추가 객체 baseline·도구 감사·실제 복원 진단 명령은 결과 보고서의 재현 기록을 따른다.
