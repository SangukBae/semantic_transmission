> 보관본: 단기 성능 개선 중심으로 정리하기 전의 전체 조사·실험안이다. 현재 실행 우선순위는 [개정 문서](../../ETRI_HALLUCINATION_LITERATURE_REVIEW_2026-09-21.md)를 따른다. 아래 표본 수·후속 후보는 현재 실행 약속이 아니다. 상대 링크만 보관 위치에 맞췄다.

# ETRI 후속 요구 기반 영상 복원 할루시네이션 검출·완화 심층 조사

조사 기준일: **2026-09-21 (KST)**. 대상: LGVSC의 SKEM–NTSCC–DSA/Open-Sora 경로.

**권고는 기존 LGVSC에 장면별 조건 수명 관리와 수신 근거 기반 제어를 먼저 추가하고, 독립 정답으로 오류 감소를 검증하는 것이다.** SparkVSR·STCDiT의 키프레임 활용, HalluGen의 검출 평가, DynamicDPS의 관측 제약이 가장 유용한 선행 근거다. 다만 어느 논문도 현재 LGVSC의 희소 수신 정보와 AWGN 조건에서 60초 이상 Added/Missing/Distorted 감소를 대신 입증하지 않는다.

실행 설계는 [LGVSC 적용 실험안](ETRI_LGVSC_HALLUCINATION_EXPERIMENT_PLAN_2026-09-21.md), 코드 버전·줄 근거는 [공개 코드 감사](ETRI_HALLUCINATION_CODE_AUDIT_2026-09-21.md)에 정리했다.

## 1. 조사 방법과 근거의 수준

이번 세션에는 전용 Deep Research 실행 도구가 연결되어 있지 않아, **웹 검색 → 논문 원문 → 저자 저장소 → 핵심 코드 정적 대조 → 로컬 LGVSC 적용 검토** 방식으로 심층 조사했다. 전용 Deep Research 제품을 실행한 결과로 표시하지 않는다.

- 2025–2026년 연구를 중심으로 검색하고 2026년 8–9월 신작까지 확인했다. 최근 검색에 걸린 논문을 모두 포함하는 체계적 문헌고찰이나 완전한 최신 목록은 아니다.
- 핵심·인접 논문 **13편**, LGVSC 원문 1편을 검토했다. 저자 공개 저장소 **9개**의 커밋을 고정하고 **35개 선택 소스 파일**을 확보하여 핵심 실행·평가 경로를 대조했다. SAVER는 원문·저장소 설명까지 확인한 인접 연구다.
- arXiv HTML의 방법·실험·한계와 CVF/학회 페이지를 사용했다. 버전은 [논문 manifest](../../../../outputs/research/etri_hallucination_20260921/paper_manifest.json)에 기록했다. sFRC·SeedVR2·SAVER·LGVSC는 확인된 v2를 반영했다.
- 논문 성능 수치는 **저자 보고값**이다. 공개 코드는 정적으로 확인했으며, 외부 모델의 가중치 다운로드·GPU 재현 실험·새 LGVSC 복원은 실행하지 않았다. 코드가 공개되어 있다는 사실과 논문 수치의 재현 가능성은 별개다.
- 검색 결과·리뷰 모음은 후보 발견에만 사용하고, 핵심 결론은 원문·저자 코드·로컬 파일에 연결했다. 코드 미확인은 공개 원문과 저자 연결 경로 및 제목 검색에서 확인하지 못했다는 뜻이다.

대표 검색어는 `video restoration hallucination detection mitigation 2025 2026`, `image restoration measurement consistency hallucination`, `video super-resolution faithfulness sparse keyframe`, 그리고 후보별 `paper / official code / github`다. 검색 뒤 초해상도·생성 영상 보정과 VideoLLM의 **언어 출력 환각**을 구별했다.

## 2. ETRI 요구사항을 연구 질문으로 변환

요구사항의 기준은 [후속 메일 요약](../../../ETRI_FOLLOWUP_EMAIL_SUMMARY.md)과 [후속 프로토콜](../../../ETRI_FOLLOWUP_PROTOCOL.md)이다. 아래 우선순위와 구체 실험 수치는 이번 조사자의 제안이다.

| 요구 | 검증할 연구 질문 | 연결 논문·설계 |
|---|---|---|
| R1: AWGN 중심 검출·완화 | 실제 수신 정보만으로 위험을 찾고 낮출 수 있는가? | HalluGen, DynamicDPS, RPU; RX 전용 위험 판정과 독립 원본 평가 분리 |
| R2: 동일 조건 전후 영상·성공/실패/새 왜곡 | 보기 좋아진 결과가 내용도 정확해졌는가? | SHAFE·sFRC의 영역 평가, SparkVSR의 충실도–지각품질 상충, RAR의 평가 반복 구조 |
| R3: 실제 연속 60초 이상 확장 검토, 3유형 | 메모리·시간축·조건 상태를 유지하며 복원 가능한가? | FlashVSR의 순차 처리, STCDiT의 구간 처리; 실제 길이·상태 계승 별도 검증 |
| R4: drift·깜빡임·씬 체인지·검출 실패 대안 | 이전 장면의 객체·캡션·참조가 언제까지 남는가? | 경계별 초기화, 최대 갱신 간격, 패킷 유효기간; STCDiT의 운동 분할과 shot detection 구별 |
| R5: 학습 이력·동적 의미정보 | 정보 누락과 생성기의 시간축 표현 부족 중 무엇이 원인인가? | SparkVSR·STAR의 영상 학습, PLLaVA·Open-Sora 이력, 사건 시각·방향·속도 보강 |
| R6: 기존 방향 유지·내부 원본 사용 | 기존 지표 연구를 실제 오류 감소에 연결할 수 있는가? | FSO/UEP/EOI의 한계를 유지한 채 독립 정답으로 검증; 원본 외부 업로드 없는 조건 |

**60초는 확장 검토·달성 목표이고, 5분은 필수 조건이 아니다.** 짧은 영상을 반복하거나 무관한 클립을 연결해 달성했다고 보고하지 않는다. 채널 종류 비교는 이번 범위에서 제외한다.

## 3. 가장 중요한 구분: 원본 오류, 수신 근거, 영상 품질

원본을 $x$, 수신 정보를 $y$, 복원을 $\hat{x}$라고 하면 서로 다른 세 질문이 있다.

1. **원본 충실도:** $\hat{x}$가 $x$의 객체·행동·관계를 보존했는가? ETRI의 Added/Missing/Distorted 정답은 여기에 속한다.
2. **수신 근거 일치:** $\hat{x}$가 실제 $y$와 모순되는가? 수신단 제어에 쓸 수 있으나, 관측되지 않은 중간 사건의 진실을 모두 알 수는 없다.
3. **지각·시간 품질:** 선명하고 자연스러우며 깜빡임이 적은가? 틀린 객체를 일관되게 유지해도 좋아질 수 있다.

HalluGen은 관측과 모순되는 intrinsic 오류와 관측에는 맞지만 원본과 다른 extrinsic 오류를 나눈다. 이는 ETRI의 Added/Missing/Distorted와 **서로 다른 분류 축**이다. 객체 추가가 항상 intrinsic이고 누락이 항상 extrinsic인 대응은 성립하지 않는다. [HalluGen §3](https://arxiv.org/html/2512.03345v1)

일례로 두 키프레임 사이에만 자전거가 등장했다가 사라지고 이 사건이 캡션에도 없으면, 자전거가 있는 원본과 없는 원본이 같은 패킷을 만들 수 있다. 수신단은 추가 관측 없이 둘을 확정적으로 구별하지 못한다. 따라서 **RX에 없다는 이유만으로 Added라고 판정하거나 객체를 삭제하면 안 된다.** 이 경우는 근거 부족으로 보류하고, 사건 정보나 키프레임 추가 비용을 연구해야 한다. 이는 희소 관측에서 도출한 본 조사자의 분석이다.

## 4. 논문별 선택 결과

`A`는 핵심 코드까지 대조, `B`는 원문·공개 자료 확인이나 직접 이식에 제약, `C`는 개념적 인접 연구다. 등급은 논문 품질 순위가 아니다.

| 논문·공개 시점/학회 | 실제 대상과 기여 | LGVSC 활용 | 판단·근거 |
|---|---|---|---|
| **HalluGen**: 2025-12, CVPR 2026 | 통제 가능한 복원 환각 합성, SHAFE, 관측–복원 검출기 | 검출기의 오탐·일반화·위치 평가 | A. [원문](https://arxiv.org/html/2512.03345v1) · [코드](https://github.com/edshkim98/HalluGen) |
| **DynamicDPS**: 2025-03/v2 2025-07, MICCAI 2025 | 조건부 MRI 복원을 관측 제약과 적응형 확산으로 보정 | 수신 관측에 묶인 제한적 보정 원리 | A. [원문](https://arxiv.org/html/2503.01075v2) · [코드](https://github.com/edshkim98/DynamicDPS) |
| **sFRC**: 2026-03/v2 2026-05 | 참조 영상과 국소 주파수 상관으로 구조 오류 위치 검출 | 학습 없는 영역 오류 비교 기준선 | A. [원문](https://arxiv.org/html/2603.04673v2) · [코드](https://github.com/DIDSR/sfrc) |
| **SparkVSR**: 2026-03, 저장소에 ECCV 2026 표시 | 전체 LR 영상+희소 HR 키프레임, 참조 강도 제어 | 수신 키프레임의 선택적 전파 아이디어 | A. [원문](https://arxiv.org/html/2603.16864v1) · [코드](https://github.com/taco-group/SparkVSR) |
| **STCDiT**: 2025-11, CVPR 2026 | 운동별 VAE 분할과 anchor-frame guidance | 경계·VAE·참조 전파 원인 분리 | A. [원문](https://arxiv.org/html/2511.18786v1) · [코드](https://github.com/JyChen9811/STCDiT) |
| **SeedVR2**: 2025-06, ICLR 2026 | 한 단계 영상 복원, 적대적 후학습, 가변 window | 복원 후 보정의 품질·비용 비교 후보 | A. [원문](https://arxiv.org/html/2506.05301v2) · [코드](https://github.com/ByteDance-Seed/SeedVR) |
| **FlashVSR**: 2025-10, CVPR 2026 | 순차 영상 초해상도, 희소 attention, 경량 decoder | 장시간 처리 구조와 비용 측정 | A. [원문](https://arxiv.org/html/2510.12747v1) · [코드](https://github.com/OpenImagingLab/FlashVSR) |
| **STAR**: 2025-01, ICCV 2025 | T2V prior, 국소 정보 모듈, 동적 주파수 손실 | 영상 미세조정이 필요한 경우의 손실 설계 | A. [원문](https://arxiv.org/html/2501.02976v1) · [코드](https://github.com/NJU-PCALab/STAR) |
| **RPU**: 2026-06 사전공개 | DPS의 prior update를 국소 안정성 탐색으로 수정 | 관측 보정 외에 prior 자체 제어 필요성 | B. [원문](https://arxiv.org/html/2606.02331v1); 저자 코드 미확인 |
| **RAR**: 2026-03, CVPR 2026 | 잠재 공간에서 복원–평가–반복, 정지 결정 | 위험 구간만 제한적으로 재생성하는 구조 | A/C. [원문](https://arxiv.org/html/2603.26385v1) · [코드](https://github.com/saic-fi/RAR) |
| **Zero-Shot VR with Multi-Modal References**: 2026-08-27 사전공개 | 이미지 확산+텍스트/이미지 참조+시간 token merging | 무학습 시간 일관성 보강의 최신 후보 | B. [원문](https://arxiv.org/html/2608.26476v1); 해당 신작 코드 미확인 |
| **VidHalLoc**: 2026-09-09 사전공개 | VideoQA·caption의 환각 검출기 평가 | VLM 판정기를 독립 정답으로 삼지 말아야 할 근거 | C. [원문](https://arxiv.org/html/2609.09895v1) · [공식 데이터 연결](https://arxiv.org/abs/2609.09895) |
| **SAVER**: 2025-08/v2 2026-07, AAAI 2026 | 양식 변화 이미지에 대한 LVLM 언어 환각 완화 | 송신 캡션 오류 연구에 한정 | C. [원문](https://arxiv.org/html/2508.03177v2) · [코드](https://github.com/llizhaoxu/SAVER) |

### 4.1 HalluGen — 검출 평가에 가장 직접적

4,350개 MRI 파생 이미지로 통제 오류를 구성하고, 원본 참조형 SHAFE와 원본 불필요 검출기를 구별한다. 후자는 **측정값과 복원값 둘 다** 입력한다. 저자 보고 AUC는 합성 intrinsic 0.91, extrinsic 0.77, 실제 복원 0.73이다. 실제 복원 결과로 전이할 때의 성능 저하를 설계에 반영해야 한다. SHAFE는 low-pass 처리, 얕은 특징의 위치별 거리, softmax 집계를 사용한다. [§4.3·보충 §8–9](https://arxiv.org/html/2512.03345v1)

코드 `SHAFE.forward(pred, gt)`는 원본이 필요하고, 기본 추출기는 `resnetaa50d.d_in12k`다. 입력을 무조건 3회 채널 반복하므로 **1채널 MRI 전제**를 수정하지 않고 3채널 영상을 전달하면 맞지 않는다. FFT cutoff 60과 softmax 온도 0.02도 자연 영상 해상도에 맞춰 교정해야 한다. 이는 논문 기각 사유가 아니라 재현·전이 조건이다. [고정 코드](https://github.com/edshkim98/HalluGen/blob/924c40b0b2718cef9d7e079aaf56a7df85679490/SHAFE.py#L402)

LGVSC에서는 SHAFE를 **원본을 사용하는 외부 평가 기준선**으로 먼저 적용한다. RX 전용 검출기를 새로 만들 경우 측정 입력을 희소 키프레임·패킷으로 재정의하고, 합성 오류와 실제 생성 오류를 분리 평가한다. 검출기가 스스로 만든 라벨을 정답으로 다시 쓰지 않는다. 의료 특징을 자연 영상의 객체·행동 정답으로 직접 이식하는 것은 권하지 않는다.

### 4.2 DynamicDPS와 RPU — 관측 제약이 필요하지만 충분하지는 않음

DynamicDPS는 초기 조건부 복원에서 시작해 데이터 일관성에 따른 시작 시점 선택과 Wolfe line search로 보정한다. 저자의 약 5% sampling steps 및 특정 조직 부피 추정 개선은 MRI 조건의 결과다. LGVSC 지연이나 환각 감소율로 환산할 수 없다. [DynamicDPS §2–3](https://arxiv.org/html/2503.01075v2)

공개 `test.py`는 memory bank로 시작 시점을 고르지만, `test_src/test.py`의 추가 경로는 `closest_time=299`로 고정한다. 기본 conditioning의 line search도 `t>5`이고 `t%5==0`일 때 실행한다. 따라서 최신 코드 전체가 논문의 매 단계 설명과 동일하다고 할 수 없다. 실행 경로를 고정해야 한다. [시작 시점](https://github.com/edshkim98/DynamicDPS/blob/0a29e70112b993e637c2237f1d5764794347bda4/test_src/test.py#L222) · [line search](https://github.com/edshkim98/DynamicDPS/blob/0a29e70112b993e637c2237f1d5764794347bda4/guided_diffusion/condition_methods.py#L390)

RPU는 관측 보정을 유지하면서 prior update를 교란·재고정한다. FFHQ inpainting의 GT 보조 선호율 91.1%는 **동률을 제외한 사람 비교 결과**이며 환각 검출 정확도도, 영상 전송 성능도 아니다. 저자 스스로 국소 가정에 따른 해석과 특정 inverse solver의 검증 범위를 명시한다. 공식 구현은 이번 조사에서 찾지 못했다. [RPU §4–7·부록 D](https://arxiv.org/html/2606.02331v1)

LGVSC의 채널 잡음은 픽셀에 직접 더해지는 MRI/일반 복원 열화가 아니다. NTSCC 부호화 심벌에 AWGN이 더해진다. 또한 현재 decoder는 `rflow`이므로 DDPM/DPS 업데이트를 그대로 붙일 수 없다. 먼저 수신 키프레임 재현 오차를 사용하는 약한 제어를 시험하고, 심벌 일관성 최적화는 별도 연구로 남기는 것이 합리적이다. [로컬 설정](../../../../configs/rtx4080_opensora.py) · [전송 구현](../../../../src/semantic_transmission/codec_transport.py)

### 4.3 sFRC — 별도 학습 없이 위치를 비교하는 기준선

sFRC는 복원·참조 영상의 작은 패치를 훑으며 Fourier Ring Correlation을 측정한다. 서로 다른 주파수에서의 상관 저하를 구조 오류 위치에 연결하고, 전문가 주석 또는 영상화 이론으로 문턱을 교정한다. 연구 대상은 CT/MRI이며 자연 영상의 객체 의미 판정 능력을 입증한 것은 아니다. [sFRC §IV·VI](https://arxiv.org/html/2603.04673v2)

공개 코드는 patch 크기, FRC 상관 문턱, 환각 판정 문턱 `ht`를 별도 인자로 받는다. **학습이 없다는 뜻이 교정이 없다는 뜻은 아니다.** 원본–복원 시간 대응을 먼저 고정하고 명도 변화·정상 텍스처·압축·약한 정렬 오차를 음성 대조군으로 넣는다. ETRI 표시 영상의 후보 영역 생성에는 유용하나 Added/Missing/Distorted는 독립 주석으로 결정한다. [CLI와 문턱](https://github.com/DIDSR/sfrc/blob/527fbf56a3b87552047971b6bb9b5b909c5cddfa/main.py#L27)

### 4.4 SparkVSR — 희소 키프레임 활용은 가깝지만 수신 입력은 다름

SparkVSR의 조건은 전체 LR 영상과 희소 HR 키프레임이다. 참조를 제거한 분기도 **LR 영상은 유지**한다. 참조 강도 RFG를 높이면 키프레임 특징을 강하게 전파한다. 원문 Table 4의 Nano-Banana-Pro 조건에서 RFG 0→1.5는 PSNR 26.62→20.89 dB, LPIPS 0.283→0.412로 변한다. 지각 점수 상승을 원본 충실도 향상으로 읽을 수 없다는 중요한 사례다. [§3.3·4.3](https://arxiv.org/html/2603.16864v1)

공개 코드는 conditional/unconditional 출력 보간, `no_ref/gt/api/pisasr` 모드, 시간 chunk 처리를 제공한다. `gt`는 일반 LGVSC 비교에서 사용할 수 없고, 외부 API 참조는 내부 원본 관리 방침과 맞지 않는다. 수신 키프레임을 직접 넣는 로컬 인터페이스도 별도 구현·검증해야 한다. [RFG 구현](https://github.com/taco-group/SparkVSR/blob/a082284b80005bb5615c0f5f5f5ed66650b1b1e7/sparkvsr_inference_script.py#L572) · [입력 모드](https://github.com/taco-group/SparkVSR/blob/a082284b80005bb5615c0f5f5f5ed66650b1b1e7/sparkvsr_inference_script.py#L623)

우선 활용할 것은 ‘키프레임을 강하게 따를수록 좋다’가 아니라 **키프레임 신뢰도·나이에 따라 전파 강도를 검증한다**는 가설이다. 잘못 복원한 키프레임의 오류가 전 구간으로 퍼지는 부작용을 함께 측정한다. Open-Sora의 text CFG 조절과 SparkVSR의 RFG를 같은 기법으로 부르지 않는다.

### 4.5 STCDiT — 생성기 이전의 VAE 왜곡을 분리

STCDiT는 카메라 이동·회전·확대 변화를 이용해 VAE 처리 구간을 나누고, 시간 압축에서 상대적으로 구조가 잘 보존되는 첫 프레임 잠재값을 활용한다. REDS30의 VAE 자체 복원 비교 27.22→31.42 dB는 **VAE 실험의 +4.20 dB**이며 전체 전송 개선이 아니다. [§3·5.1](https://arxiv.org/html/2511.18786v1)

저장소에서 운동 분석, 구간별 VAE 인코딩, anchor index 구성을 확인했다. tiny 추론 경로는 Wan2.1 기반이며, 저자 안내의 24GB 추론 조건을 현재 16GB GPU에 바로 적용 가능한 조건으로 볼 수 없다. 카메라 운동 분할은 scene cut의 정답 검출과도 다르다. [운동 분석](https://github.com/JyChen9811/STCDiT/blob/7c4be6e2774b1bdf51658d1e495a0a3a3ace3772/Inference/video_motion_detection.py#L7) · [구간 인코딩](https://github.com/JyChen9811/STCDiT/blob/7c4be6e2774b1bdf51658d1e495a0a3a3ace3772/diffsynth/pipelines/wan_video_t2v_tiny.py#L638)

LGVSC에서는 수신 키프레임→VAE 왕복과 실제 생성 결과를 분리 진단한다. 원본 전체 영상을 VAE에 넣는 실험은 oracle 진단이며 수신단 성능으로 보고하지 않는다. 무조건 모든 구간을 초기화하면 drift가 줄어 보일 수 있으므로 진짜 경계와 정상 연속 구간을 나눠 비교한다.

### 4.6 SeedVR2·FlashVSR·STAR — 영상 prior와 실행 비용의 비교군

**SeedVR2:** 한 단계 추론을 위해 적대적 후학습·feature matching·가변 window를 사용한다. 코드는 `sample_steps=1` 경로를 제공하지만 저자는 공개 prototype이 논문 수치와 완전히 맞지 않을 수 있고 약한 열화에서 세부를 과생성할 수 있다고 명시한다. ‘한 단계’는 모델 평가 횟수이며 전체 파이프라인의 실시간성을 의미하지 않는다. [원문](https://arxiv.org/html/2506.05301v2) · [공개 모델 한계·GPU 안내](https://github.com/ByteDance-Seed/SeedVR#-notice) · [한 단계 경로](https://github.com/ByteDance-Seed/SeedVR/blob/e4de8c24441a67e1b7df56abea10645059bb1185/projects/inference_seedvr2_3b.py#L140)

**FlashVSR:** 순차 추론·희소 attention·경량 decoder로 비용을 낮춘다. 저자 약 17 FPS는 768×1408, 단일 A100 조건이다. 공개 full 예제는 프레임을 리스트에 모아 tensor로 만든 뒤 모델에 전달한다. 모델 내부의 순차 처리와 **입출력까지 메모리가 길이에 무관한 streaming**은 구별해야 한다. 60초 실험에서는 GPU뿐 아니라 호스트 메모리도 기록한다. [원문](https://arxiv.org/html/2510.12747v1) · [입력 적재](https://github.com/OpenImagingLab/FlashVSR/blob/cf910c61a60733e610e9c6e8b607f80c3a6c202b/examples/WanVSR/infer_flashvsr_v1.1_full.py#L143) · [순차 경로](https://github.com/OpenImagingLab/FlashVSR/blob/cf910c61a60733e610e9c6e8b607f80c3a6c202b/diffsynth/pipelines/flashvsr_full.py#L357)

**STAR:** 이미지 prior만으로 부족한 시간 정보를 영상 생성 prior로 보강하면서, 국소 정보 모듈과 diffusion 시간에 따른 주파수 손실로 충실도를 강화한다. 이는 **학습 방법**이다. 공개 손실의 주파수 분해 및 시간별 가중치를 함께 확인해야 한다. LGVSC에 적용하려면 수신 키프레임·희소 패킷 열화 분포로 학습 조건을 바꾸어야 한다. [원문](https://arxiv.org/html/2501.02976v1) · [공개 DF loss](https://github.com/NJU-PCALab/STAR/blob/69b8bc53e4bb04267265fa9a288ee30882c5b2ed/cogvideox-based/sat/sgm/modules/diffusionmodules/loss.py#L247)

세 모델 모두 LGVSC 복원 뒤 보정기로 넣을 수 있는 후보지만, 그때 입력은 이미 오류가 생긴 LGVSC 영상이다. 구조적 오류를 더 선명하게 만들 수 있으므로 독립 의미 정답이 필요하다. 송신 원본의 전체 LR 영상을 주는 비교는 추가 전송량을 포함한 별도 시스템으로 분리한다.

### 4.7 RAR — 반복 구조는 유용하지만 IQA 판정은 의미 정답이 아님

RAR는 잠재 공간에서 열화 식별·복원·품질 평가를 통합하고 반복한다. 코드는 현재 결과와 직전 결과를 `quality_compare_noref`로 비교해 직전 결과를 선택하거나 반복을 끝낸다. 이는 ETRI의 원본 내용 일치 판정과는 다른 목적함수다. [원문](https://arxiv.org/html/2603.26385v1) · [정지 결정](https://github.com/saic-fi/RAR/blob/ce1885cf06a61dc23ba2232472d766807ad0870b/run.py#L625)

LGVSC에서는 ‘위험 구간 발견→한 번 보수적으로 재생성→근거 일치 확인→수용 또는 기존 결과 유지’ 형태만 우선 차용한다. 반복 횟수에 상한을 두고, 재생성 후 자연스러움만 좋아진 결과를 자동 수용하지 않는다. 실제 의미 개선은 제어에 쓰지 않은 평가기로 확인한다.

### 4.8 2026년 8–9월 신작과 언어 환각 연구의 적용 경계

**Multi-Modal References (8월 27일):** 이미지 diffusion의 프레임 간 깜빡임을 dual prompt tuning, texture-aware temporal token merging, 참조 attention으로 다룬다. 참조를 이용한 복원·편집 연구이며 원본 기준 객체 오류의 포괄적 검출기를 제시한 것은 아니다. 같은 저자의 [ZVRD 공개 코드](https://github.com/cao-cong/ZVRD)는 AAAI 2025 선행작이다. 이를 신작의 공식 구현으로 대체 표시하지 않는다. [신작 원문 §4–5·보충 한계](https://arxiv.org/html/2608.26476v1)

**VidHalLoc (9월 9일):** 2,000개 VideoQA·caption 환각 사례에서 검출기를 평가한다. 전용 검출기들의 최고 Overall 34.63%라는 결과는 **그 벤치마크의 다종 오류 판정**에 관한 것으로 영상 픽셀 복원 검출 정확도가 아니다. LGVSC에서는 VLM 판정기를 원본 정답으로 그대로 쓰지 않는 근거로 활용한다. [원문](https://arxiv.org/html/2609.09895v1)

**SAVER:** 양식 변화 이미지에서 LVLM의 얕은 층 시각 attention을 활용해 언어 출력 환각을 줄인다. 송신단 caption 품질 개선 후보일 수 있으나 Open-Sora 영상의 객체 삭제·시간 drift를 직접 해결하는 복원 모델은 아니다. 현재 InternVL2·PLLaVA에 적용하려면 해당 모델의 attention·출력 생성 경로를 다시 검증해야 한다. [원문](https://arxiv.org/html/2508.03177v2) · [저자 저장소](https://github.com/llizhaoxu/SAVER)

## 5. 현재 LGVSC에서 확인한 적용 제약

| 항목 | 확인한 현재 상태 | 실험 설계에 미치는 영향 |
|---|---|---|
| 기준선 | `etri_official.json`: 240 frames, 24 FPS, 576×320, AWGN 10 dB, 30 steps, `paper_reproduction=false` | 로컬 공개 이식 baseline과 원 논문 재현을 구별 |
| 길이 | `lgvsc_variable.json`: `max_frames=384` | 24 FPS에서 최대 16초 설정이므로 60초 결과로 보고 불가 |
| 구간 caption | PLLaVA가 구간당 네 프레임 표집 | ‘중간 caption이 전혀 없다’는 설명은 틀림. 사건 포착 충분성을 실험 |
| 움직임 | UniMatch 결과의 절대값 평균 등을 구간 스칼라로 집계 | 객체별 방향·속도·등장/퇴장 시각은 명시적으로 보존하지 않음 |
| 수신 경계 | header에 구간·키프레임·전송 비용 정보, 수신 모듈은 수신 자료로 복구 | scene ID·유효기간·갱신 사유를 새로운 packet 계약으로 추가 |
| decoder | STDiT3·OpenSoraVAE1.2·T5, rectified-flow scheduler | DPS/RPU의 noise parameterization·gradient 경로를 그대로 적용 불가 |
| 시간 정렬 | 공개 연결과 endpoint-exact 연결·conditioning 정책이 다름 | 정렬 변경과 완화 변경을 별도 arm으로 고정 |
| GPU | 조사 시 RTX 4080, 16,376 MiB | 기존 LGVSC의 순차 추론 우선, 대형 대체 모델 성능·메모리는 미측정 |

근거: [설정](../../../../configs/etri_official.json), [가변 길이](../../../../configs/lgvsc_variable.json), [caption/flow 코드](../../../../src/semantic_transmission/workers.py), [송수신](../../../../src/semantic_transmission/codec_transport.py), [decoder 설정](../../../../configs/rtx4080_opensora.py), [기존 논문·구현 감사](../../../LGVSC_PAPER_IMPLEMENTATION_AUDIT.md). 현재 사용자 작업 트리에 미커밋 변경이 있어 로컬 근거는 조사 당시 파일 내용 기준이며, 일부 과거 문서의 줄 번호는 달라질 수 있다.

### 학습 데이터 질문에 대한 현재 답변

‘전체 시스템이 이미지 데이터만으로 학습됐다’고 답할 근거는 없다. **모듈별 사전학습과 이번 프로젝트의 추가 학습**을 나눠 설명해야 한다.

| 모듈 | 지금 설명할 수 있는 내용 | 미확인·보완할 항목 |
|---|---|---|
| NTSCC | 이미지 코덱 가중치 사용 | 로컬 공개 quality-4와 논문 학습 가중치의 동일성·정확한 데이터 이력 |
| InternVL2-8B | 키프레임 의미 선택용 사전학습 모델 | 해당 revision의 학습 자료와 장시간 사건 판별 신뢰도 |
| PLLaVA-7B | 영상 caption 모델이며 로컬에서 구간 caption 실행 | 특정 체크포인트의 학습 목록·4프레임 표집으로 놓치는 사건 |
| UniMatch | 사전학습 광류 추론 | 실제 사용 가중치의 학습 자료, 집계 전후 정보 손실 |
| OpenSora-STDiT-v3·VAE | 영상 생성 구조·모델 카드가 존재 | 논문 가중치와 로컬 revision의 대응, 목표 장면에서의 충분성 |
| 로컬 개선 | 기존 정렬·키프레임 추가는 무학습 변경으로 기록 | 새로운 영상 미세조정은 필요성 판단과 별도 실행 증거 필요 |

모델 ID·revision은 [models.json](../../../../configs/models.json), 설명 근거는 [로컬 모델 구조](../../../MODEL_ARCHITECTURE.md), [PLLaVA 저자 저장소](https://github.com/magic-research/PLLaVA), [OpenSora-STDiT-v3 모델 카드](https://huggingface.co/hpcai-tech/OpenSora-STDiT-v3)다. 영상 모델을 사용한다는 사실만으로 60초 drift나 모든 사건 보존이 해결됐다고 말할 수는 없다.

## 6. 새 지표·논문 기여에 대한 판단

기존 [지표 개발 지침](../../../METRIC_DEVELOPMENT_STRATEGY.md)의 FSO·UEP·EOI는 최종 채택된 정답 평가기가 아니다. v6 기록에는 자연 영상 UEP 오경보 **32/40**, 실제 복원 **12/12에서 1.0 포화**, EOI의 사건 대응 부족이 남는다. 이 지표 하나를 제어·학습·평가에 동시에 사용하면 점수만 개선하는 폐회로가 생길 수 있다.

추천하는 기여 후보는 다음 두 가지다. 모두 **연구 가설**이며 신규성 확정이 아니다.

- **갱신 이후 잘못된 의미 상태가 얼마나 오래 지속되는지 측정:** 이벤트의 올바른 유효기간과 실제 복원 상태를 대응시켜 잔존·조기 등장·멈춤 후 이동을 초 단위로 평가한다. 기존 FSO/ghost 계열과 수식이 같으면 새 이름으로 신규성을 주장하지 않는다. 잘못된 수신 조건 때문인지, 갱신 후에도 생성기가 무시했는지 분해하는 추가 정보가 있는지 검증한다.
- **관측 가능성을 명시한 RX 위험 판정:** 단순 ‘RX에 없음’ 대신 관측 가능/가려짐/미수신/판정 불가를 구별하고, 보류를 포함한 위험–coverage 곡선을 평가한다. 알려진 불확실성·선택적 예측 개념과 구별되는 기여는 희소 패킷·시간 유효성·통신 비용의 결합에서 입증해야 한다.

HalluGen·sFRC의 지역 차이 점수, SparkVSR의 참조 가중, 기존 시간 지표를 합친 것만으로 신규 지표라고 할 수 없다. 본 조사는 유력한 선행 방법을 찾은 것이며, 전 분야의 신규성 부재를 증명하거나 신규성을 인증한 조사는 아니다.

## 7. 개발 우선순위와 보류 후보

| 순서 | 제안 | 먼저 하는 이유 | 진행 판단 |
|---|---|---|---|
| P0 | 60초 시간축 지원 점검, 독립 오류 주석·평가 계약 | 짧은 영상·점수 포화로 효과를 오판하지 않도록 함 | 실제 길이·수신 입력·원본 분할 검증 |
| P1 | scene ID·TTL·경계 초기화·최대 갱신 간격 | ETRI R4와 직접 연결, 기존 경로에 국소 변경 가능 | 오류 지속 감소와 추가 전송량 동시 확인 |
| P1 | 동작·방향·등장/퇴장 시각을 구간 패킷에 보강 | LGVSC에서 전송 중 사라진 정보인지 분리 | 같은 비용의 키프레임 추가보다 유효한지 비교 |
| P2 | 수신 anchor·조건 유효성으로 보수적 재생성·보류 | 무조건 반복 복원의 부작용과 비용 제한 | 독립 의미 오류 감소, 정상 객체 삭제·정지 증가 확인 |
| P3 | SparkVSR/STCDiT 구조 차용 또는 STAR식 영상 미세조정 | 입력 조건·가중치·메모리 변경이 큼 | P1/P2에서 남은 실패가 조건 반영/시간 표현 문제일 때 |
| 연구 보류 | RPU 직접 이식, 의료 diffusion 가중치 직접 적용, 미확인 신작 코드 재현 | sampler·관측 모델·도메인 차이 또는 구현 근거 부족 | 작은 원인 분리 실험 이후 재판단 |

첫 결과물은 **동일 AWGN·입력·시간축의 원본/기준선/완화 영상**, 오류 구간·유형·가능한 영역, 실패와 새로운 왜곡 사례, 전송량·실행 비용이다. 제안한 모든 수치·정책의 실제 효과는 [별도 실험안](ETRI_LGVSC_HALLUCINATION_EXPERIMENT_PLAN_2026-09-21.md)에 따라 확인해야 한다.
