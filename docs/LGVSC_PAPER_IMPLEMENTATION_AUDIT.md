# LGVSC 논문·현재 구현 대조 — 2026-09-18

현재 구현은 **공개 코드에 기반한 SKEM–DSA 파이프라인을 이 PC에서 실행하도록 이식하고 확장한 구현**이다. 핵심 알고리즘은 논문 설명과 대응하지만, 논문 전체의 학습·전송·평가 조건까지 동일하게 재현했다는 판정은 불가능하다. 최근 ETRI의 `정렬+12장`은 원 논문의 기본 설정과 구분해야 하는 로컬 개선 실험이다.

검토 기준은 제공된 [논문 LaTeX](LGVSC_paper/main.tex), 현재 작업 트리, 저장소에 보관된 공개 코드 기준 커밋 `50c9ff98fb1aaa8136aef2e41e02bbb0d1a2d1c3`이다. 현재 HEAD는 `710151f3d749cb64d03d73304ff7ae16e4c77fe3`이며 커밋되지 않은 변경도 검토 대상에 포함했다. 공개 코드의 동작을 유지하는 것과 논문의 실험 조건을 충족하는 것은 별개로 판정했다.

| 검토 항목 | 논문과 현재 코드의 대응 | 판정 |
|---|---|---|
| PSSS 의미 점수 | 논문 식의 `P(No) − P(Yes)`를 실제 InternVL 출력의 첫 토큰 확률로 계산한다. 로그에 비율도 출력하지만 선택 조건은 차이값이다. | 핵심 수식 일치 |
| SKEM 선택 | 최신 선택 키프레임과 후보 프레임을 비교하고 임계값 초과 시 갱신한다. 공식 프로파일은 threshold=0.35, stride=1이다. | 핵심 선택 절차 일치 |
| 의미 초점·목표 CBR 조절 | 논문은 의미 초점과 임계값 변경 및 검증 세트 기반 목표 CBR 보정을 설명한다. 현재 실행 경로는 공개 기본 프롬프트를 사용하며 목표 CBR을 입력받아 임계값을 보정하는 일반 기능은 없다. | 논문에서 설명한 조절 범위 일부 미구현 |
| 구간 캡션·부가 정보 | 실제 PLLaVA와 UniMatch를 사용한다. 공개 경로처럼 광류를 평균 절댓값의 스칼라로 요약한다. 포즈·객체 궤적을 직접 전달하는 시스템은 아니다. | 공개 구현의 방식과 대응 |
| NTSCC 키프레임 전송 | 실제 학습된 NTSCC 인코더·디코더와 AWGN을 사용한다. 현재 로더는 quality-4 체크포인트를 고정해 읽는다. | 구성 일치, 논문 학습 가중치 동일성 미확인 |
| DSA | 구간마다 프레임 수에서 VAE latent 크기를 산출하고 그 크기로 Open-Sora 모델을 구성하며 사전학습 파라미터를 재사용한다. | 논문의 DSA 설명과 대응 |
| 구간 연결·출력 길이 | `official_release`는 양쪽 끝점을 모두 포함하고 다음 구간의 오버랩 17장만 제거한다. 내부 경계가 중복돼 출력은 원본 F장보다 많다. | 공개 동작 유지, 논문의 F장 출력 정의와 차이 |
| 디지털 전송·CBR | 로컬 패킷의 메타데이터·rate 정보 등을 LDPC/16QAM으로 실제 처리하고 시각 심볼과 합산한다. 공개 NTSCC 스크립트에는 side information의 이상적 용량 근사 산식이 있다. | 논문 전략과 양립하지만 산정 범위·패킷 형식 동일하지 않음 |
| 기본 화질 지표 | PSNR·SSIM·LPIPS·CLIP·DISTS 계산을 공개 코드와 대조했다. CLIP은 `(cosine + 1)/2`이다. | 검증한 이미지 쌍에서 계산 일치 |
| 논문 전체 실험 | WebVid 55편·Kinetics-400 14편, 여러 SNR, downstream 3종 및 비교 방법 전체를 현재 실행 경로에서 재검증한 상태는 아니다. | 전체 실험 재현 미완료 |
| 메모리·실행 환경 | CPU offload, 텍스트 인코더 배치, 모듈별 환경 분리 등 16GB GPU용 변경이 있다. 공식 프로파일의 InternVL은 8bit가 아니다. | 실행 이식; 전체 결과의 비트 단위 동일성 미보장 |

근거 위치: 논문 `main.tex:232–245, 264–280, 288–302, 390–398`; 선택 `02_semantic_encoder/skem/MLM-keyframe-internvl.py:265–289`; 실행 연결 `src/semantic_transmission/workers.py:84–122, 153–273`; DSA `04_semantic_decoder/scripts/mydemo_new_align_sh.py:515–533`; 지표 `src/semantic_transmission/official_quality.py:19–91`.

특히 주의할 차이는 다음과 같다.

1. **NTSCC 가중치의 학습 이력을 동일하다고 확인할 수 없다.** 논문 `main.tex:290`은 10만 프레임, 학습률 1e-4, batch 64, 100 epochs를 명시한다. 현재 `codec_transport.py:20–38`은 공개 `ntscc_hyperprior_quality_4_psnr.pth`를 읽으며, 동봉 NTSCC README의 공개 사전학습 설명은 OpenImages 50만 이미지다. 공개 LGVSC 스크립트의 10dB 경로도 같은 이름의 quality-4 파일을 사용하므로 이것을 우리 이식 과정에서 잘못 고른 가중치라고 단정하지 않는다. 논문 실험 가중치와의 동일성을 입증할 저자 체크포인트·해시·학습 기록이 필요하다. 공개 스크립트의 0–8dB 경로는 별도 학습 가중치를 가리키지만 현재 로더는 SNR별로 가중치를 교체하지 않는다.

2. **공개 연결 정책의 프레임 수는 논문 정의와 다르다.** 논문 `main.tex:213,220`은 F장 입력을 F장으로 복원한다. 현재 `temporal.py:4–26`의 공개 정책은 키프레임 수가 K일 때 `F + K − 2`장이 된다. 저장된 키프레임 목록으로 확인하면 ETRI 기본 240장·3키는 241장, WebVid 기본 337장·14키는 349장이다. `concatenation_policy=endpoint_exact`는 중복 제거 후 F장을 만든다. 이 계산은 생성 품질 개선 여부를 뜻하지 않는다.

3. **출력 연결과 생성 조건 정렬은 다른 변경이다.** 공개 Open-Sora의 `apply_mask_strategy`는 align 값이 있으면 잠재 공간의 조건 시작 위치를 정렬한다. 로컬 `conditioning_alignment=endpoint_exact`는 그 정렬을 끄고 다음 구간에 전달할 이전 출력도 오버랩 길이로 제한한다. 논문에는 이 세부 마스크 규약이 없고 VAE 블록 경계와도 관련되므로, 이를 논문에서 증명된 오류의 수정이라고 부르기는 어렵다. `concatenation_policy`만 바꿔서는 생성 조건이 바뀌지 않는다. 근거: `.local/vendor/Open-Sora/opensora/utils/inference_utils.py:175–199`, `mydemo_new_align_sh.py:387–392,545–559,583–585`.

4. **CBR 분모가 같아도 분자에 포함하는 비용이 다르다.** 현재 `codec_transport.py:157–169`는 시각 신호와 실제 디지털 패킷 채널 사용량을 합한다. `03_jscc_transmission/ntscc/main_save.py:143–155`는 rate side information에 이상적인 채널 용량을 가정한다. 현재 비용은 물리 링크 헤더 등 모든 무선 오버헤드를 포함하지도 않는다. 논문 CBR이나 성능 곡선에 현재 값을 그대로 겹쳐 동등 조건이라고 주장해서는 안 된다.

5. **프로파일을 지정해야 비교 기준이 명확하다.** `configs/etri_official.json`은 공개 정책을 명시하고 `paper_reproduction: false`로 구분한다. 반면 `research.py:34`의 프로파일 미지정 기본값은 `etri_hq.json`이다. `정렬+12장`은 원래 SKEM 결과에 최대 간격 1초 키프레임을 추가하고 생성 조건·연결을 바꾼 실험이다. 원본 키프레임을 직접 넣는 `clean_keys`도 통신 성능 보고용 기본 조건이 아닌 원인 분리 실험이다. 원 논문 baseline, 로컬 이식 baseline, 로컬 개선 조건을 결과 표에서 구분해야 한다.

이번 감사에서 기존 테스트 네 파일을 실행해 **25 passed, 2 warnings, 11.17초**를 확인했다. GPU 지표 검증은 실제 출력에서 가져온 이미지와 변형 이미지로 구성한 두 쌍을 공개 함수와 비교하며 허용 오차는 1e-6이다. 프로파일, 출력 길이, 조건 마스크 관련 검증도 포함한다. 이것은 전체 영상의 동일성, NTSCC 학습 이력, 전체 데이터셋 성능의 재현을 입증하지 않는다. 새 영상 복원·재학습은 실행하지 않았고 모델 코드는 수정하지 않았다.

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 LGVSC_GPU_TESTS=1 \
  /home/sangukbae/anaconda3/envs/lgvsc/bin/python -m pytest \
  tests/test_lgvsc_alignment.py tests/test_official_profile.py \
  tests/test_research_runtime.py tests/test_decoder_conditioning.py -q
```

[테스트 로그](../outputs/diagnostics/lgvsc_paper_audit_20260918/tests.log) · [논문 해시·프레임 수 검증 기록](../outputs/diagnostics/lgvsc_paper_audit_20260918/evidence.json)

논문과 동일한 기준점을 더 강하게 주장하려면 저자 가중치 및 SNR별 가중치의 이력을 확인하고, 데이터 목록·전처리·프레임 매칭·CBR 비용 항목을 고정한 뒤 논문 실험을 재실행해야 한다. 현재의 낮은 복원 품질을 전부 논문 모델 자체의 한계로 귀속할 근거는 아직 부족하다.
