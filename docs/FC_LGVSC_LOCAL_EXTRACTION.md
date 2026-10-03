# FC-LGVSC 로컬 자동 캡션·키프레임 추출

이 구현은 대화형 assistant가 관찰해서 작성하던 **후보 프레임과 캡션 생성**을 로컬 공개 가중치 VLM으로 대체한다. 기존 FC-LGVSC의 순차 SKEM 판정과 최대 24프레임 간격은 유지한다. 원본 복원, 채널 전송, ETRI Added/Missing/Distorted 평가까지 완료한 결과는 아니다. 대화형 assistant와 동등한 품질도 아직 입증하지 않았다.

<a id="status"></a>
## 현재 상태 — 2026-10-03

- 기존 103편 배치는 Qwen3-VL-8B INT8을 사용했으며 **7편 완료·1편 실패 후 중단** 상태다. 이후 전체 실행 완료 근거는 없다. [배치 상태](../outputs/fc_lgvsc_auto_20261002/status.json) · [중단 기록](../outputs/fc_lgvsc_auto_20261002/paused_for_faithful_correction.json)
- assistant 작성 전체 원본과 순차 SKEM 보정본은 별도 경로이며 현재 미완료다. [데이터별 상태](DATA.md#full-extraction)
- Qwen3.5-9B의 캡션·문맥 비교는 [별도 실행기](QWEN35_LOCAL_SETUP.md)로 진행했다. 이 Qwen3-VL 배치를 교체하거나 재개한 결과가 아니다.

## 처리 범위와 구조

- 입력: `outputs/fc_lgvsc_webvid_tvsum_full_20261001/inventory.json`의 WebVid 원본 53편 + TVSum 원본 50편. 기존 전체 길이 24 FPS, 576×320 정규화 영상과 PNG를 읽기 전용으로 재사용한다. 짧게 잘린 WebVid 사본이나 TVSum 60초 벤치마크만 처리하는 방식이 아니다.
- 후보 탐지: 정규화 영상 전체에 TransNetV2를 실행한다. 매 프레임의 급격한 픽셀 변화도 별도 검출하여 화면 일부만 바뀌는 전환의 누락을 보완한다. Qwen3-VL은 2초씩 최대 13장, 6 FPS의 개별 이미지를 관찰한다. 인접 창은 경계 프레임을 공유한다. 마지막 프레임도 포함한다.
- 키프레임 선택: 처음/끝, 탐지된 장면전환, 객체 출입·가림의 관찰 전후 프레임을 보호한다. 나머지 후보는 기존 InternVL2-8B SKEM의 `P(No)-P(Yes)>0.35`를 적용한다. 비교 기준은 직전 **선택된** 프레임이며, 24프레임 최대 간격을 보장한다. 후보 생성기가 달라졌으므로 기존 수동 FC-LGVSC와 완전히 같은 선택기가 아니다.
- 캡션: 선택된 각 `[시작, 다음 키프레임)` 구간에서 기존 PLLaVA 방식의 4개 정확한 인덱스를 계산한다. Qwen3-VL이 원본 PNG를 보고 영어 캡션을 작성하고, 별도 호출에서 같은 이미지와 다시 대조한다. 마지막 키프레임은 종단 경계로 별도 저장된다.
- 검증: 실제 키프레임 이미지 해시, 순차 SKEM 선택 재현, 전체 구간 캡션 개수/샘플 인덱스/추론 기록을 검사한다. 같은 모델의 재검토는 독립적인 사실 검증이나 할루시네이션 제거 보장이 아니다.

모델은 [Qwen3-VL 공개 가중치](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)를 사용한다. 4B BF16과 8B INT8의 실제 로컬 표본 결과는 `outputs/fc_auto_validation_20261002/`에 저장하며, **배치에 실제 사용된 모델 ID·revision·양자화 설정은 해당 배치의 `run_config.json`이 기준**이다. INT8은 BF16과 수치적으로 같은 실행이 아니다. 기존 InternVL SKEM은 BF16 및 기존 CPU offload 설정을 유지한다.

TransNetV2는 기존 로컬 자산을 재사용한다. 해당 체크포인트는 제3자가 변환한 PyTorch 가중치이고 공식 TensorFlow 가중치와의 동등성은 독립적으로 확인하지 않았다. 출처와 해시는 `data/etri_benchmark_v1_20260924/metadata/transnetv2/provenance.json`에 있다. 장면전환 후보를 만드는 용도이며 정답 라벨이 아니다.

픽셀 보완 검사는 48×27 RGB에서 인접 프레임의 정규화 평균 절대 차이가 0.05 이상이고, 직전 24프레임 중앙값(하한 0.01)의 4배 이상일 때 양쪽 프레임을 보호한다. 플래시·빠른 움직임도 추가 키프레임이 될 수 있으며, 전송 비용 증가를 확인해야 한다. 이 수치는 개발 표본으로 정한 휴리스틱이고 전체 데이터셋에서 최적화된 임계값이 아니다.

## 이번 개발 검증

- 4B BF16, 8B INT8, 8B NF4를 실제 RTX 4080에서 실행했다. 배치에는 **8B INT8**을 선택했다. 더 큰 모델을 로컬에서 실행할 수 있고 NF4보다 정밀도를 유지하는 선택이며, 표본에서 우수성이 통계적으로 입증됐다는 뜻은 아니다.
- WebVid 애니메이션·등갓 2편의 전체 추출이 완료되어 키프레임 17개·캡션 15개가 생성되고 구조 검증을 통과했다. `outputs/fc_auto_smoke_20261002_v1/`에 보존한다.
- 기존 InternVL SKEM 72→92 프레임 쌍을 다시 실행했으며 `P(Yes)`와 `P(No)` 모두 기존 값과 정확히 일치했다. 한 쌍의 수치 회귀 검사이지 선택 품질 검증은 아니다.
- TVSum 개발 원본의 확인 대상 전환 4개 중 신경망 장면 검출기만으로는 1개를 놓쳤다. 상품 사진이 겹쳐진 상태에서 배경만 바뀌는 전환이며, 픽셀 보완 검사를 추가했다. 원본 PNG를 다시 확인한 실제 첫 새 프레임은 1668이고, 앞선 기록의 1667은 한 프레임 이르다. 기존 수동 결과는 수정하지 않았다.
- 픽셀 보완 검사 추가 후의 확인 결과는 `outputs/fc_auto_validation_20261002/tvsum_known_cuts_with_guard.json`, 앞선 2편에 추가 후보가 생기지 않는지의 검사는 `guard_smoke_equivalence.json`에 저장한다. 전체 실행 시작 확인은 `launch_validation.json`이 기준이다.
- 표본에서도 VLM이 재질·동작 원인·공간 종류를 과하게 추측하거나 사건 시점을 잘못 지목하는 사례가 남았다. `development_assessment.json`에 한계를 기록했다. 동등 품질이나 할루시네이션 해결로 보고하지 않는다.

## 실행 및 재시작

기존 LGVSC 환경을 변경하지 않는 별도 `.local/fc_auto_env`를 사용한다. 설치/가중치 다운로드는 네트워크를 사용하지만 추출은 `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`로 실행하며 외부 추론 API를 호출하지 않는다.

```bash
# 별도 환경 준비. 이미 준비된 환경/모델에는 다시 할 필요가 없다.
bash scripts/bootstrap_fc_auto.sh

# 새 결과 폴더 초기화. 기본은 4B BF16이며 아래처럼 8B INT8도 지정 가능.
bash scripts/bootstrap_fc_auto.sh Qwen/Qwen3-VL-8B-Instruct 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
bash scripts/run_fc_auto_extraction.sh init --root outputs/fc_lgvsc_auto_20261002 \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b --quantization int8

# 선택한 배치를 의도적으로 재개할 때만 실행. 현재 중단 상태를 자동 해제하지 않는다.
bash scripts/run_fc_auto_extraction.sh start --root outputs/fc_lgvsc_auto_20261002

# 사용자가 원할 때만 확인. 자동 외부 LLM 호출이나 Codex 감시 루프는 없다.
bash scripts/run_fc_auto_extraction.sh status --root outputs/fc_lgvsc_auto_20261002
```

출력 폴더 초기화는 한 번만 한다. 캐시는 모델·프롬프트·코드·입력 프레임·선택 결과에 결합되어 있으므로 설정이나 코드를 바꾸면 기존 폴더를 그대로 재사용하지 않는다. GPU 오류 등은 해당 단계부터 한 번 다시 시도하고, 계속 실패한 영상은 `error.json`에 남긴 후 나머지 영상을 진행한다. 다음 실행에서는 성공한 영상을 검증하여 건너뛰고, 실패 영상은 저장된 구간부터 다시 시도한다. 같은 GPU를 쓰는 기존 검증 파이프라인과 파일 잠금을 공유한다.

백그라운드 실행은 WSL과 PC가 실행 중인 동안 유지된다. PC 절전·종료 이후 자동 부팅 재시작을 설치하는 기능은 포함하지 않는다.

## 결과와 보고 기준

배치 폴더의 `index.html`, `status.json`, `batch.log`가 전체 진행 상태다. 처리 중인 영상의 `progress.json`은 현재 단계와 완료 구간 수를 기록한다. `RUNNING`은 완료를 뜻하지 않는다. 전체 완료는 103개 영상의 구조 검증이 모두 통과한 `COMPLETE`일 때만 선언한다.

각 영상 폴더에는 다음이 저장된다.

| 파일 | 의미 |
|---|---|
| `shot_candidates.json`, `observations/*.json` | 장면 검출 및 VLM 후보 추론의 원문·입력 인덱스·시간·VRAM |
| `proposals.json`, `skem_scores/*.json` | 보호 후보와 순차 SKEM 실제 점수 |
| `keyframes.json`, `selected_frames/` | 선택 인덱스와 원본 정규화 PNG의 동일 바이트 사본 |
| `captions_draft/`, `captions_review/` | 캡션 초안과 같은 모델의 이미지 대조 결과 |
| `captions.json` | 시간/프레임이 결합된 최종 자동 캡션 |
| `result.json`, `index.html` | 구조 검증 상태와 검토 화면 |

기존 수동 `assistant_captions` 묶음으로 가장하거나 그 author 필드를 바꾸지 않는다. 신규 자동 캡션은 별도 스키마이며, 기존 수동 캡션 import 경로와 자동으로 호환된다고 주장하지 않는다. 후속 복원에 연결할 때는 이 스키마의 모델/샘플 provenance를 보존하는 importer가 필요하다.

`keyframe_png_bytes`와 `caption_utf8_bytes`는 산출물 크기다. NTSCC 채널 사용량이나 실제 CBR을 측정한 값이 아니다. 최대 1초 간격과 보호 프레임 증가에 따른 전송 비용도 후속 복원/전송 평가에서 별도 비교해야 한다.

ETRI에는 실제 `run_config.json`과 완료 상태에 근거하여 “로컬 Qwen3-VL 기반 시각 후보·구간 캡션 생성, TransNetV2 장면전환 후보, InternVL2-8B 순차 SKEM을 결합한 자동 전처리”로 기술할 수 있다. 기존 대화형 assistant 결과는 별도 참고 자료이고 자동 모델의 성능 정답이나 학습 데이터로 사용하지 않았다. 표본을 보면서 설정을 정했으므로 해당 표본에서의 확인은 개발 검증이며 독립적인 성능 평가가 아니다. 프롬프트만으로 객체 누락/추가/왜곡이 해결되었다고 보고해서는 안 된다.

검사 명령:

```bash
PYTHONPATH=src .local/fc_auto_env/bin/python -m pytest \
  tests/test_auto_extraction.py tests/test_hybrid_selection.py tests/test_assisted_captions.py -q
```
