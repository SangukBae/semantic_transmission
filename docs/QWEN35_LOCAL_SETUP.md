# Qwen3.5-9B 로컬 캡션 실행

Ubuntu 22.04 / RTX 4080 16GB용 별도 환경이다. 기존 LGVSC·PLLaVA·InternVL·Qwen3-VL 환경과 분리되어 있다. 기본 실행은 **공식 Qwen3.5-9B 가중치의 NF4 4비트 추론**이다. 모델 전체를 BF16으로 GPU에 올리는 구성이 아니다.

## 설치 및 검증 기록

- 환경: `.local/qwen35_env`
- 가중치 캐시: `.local/qwen35_hf`
- 모델: `Qwen/Qwen3.5-9B`
- 고정 revision: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- 직접 의존성: [requirements-qwen35.txt](../environment/requirements-qwen35.txt)
- 전체 설치 버전: [requirements-qwen35.lock.txt](../environment/requirements-qwen35.lock.txt). 설치기는 이 고정 목록을 사용하며, 설치 후 freeze도 `.local/qwen35_environment.lock.txt`에 저장한다.
- 가중치 파일 SHA-256·tensor index 검증: `outputs/qwen35_setup/model_manifest.json`
- 패키지·설정·4장 이미지 전처리·CUDA 상태: `outputs/qwen35_setup/doctor.json`

`doctor` 성공은 전체 모델 추론 성공과 구분한다. 실제 캡션 JSON이 생성돼야 해당 장치에서의 전체 추론이 확인된다. 실행 중인 SKEM이 GPU 메모리를 사용하는 경우 GPU 검증은 보류되며, 프로세스를 중단하지 않는다.

2026-10-02 검증 결과:

- 공식 가중치 4개 shard, 총 19,306,310,880 bytes 다운로드 완료. 775개 tensor의 파일 매핑과 공식 Hub LFS SHA-256 일치 확인.
- CUDA 13.0 / RTX 4080 인식 및 4장 이미지 전처리 통과.
- 원본 `tv_low_08`의 프레임 `[1, 4, 7, 9]`를 사용한 **CPU NF4 전체 모델 추론 통과**. 결과: [cpu_caption_smoke.json](../outputs/qwen35_setup/cpu_caption_smoke.json).
- CPU 2 threads, 이미지당 픽셀 상한 65,536에서 로딩 약 115초, 생성 약 344초, 출력 27 tokens, `truncated=false`. 이는 CPU 검증 시간이며 GPU 속도 추정값이 아니다.
- 이후 사용자의 명시적 요청으로 SKEM 작업을 재개 가능한 상태로 중단하고 **RTX 4080 GPU에서 78개 캡션 생성 완료**. 출력 잘림·재시도 0개. CPU 표본은 별도의 환경 동작 확인 기록이다.
- GPU 입력·시간·캡션 오류 비교는 [실험 결과](ETRI_COMBINED_REFERENCE_DIAGNOSIS.md#qwen-caption), 환경 검증은 [GPU_VALIDATION.json](../outputs/qwen35_setup/GPU_VALIDATION.json)에서 확인한다. 독립 정답·복원 품질 검증은 미완료다.

중단한 SKEM의 체크포인트와 재개 명령은 `outputs/fc_lgvsc_webvid_tvsum_faithful_20261002/paused_for_qwen_caption_comparison.json`에 있다. 비교 종료 후 자동 재개하지 않았다.

처음 설치하거나 같은 revision을 다시 준비할 때:

```bash
cd /home/sangukbae/semantic_transmission
bash scripts/bootstrap_qwen35.sh
```

설치와 다운로드에는 인터넷을 사용한다. 이후 아래 실행기는 오프라인 모드이며 외부 추론 API를 호출하지 않는다.

## 영상 구간의 캡션 생성

현재 두 shell 실행기의 기본 지시문은 FC-LGVSC `faithful_v2`의 관찰 기준에 맞춘 [faithful_visible_under80.txt](../configs/captions/faithful_visible_under80.txt)다. 기존 지시문의 핵심인 객체·위치·크기·방향·가림, 카메라/객체 이동 구분을 유지하고, 사용자 요청에 따라 **80단어 미만(최대 79단어)** 및 추측·대안 나열 금지를 명시했다. 4장 이외 입력에도 사용할 수 있도록 프레임 수 표현은 일반화했다. 과거 캡션 작성 과정 전체나 결과 품질이 동일해졌다는 뜻은 아니다.

실행 후 공백 기준 단어 수를 확인한다. 80단어 이상·빈 출력·잘린 출력은 `CAPTION_POLICY_CHECK_FAILED`로 기록하고 종료 코드 2로 반환하며, 문장을 임의로 잘라 성공 처리하지 않는다. 검사는 길이·완결성에 한정되며 시각적 사실 여부를 자동 인증하지 않는다. `--prompt`를 명시하면 사용자 지시문이 우선하지만 79단어 검사는 유지된다.

```bash
bash scripts/run_qwen35_caption.sh --show-prompt
bash scripts/run_qwen35_advanced.sh detail --show-prompt
```

이 지시문과 형식 검사는 새로운 일반 실행에 적용한다. 기존 78개 캡션 비교와 48개 상향 설정 시험은 당시 지시문·실행 파일을 보존한다. 새 지시문만의 정확도 효과는 분리하지 않았다. 문맥 전략의 대응 비교는 별도 [실험 결과](ETRI_COMBINED_REFERENCE_DIAGNOSIS.md#qwen-context)를 따른다.

```bash
bash scripts/run_qwen35_caption.sh caption \
  --video /absolute/path/input.mp4 --start 0 --end 2 \
  --output outputs/qwen35_example/caption_0_2.json
```

기본적으로 지정 구간에서 PLLaVA의 기존 샘플링식과 같은 4개 프레임을 선택한다. `--frames 8`로 늘릴 수 있다. 4장이 아닌 경우에는 구간을 균등 분할한 중앙 지점을 사용한다. 원본 전체 영상 대신 짧은 구간부터 검증한다. OpenCV로 디코딩하며 가변 FPS 영상의 정확한 시각 비교에는 고정 FPS로 정규화한 원본을 사용한다.

이미 추출한 원본 이미지 4장을 시간 순서대로 전달할 수도 있다.

```bash
bash scripts/run_qwen35_caption.sh caption \
  --images /absolute/path/0.png /absolute/path/1.png \
           /absolute/path/2.png /absolute/path/3.png \
  --output outputs/qwen35_example/caption_frames.json
```

결과 JSON에는 캡션, 원본 파일 해시, 사용 프레임, 프롬프트, 모델 revision, 양자화, 패키지 버전, 로딩·생성 시간과 GPU 최대 할당량이 남는다. 이미 존재하는 결과 파일은 덮어쓰지 않는다. `truncated: true`라면 출력 토큰 제한에 도달했으므로 캡션 완결성을 확인한다.

## GPU 여유를 활용한 상향 설정

개발 기본 후보는 **NF4·최대 16장 `detail`**, INT8·최대 16장 `max`는 비교 실험용이다. 6설정·48개 진단 캡션의 시간·메모리·오류와 한계는 [상향 설정 결과](ETRI_COMBINED_REFERENCE_DIAGNOSIS.md#qwen-capacity)에 기록한다. 576×320 원본의 표본 검사이며 다른 해상도·긴 입력의 최대 용량을 보장하지 않는다.

```bash
# 권장: 4비트 정밀도를 유지하고 시간 순서의 입력을 최대 16장으로 확대
bash scripts/run_qwen35_advanced.sh detail \
  --video /absolute/path/input.mp4 --start 0 --end 2 \
  --output outputs/qwen35_example/detail_0_2.json

# 메모리를 더 사용하는 선택: 8비트 + 최대 16장
bash scripts/run_qwen35_advanced.sh max \
  --video /absolute/path/input.mp4 --start 0 --end 2 \
  --output outputs/qwen35_example/max_0_2.json
```

상향 실행기는 기존 기본 실행기·기존 캡션 파일을 교체하지 않는다. 일반 영상 실행은 구간을 균등 분할한 중앙 프레임을 사용한다. 위 비교 실험은 기존 네 장을 유지한 채 추가 프레임을 넣었으므로 같은 구간이라도 샘플링 위치와 최종 캡션이 완전히 같지는 않다.

- 8비트는 공식 `bitsandbytes`의 `load_in_8bit=True`, outlier threshold 6.0을 사용한다. 시각 인코더와 출력 head는 양자화에서 제외한다. [공식 설명](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes#llmint8).
- 입력 상한 8192 tokens, 최대 16장, 기본 출력 상한 256 tokens, thinking 비활성화·greedy decoding. 프레임을 더 넣거나 출력 상한을 늘린다고 정확성이 보장되지는 않는다.
- 로딩 전에 12000 MiB 이상 여유를 확인하고, 현재 여유에서 1024 MiB를 뺀 값으로 PyTorch 할당 예산을 제한한다. 다른 GPU 작업을 중단하지 않는다. 실행 중 다른 프로그램이 메모리를 사용하면 추가 여유가 줄어들 수 있다.
- 래퍼의 기본 이미지당 픽셀 상한은 262144다. 상향 Python 실행기는 `--max-pixels`로 최대 1048576까지 허용하지만 별도 조합 시험이 필요하다. 이번 576×320 원본에서는 두 상한 모두 같은 20×36 patch grid가 생성되어, 상한만 올려도 시각 정보나 입력 토큰은 늘지 않았다.
- 전체 BF16 가중치를 GPU에 올리는 구성과 다르다. 기존 78개/65구간 평가와 그 지시문·파일은 보존했다.

## 16GB 기본 설정

- NF4 + double quantization, BF16 계산, SDPA attention.
- Gated DeltaNet은 기본 PyTorch 구현을 사용한다. 선택적 `causal_conv1d` / `flash-linear-attention` 최적화 커널은 미설치이며, GPU 처리속도 최적화는 별도 검증 대상이다.
- 시각 인코더와 출력 head는 양자화 대상에서 제외.
- 호출당 최대 8장, 이미지당 최대 256 visual tokens에 해당하는 픽셀 상한, 입력 4096 tokens 상한.
- 기본 최대 출력 160 tokens, thinking 비활성화, greedy decoding.
- CPU 2 threads. GPU 여유 메모리 최소 12000 MiB를 확인한 뒤 로딩.
- GPU 메모리가 부족하거나 GPU 조회가 차단되면 **종료 코드 75**로 종료. 다른 프로세스를 종료하지 않는다. 실행 직후 다른 작업이 GPU를 점유하는 경쟁 상황까지 예약·차단하지는 않는다.
- 선택 사항인 `--device cpu`는 시스템 가용 RAM 11 GiB 이상일 때만 실행하며 GPU보다 느리다.

```bash
bash scripts/run_qwen35_caption.sh doctor
```

전용 환경의 Python을 직접 사용할 수도 있다: `.local/qwen35_env/bin/python scripts/qwen35_faithful_caption.py --engine basic caption ...` 또는 `--engine advanced --video ...`. `qwen35_caption.py`와 `qwen35_advanced_caption.py`를 직접 호출하면 과거 재현용 지시문/동작이 적용되므로, 현재 정책에는 shell 실행기 또는 `qwen35_faithful_caption.py`를 사용한다.

기존 FC-LGVSC 자동 배치에는 연결하지 않았다. 이 실행기의 출력은 독립 캡션 JSON이며 기존 caption bundle과 다른 형식이다. 위 결과는 기존 캡션과의 단일 영상 비교다. 모델 자체의 우위와 4비트 양자화의 품질 영향은 동일 프롬프트의 PLLaVA 재추론, BF16 대조 및 독립 정답 평가가 필요하다.

이번 고정 입력 비교의 재개와 수동 판정 집계 명령:

```bash
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  .local/qwen35_env/bin/python scripts/run_qwen_caption_comparison.py \
  --root outputs/caption_compare_qwen_tv_low_08_20261002 \
  --confirmed-model Qwen/Qwen3.5-9B
python3 scripts/report_qwen_caption_comparison.py \
  --root outputs/caption_compare_qwen_tv_low_08_20261002
```

완료된 78개 캡션은 재생성하지 않는다. 보고서 스크립트는 원본·캡션 해시와 정렬을 검증하고 이미 작성한 수동 주석을 집계하며, 정확도를 자동 판정하는 모델이 아니다.

## 직전 원본 문맥 비교 실행

현재 구간만/직전 최대 2초 문맥 적용의 고정 78구간 실행이다. 두 조건의 이미지 수·토큰 예산을 맞추고 기존 원본 4개 샘플을 유지한다. [처리 구조](MODEL_ARCHITECTURE.md#caption-providers) · [생성·후속 오류 평가 결과](ETRI_COMBINED_REFERENCE_DIAGNOSIS.md#qwen-context)

```bash
# 완료된 항목은 해시/형식을 검증하고 건너뛴다.
.local/qwen35_env/bin/python scripts/run_qwen35_context.py run \
  --root outputs/caption_context_qwen_tv_low_08_20261003
python3 scripts/report_qwen35_context.py \
  --root outputs/caption_context_qwen_tv_low_08_20261003
```

새 실험은 별도 출력 경로에서 `prepare --root ... --inputs ...`로 계획을 먼저 고정한다. 입력 파일은 기존 비교 입력 스키마가 필요하다. 일반 영상 CLI에 문맥 기능을 자동 적용하는 변경은 아니다.

<a id="selector-timing"></a>
## 선택 판정 30쌍 시간 측정

원본 PNG와 준비된 `single_subject` 기준선이 필요하다. 같은 출력 폴더를 재사용해 과거 측정값을 덮어쓰지 않고 새 경로를 지정한다.

```bash
.local/qwen35_env/bin/python scripts/benchmark_qwen35_keyframe_timing.py \
  --root outputs/qwen35_keyframe_timing_new
```

GPU 공유 잠금 후 30쌍을 실제 추론하며 `RESULT.json`에 쌍별 시간과 전체 시간 추정을 구분해 저장한다. 전체 영상 선택·품질 평가는 실행하지 않는다. [10월 3일 측정](ETRI_COMBINED_REFERENCE_DIAGNOSIS.md#qwen-selector-timing)

공식 참고: [Qwen 모델](https://huggingface.co/Qwen/Qwen3.5-9B) · [Transformers 구현](https://huggingface.co/docs/transformers/en/model_doc/qwen3_5) · [bitsandbytes 양자화](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes).
