# 설치·실행·학습·평가

[프로젝트](../README.md) · [데이터](DATA.md) · [구조](MODEL_ARCHITECTURE.md) · [결과](RESULTS.md)

**2026-10-19 SPARC 브랜치:** `run_bsu.sh`와 `run_sparc.sh`는 SPARC를 실행한다. 기존 생성·예측·혼합 모델은 각각 `run_bsu_gen.sh`, `run_bsu_predict.sh`, `run_bsu_hybrid.sh`로 유지한다.

<a id="bsu-default"></a>
<a id="sparc"></a>
## SPARC 실행·재개·대조 실험

Qwen 의미 키 선택 → 원본 움직임 기반 보조 프레임 → 장면별 DCVC → 메타데이터 무손실 압축 → LDPC/16QAM/AWGN → 독립 복호화 → RIFE → 최종 MP4의 순서다. 새 학습·캡션 생성·생성형 보충 경로는 없다. [설정](../configs/sparc.json) · [구조](MODEL_ARCHITECTURE.md#sparc)

```bash
bash scripts/run_sparc.sh profile
bash scripts/run_sparc.sh check --video /absolute/path/input.mp4 --output outputs/sparc_video
bash scripts/run_sparc.sh run --video /absolute/path/input.mp4 --output outputs/sparc_video
bash scripts/run_sparc.sh status --output outputs/sparc_video

# 정규화된 연속 0.png, 1.png, ... 입력; 실제 FPS를 지정
bash scripts/run_sparc.sh run --source-frames /absolute/path/frames --fps 24 --output outputs/sparc_adaptive

# 같은 실제 Qwen 응답을 검증·재사용하여 새 모델 호출 없이 대조
bash scripts/run_sparc.sh run --source-frames /absolute/path/frames --fps 24 --semantic-replay outputs/sparc_adaptive/semantic.json --anchor-policy uniform-matched --output outputs/sparc_uniform
bash scripts/run_sparc.sh run --source-frames /absolute/path/frames --fps 24 --semantic-replay outputs/sparc_adaptive/semantic.json --anchor-policy fixed-stride2 --output outputs/sparc_fixed

# 실제 최종 MP4를 원본 PNG와 비교; 새 출력 폴더 사용
PYTHONPATH=src .local/bsu_codec_eval_env/bin/python scripts/compare_sparc.py --adaptive outputs/sparc_adaptive --uniform outputs/sparc_uniform --fixed outputs/sparc_fixed --output outputs/sparc_comparison
```

`--video`는 전체 길이를 576×320·24FPS로 정규화한다. `--width`, `--height`, `--fps`로 변경할 수 있다. PNG 입력은 기존 크기를 유지하며 최소 2장, 양의 짝수 크기를 요구한다. 길이 자르기·반복 연결·오디오 전송은 없다. 출력과 입력·재생 자료는 겹칠 수 없다.

Qwen3.5-9B NF4 고정 snapshot, DCVC-RT I/P 가중치·소스, RIFE4.25 가중치·소스, Sionna 채널 환경, FFmpeg/ffprobe가 필요하다. [Qwen 준비](#public-start) · [DCVC/RIFE 출처](../THIRD_PARTY_NOTICES.md#bsu-predictive-dependencies). 새 장비용 DCVC/RIFE 자동 설치기는 제공하지 않는다. SPARC는 캡션 LoRA·Open-Sora·NTSCC·LTX를 요구하지 않는다.

기본 런타임은 `.local/qwen35_env/bin/python`, `.local/dcvc_rt_env/bin/python`, `.local/bsu_codec_eval_env/bin/python`, `.local/settings.json`의 `channel_python`이다. `--selector-python`, `--codec-python`, `--receiver-python`, `--channel-python`으로 바꿀 수 있다. 셸 실행기 Python은 `SPARC_PYTHON`으로 지정한다. 비교 환경에는 기존 공식 평가용 LPIPS-VGG·CLIP·DISTS 가중치도 필요하다.

`check`는 입력·설정·실행 경로·코드·DCVC/RIFE 가중치 해시와 새 선택에 사용할 Qwen snapshot 파일을 검사한다. CUDA 실행·패키지 import·최종 품질을 의미하지 않으며 설치나 다운로드를 시작하지 않는다. `--semantic-replay`에는 완료된 `semantic.json`과 같은 위치의 `semantic_cache/`가 함께 필요하다. 원본 해시·선택 설정·모델 출처·응답 캐시를 검증하며, 캐시만으로 같은 선택을 재생할 수 없으면 실패한다.

진행 화면에는 단계, 완료 수/전체 수, 비율, 경과 시간, 로그, 실시간 GPU/VRAM이 고정 행으로 표시된다. 전체 ETA는 근거 없이 표시하지 않는다. `Ctrl+C`는 작업 프로세스 그룹을 중단하고 `INTERRUPTED`를 기록한다. **같은 명령**으로 입력·코드·모델과 완료 단계 해시를 검사해 재개한다. 중단 단계의 부분 산출물은 보관하고 다시 실행하며 Qwen 정상 호출 캐시는 재사용한다. 코드가 바뀌면 새 출력 폴더가 필요하다. 학습을 수행하지 않으므로 optimizer checkpoint는 없다.

| 출력 | 내용 |
|---|---|
| `protocol.json`, `source.json` | 설정·코드·모델 해시·입력·재개 계약 |
| `semantic.json`, `semantic_cache/` | 실제 Qwen 선택·원문·실패 응답·재사용 근거 |
| `selection.json` | 의미 키·보조 키·장면 전환·선택 지표 |
| `sender/`, `received/` | 압축 전후 패킷, 실제 전송량·비트 오류·바이트 일치 |
| `decoded/`, `reconstruction/` | 수신 앵커, 전체 복원 PNG와 MP4 |
| `stages/`, `status.json`, `RESULT.json` | 단계별 해시 검사·중단/실패/복원 완료 |
| 비교 폴더의 `COMPARISON.json`, `*_per_frame.csv` | 최종 MP4의 PSNR·SSIM·LPIPS·CLIP·DISTS |

`COMPLETE_SPARC`는 전체 송수신·복원 완료다. 화질 평가 완료는 별도 `COMPLETE_SPARC_MATCHED_COMPARISON`이며 독립 행동·의미 검수 완료를 뜻하지 않는다. `uniform-matched`는 적응 선택과 같은 프레임 수·장면 전환 조건을 유지하지만 의미 키 보존을 강제하지 않는 대조군이다. `fixed-stride2`도 SPARC의 장면 초기화·패킷 형식을 사용하므로 과거 BSU 패킷과 바이트 단위로 같은 조건은 아니다. 같은 QP·키 수를 같은 실제 전송량으로 해석하지 않는다.

<a id="sparc-motion"></a>
### SPARC 희소 궤적·가시성 실험

`run_sparc_motion.sh`는 완료된 SPARC 의미 선택과 기존 BSU 궤적을 검증·재사용하는 별도 실험이다. 기본 `run_sparc.sh`의 보조 프레임 정책은 바꾸지 않는다. [실험 설정](../configs/sparc_motion.json) · [구조](MODEL_ARCHITECTURE.md#sparc-motion) · [결과](RESULTS.md#sparc-motion)

```bash
bash scripts/run_sparc_motion.sh profile
bash scripts/run_sparc_motion.sh run \
  --reference outputs/kinetics_001_adaptive_v2_20261019 \
  --trajectory-run /absolute/path/bsu_kinetics_context/benchmark/snr_10/kinetics_001_a393e9d3 \
  --trajectory-source /absolute/path/bsu_kinetics_context/inputs/kinetics/kinetics_001_a393e9d3/source_prepared.json \
  --historical-comparison outputs/kinetics_001_comparison_20261019/COMPARISON.json \
  --output outputs/kinetics_001_sparse_motion_20261019
bash scripts/run_sparc_motion.sh status --output outputs/kinetics_001_sparse_motion_20261019
```

`run`을 `check`로 바꾸면 읽기 전용 사전 검사를 수행한다. 기존 완료 단계·원본 파일 해시·시간표·키 시점·궤적 바인딩을 검사하며 PNG 인코딩이 달라도 모든 디코딩 RGB가 같으면 재사용한다. `--historical-comparison`은 선택 사항이며 기존 간격2/적응 선택의 완료 영상·공식 지표를 해시 검증 후 비교에 추가한다. 궤적 예산은 새 키 패킷의 실제 채널 사용량 대비 최대 10%다. 등록된 표현 중 예산에 맞는 것이 없거나 장면 전환·64프레임 초과 키 간격이 있으면 실행하지 않는다.

`prepare → encode → channel → decode → restore → evaluate`의 여섯 단계가 완료되면 `COMPLETE_SPARC_MOTION_PILOT`를 기록한다. `reuse.json`은 재사용 근거, `sender/sender.json`은 궤적 예산·보조 프레임 후보별 비용, `received/channel.json`은 실제 전송, `comparison.json`은 네 조건의 최종 MP4 지표를 담는다. 영상은 `reconstruction/{keys,motion,motion_no_visibility,auxiliary}/reconstruction.mp4`에 저장한다. 중단 후 같은 명령으로 완료 단계 해시를 검사해 재개한다. 코드·설정을 바꾸면 새 출력 폴더를 사용한다. 가림 정보는 추적점의 유효성 비트이며 별도 가림 영역 분할 모델을 실행하지 않는다.

<a id="sparc-auxiliary-search"></a>
### SPARC 보조 프레임 선택 개선 실험

완료된 SPARC 기준 영상·의미 선택·모델·채널 환경과 선택적으로 기존 궤적 캐시가 필요하다. [설정](../configs/sparc_auxiliary_search.json)의 최대 32회 제거·이동 탐색과 동일 개수 균등 대조 1회를 수행한다. 기본 프로필을 바꾸는 명령이 아니다.

```bash
bash scripts/run_sparc_auxiliary_search.sh check \
  --reference outputs/kinetics_001_adaptive_v2_20261019 \
  --motion-reference outputs/kinetics_001_sparse_motion_20261019 \
  --output outputs/kinetics_001_auxiliary_search_20261020
bash scripts/run_sparc_auxiliary_search.sh run
bash scripts/run_sparc_auxiliary_search.sh watch --interval 1
```

위 `run`·`watch`는 `check`에 표시된 기본 경로를 사용한다. 다른 경로를 검사했다면 실행·재개에도 같은 인수를 지정한다. 중단 뒤 같은 `run` 명령으로 기록된 단계·측정값을 해시 검증해 재사용한다. `history.json`, `best.json`, `trials/`, `*_official.json`, `RESULT.json`에 후보·채택·실제 비용·최종 MP4 품질을 남긴다. 총 후보 수가 적응적으로 바뀌므로 진행 표시는 평가 횟수와 사전 상한을 구분하고 ETA를 추정하지 않는다. [결과와 허용치](RESULTS.md#sparc-auxiliary-search)

<a id="bsu-gen"></a>
## BSU-Gen 실행

```bash
# 설치·GPU 없이 설정 확인
bash scripts/run_bsu_gen.sh profile

# 읽기 전용 입력·환경·모델 설치 검사; 출력 생성이나 모델 추론 없음
bash scripts/run_bsu_gen.sh check \
  --video /absolute/path/input.mp4 --output outputs/bsu_gen_my_video

# 전체 영상 → Qwen 선택·캡션 → NTSCC/AWGN → Open-Sora → 수신 보정·정량 평가
bash scripts/run_bsu_gen.sh run \
  --video /absolute/path/input.mp4 --output outputs/bsu_gen_my_video

# 기존 전체 길이 정규화 입력을 사용하는 대안 (--video와 동시에 사용하지 않음)
bash scripts/run_bsu_gen.sh check \
  --prepared /absolute/path/source_prepared.json --output outputs/bsu_gen_prepared_video

bash scripts/run_bsu_gen.sh status --output outputs/bsu_gen_my_video
```

`run_bsu_gen.sh`는 기존 생성 전용 명령이다. 원본 전체를 576×320·24FPS로 처리하며 길이 상한이나 반복 연결을 넣지 않는다. `--prepared`의 시작 시점은 0, `duration_limit`은 null이어야 한다. 다른 입력·코드·모델 설치 상태로 기존 출력에 이어 쓰는 것은 거부한다. 과거 생성 실험의 `protocol.json`이나 코호트 manifest를 요구하지 않고 설치된 자산과 공개 설정에서 새 실행 계약을 만든다.

기본 환경은 `.local/settings.json`의 `python`·`channel_python`, `.local/qwen35_env/bin/python`이다. 주 실행기 Python만 `BSU_GEN_PYTHON`으로 지정할 수 있으며 하위 모델 실행 환경은 위 설정을 따른다. Qwen 설치 manifest(`outputs/qwen35_setup/model_manifest.json`), 고정 rank-32 캡션 LoRA·PEFT, `.local/model_paths.json`의 STDiT·VAE·T5·2D VAE, Open-Sora·NTSCC 외부 코드, NTSCC·UniMatch 가중치가 필요하다. [공통 준비](#public-start) · [외부 출처](../THIRD_PARTY_NOTICES.md)

`check`의 `READY_FILES_VERIFIED`는 입력 해시·코드·어댑터/코덱 파일·실행 경로·모델 파일 크기/수정 시각 검사다. Qwen 및 생성기 전체 가중치 해시 재검사, 패키지 import, CUDA 실행, 전체 송수신·복원 성공은 의미하지 않는다. 설치나 대용량 모델 다운로드를 자동으로 시작하지 않는다.

| 출력 | 내용 |
|---|---|
| `gen_protocol.json`, `runtime_template.json`, `protocol.json` | 고정 모델 설정·입력·코드·설치 자산·단일 영상 실행 계약 |
| `source/` | `--video` 사용 시 전체 정규화 영상·PNG·원본 대응 |
| `benchmark/snr_10/video/base/` | Qwen 선택·캡션·NTSCC 송수신·Open-Sora 출력과 단계별 검증 |
| `benchmark/snr_10/video/sender`, `received` | 유료 궤적 송신·채널 결과 |
| `benchmark/snr_10/video/refined/` | 보정 PNG·`reconstruction.mp4`·최종 정량 결과 |
| `status.json`, `RESULT.json` | 실패/중단/완료 상태, 최종 영상 해시, 총 채널 사용량 |

고정 터미널 화면에 현재 단계·처리량·경과 시간·로그·실시간 GPU/VRAM을 표시한다. 전체 남은 시간을 근거 없이 추정하지 않는다. 같은 명령으로 완료 파일을 해시 검증해 재개하며 중단된 단계의 부분 파일은 보관 후 재실행한다. `COMPLETE_BSU_GEN`은 복원·등록 화질 평가 완료이며 독립 의미 오류 검수는 `PENDING`이다. Qwen·Open-Sora·채널 오류 시 DCVC/RIFE로 전환하지 않는다.

<a id="bsu-ltx"></a>
## LTX-2.5 생성기 교체 비교

`configs/bsu_gen_ltx25.json`에 공식 LTX-2.5 22B Distilled 가중치·소스 리비전, FP8 cast·CPU offload, 8+3단계 sigma schedule을 고정한다. 모델 다운로드는 해당 Hugging Face 계정의 접근 승인이 필요하다. 기존 모델 환경과 분리된 `.local/envs/bsu-ltx25`에 설치한다.

```bash
bash scripts/setup_bsu_ltx.sh

bash scripts/compare_bsu_gen.sh prepare \
  --bsu-root /absolute/path/outputs/bsu_kinetics_context \
  --lgvsc-root /absolute/path/outputs/lgvsc_kinetics_matched \
  --output outputs/bsu_ltx_comparison_20261018

bash scripts/compare_bsu_gen.sh run \
  --bsu-root /absolute/path/outputs/bsu_kinetics_context \
  --lgvsc-root /absolute/path/outputs/lgvsc_kinetics_matched \
  --output outputs/bsu_ltx_comparison_20261018

bash scripts/compare_bsu_gen.sh status --output outputs/bsu_ltx_comparison_20261018
```

진행 화면에는 완료 영상 수, 현재 영상·단계, 경과 시간과 GPU/VRAM을 표시한다. 완료 수는 복원·보정·평가가 모두 끝난 영상만 세므로 첫 영상 처리 중에는 `0/14`다. 별도 터미널에서 `watch -n 5 cat outputs/bsu_ltx_comparison_20261018/status.json`으로 읽기 전용 상태를 확인할 수 있다. `RUNNING`은 처리 중, `FAILED`는 오류로 중단된 상태이며, 전체 완료 상태는 `COMPLETE_THREE_WAY_RECONSTRUCTION_COMPARISON`이다. tmux에서 실행했다면 `Ctrl+B` 다음 `D`로 화면만 분리한다.

이 명령은 **실제 수신 조건 재생에 의한 생성기 교체 실험**이다. Qwen·NTSCC를 다시 실행하거나 원본 키를 추가하지 않는다. Open-Sora BSU가 실제로 받은 키 PNG·캡션·움직임·궤적과 전송 비용을 고정하고 LTX 생성 및 같은 `combined_context` 보정을 새로 실행한다. 기존 패킷의 Open-Sora 샘플러 설정은 LTX에 적용하지 않으며, 등록한 공통 수신기 프로필을 사용한다. 독립 수신기 명령 `run_bsu_gen_ltx.sh`는 비교 실행기가 만든 `--request .../conditions/request.json --output DIR`를 받는다. 기존 전체 영상 입력의 `run_bsu_gen.sh`는 Open-Sora 기준선으로 보존한다.

전체 14편을 원래 manifest 순서로 평가한다. 입력은 모두 576×320·24FPS·AWGN 10dB이고 기본 생성 seed는 2025다. LTX는 각 키 구간의 길이를 8k+1로 올려 생성한 뒤 원래 FPS에서 여분의 끝부분만 버린다. 마지막 키의 시점은 원본과 동일하며 시간축을 늘이거나 줄이지 않는다. 수신부는 원본을 읽지 않고, 별도 평가 프로세스만 원본에 접근한다.

`raw/`에는 LTX 생성 결과, `refined/`에는 수신 보정 결과와 PNG/MP4별 PSNR·SSIM·LPIPS·CLIP·DISTS·시간 지표를 기록한다. 구간별 완료 파일을 해시 검증하여 재개한다. 최종 `COMPLETE_THREE_WAY_RECONSTRUCTION_COMPARISON`은 14편 전체의 비교 완료만 의미하며 독립 의미 오류 평가와 60초 이상 일반화는 포함하지 않는다. LGVSC는 자체 키 선택·캡션과 전송량을 유지하므로 LGVSC 대비 동일 비트율 성능 비교가 아니다.

<a id="bsu-predict"></a>
## BSU-Predict 실행

영상 파일 또는 `0.png, 1.png, ...`로 연속된 정규화 RGB PNG를 입력으로 받는다. Qwen 의미 추출·NTSCC 전송·궤적·Open-Sora 산출물은 필요하지 않다. QP32·기본 간격2·AWGN 10dB·seed42의 예측 전용 프로필은 [`configs/bsu_predict.json`](../configs/bsu_predict.json)이다.

```bash
# 설치·GPU 없이 모델 설정 확인
bash scripts/run_bsu_predict.sh profile

# 전체 영상의 입력·런타임 경로·고정 모델 파일 검사; 추론하지 않음
bash scripts/run_bsu_predict.sh check \
  --video /absolute/path/input.mp4 --output outputs/bsu_predict_my_video

# 전체 영상 정규화 → DCVC 송신 → 실제 채널 → 독립 복호화 → RIFE → MP4
bash scripts/run_bsu_predict.sh run \
  --video /absolute/path/input.mp4 --output outputs/bsu_predict_my_video

# 기존 정규화 프레임을 쓸 때 FPS를 실제 원본 시간표에 맞춘다
bash scripts/run_bsu_predict.sh run \
  --source-frames /absolute/path/source_frames --fps 24 \
  --output outputs/bsu_predict_my_frames

bash scripts/run_bsu_predict.sh status --output outputs/bsu_predict_my_video
```

`--video`는 전체 길이를 기본 576×320·24fps로 정규화한다. `--width`·`--height`·`--fps`로 이를 명시할 수 있다. `--source-frames`는 기존 크기를 유지하고 `--fps`의 시간표를 사용한다. 크기는 양의 짝수, 프레임은 최소 2장이어야 한다. 오디오 전송은 포함하지 않는다. 입력 폴더와 출력은 서로 겹치지 않아야 한다.

필요한 자산은 아래 이전 혼합 모델 표의 **DCVC-RT·RIFE·복원 Python·채널 Python·FFmpeg/ffprobe**다. BSU-Predict에는 Qwen·NTSCC·Open-Sora·T5·생성 어댑터가 필요하지 않다. 기본 Python 경로는 `.local/dcvc_rt_env/bin/python`, `.local/bsu_codec_eval_env/bin/python`, `.local/settings.json`의 `channel_python`이다. 다른 실행 환경은 `--codec-python`·`--receiver-python`·`--channel-python`으로 지정한다. 셸 실행기 자체의 Python은 `BSU_PREDICT_PYTHON=/path/to/python`으로 지정할 수 있다. DCVC 소스는 `.local/vendor/DCVC-RT`에 준비한다.

`check`의 `READY_FILES_VERIFIED`는 입력·파일·해시 검사이며 패키지 import·CUDA 커널·실제 송수신 성공은 검증하지 않는다. 새 장비용 DCVC/RIFE 자동 설치기는 제공하지 않는다. [외부 출처와 고정 버전](../THIRD_PARTY_NOTICES.md#bsu-predictive-dependencies)

| 출력 | 내용 |
|---|---|
| `protocol.json`, `source.json` | 설정·원본·코드·모델·정규화·재개 계약 |
| `sender/packet.bin`, `sender/sender.json` | 예측 스트림·유료 시점·예상 실제 채널 사용량 |
| `received/packet.bin`, `received/channel.json` | 단일 seed42 채널 전달·실측 심벌·비트 오류 |
| `decoded/`, `decoded.json` | 받은 패킷만으로 독립 복호화한 앵커 |
| `reconstruction/reconstruction.mp4`, `reconstruction/receiver.json` | RIFE 복원 및 최종 MP4 시간표 검사 |
| `stages/`, `status.json`, `RESULT.json` | 완료 단계 해시·진행·실패/중단·최종 결과 |

이 브랜치에서 예측 모델은 `run_bsu_predict.sh`로 명시해 실행한다. 고정 터미널 진행 화면에 완료 단계 수·경과 시간·GPU 사용률·사용/전체 VRAM·로그를 표시한다. 단계별 작업량이 다른 동안 전체 남은 시간을 추정하지 않는다. 같은 명령으로 재개하면 완료 단계와 의존 산출물을 해시 검증하며, 변경된 입력·코드·모델은 기존 출력 재사용을 거부한다. 중단된 단계의 부분 파일은 보관 후 다시 실행한다. 화질·행동 평가와 새 학습은 이 복원 명령에 포함하지 않는다.

<a id="bsu-hybrid"></a>
## 이전 혼합 BSU 실행 — 2026-10-16 프로필

이 절은 `configs/bsu_default.json`·`bsu.py`를 사용하는 과거 모델의 재현용이다. 의미 키 ≥5개이고 예측 경로 비용이 더 작을 때 예측 경로를 택하고 나머지는 `combined_context`로 생성한다.

이전 혼합 명령은 **기존 Qwen 의미 추출·전송 산출물과 정규화 원본 PNG**를 받아 새 송수신·복원을 실행한다. 새 영상 파일에서 의미 추출까지 한 번에 자동 수행하는 명령은 아니다. 원본 PNG는 송신부만 읽으며 수신부는 선택된 패킷을 독립 복호화한다.

```bash
# 설치·GPU 없이 채택 설정 확인
bash scripts/run_bsu_hybrid.sh profile

# 입력·전송 조건·고정 가중치·실행 환경 검사 (추론하지 않음)
bash scripts/run_bsu_hybrid.sh check \
  --semantic-run /absolute/path/event_pairs/run \
  --source-frames /absolute/path/source_frames \
  --output outputs/bsu_default_my_video

# 실제 송신·채널·독립 수신·복원; 같은 명령으로 완료 단계 검증 후 재개
bash scripts/run_bsu_hybrid.sh \
  --semantic-run /absolute/path/event_pairs/run \
  --source-frames /absolute/path/source_frames \
  --output outputs/bsu_default_my_video

bash scripts/run_bsu_hybrid.sh status --output outputs/bsu_default_my_video
```

`semantic-run`에는 `run_config.json`, `transmitter/metadata.bin`, `transmitter/visual.c64`가 필요하다. 원본은 같은 영상·정규화 시간표의 `0.png`, `1.png`, …여야 한다. 현재 구현은 **10dB·프레임 시점별 고정 AWGN** 입력을 요구한다. 궤적은 원래 배치의 `sender/trajectory.bin`을 찾으며 다른 배치 구조에서는 `--trajectory PATH`로 명시한다. 입력 폴더 안이나 기존 산출물 위에 출력을 만들지 않는다. 다른 SNR을 이 설정으로 조용히 덮어쓰지 않는다.

검증된 설치는 원격 `sangukcom`의 `/home/sangukbae/ETRI/Semantic/semantic_transmission`이다. 해당 저장소에서 골프 입력은 `outputs/bsu_kinetics_context/benchmark/snr_10/kinetics_005_11836d73/base/reconstruction/event_pairs/run`, 원본은 `outputs/bsu_kinetics_context/inputs/kinetics/kinetics_005_11836d73/source_frames`다. 각 체크아웃의 배포 여부와 `profile`을 확인한 뒤 실행한다.

| 필요한 설치 | 위치 / 고정 버전 |
|---|---|
| DCVC 실행 환경·소스 | `.local/dcvc_rt_env/bin/python`, `.local/vendor/DCVC-RT` · revision `cbdae87a5445114cdc7f48816da63ea80bdeac40` |
| DCVC 가중치 | `.local/checkpoints/dcvc_rt/dcvc_rt_intra.tar`, `dcvc_rt_inter.tar` · [검사 해시](../src/semantic_transmission/bsu_dcvc_keys.py) |
| RIFE 소스·모델 | `.local/vendor/Practical-RIFE` · revision `bbfd2ea90910789a860ea3e2b32a240cd577b75e`; `.local/checkpoints/rife/v4.25/{flownet.pkl,IFNet_HDv3.py}` · [검사 해시](../src/semantic_transmission/dense_temporal_receiver.py) |
| 복원 실행 환경 | `.local/bsu_codec_eval_env/bin/python`와 기존 BSU/Open-Sora 모델·T5 캐시 |
| 실제 디지털 채널 환경 | `.local/settings.json`의 `channel_python` · LDPC(6144,9216), 16QAM |

외부 코드·가중치·원본·환경은 Git에 포함하지 않으며 새 장비용 DCVC/RIFE 자동 설치 명령은 아직 제공하지 않는다. [외부 출처](../THIRD_PARTY_NOTICES.md#bsu-predictive-dependencies)를 따라 고정 자산을 준비하고 `check`로 검사한다. `READY`는 환경·파일 검사이지 전체 추론 성공이 아니다.

| 출력 | 내용 |
|---|---|
| `default_profile.json`, `sender_protocol.json`, `delivery/protocol.json` | 채택 설정·입력·실행 코드와 재개 계약 |
| `predictive_sender/sender/` | 송신부가 시험 부호화한 유료 예측 후보 |
| `delivery/selection.json`, `delivery/channel.json` | 실제 비용에 따른 선택, 세 디지털 시드 채널 검사·전송량 |
| `delivery/refined/reconstruction.mp4` | 최종 전달 영상 |
| `RESULT.json` | 완료 상태·선택 경로·전송량·영상 SHA256 |

이 이전 명령은 복원까지 수행하며, 원본 대비 화질·TimeSformer 평가를 자동 실행하지 않는다. 고정 진행 화면·GPU/VRAM·상세 로그는 `.local/terminal_logs/bsu/`에 표시한다. 기존 완료 단계를 해시로 검증하고 입력·코드·모델이 달라지면 재사용을 거부한다. 기존 실험 결과를 다른 모델 결과로 재명명하지 않는다.

**연구 옵션:** `run_bsu_adaptive_dense.sh`는 간격·QP 등을 명시해 바꾸는 기존 실험용 인터페이스다. 이름의 `adaptive_dense`는 비용에 따른 두 경로 선택이며 구간별 간격+QP 탐색과 다르다. 그 탐색은 별도 `experiment_bsu_adaptive_rate.sh`에만 남긴다. [미채택 근거](RESULTS.md#bsu-adaptive-rate)

<a id="lgvsc-kinetics-matched"></a>
## 완료된 BSU와 같은 Kinetics-400 14편을 LGVSC로 비교

```bash
cd /home/sangukbae/semantic_transmission
bash scripts/run_lgvsc_kinetics.sh
```

기본 명령은 **LGVSC 키 선택·캡션·전송 → 복원 → 화질 → TimeSformer 행동 평가**를 수행한다. `outputs/bsu_kinetics_context`의 완료 결과를 먼저 검증하고, BSU가 실제 사용한 정규화 영상과 전체 PNG를 읽기 전용으로 연결한다. 원본 재변환·BSU 재추론 없이 14편·3,333프레임·576×320·24fps·AWGN 10dB를 맞춘다. 기본 출력은 `outputs/lgvsc_kinetics_matched/`다. 오래된 16초 전용 `run_lgvsc_qwen_baseline.sh`를 이 14편 실행에 사용하지 않는다.

```bash
# 모델 추론 없이 입력·BSU 완료 파일·모델·설정 검사
bash scripts/run_lgvsc_kinetics.sh check
# 입력 연결과 재개 설정만 준비
bash scripts/run_lgvsc_kinetics.sh prepare
# LGVSC 송신·채널 처리까지만 실행하고 전송량 비교
bash scripts/run_lgvsc_kinetics.sh rate
# rate 완료 후 같은 명령으로 복원·평가까지 이어서 실행
bash scripts/run_lgvsc_kinetics.sh
# 저장된 진행 상태 확인
bash scripts/run_lgvsc_kinetics.sh status
# SSH 연결 종료 후에도 유지되는 전용 tmux 세션에서 run 실행
bash scripts/run_lgvsc_kinetics.sh start
# 실행 중인 고정 8줄 진행 화면에 읽기 전용 접속
bash scripts/run_lgvsc_kinetics.sh watch
```

`start`와 `watch`에는 `tmux`가 필요하다. `watch`는 새 실험을 시작하지 않으며, **Ctrl+b를 누른 뒤 d**로 화면에서 나와도 실험은 계속된다. 화면에는 단계별 진행률·경과 시간·GPU 사용률·사용/전체 VRAM·상세 로그가 표시된다. 완료·실패 후에도 세션의 마지막 화면과 종료 코드를 보존하며, 같은 세션이 있으면 `start`는 중복 실행을 거부한다. 기본 세션은 `lgvsc_kinetics`, 전용 tmux 소켓 이름은 `semantic-transmission`이다. 다른 세션 이름은 `LGVSC_KINETICS_SESSION`으로 지정한다.

로컬에서 원격 진행 화면에 접속하려면 아래의 계정·호스트·체크아웃 경로를 실제 값으로 바꾼다. SSH 연결을 닫아도 원격 실험은 유지된다.

```bash
ssh -t USER@HOST 'cd /path/to/semantic_transmission && bash scripts/run_lgvsc_kinetics.sh watch'
```

`--bsu-root PATH`로 검증할 완료 BSU 배치를, `--root PATH`로 별도의 LGVSC 출력 위치를 지정한다. 기존 출력의 입력·코드·모델이 달라지면 중단하며 새 출력 위치가 필요하다. GPU 공유 잠금과 실행 잠금으로 중복 작업을 막는다. Ctrl+C 후 같은 명령으로 재개하면 완료 단계의 해시를 검증해 재사용하고, 실패한 단계의 부분 출력은 보존한 뒤 재시도한다. SKEM 비교·PLLaVA 캡션·개별 행동 분류의 검증된 체크포인트도 재사용한다.

**이전 실행의 설정 경로 오류 복구:** 복원·화질 14편을 완료한 뒤 행동평가에서 `action policy/BSU reference/config changed`로 중단됐고, BSU가 기록한 설정과 현재 설정의 내용은 같지만 절대 경로만 다른 경우에 사용한다.

```bash
bash scripts/recover_lgvsc_kinetics_action.sh --check
bash scripts/recover_lgvsc_kinetics_action.sh
```

이 복구는 알려진 이전 행동평가 코드에서 이번 경로 검증 수정으로의 변경만 허용한다. 기존 입력·모델·나머지 코드와 14편의 완료 단계·의존성·파일 해시를 모두 확인한 후 행동평가만 실행한다. 기록된 설정 파일과 실제 평가기가 읽는 설정 파일은 모두 동결된 SHA-256과 같아야 한다. 원래 `protocol.json`, `action_job.json`, 영상·화질·전송 결과는 보존하며, 새 평가 작업·수정 이력·평가 결과는 `recovery/action_config_path_v1/`에 저장하고 최종 `RESULT.json`에서 연결한다. 이 과거 실행을 다시 확인하거나 재개할 때도 복구 명령을 사용한다. 다른 코드·입력 변경을 허용하는 일반 복구 기능은 아니다.

**비교에서 유지하는 차이와 공통 조건:** LGVSC는 InternVL2-8B BF16 SKEM/PSSS 임계값 0.35와 실제 PLLaVA-7B 추론을 사용한다. Qwen·캡션 LoRA·BSU 궤적·시간 보정·새 구간 연결은 LGVSC에 적용하지 않는다. 키 개수를 강제로 맞추지 않는다. NTSCC 가중치·통신 부호·변조·SNR·프레임별 잡음 시드는 BSU와 같다. 두 방식에 같은 원본 시점의 키가 있으면 같은 시각 채널 잡음을 쓴다.

**공개 LGVSC 코드와의 차이:** 4프레임 후보 격자에 EOF 프레임을 포함하고, BSU와 같은 `frame_exact` 구간 추출·`endpoint_exact` 연결을 사용한다. 기본 Open-Sora는 BSU와 같은 tail17·T5 FP32 캐시·키 조건 충돌/짧은 구간 시간표 수정·메모리 실행 보정을 사용한다. 서로 다른 구간 길이의 확산 난수가 같은 텐서라는 뜻은 아니다. 이 공통 실행 보정과 마지막 프레임 규칙을 `protocol.json`에 기록하며 **수정하지 않은 논문 원본 실행의 재현값으로 표시하지 않는다.** 목적은 동일 입력과 전송 회계에서 현재 BSU를 로컬 LGVSC 대조와 비교하는 것이다.

전송량은 실제 NTSCC 복소 심벌과 캡션·광류·인덱스·헤더·CRC·LDPC 패딩/패리티를 포함한 디지털 심벌을 합산한다. BSU 쪽에는 추가 궤적 심벌도 포함한다. 물리 링크의 별도 헤더·모델 사전 설치 비용은 양쪽 모두 제외한다. CBR은 전체 심벌 합계 / `(3 × 높이 × 너비 × 전체 프레임 수)`이며 MP4 저장 바이트나 complex64 저장 바이트를 무선 비트 수로 바꾸지 않는다. 논문 표의 0.00065를 실제 LGVSC 측정값 대신 넣지 않는다.

| 출력 파일 | 내용 |
|---|---|
| `protocol.json` | 공통 조건·모델별 차이·원본/BSU 결과/코드·모델 상태 동결 |
| `status.json` | 전송·복원·평가 진행 및 실패/중단 상태 |
| `RATE_RESULT.json`, `rate_comparison.csv` | 14편별·전체 전송량, BSU/LGVSC 배율·증감률 |
| `videos/<id>/run/` | 실제 LGVSC 선택·캡션·송수신·복원·프레임별 화질 기록 |
| `action/` | BSU와 같은 공식 TimeSformer 설정으로 평가한 LGVSC 14편 |
| `RESULT.json` | 양쪽 전송량·5개 화질 지표·원본/BSU/LGVSC Top-1·Top-5 비교 |

터미널은 BSU 명령과 같은 **고정 8줄**이며, 아래 값은 화면 형식 예시다. GPU 사용률·사용/전체 VRAM은 작업 로그와 독립적으로 약 1초마다 실제 측정해 갱신한다. 작업량을 모르는 모델 로딩 단계에는 진행률·ETA를 만들지 않는다. 일반 파일/파이프에서는 로그 형태로 출력하며 `BSU_PROGRESS=live`로 고정 화면을 강제할 수 있다. 상세 로그는 `.local/terminal_logs/lgvsc_kinetics/`와 영상별 `logs/`에 남고 실패 종료 코드를 그대로 반환한다.

```text
전체 진행 : LGVSC 전송 3/14 (21.4%) | 복원·화질 2/14 (14.3%)
현재 작업 : kinetics_003_5d588509 (3/14)
단계 진행 : LGVSC reconstruct 15/30 (50.0%)
경과 시간 : 00:25:10 | 전체 ETA 미확정
GPU 사용률 : GPU 0 98%
GPU 메모리 : GPU 0 13.20/15.99 GiB (82.6%)
실행 상태 : 실행 중
상세 로그 : /.../.local/terminal_logs/lgvsc_kinetics/...
```

아래는 실행 상태를 판별하는 규칙이다. 10월 16일 현재 14편의 실제 완료 비교는 [결과표](RESULTS.md#bsu-predictive-default)에 반영했다. `READY`나 `prepare`는 LGVSC 모델 추론 완료가 아니며, 실제 전송량은 `COMPLETE_RATE_COMPARISON`, 복원·화질·행동 전체 비교는 `COMPLETE_RATE_QUALITY_ACTION_COMPARISON` 결과가 생성된 뒤 확인한다.

**구현 검증(2026-10-14):** 관련 테스트 72개 통과. 실제 14편·3,333프레임의 원본 픽셀 일치, 준비 재실행 시 3,403개 파일의 해시 유지, 고정 8줄과 실제 GPU/VRAM 갱신을 확인했다. 기존 BSU 패킷을 검증용으로 다시 집계해 기본 전송량 4,217,279심벌과 일치함을 확인했고, BSU 파일은 변경되지 않았다. 이 검증에서는 LGVSC 모델 추론을 시작하지 않았다. [검증 기록](validation/2026-10-14-lgvsc-kinetics-command.json)

<a id="bsu-kinetics-benchmark"></a>
## 이전 combined_context BSU의 Kinetics-400 배치 재현·행동 평가

아래 명령은 **이전 `combined_context` 모델**의 준비부터 최종 평가까지 수행한다. 현재 예측 전용 모델은 [기본 BSU 실행](#bsu-default)을 사용한다. 중단 후에도 같은 명령을 사용한다.

```bash
bash scripts/run_bsu_kinetics.sh
```

[`실행기`](../scripts/run_bsu_kinetics.sh)는 저자 공개 [14편 목록](eval_manifest_kinetics.csv)의 파일·라벨·원본 SHA256을 확인하고, 원본 전체 길이를 576×320·24fps로 준비한다. 이어 Qwen 의미 선택·고정 캡션 LoRA → AWGN **10dB** 송수신 → Open-Sora 복원 → **새 구간 연결(`combined_context`)** → PNG/전달 MP4 화질·전송량 → **TimeSformer Top-1·Top-5** 평가를 실행한다. 기존 원격 WebVid 결과가 필요하지 않다.

TimeSformer는 `.local/bsu_kinetics_runtime/`에 별도 환경·고정 소스·공식 K400 가중치를 자동 준비한다. 최초 준비에 네트워크가 필요하며 기존 BSU/Qwen 환경은 수정하지 않는다. [저자 설정](../07_downstream/timesformer/TimeSformer_divST_8x32_224_TEST.yaml)의 **8프레임·간격 32·공간 크롭 3개·시간 뷰 1개·224 크롭**과 공식 테스트 데이터 로더를 사용한다. 디코더의 기준 FPS는 30이다. 테스트 코드와 같이 3개 크롭의 FP32 로짓을 합쳐 **영상별** Top-1·Top-5를 계산한다. 추론 배치는 한 영상의 3개 크롭이다. 정규화 원본 14편과 최종 BSU 전달 MP4 14편을 같은 방식으로 평가하며, 레이블은 행동 평가에만 사용한다. 학습이나 실패 입력의 임의 대체는 하지 않는다.

**고정 화면:** 일반 터미널은 아래 **같은 8줄**에서 값만 갱신한다. GPU 사용률과 사용/전체 VRAM은 작업 출력이 없는 동안에도 약 1초마다 갱신된다. GPU 값은 해당 장치 전체의 사용량이며 다른 프로그램 사용량도 포함한다. 조회 실패는 `조회 불가`로 표시한다.

```text
전체 진행 : 복원 0/14 (0.0%) | 행동평가 0/28 (0.0%)
현재 작업 : 현재 영상·처리 대상
단계 진행 : 현재 단계·완료/전체·백분율·측정 가능한 ETA
경과 시간 : 00:00:00 | 전체 ETA 미확정
GPU 사용률 : GPU 0 0%
GPU 메모리 : GPU 0 0.00/16.00 GiB (0.0%)
실행 상태 : 실행 중
상세 로그 : 실행별 전체 로그 위치
```

위 숫자는 표시 예시다. 환경 준비·모델 로딩 등 총작업량을 모르는 단계는 미확정으로 표시한다. 파일·파이프 출력에서는 일반 로그로 전환하고, 실제 터미널에서 강제 갱신이 필요하면 `BSU_PROGRESS=live`를 붙인다. 상세 로그는 `.local/terminal_logs/bsu_kinetics/`와 출력 폴더의 `logs/`에 보존한다. 실패 시 종료 코드·마지막 오류·로그 경로가 남고, Ctrl+C로 중단한 뒤 같은 명령으로 재개할 수 있다. 검증된 원본 준비·완료 복원·개별 행동 분류 결과를 재사용하며, 입력·코드·가중치·옵션이 바뀌면 새 `--root`를 요구한다.

**상태 설명 길이 재시도 수정(2026-10-14):** 첫 영상의 Qwen 키 선택에서 두 상태 설명이 45단어를 초과해 중단됐다. Kinetics 전용 [`bsu_kinetics_base.py`](../src/semantic_transmission/bsu_kinetics_base.py)는 해당 형식 오류의 재시도에 실제 단어 수와 상태별 12단어 이하의 작성 지침을 추가한다. 첫 요청·45단어 검증·의미 판정·재시도 횟수는 유지한다. 기존 WebVid 실행 코드에는 적용하지 않는다. 실패 당시의 설정·원문·로그는 `outputs/bsu_kinetics_context/repairs/selection_length_20261014/`에 보존했고, 완료된 16개 판정만 입력·모델·원문 검증 후 원래 출처를 표시해 재사용했다. 이 복구는 복원 시작 전의 해당 실패 건에 한정하며 일반적인 해시 검증 우회 옵션이 아니다. [검증 기록](validation/2026-10-14-bsu-kinetics-selection-repair.json)

수정 직후 검증에서 첫 영상 키 선택 **60/60**을 완료했고, 실패했던 60→68프레임 판정은 한 차례 재시도로 13단어 설명을 반환했다. 완료 상태 재개는 **60개 전부 캐시 재사용·새 추론 0회**, 관련 테스트는 **119개 통과**였다. 이후 2026-10-14 04:40 KST에 `outputs/bsu_kinetics_context/RESULT.json`의 14편·3,333프레임 복원·화질·원본/BSU 행동 평가가 모두 완료됐다.

| 기본 출력 `outputs/bsu_kinetics_context/` | 내용 |
|---|---|
| `RESULT.json`, `action_accuracy.csv` | 14편 완료 후 화질·전송량, 원본/BSU 행동 정확도 |
| `benchmark/snr_10/<video_id>/refined/reconstruction.mp4` | 새 구간 연결까지 적용한 전체 길이 복원 |
| `action/predictions/*.json`, `action/RESULT.json` | 입력 해시·영상별 400개 로짓·상위 5개·정답 여부·집계 |
| `benchmark/protocol.json`, `config/` | 소스·환경·모델·수신부·평가 설정 및 해시 |
| `status.json`, `action/status.json` | 복원/전체 단계 상태, 행동 평가의 개별 진행 상태 |

```bash
# 읽기 전용 입력·프로토콜 확인 / 준비만 수행 / 저장 상태 확인
bash scripts/run_bsu_kinetics.sh check
bash scripts/run_bsu_kinetics.sh prepare
bash scripts/run_bsu_kinetics.sh status
```

**구현 검증(2026-10-13):** 저자 원본 14편, 새로 정규화한 **3,333프레임·138.875초**, 공식 가중치 엄격 로딩, 원본 1편의 실제 CUDA 3크롭 추론을 확인했다. 이 준비본은 앞서 시간 추정에 사용한 기존 `processed/`의 3,346프레임과 별개이며 원본 시작부터 EOF까지 사용한다. **당시에는 14편 복원 전이었다. 이후 이전 BSU·LGVSC·당시 혼합 BSU의 비교가 완료됐으며 [당시 결과](RESULTS.md#bsu-predictive-default)에 별도로 기록한다.** 테스트와 재개 검증은 [검증 기록](validation/2026-10-13-bsu-kinetics-command.json)에 남긴다. 논문과 맞춘 범위는 원본 목록·10dB·행동 평가 설정이다. BSU의 선택·복원·전송 정책 및 원본 정규화를 포함한 논문 전체 수치의 정확한 재현이나 신규 LGVSC/코덱 대조 실험으로 표시하지 않는다.

<a id="bsu-webvid-benchmark"></a>
## 이전 combined_context BSU의 WebVid 전체 영상 복원·평가

[`run_bsu_webvid.sh`](../scripts/run_bsu_webvid.sh)는 [`bsu_default.py`](../src/semantic_transmission/bsu_default.py)를 통해 준비된 WebVid 원본 전체 길이를 **576×320·24fps**로 처리한다. 기본은 AWGN **6·8·10dB**, Qwen 의미 변화 선택 + 고정 캡션 LoRA + 수정 Open-Sora + **궤적·수신 키 합성 → 시간 보정 → 새 구간 연결(`combined_context`)**이다. 검증한 원래 `context_only` 연산을 사용하며 후속 경로·잔차·윤곽 게이트는 추가하지 않는다. 기존 단일 영상 생성 명령과 과거 동결 실행기도 보존한다.

준비 inventory·실행 템플릿·모델이 있는 환경에서 다음을 실행한다. 이 컴퓨터의 프로젝트는 `/home/sangukbae/semantic_transmission`이며 설정된 전용 Python 환경을 사용한다. 아래는 이전 생성 배치의 설정이다. 10월 14일 원격에도 해당 코드가 배포됐으며 기존 실행의 저장된 프로토콜은 유지한다.

```bash
cd /home/sangukbae/semantic_transmission

# 전체 준비본·코드·어댑터 확인. 추론 및 결과 폴더 생성 없음.
bash scripts/run_bsu_webvid.sh check

# 준비된 53편 × SNR 3개 = 159조건. 중단 후 같은 명령으로 재개.
bash scripts/run_bsu_webvid.sh run

# 다른 터미널에서 진행 상태 확인 / 완료된 조건의 성능표 재집계
bash scripts/run_bsu_webvid.sh status
bash scripts/run_bsu_webvid.sh report
```

**진행률 표시:** `run`·`check`는 일반 터미널에서 전체 완료 건수, 현재 영상·SNR, 단계별 진행률, 경과 시간, 상태, 로그 위치를 **같은 6줄에 갱신**한다. 완료 자료를 재개할 때는 검증된 완료 건수를 유지한다. 모델 로딩처럼 총작업량을 모르는 단계는 미확정으로 표시한다. 전체 출력은 `.local/terminal_logs/bsu_webvid_*.log`에 실행별로 보존하고, 실패 시 마지막 오류와 로그 경로를 터미널에 남긴다. 파일 리다이렉션·파이프 또는 `TERM=dumb`에서는 줄을 누적하는 방식으로 자동 전환한다. 터미널에서도 기존 방식이 필요하면 `BSU_PROGRESS=plain bash scripts/run_bsu_webvid.sh run`을 사용한다. `status`·`report`의 출력 형식과 기존 실행의 해시 검증을 유지하며, 새 구간 연결의 두 단계도 같은 표시기를 사용한다.

**프레임 수 정밀도 수정(2026-10-12):** Open-Sora의 BF16 메타데이터가 721을 720으로 반올림해 복원 검사가 실패하던 문제를 수정했다. 정확히 표현되지 않는 프레임 수만 원래 정수에서 FP32로 다시 만들고, 프레임 수·잠재 길이 검사는 유지한다. 수정 전 코드 해시가 고정된 실패 실행은 보관하고 새 템플릿·출력으로 시작해야 한다.

**긴 구간의 메모리 처리:** 721프레임 실복원에서 시간 어텐션의 임시 행렬로 VRAM 부족이 확인되어, 큰 어텐션은 독립적인 위치 묶음으로, 큰 MLP는 독립적인 토큰 묶음으로 계산한다. 전체 시간 구간·가중치·정밀도·샘플링 횟수·기존 어텐션 커널 선택은 유지하며, 연산 묶음 크기에 따른 반올림 차이는 가능하다. 분할 실행 횟수는 `base/reconstruction/event_pairs/run/receiver/memory_execution.json`에 기록한다. [수정·검증 기록](validation/2026-10-13-bsu-frame-count-memory-fix.json): 로컬·원격 각 194개 테스트 통과, 원격 첫 영상 721프레임·6dB의 복원·보정·평가 완료. 이 기록은 첫 조건의 검증 당시 상태이며, 현재 누적 완료 건수는 `status`·`report`로 확인한다.

**Qwen 응답 형식 복구(2026-10-13):** 의미 선택에서 `entry_exit`의 하위 표기인 `entry`·`exit`·`entry/exit`를 허용하되 원문·판정·근거 프레임을 보존한다. 알 수 없는 종류, 판정과 모순된 종류, 잘못된 근거, 미해결 불확실성은 계속 거부한다. 형식 재시도에는 이전 응답과 구체적인 검증 오류를 누적해 전달하고, 실패 원인·원문은 기록한다. 정상 완료된 조건은 해시 검증 후 재사용한다. 이번 배포는 기존 성공 응답 재검증 후 미완료 선택 디렉터리만 보관하고 다시 생성하며, 완료 결과의 과거 코드 출처를 덮어쓰지 않는다. [수정·검증 기록](validation/2026-10-13-bsu-response-recovery.json)

기본 입력은 `outputs/remote_webvid_full_20261012/inventory.json`, 템플릿은 `.local/deployment/bsu_template.json`, 새 기본 출력은 **`outputs/bsu_webvid_context/`**다. 입력이 다른 위치에 있으면 `--inventory`, `--template`, `--root`를 지정한다. 복사한 비교용 수신 산출물만으로 전체 송신·생성 입력이 준비됐다고 가정하지 않는다. 준비본·선택·기본 복원·보정·지표의 해시를 확인하며, 옵션이나 코드가 달라지면 새 출력 폴더를 요구한다. **개별 조건 실패 시 종료**하고 완료 조건은 보존한다. `--keep-going`은 실패 목록과 `PARTIAL`을 남긴다.

기존 `--root`에 프로토콜이 있으면 생략한 옵션은 저장된 값으로 재개한다. 예전 `combined` 배치에 새 기본값을 소급하지 않는다. 명시한 모드·SNR·영상·입력이 저장값과 다르면 새 폴더를 요구한다. 예전 실행의 상태·보고서도 `--root outputs/bsu_webvid_paper`로 명시한다. 입력을 다시 생성하지 않고 새 방식만 비교하려면 검증된 `--reuse-base` 또는 앞서 가져온 산출물의 [수신부 비교](#bsu-context-extension)를 사용한다.

**디지털 링크 설정:** 실제 시험에서 기존 LDPC(6144,9216)·16-QAM이 6/8dB에서 메타데이터 복호에 실패했다. 이 배치의 기본·추가 메타데이터는 **6/8dB QPSK, 10dB 16-QAM**을 사용한다. SNR 자체와 시각 NTSCC 채널은 유지하고, QPSK에서 늘어난 디지털 심벌도 모두 계상한다. 이 설정은 명시적인 로컬 링크 정책이며 논문의 의미 메타데이터 링크와 같다고 검증된 것은 아니다. 오류 패킷을 원문으로 바꾸거나 비용을 숨겨 재전송하지 않는다.

```bash
# 먼저 1편·10dB 실행
bash scripts/run_bsu_webvid.sh run --ids webvid_008_852d56f2 --snr 10 \
  --root outputs/bsu_webvid_single

# 시간 보정만 비교하는 별도 실행
bash scripts/run_bsu_webvid.sh run --mode cadence --snr 10 \
  --root outputs/bsu_webvid_cadence

# 새 구간 연결을 끈 이전 결합 방식
bash scripts/run_bsu_webvid.sh run --mode combined --snr 10 \
  --root outputs/bsu_webvid_combined
```

`--mode none`은 보정 없는 비교이며 원래 MP4 바이트를 보존한다. `--reuse-base /absolute/path/completed_run`은 단일 영상·단일 SNR에서만 허용하며, 완료된 동일 BSU 실행의 원본·정규화 프레임·모델·캡션·코드·SNR·출력 해시를 검증해 새 추론 없이 보정과 평가를 시험한다. 이를 새 전체 모델 추론 결과로 표기하지 않는다. 서로 다른 SNR의 일반 실행은 각자 선택·캡션·채널·복원 기록을 만들며, 반복 추론 시간 절감용 SNR 간 캐시 공유는 구현하지 않았다.

| 산출물 | 내용 |
|---|---|
| `SUMMARY.json` | SNR별 기본/보정 출력의 영상 동일 가중 평균·프레임 가중 평균, 전송량·CBR, 완료/미완료 건수 |
| `per_video.csv`, `comparison.tsv` | 영상별 5개 지표와 전송량, SNR별 비교표 및 논문 CBR 참고값 |
| `snr_10/<video_id>/refined/reconstruction.mp4` | 전체 길이의 최종 복원 영상 |
| `snr_10/<video_id>/refined/quality_*.csv` | PNG와 전달 MP4 각각의 모든 프레임 점수 |
| `snr_10/<video_id>/sender`, `received` | 송신 궤적 패킷·실제 LDPC/QAM/AWGN 수신 기록·비용 |
| `status.json`, 개별 `logs/`, `stages/` | 현재 영상/SNR, 단계별 진행률, 실패 로그와 재개 기록 |

궤적 예산 기본값 `--extra-rate-ratio 0.05`는 **해당 영상의 실제 기본 BSU 전송량 대비 최대 +5%**다. 영상 전체의 별도 궤적 패킷에 헤더·CRC·LDPC 패딩·패리티까지 포함한다. `combined_context`의 송신 패킷·비용은 기존 `combined`와 같으며 새 연결 자체의 추가 전송은 0이다. 유효 궤적이 없으면 합성을 생략하고 시간 보정·새 구간 연결을 수행한다. 각 영상의 LGVSC 전송량 대비 +5%를 보장한다는 뜻은 아니다.

논문은 WebVid **55편**을 사용하지만 현재 보유 입력은 **53편**이다. 결과는 누락 파일명과 `53/55`를 표시한다. `--require-complete-manifest`를 쓰면 누락 상태에서 시작하지 않는다. 논문과 같은 다섯 화질 지표·전체 프레임 평균을 계산하지만, 준비 해상도·NTSCC 가중치·부가정보 비용까지 논문과 동일하다고 확인된 재현은 아니다. 논문의 정확한 화질 수치는 곡선으로 제시되어 표에 임의의 값을 채우지 않으며, 공개된 CBR 표 값만 출처와 함께 제공한다. [평가·집계 기준](METRICS.md#bsu-webvid-metrics) · [논문 기준 설정](../configs/bsu_webvid_paper.json)

**과거 실험의 산출물 가용성:** 2026-10-08 성능 문제로 보류된 후보의 일부 가중치·캐시·영상은 사용자 요청으로 삭제했다. 현재 기본 모델과 실행 중 LGVSC/Qwen 비교의 의존 파일은 보존했다. 아래 과거 명령은 삭제된 산출물이 있으면 바로 재개할 수 없으며, 해당 실행의 `ARTIFACT_CLEANUP.json`과 [정리 기록](validation/2026-10-08-rejected-artifact-cleanup.json)을 확인하고 원래 추출·학습·생성 단계부터 복구해야 한다.

<a id="bsu-adapter-training"></a>
### 생성 내부 궤적 어댑터의 로컬 학습

현재 학습의 최신 체크포인트에서 **추가 약 7시간**만 이어 학습할 때는 [train_bsu_adapter_7h.sh](../scripts/train_bsu_adapter_7h.sh)를 사용한다. 2026-10-15 로컬 실행은 221번째 업데이트를 끝내고 가중치·옵티마이저·스케줄러·난수·데이터 순서를 저장한 뒤 안전하게 중단했다. 원래 실행의 코드·설정·체크포인트는 보존하고, 별도 `outputs/bsu_adapter_training_7h_v1/`에 이어 학습 상태를 둔다.

```bash
# 원본 최신 체크포인트와 데이터 해시 검사·준비. GPU 학습은 시작하지 않는다.
bash scripts/train_bsu_adapter_7h.sh check

# 검증된 체크포인트부터 추가 7시간. 중간 중단 후에는 남은 시간만 재개한다.
bash scripts/train_bsu_adapter_7h.sh run

# 별도 터미널에서 진행 상황과 최신 체크포인트를 확인한다.
bash scripts/train_bsu_adapter_7h.sh status
```

이 명령은 기존 학습 시간과 별도로 **25,200초의 실행 시간 예산**을 사용한다. 시작 시 검사·모델 로딩·250회마다의 고정 잡음 검증도 예산에 포함하고, 중단해 둔 시간은 제외한다. 시간 만료나 Ctrl-C를 받으면 진행 중인 누적 4회의 업데이트를 끝내고 저장하므로 해당 업데이트와 저장 시간만큼 초과할 수 있다. 검증 중에는 현재 순전파를 마치고 멈추며 미완료 검증을 학습률 스케줄러에 반영하지 않는다. 매 업데이트를 저장하고, 같은 명령의 재실행은 남은 예산만 사용한다. 완료한 명령을 다시 실행해도 7시간이 새로 추가되지 않는다. 원본 `run` 명령의 20,000회 설정은 이 시간 제한 실행에 적용하지 않는다.

시간 예산·백분율·추가/누적 업데이트·남은 시간·GPU·VRAM·로그를 고정 행으로 표시한다. 데이터 캐시는 해시를 검증해 읽기 전용으로 재사용하고, `continuation.json`에 시작 체크포인트와 추가 실행 코드를 고정한다. 최종 가중치는 `outputs/bsu_adapter_training_7h_v1/final.pt`, 재개 상태는 같은 폴더의 `latest.json`이다. 전체 영상 품질 검증과 최종 시험은 추가 시간이 필요하므로 자동 실행하지 않는다. `RESULT.json`의 `TRAINING_BUDGET_COMPLETE_EVALUATION_PENDING`은 시간 예산을 마쳤다는 뜻이며 성능 개선이나 수렴을 입증하지 않는다. 종료 시점의 가중치와 검증으로 선택한 최적 가중치를 구분한다.

아래는 원래의 장기 학습 명령과 설정이다.

실행 명령은 [train_bsu_adapter.sh](../scripts/train_bsu_adapter.sh), 기본 학습 설정은 [bsu_adapter_training.json](../configs/bsu_adapter_training.json)이다. 현재 로컬의 `bsu_webvid_completed_20261015`, `bsu_receiver_improvement_20261015_v1`, `bsu_motion_validation_20261015_v1`, `bsu_kinetics_context` 산출물과 고정 모델 캐시가 필요하다. Git 코드만 복사하면 입력이 준비되는 명령은 아니다. 기존 600회 실험의 가중치·결과와 `combined_context` 생성 경로는 변경하지 않는다. 학습 명령의 구현·준비는 충분한 학습 완료나 기본 하이브리드에 대한 어댑터 채택을 뜻하지 않는다.

```bash
# 입력·원본별 분리·모델·코드 검사. 학습이나 모델 추론을 시작하지 않는다.
bash scripts/train_bsu_adapter.sh check

# 준비 → 학습 → 주기적 검증 → 최종 평가. 중단 후 같은 명령으로 재개한다.
bash scripts/train_bsu_adapter.sh run

# 다른 터미널에서 저장된 상태를 읽는다. 실행 중인 학습에는 영향을 주지 않는다.
bash scripts/train_bsu_adapter.sh status
```

기본 출력은 `outputs/bsu_adapter_training_v1/`이다. 준비 단계는 실제 수신 메타데이터, FP32 T5 캐시, 원본 목표 잠재값과 수신 조건 잠재값을 만든다. 학습 24개 원본의 128개 구간, 검증 8개 원본, 최종 평가 18개 원본을 분리한다. 이전 어댑터 평가에 사용했던 8개 원본은 이제 개발용 검증 세트이며 새로운 독립 시험으로 부르지 않는다. 최종 평가 원본도 다른 과거 진단에서 관찰됐을 수 있어 완전히 눈가림된 모집단 시험은 아니다.

학습은 **576×320·24fps의 실제 구간 전체**를 사용한다. 균등 33프레임 추출이나 긴 구간 잘라내기는 하지 않는다. 시작·끝 키의 실제 잠재 위치, 충돌 보정, RFLOW 시간 변환과 이전 생성 구간의 마지막 17프레임을 반영한다. 이전 구간 조건은 동결된 기존 생성 결과에서 가져오며 원본 정답 프레임을 조건으로 넣지 않는다. 학습 중 갱신된 어댑터가 직접 생성한 이력과의 차이는 남으므로, 검증에서는 전체 영상을 실제 구간 순서로 다시 생성한다.

| 기본 설정 | 동작 |
|---|---|
| 최소 3,000 / 최대 20,000 업데이트 | 횟수 도달 자체를 충분한 학습으로 판정하지 않음 |
| 배치 1, 누적 4회 | 어댑터만 AdamW로 학습, 생성 본체는 고정 |
| 250회마다 검증 | 고정 원본·잡음 2개·시간 조건 2개로 검증 손실 비교, 정체 시 학습률 감소 |
| 2,000회마다 전체 영상 검증 | 같은 수신 정보·생성 잡음으로 기준/후보의 화질·동작·경계 비교 |
| 손실 4회·영상 품질 3회 연속 정체 | 최소 업데이트 이후 종료. 운영상 기준이며 최적성의 증명이 아님 |
| 25회마다 체크포인트 | 가중치·옵티마이저·스케줄러·난수·데이터 순서를 원자적으로 저장 |

추가 출력이 0이 되도록 초기화한 새 어댑터를 학습한다. 체크포인트 선택은 품질 기준 통과 여부를 우선하고, 그 안에서 LPIPS·흐름 오차·시간 변화 오차의 고정 복합 점수를 사용한다. 최대 횟수에 먼저 도달하면 `MAX_STEPS_WITHOUT_CONFIRMED_PLATEAU`로 기록한다. `best.pt`가 있어도 성능 개선이 보장되지는 않는다. `RESULT.json`의 검증·최종 평가 판정과 제한을 확인해야 하며, 기본 BSU 모델에 자동 적용하지 않는다. 데이터는 여전히 소규모이므로 업데이트 횟수만으로 범용성이나 내용·동작 오류 해결을 입증할 수 없다.

`latest.json`, `checkpoints/`는 재개 상태이고 `exports/`는 시점별 추론용 가중치다. `validation_loss.json`, `validation/step_*/quality.json`, `best.json`, `final_test/`, `RESULT.json`에 학습 곡선과 선택·평가 근거를 남긴다. 고정 행의 진행 화면에는 단계, 업데이트 수/상한, 경과 시간, 로그, 실패·중단 상태, GPU 사용률과 사용/전체 VRAM을 표시한다. 조기 종료 시점과 검증 비용을 미리 알 수 없으므로 전체 ETA를 확정하지 않는다. 로그는 `.local/terminal_logs/bsu_adapter_training/`에 남는다.

환경에서 실제 역전파·저장 시험만 실행하려면 아래 명령을 사용한다. 가장 긴 학습 구간과 이전 구간 연결이 있는 가장 긴 구간을 검사하며, 이 시험의 가중치는 본 학습에 사용하지 않는다.

```bash
bash scripts/train_bsu_adapter.sh smoke --root outputs/bsu_adapter_training_smoke
```

출력 폴더와 설정은 `--root`, `--config`로 지정한다. 같은 폴더에서 코드·모델·입력·설정이 달라지면 재개를 거부하므로 변경 실험에는 새 폴더를 사용한다. 메모리를 줄이기 위해 체크포인트 재계산, CPU 활성값 보관, 독립 어텐션 배치·MLP 토큰의 미분 가능한 분할을 사용한다. 시공간 어텐션 문맥은 유지한다. 전체 생성 검증이 학습 업데이트 자체보다 오래 걸릴 수 있다.

<a id="public-start"></a>
## 완료된 WebVid 4편의 수신부만 로컬 비교

```bash
bash scripts/compare_bsu_webvid.sh
```

[`compare_bsu_webvid.sh`](../scripts/compare_bsu_webvid.sh)는 원격 `sangukbae@155.230.15.77`의 `outputs/bsu_webvid_paper`에서 첫 4편의 완료된 **6·8·10dB** 산출물만 가져와 **현재 결합 / 시간 보정만 / 윤곽 검사와 수신 키 보존**의 36개 결과를 비교한다. 기존 생성 PNG를 재사용하며 선택·캡션·Open-Sora·채널을 다시 실행하지 않는다. 기본 출력은 `outputs/bsu_webvid_receiver_comparison_20261013/`이다. SSH 인증이 필요하면 터미널에서 비밀번호를 입력하며, 명령·로그·설정에 비밀번호를 저장하지 않는다.

복사 대상은 평가용 원본 PNG·정규화 MP4, 보정 전·현재 보정 PNG/MP4, 수신 키·메타데이터·궤적 패킷, 설정·완료 기록이다. 공유 원본은 한 번만 복사하고 모델·전체 WebVid·전송용 대형 비트스트림은 제외한다. 원격 기록에서 고정한 SHA256과 복사본을 대조한다. 로컬 화질 평가 가중치와 `.local/settings.json`의 Python 환경이 필요하며 `check`로 확인한다. 원본은 별도 평가 작업에만 전달한다.

일반 터미널은 **같은 6줄**에서 전체 `완료/36`, 현재 영상·SNR·방식, 단계별 건수·백분율·ETA, 경과 시간, 상태와 로그 위치를 갱신한다. 처음 연결 여부를 판단하는 로컬 해시 검사는 한 줄을 갱신한다. 전체 ETA는 작업량이 서로 달라 미확정으로 표시한다. 상세 출력은 `.local/terminal_logs/bsu_receiver_compare/`에 남으며, 파이프·파일 출력은 일반 로그 형식이다.

중단 후 **같은 명령**으로 재개한다. 입력과 완료 결과를 해시로 검증하고, 저장된 구간 PNG 및 완료 평가를 재사용한다. 완료 전 중단된 평가 단계는 다시 계산한다. 완료 파일이 변조되거나 코드·옵션·환경이 바뀌면 중단하므로 다른 실험은 새 `--root`를 지정한다. 원격 본 실험은 수정하지 않는다.

```bash
# 다운로드 없이 원격 완료 조건·필요 용량·로컬 평가 환경 확인
bash scripts/compare_bsu_webvid.sh check
# 산출물만 복사 / 복사 완료 후 네트워크 없이 비교
bash scripts/compare_bsu_webvid.sh fetch
bash scripts/compare_bsu_webvid.sh --offline
# 실행 종료·중단 후 집계 상태 확인
bash scripts/compare_bsu_webvid.sh status
```

`--ids`, `--snr`, `--root`, `--host`, `--remote-repo`, `--remote-root`, `--device cpu|cuda`로 범위를 바꿀 수 있다. 기본값은 CUDA다. `report.html`에 비교 영상·표본·지표를 모으며 `summary.csv`, `RESULT.json`, `protocol.json`에 수치·누락 조건·동결 설정을 기록한다. PSNR·SSIM·LPIPS·CLIP·DISTS를 PNG와 전달 MP4에서 평가하고, 시간 변화·광류·정지 오판·수신 키 충실도·키 경계 변화도 확인한다. 현재 결합 방식의 기존 화질 지표는 검증한 완료 기록에서 재사용한다. 세 방식 모두 완료된 같은 영상끼리만 평균을 내며 SNR 반복을 독립 영상으로 세지 않는다.

수신 키 보존은 PNG의 키 시점에서 수신 RGB를 정확히 유지하고 가까운 프레임에는 정합 신뢰도에 따라 보정을 줄이는 **실험 후보**다. 윤곽 검사와 키 보존의 개별 효과를 분리한 대조는 아니다. 주 비교의 전송량은 세 방식 모두 같은 실제 수신 패킷 비용을 사용한다. 시간 보정만의 궤적 생략 비용은 별도 회계상 참고치이며 채널 재실험 결과가 아니다. 정량 완료 상태도 `COMPLETE_QUANTITATIVE_REVIEW_PENDING`이며, 이 4편의 개발 비교만으로 독립 의미 검수·ETRI 목표 달성이나 전체 WebVid 성능을 확정하지 않는다.

[구현 검증 기록](validation/2026-10-13-bsu-receiver-compare.json): 필요한 18,274파일·약 4.08GiB 복사·해시 검사 완료. 4번째 영상 494프레임·10dB에서 세 방식의 실제 보정·평가·보고서와 재개를 검증했다. 재실행 시 완료 산출물 1,006개의 수정 시간이 유지됐다. 전체 36조건 비교는 이 검증에 포함하지 않았다.

<a id="bsu-key-context"></a>
### 수신 키 전후 연결과 윤곽 검사의 개별 효과 검증

```bash
bash scripts/experiment_bsu_key_context.sh
```

완료된 위 WebVid 비교와 `outputs/bsu_adaptive_visibility_20261012_v1/`의 DAVIS 4편을 사용한다. **윤곽 검사 유무 × 키 보정 없음·기존 키 보존·새 구간 연결**의 6방식이며, WebVid 4편×3 SNR와 DAVIS 4편×10dB를 합쳐 96조건이다. DAVIS는 과거 실험에도 사용된 96프레임 자료로, 완전히 미사용한 영상이나 현재 Qwen 전체 경로의 추가 검증을 뜻하지 않는다.

새 구간 연결은 수신 키와 주변 생성 프레임을 양방향 광류로 보간한다. 공통 규칙으로 범위를 8~24프레임에서 정하며 짧은 구간과 장면 전환에서는 줄인다. 원본은 평가 작업에만 제공한다. 선택·캡션·생성기·채널을 다시 실행하지 않고 기존 수신 정보와 전송 비용을 유지한다. 최초에는 기존 화질의 프레임별 CSV 24개만 원격에서 추가로 가져오며, 이후에는 로컬로 재개할 수 있다.

출력은 `outputs/bsu_key_context_20261013_v1/`이다. `check`는 입력·코드·설정·평가 기준만 동결하고, 기본 `run`은 완료 기록을 검증해 생성·평가를 재개한다. 터미널은 위와 같은 고정 6줄 진행 화면을 사용한다. PNG와 전달 MP4에서 공식 5개 화질 지표, 전체 시간 지표, 키·비키 화질과 키 경계를 평가한다. 기존 지표는 **평가 원본의 PSNR과 출력 RGB가 일치하는 프레임에만** 재사용하며, 달라진 프레임의 화질과 모든 시간 전환은 다시 계산한다. 설정·입력·코드가 달라지면 새 `--root`가 필요하다.

`RESULT.json`에는 2×3 대조와 사전 공학적 선별 기준 판정이 기록된다. 이 기준은 ETRI의 공식 합격 기준이나 통계적 비열등성 검정이 아니다. 자동 기본값 변경은 없으며 독립 의미 검수·연속 60초·미사용 원본 검증은 별도로 남는다.

<a id="bsu-context-extension"></a>
### 추가 완료 WebVid 영상에서 동결된 구간 연결 비교

```bash
bash scripts/experiment_bsu_context_extension.sh --offline
```

`outputs/bsu_context_extension_20261013_v1/cohort.json`에 고정한 추가 12편·35 영상/SNR 조건을 사용한다. 원격의 현재 결합 출력과 수신 키·패킷, 평가 원본 PNG·프레임별 지표·완료 영수증만 복사했다. 원래 생성 PNG·모델 가중치는 필요 없다. `--offline`을 빼면 누락되거나 손상된 입력만 원격에서 다시 복사하며 기존 SSH 인증을 사용한다. 원격 실행은 수정하지 않는다.

원래 `context_only` 코드와 설정을 그대로 적용해 현재 결합과 비교한다. 움직임 잔차·프레임별 경로·적용 게이트는 추가하지 않는다. 이전 개발 원본과의 해시 중복을 거부하고, 평가 환경과 수신부 코드가 앞선 실험과 같은지 검사한다. 6·8·10dB가 모두 완료된 11편을 주 집계로, 6·8dB만 완료된 1편을 별도로 집계한다. SNR 평균 후 영상별 동일 가중치를 사용하며 실행 도중 새로 완료되는 원격 조건은 포함하지 않는다.

고정 6줄 진행 화면, `check`, 입력·출력 해시 검사와 완료 결과 재개를 지원한다. 새 출력 폴더에는 `cohort.json`을 먼저 준비해야 한다. 결과는 `RESULT.json`, 생성·평가 영수증과 프레임별 CSV에 남기며 분석·표본은 로컬 `analysis/index.html`에 보존한다. 기본 BSU 설정은 변경하지 않는다.

<a id="bsu-motion-retention"></a>
### 키 연결에서 지워진 움직임 복원과 적용 제한

완료된 수신 키 연결 실험과 로컬 개발 진단 기록을 사용하는 후속 명령이다. 현재 작업 공간에서는 제한형 결과까지 완료되어 같은 명령으로 검증·재개할 수 있다.

```bash
# 보수적 적용 제한: 기존 움직임이 감소하는 창은 이전 연결 출력 유지
bash scripts/experiment_bsu_motion_retention.sh
# 적용 제한 없는 잔차 복원 대조
bash scripts/experiment_bsu_motion_context.sh
```

각각 `outputs/bsu_motion_retention_20261013_v1/`, `outputs/bsu_motion_context_20261013_v1/`에 48조건을 기록한다. 현재 결합·이전 연결 32조건을 재사용하고 16조건을 새로 평가한다. 제한형은 완료된 무제한 후보의 PNG를 받아 원본 없이 적용 창을 선택한다. 고정 6줄 진행 표시, 입력·코드·완료 산출물 해시 검사, 중단 후 같은 명령 재개를 지원한다. `check`는 동결 검증만 수행한다. 기본값은 CUDA이며 기존 평가 가중치를 오프라인으로 사용한다.

두 명령은 임의 MP4를 받는 일반 배치 인터페이스가 아니다. 앞 절의 동결 결과와 `bsu_motion_context_20261013_pilot1`, `bsu_motion_gate_20261013_pilot1`, `bsu_motion_retention_20261013_pilot1`의 로컬 개발 기록이 필요하다. 이 자료는 Git에 없으므로 새 장비에서는 입력 인계가 필요하다. 원격 재접속·추가 다운로드·생성기 재실행·추가 전송은 하지 않는다. `RESULT.json`의 `motion_context`는 폴더별 정책을 확인해야 한다. 제한형 폴더에서는 보수적 적용 조건까지 포함하며, 정책은 `protocol.json`과 `generation_info.json`에 기록된다. 기본 BSU 명령에 자동 적용하지 않는다. [결과와 한계](RESULTS.md#bsu-motion-retention)

<a id="bsu-path-context"></a>
### 프레임별 이동 경로를 반영한 키 연결 검증

```bash
bash scripts/experiment_bsu_path_context.sh
```

완료된 `outputs/bsu_key_context_20261013_v1/`의 같은 8원본·16 영상/SNR 조건을 사용한다. 현재 결합·이전 키 연결 32조건을 재사용하고 `path_context` 16조건을 새로 생성·평가해 `outputs/bsu_path_context_20261013_v1/`에 기록한다. 인접 프레임의 기존 DIS 광류로 이동 경로만 바꾸는 대조이며 새 추적 모델·RGB 잔차·영역별 보정·구간 선택은 추가하지 않는다. 별도의 과거 움직임 복원 파일럿 결과는 필요 없다.

기존 수신 데이터만 사용하고 원본은 평가 작업에만 제공한다. 평가 환경·모델은 `.local/settings.json`의 로컬 Python과 기존 캐시를 사용하며 네트워크·생성기·채널 재실행은 없다. 고정 6줄 진행 표시와 `check`, 같은 명령 재개, `--root`, `--parent`, `--device cpu|cuda`를 지원한다. 부모와 평가 환경이 같아야 하며 코드·설정·입력이 바뀌면 새 출력 폴더가 필요하다. 출력 정책은 실행 전에 동결한다.

수신 키·보정 범위·전체 PNG와 MP4 화질·시간 지표를 확인한다. 생성 작업 안에서 이전 연결도 동일 프레임으로 재계산해 동결 출력과 RGB를 대조하고, 두 보정의 실행 순서를 교대해 연산 시간을 기록한다. `RESULT.json`의 `timings`는 입출력·인코딩·평가·생성기 시간을 제외한 단일 측정이며, 재개 시 원래 측정 기록을 유지한다. 비교 HTML은 로컬 `analysis/index.html`에 있다. 기본 BSU 경로는 자동 교체하지 않는다. [결과와 한계](RESULTS.md#bsu-path-context)

## 새 장비와 ETRI 인계에서 시작하기

1. Linux/WSL2·NVIDIA 드라이버·Conda·FFmpeg를 준비합니다. 통합 환경은 `bash scripts/bootstrap_official.sh`, 독립 Qwen 환경은 `bash scripts/bootstrap_qwen35.sh`로 준비합니다. 각 스크립트의 다운로드·설치 대상과 디스크 용량을 확인합니다.
2. `.local/settings.json`은 새 장비의 실제 Python·모델 경로로 생성합니다. 연구자의 절대 경로가 기록된 과거 설정을 새 장비에 그대로 복사하지 않습니다.
3. 원본·정규화 프레임·source hash·prepared inventory는 [데이터 인계](DATA.md#handoff)에 따라 준비합니다. 전체 FC 실행에는 WebVid template protocol을 포함한 준비 inventory가 필요합니다. 단일 임의 MP4를 `init`에 직접 전달하는 인터페이스는 아닙니다.
4. 보강 LoRA를 아래 절차로 설치합니다. 없어도 비교용 `init --base-caption`은 선택할 수 있지만 같은 모델의 재현으로 표시하지 않습니다.
5. [현재 Qwen 영상 경로](RUN_GUIDE.md#qwen35-fc)에 24fps 준비본과 새 출력 폴더를 지정하고 `--check`로 검사합니다. `--select-only`로 의미 선택을 확인한 뒤 같은 입력 옵션으로 캡션·송수신·복원을 진행할 수 있습니다. GPU 공유 잠금을 유지합니다.

기존 환경을 재사용할 때는 별도 Conda 환경으로 복제한 뒤 이 checkout을 `pip install -e . --no-deps --no-build-isolation`로 등록합니다. 하위 복원 프로세스가 다른 저장소의 예전 Python 패키지를 읽지 않는지 확인하고, `.local/settings.json`에 복제한 환경을 지정합니다. Qwen은 고정 lock으로 별도 환경을 생성합니다.

인계받은 기준 `protocol.json`은 그대로 보존하고, 새 장비의 경로와 파일 상태를 담은 실행 템플릿을 따로 만듭니다. 다음 명령은 기준 코드·Qwen 가중치·캡션 LoRA를 확인하며 추론은 시작하지 않습니다. `--snr-db`를 지정하면 해당 채널 조건의 새 템플릿을 만들 수 있습니다. 과거 프로토콜 전체를 수정해 해시 검사를 우회하지 않습니다.

```bash
PYTHONPATH=src python3 scripts/prepare_bsu_runtime.py \
  --reference outputs/qwen35_pairs_caption_comparison_20261007/protocol.json \
  --output .local/deployment/bsu_template.json
bash scripts/run_qwen35_pairs_video.sh \
  --template .local/deployment/bsu_template.json \
  --prepared /absolute/path/source_prepared.json \
  --root outputs/bsu_new_machine --check
```

같은 명령에서 `--check`를 빼면 실행합니다. 새 계획은 `--template`의 경로와 해시를 고정하며, 별도 worker도 그 템플릿을 사용합니다. 이 경로의 기본 모델은 Qwen 의미 선택·캡션 보강과 기존 BSU 복원입니다. 궤적·키 합성 및 시간 보정의 다중 구간 일괄 적용은 아래 연구 옵션의 구현 범위와 별도로 확인해야 합니다.

Qwen 환경에 학습/어댑터 의존성을 추가할 때 기존 Torch·Transformers 설치를 교체하지 않습니다.

```bash
.local/qwen35_env/bin/python -m pip install --no-deps \
  --target .local/qwen35_training_probe_deps \
  -r environment/requirements-qwen35-training.txt
```

### 보강 LoRA 가중치 준비

고정 대상: `qwen35_evidence_repair_v3/caption/update-000060-e002-n0000/adapter`, rank 32·alpha 64. 파일별 SHA256과 제공 상태는 [배포 명세](../configs/models/qwen35_caption_adapter.json)에 있습니다. **공개 다운로드 주소는 아직 없으며 Git에는 가중치를 넣지 않습니다.** 전달받은 패키지를 검증·설치하는 명령입니다.

```bash
# 가중치를 가진 장비: 추론용 두 파일만 내보냄 (optimizer/데이터 제외)
PYTHONPATH=src python3 -m semantic_transmission.caption_adapter_package export \
  --output outputs/handoff/qwen35-caption-repair-e002.zip

# 수신 장비: 고정 해시와 ZIP 구성 검사 후 이 checkout의 기본 위치에 설치
PYTHONPATH=src python3 -m semantic_transmission.caption_adapter_package install \
  /absolute/path/qwen35-caption-repair-e002.zip
PYTHONPATH=src python3 -m semantic_transmission.caption_adapter_package check
```

파일이 없거나 다르면 실패합니다. 기존 다른 가중치를 덮어쓰지 않으며 모델 로딩·추가 학습은 하지 않습니다. 패키지는 LoRA만 포함하므로 공식 Qwen revision·환경·데이터는 별도 준비합니다. 원래 adapter 설정의 base 경로는 학습 출처 기록이며 실제 본체는 현재 checkout의 고정 snapshot에서 로딩합니다.

### 학습과 평가의 입력 의존성

아래 학습 명령은 **동결된 원본·분할·교사 정답·부모 체크포인트를 먼저 준비한 경우**에 사용합니다. 일부 주석·이미지·완료 결과는 로컬 인계 대상이므로 Git clone만으로 즉시 모든 학습을 수행한다고 가정하지 않습니다. `prepare`/`dry-run`/입력 감사부터 실행하고 누락 자료를 임의로 생성해 정답으로 대체하지 않습니다.

```bash
bash scripts/train_qwen35_evidence_repair.sh prepare
bash scripts/train_qwen35_evidence_repair.sh dry-run
# 사용자가 학습을 시작할 때만 실행: 네 조건을 별도 프로세스로 순차 학습
bash scripts/train_qwen35_evidence_repair.sh train all
# 학습 완료 후 동결 평가 자료를 갖춘 상태에서
bash scripts/evaluate_qwen35_evidence_repair.sh generate
```

현재 `caption/evidence/judgment/preference`는 각 2 epoch·60 update의 대조 실험입니다. 기본 채택은 `caption`뿐입니다. 단계별·공동학습, 이전 전이 실험의 명령과 데이터 조건은 아래 절에 따로 보존합니다. 실행 명령의 존재는 의미 성능 개선의 증거가 아닙니다.

### GPU와 복구

- 검증 장비: RTX 4080 16GB. 기본 Qwen 로더는 여유 VRAM 12,000MiB 이상을 요구합니다. 학습 로더의 rank 32 사전 검사는 11,264MiB이며 실제 입력·환경에 따라 추가 여유가 필요합니다.
- 보강 캡션 학습 최대 할당은 약 9.51GiB, 기본 추출 연결 검사의 최대 할당은 8.28GiB였습니다. PyTorch 할당량과 장치 전체 점유량은 다릅니다.
- `train all`은 조건을 순차 실행하고 실패하면 중단합니다. 같은 명령으로 완료 조건은 건너뛰고 미완료 조건은 기존 체크포인트 규칙으로 재개합니다.
- 입력·코드가 바뀌었다면 새 실행 폴더를 사용합니다. 기존 freeze 파일의 해시를 덮어써서 통과시키지 않습니다.
- 출력 토큰 잘림·스키마 오류·빠진 source hash·가중치 불일치는 실패로 남깁니다. 같은 모델의 시각 재검토는 독립 정확도 검증이 아닙니다.

<a id="contributing"></a>
## 코드 검증과 변경

GPU 추론 없이 검사할 수 있는 테스트와 모델/자료가 필요한 실험을 구분합니다. Qwen 전용 테스트는 `.local/qwen35_env`와 PEFT/pytest 설치를, 수신단 테스트는 LGVSC 환경을 사용합니다. 각 변경은 관련 테스트·문서 링크·입력 출처를 검사하고 동결 실험에는 새 revision을 기록합니다. 파생 코드에는 기존 LICENSE가 적용되며 외부 모델·코드의 고지는 [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md)에 유지합니다.

## 실험별 실행과 재현 명령

<a id="bsu-adaptive-visibility"></a>
### BSU 적응형 시간 보정·가림 합성 2×2 대조

```bash
bash scripts/experiment_bsu_adaptive.sh
```

준비·실제 메타데이터 채널·공통 생성·4조건 보정·평가·산출물 검증을 순서대로 실행한다. 조건은 **현재 궤적·키 합성+고정 시간 보정 / 시간 보정만 적응형으로 변경 / 키 합성만 가림 방식으로 변경 / 두 변경 결합**이다. 추가 학습·조건 간 추가 전송은 없다. 전체 단계와 단계 내 진행률·경과 시간을 터미널과 로그에 표시한다.

기본 출력은 `outputs/bsu_adaptive_visibility_20261012_v1/`이다. `RESULT.json`·`summary.csv`는 원본 PNG와 전달 MP4의 화질·시간 지표를, `verification.json`은 해시·끝 프레임·프레임 수·FPS·동일 전송 검사를 기록한다. `review/<case>/comparison.mp4`는 **원본 / 현재 결합 / 적응형 시간 / 가림 합성 / 두 변경 결합** 순서다. 동일 명령 재실행은 입력·코드·산출물 해시를 확인하고 완료된 생성·평가를 재사용한다. 변경된 입력·코드로 기존 점수를 재사용하지 않는다.

이 명령은 이전 동결 실험·로컬 데이터·가중치가 필요한 연구 대조다. 원격 WebVid 전체 배치의 기본값을 변경하지 않는다. 새 장비에서 임의 원본을 받는 범용 배치 명령은 [WebVid 실행](#bsu-webvid-benchmark)을 사용한다. [자료 범위](DATA.md#bsu-adaptive-data) · [실측·판정](RESULTS.md#bsu-adaptive-visibility)

<a id="bsu-trajectory-cadence"></a>
### BSU 선택형 보정과 7원본 궤적·시간 대조

완료된 BSU 양끝 2키 창에 기본 `combined_context` 보정을 적용한다. 수신 `metadata.bin`·`key0.png`·`key1.png`와 기존 생성 `frames/`를 사용하며 원본·재전송·재생성은 필요 없다. 패킷의 유료 궤적이 없으면 합성만 생략한다. 여러 키의 전체 영상은 위 WebVid 배치를 사용한다. 수신 메타데이터와 생성 프레임 길이가 같아야 하고 출력은 입력과 별도 폴더여야 한다.

```bash
PYTHONPATH=src /home/sangukbae/anaconda3/envs/lgvsc/bin/python \
  -m semantic_transmission.bsu_refine \
  --received outputs/bsu_adaptive_visibility_20261012_v1/received/transfer_bus \
  --generation outputs/bsu_adaptive_visibility_20261012_v1/generation/transfer_bus/baseline \
  --output outputs/bsu_refine_bus_context
```

`--mode combined`는 새 연결 없는 이전 결합, `--mode cadence`는 수신 키가 필요 없는 시간 보정만 선택한다. `--mode guarded_combined`는 과거 윤곽 검사 합성을 유지하는 연구 옵션이다. 새 기본값은 PNG에서 실제 수신 키를 보존하며 기본 배치와 같은 CRF18·fast·24fps 인코딩을 사용한다. 완료된 동일 입력·코드·옵션의 반복 호출은 해시 확인 후 `COMPLETE_REUSED`로 종료한다. 초기 항구 384장 `cadence` 재현(19.6초)은 당시 코드의 기록이며 새 기본값의 시간·성능 측정이 아니다.

동결된 7원본 실험·후속 대조의 실행은 다음과 같다. 로컬 기존 입력·가중치가 필요하며 독립 장비의 새 원본 실행 명령은 아니다.

```bash
bash scripts/experiment_bsu_trajectory.sh \
  --root outputs/bsu_trajectory_cadence_20261012_v2
bash scripts/experiment_bsu_trajectory_guard.sh \
  --root outputs/bsu_trajectory_guard_20261012_v1
```

첫 명령은 준비·실제 채널·기준선·보정·평가·검증을, 후속은 기존 패킷/기준선/시간 보정을 재사용하고 윤곽 검사만 바꾼다. `--stage refine`은 완료 보정을 검증·재사용한다. 입력·코드가 바뀌면 동결 실행을 거부하며 `frozen_code/`에 원 소스를 보존한다. 원 `RESULT.json`은 초기 검토 대기 기록, 현재 선택은 `ADOPTION.json`, 실제 본 표본은 `AI_REVIEW.json`이다. 첫 폴더의 `review_cadence/<case>/comparison.mp4`는 **원본/기존/선택한 시간 보정**, `review/<case>/comparison.mp4`는 **원본/기존/궤적 결합**이다. 서로 다른 후보의 영상을 혼용하지 않는다. [결과·한계](RESULTS.md#bsu-trajectory-cadence) · [검증 요약](validation/2026-10-12-bsu-trajectory-cadence.json)

<a id="bsu-temporal-support"></a>
### BSU 후속 — 희소/연속 시간 보정 원인 대조

```bash
bash scripts/experiment_bsu_temporal_support.sh
# 완료된 4조건의 해시를 검사하고 추론 없이 재개
bash scripts/experiment_bsu_temporal_support.sh --stage generate
```

기본 출력은 `outputs/bsu_temporal_support_20261012_v1`이며 아래 저주파 주입 v1의 완료 산출물을 요구한다. 같은 수신 패킷·구조 latent·키·캡션·실제 난수와 기존 공간 보정 코드를 재사용한다. 준비 → 희소/연속 × 강도 2수준 생성 → PNG·시간·밝기 평가와 비교 영상 → 조건·산출물 검증 순서다. `--stage prepare|generate|evaluate|verify`를 지원한다. 코드·입력 변경 시 새 `--root`를 사용한다.

`RESULT.json`, `report.html`, `verification.json`, `AI_REVIEW.json`, `review/temporal_diagnostic.png`·`.svg`, `frozen_code/`에 수치·검수·시간 곡선·소스를 보존했다. 이후 손익 재평가에서 `continuous_low`를 저주파 시간 보정 후속 개발 기준안으로 선택했다. 기본 BSU 실행 경로는 아직 교체하지 않았으며, 동결 결과의 `adopted: false`는 당시 판정으로 보존한다. [현재 재평가와 결과](RESULTS.md#bsu-temporal-support) · [당시 검증 요약](validation/2026-10-12-bsu-temporal-support.json)

<a id="bsu-frequency-anchors"></a>
### BSU 후속 — 저주파 구조 주입

```bash
bash scripts/experiment_bsu_frequency_anchors.sh
# 완료된 5조건의 해시를 검사하고 추론 없이 재개
bash scripts/experiment_bsu_frequency_anchors.sh --stage generate
```

기본 출력은 `outputs/bsu_frequency_anchor_20261011_v1`이다. 아래 구조 앵커 v2의 완료 입력·수신 패킷·144×80 구조 latent·기준 출력을 먼저 요구한다. 같은 수신 바이트를 재사용하므로 새 부호화·채널·학습 단계는 없다. 준비 → 5조건 생성 → PNG·시간 지표 평가와 비교 영상 → 난수·조건·산출물 검증을 수행한다. `--stage prepare|generate|evaluate|verify`를 지원한다. 새 주입 코드와 입력을 해시로 동결하며 변경 시 새 `--root`가 필요하다. 부모 실험 코드·산출물은 보존한다.

`RESULT.json`, `report.html`, `verification.json`, `AI_REVIEW.json`, `temporal_diagnostic.json`과 `frozen_code/`에 수치·검수·실패 지점·소스를 기록했다. 정량 실행의 완료와 후속 AI 표본 검토는 독립 의미 검수 완료를 뜻하지 않는다. 2026-10-12 완료본은 구도·시간 품질 손실로 **비채택**이다. [결과](RESULTS.md#bsu-frequency-anchors) · [검증 요약](validation/2026-10-12-bsu-frequency-anchors.json)

<a id="bsu-structure-anchors"></a>
### BSU 아이디어 ①·② — 키 예산 재배분·저해상도 구조 앵커

```bash
bash scripts/experiment_bsu_anchors.sh
# 완료된 복원 산출물만 검사하고 추론 없이 재개 여부 확인
bash scripts/experiment_bsu_anchors.sh --stage generate
```

기본 출력은 `outputs/bsu_structure_anchor_20261011_v2`다. 실제 키/보조 영상 부호화와 예산 선택 → LDPC/16-QAM/AWGN → 수신 복호 → 무학습 6조건 복원 → PNG 평가·비교 영상 순서다. 기존 `outputs/bsu_zea_16s_20261009/reconstruction/event_pairs/run`의 항구 입력·송수신 키·T5 캐시·난수 기록과 설치된 NTSCC/Open-Sora가 필요하다. 학습이나 새 데이터셋 전체 평가 명령이 아니다.

`--stage prepare|channel|receive|generate|evaluate`로 단계를 지정한다. 준비·완료 채널·완료 생성은 입력/코드/산출물 해시를 검증해 재사용한다. `receive`는 키를 다시 복호하고 `evaluate`의 보조 영상 단순 확대 대조는 다시 계산한다. 코드·입력이 바뀌면 새 `--root`를 사용한다. 단계 진행률과 실패는 터미널·`status.json`에 남으며 중단된 생성은 완료한 조건을 재사용한다.

결과는 `RESULT.json`·`report.html`, 전송은 `rate_curve.json`·`channel.json`, 시각 검수는 `AI_REVIEW.json`, 소스 스냅샷은 `frozen_code/`에 있다. 첫 v1의 48→47장 복호 실패 기록은 보존했다. 이번 실행은 LGVSC 예산을 충족했으나 LPIPS가 악화해 **비채택**이다. [결과·판정](RESULTS.md#bsu-structure-anchors) · [검증 요약](validation/2026-10-11-bsu-structure-anchors.json)

<a id="bsu-motion-pilot"></a>
### BSU 캡션 분리·압축 움직임 어댑터 실험

```bash
bash scripts/experiment_bsu_motion.sh
# 위 실험 완료 후: 움직임 정렬 자체와 CNN 학습의 기여도를 분리
bash scripts/experiment_bsu_motion_ablation.sh
```

로컬 기본 출력은 `outputs/bsu_motion_pilot_20261010_v2`이다. 준비 → LDPC/16-QAM/AWGN 검증 → FP32 T5 → 같은 키·실제 난수의 캡션 비교 → 어댑터 학습 → 움직임 대조 복원 → 평가를 순차 실행한다. 현재 항구 2키 실행과 기존 원본 분할·수신 키·텍스트 캐시가 필요하므로 새로운 데이터셋용 통합 실행 명령은 아니다. `--stage prepare|channel|text|caption|train|generate|evaluate`로 단계를 지정할 수 있다. 학습·생성의 완료 영수증은 검증 후 재사용하며, 실패는 `status.json`에 기록한다. 입력·코드가 다르면 새 `--root`를 사용한다.

기본 Qwen 의미 선택기와 기존 결과는 유지한다. 15개 움직임 표본은 구간 내부의 작은 수치 배열이며 전송 키 이미지 개수가 아니다. 항구 전체 패킷은 2,249바이트로 기존 3 LDPC 블록의 2,304바이트 용량 안에 들어간다. 어댑터는 다른 원본의 VAE latent로 학습하고, 수신 시에는 수신 키와 수신 움직임만 사용한다. 생성 뒤 RGB 보간·워핑은 하지 않는다. `RESULT.json`, `training.json`, `channel.json`, `report.html`에서 실제 완료 여부·성능·전송량을 확인한다.

2026-10-10 실행은 추가 CNN 대조와 역사적 LGVSC·22키의 공통 PNG 평가까지 완료했으며 **비채택**이다. 종합 보고서는 `analysis.html`, 기준선 재실행 일치·채널·산출물 검증은 `verification.json`에 있다. [결과와 제한](RESULTS.md#bsu-motion-pilot)을 확인한다. 추가 대조는 첫 명령이 끝난 뒤 실행해 GPU 작업이 겹치지 않게 한다.

`RESULT.json`의 정량 완료 상태와 후속 `AI_REVIEW.json`의 비독립 표본 검토는 별도 기록이다. 완료 명령을 다시 실행했다고 독립 의미 검수가 완료되는 것은 아니다. 재현에 필요한 [원본·분할](DATA.md#bsu-motion-data)과 [PNG/MP4·시간 지표 경계](METRICS.md#bsu-reconstruction-evaluation)를 함께 확인한다. 최초 v1의 실패 계획·로그를 v2 결과로 덮어쓰지 않는다.

[과제 현황](../README.md) · [실험 결과](RESULTS.md)

- 실행 위치: 저장소 루트.
- 별도 자료: 데이터·가중치·기존 결과는 Git에 미포함.

<a id="receiver-history-gate"></a>
## 학습형 수신부 참조 게이트 후보

**실제 학습·평가를 완료했으나 개선이 없어 기본 경로에는 비채택.** 생성기를 고정하고 이전 생성 참조의 일부 수정 여부를 선택하는 385개 파라미터 신경망을 학습한다. 원본 3개·12개 창으로 교사 대조를 만들고 학습에서 제외한 개발 장면 4개에서 검증한다. [결과·한계](RESULTS.md#receiver-history-gate)

```bash
# 기존 수신 키·캡션·원본 분할·체크포인트 검사; 추론·학습 없음
bash scripts/experiment_receiver_gate.sh --check
# 텍스트 준비 → 학습용 두 조건 복원 → 교사 평가 → 게이트 학습 → 평가 복원 → 평가
bash scripts/experiment_receiver_gate.sh
```

기본 출력은 `outputs/receiver_gate_20261009/`다. 새로운 실험은 `--output outputs/receiver_gate_<새이름>`을 지정한다. 명령은 해당 로컬 TVSum/DAVIS·개발 장면의 기존 수신 산출물과 모델을 필요로 하며, 임의 입력 영상용 기본 복원 명령은 아니다. 평가 입력의 원본 픽셀은 생성이나 게이트 추론에 넣지 않는다. 생성기·수신 키·캡션·난수는 대조 조건 간 고정하며 추가 전송·출력 픽셀 보간은 없다.

전체 6단계와 단계 내부 진행률을 터미널에 표시한다. 같은 명령은 입력·코드·산출물 해시를 확인해 완료 단계를 재사용하고, 중간 생성 실패는 완료된 조건별 영수증을 검증한 뒤 재개한다. 실제 코드·입력이 달라지면 새 출력 폴더를 사용한다. `gate.pt`·`training.json`은 실제 학습 결과, `test_metrics.json`·`results.json`·`report.html`은 원래 정량 기록이며 후속 `AI_REVIEW.json`·`review.html`은 별도 시각 검수다. 새로 실행한 정량 완료 상태만으로 의미 검수까지 완료됐다고 표시하지 않는다.

<a id="sender-repair-pilot"></a>
## 송신부 키 예산·움직임 추출 수정 후보

**실험 후보이며 기본 경로에는 비채택.** 16초 Zea 시험에서 전송량은 감소했지만 움직임 오차와 최대 튐은 증가했다. [결과·한계](RESULTS.md#sender-repair-pilot)

```bash
# 완료된 동일 입력 Qwen 실행·모델·캡션 출처 검사만 수행
bash scripts/experiment_sender_repair.sh --check
# 사건 재검토 → 최소 필요 키 선택 → 두 조건 복원 → 정량 비교
bash scripts/experiment_sender_repair.sh
```

기본 부모는 `outputs/lgvsc_qwen_zea_16s_20261008/qwen`, 출력은 `outputs/sender_repair_20261008`이다. 다른 완료 Qwen 실행에는 `--parent PATH --output NEW_PATH`를 지정한다. `--cap`은 키 개수 상한, `--max-gap`은 최대 프레임 간격이며 기본은 부모 키 개수와 24다. 객체 변화·장면 전환·불확실 사건 쌍의 점수를 먼저 보존하고 키 수를 최소화한다. 최대 간격을 장면별로 학습하는 선택기는 아니다.

`selector`는 선택·구간 캡션 변경, `selector_flow`는 동일 키·캡션에서 움직임 추출만 추가 변경한 조건이다. 같은 구간·프레임의 캡션만 해시 검증해 재사용하며 나머지는 고정 Qwen/LoRA로 재추론한다. 두 조건의 수신 키·난수 텐서 일치를 검사하고, 공통 비키 화질·전체 전환 움직임·실제 전송량을 비교한다. 전체 단계 수와 개별 단계 진행률을 별도로 출력하며 초기화 중에는 경과 시간을 표시한다.

같은 명령의 재실행은 완료 단계의 산출물·입력·코드를 검증해 재사용한다. 실패한 단계는 로그·부분 산출물을 보존하고 재개하며 코드/입력 변경 시 새 출력 경로를 사용한다. 자동 완료 상태는 `COMPLETE_QUANTITATIVE_PILOT_VISUAL_REVIEW_PENDING`이다. 이번 실행의 후속 AI 표본 검수는 원래 정량 영수증을 덮어쓰지 않고 `AI_REVIEW.json`·`review.html`에 보존했다. 독립 의미 검수나 범용 채택을 뜻하지 않는다.

<a id="qwen35-caption-holdout"></a>
## 독립 정답 검수와 미사용 원본의 캡션 비교

사용자 승인 후 별도 GPT6-Astra 문맥의 원본 재검수·새 정답 교차 검수·48개 익명 캡션 판정을 완료했다. 새 24구간의 지지 사실 83개를 고정한 비교에서 보강 캡션은 사실 51→54/83, 동작 17/25·오류 7/24 유지다. 추가 학습은 없다. [결과와 한계](RESULTS.md#qwen35-caption-holdout-result) · [자료](DATA.md#qwen35-caption-holdout) · 원본과 실제 캡션（로컬 산출물 `outputs/qwen35_caption_holdout_20261007/report.html`）

```bash
bash scripts/evaluate_qwen35_caption_holdout.sh verify
```

별도 평가자가 같은 폴더의 `source_packet_fresh.json`, `source_packet_train.json`, `source_packet_development.json`을 원본과 대조해 각 `source_review_*.json`을 작성한다. `*.template.json`은 미작성 양식이며 정답이 아니다. 다른 문맥의 평가자가 새 원본 정답을 교차 검수해 `source_reference_cross_audit.json`을 작성한다. 새 원본의 사실·동작 정답과 교차 검수는 모델 출력을 보기 전에 작성해야 하며 지지된 사실만 주 평가에 사용한다. 작성자는 실제 모델·문맥 또는 사람 신원을 `annotator`에 남긴다.

```bash
# 세 원본 검수와 새 정답 교차 검수 파일 작성 뒤에만 실행 가능
bash scripts/evaluate_qwen35_caption_holdout.sh freeze-reference
bash scripts/evaluate_qwen35_caption_holdout.sh generate
bash scripts/evaluate_qwen35_caption_holdout.sh packet
```

`generate`는 이전 Astra 정답 rank 32 Qwen과 완료된 `caption` epoch 2만 같은 입력으로 비교한다. GPU 락·동결 입력·가중치 검증과 저장 결과 재사용을 적용하며 optimizer는 없다. `caption_packet.json`은 조건 이름을 숨긴다. 새 원본 정답 작성자와 다른 검수자가 이 묶음을 원본과 대조해 `caption_review.json`을 작성한 뒤 집계한다.

```bash
bash scripts/evaluate_qwen35_caption_holdout.sh score
```

집계는 수동 판정을 검사할 뿐 의미 정답을 자동 생성하지 않는다. 결과는 `RESULT.json`과 `report.html`에 기록하고 원본 단위 bootstrap 구간도 제시한다. 작은 ClipShots 표본·동일 계열 AI 검수의 한계를 유지하며 학습·기본 경로 변경·복원은 수행하지 않는다.

<a id="qwen35-evidence-repair"></a>
## 중간 근거·순서·움직임 보강 — 학습 및 완료 모델 평가

**2026-10-07 현재 네 조건 모두 사용자 학습 2 epoch·60 update와 개발 16구간 캡션 비교를 완료했다.** 추가 학습 없이 이전 Astra rank 32와 네 마지막 epoch 모델의 80개 캡션을 생성했다. 새 독립 최종 평가가 아니며, 주석 정정 후에도 전반 개선 기준은 통과하지 못했다. [결과·원본 정정](RESULTS.md#qwen35-evidence-repair-result)

완료 모델의 평가·검수 재집계 명령은 다음과 같다. `generate`는 동결된 기존 출력이 있으면 검증 후 재사용한다. `score`는 저장된 수동 원본 검수를 검증·집계하며 자동으로 의미 판정을 만들지 않는다. 학습 optimizer를 실행하지 않는다.

```bash
bash scripts/evaluate_qwen35_evidence_repair.sh generate
bash scripts/evaluate_qwen35_evidence_repair.sh packet
bash scripts/evaluate_qwen35_evidence_repair.sh score
```

평가 결과는 `outputs/qwen35_evidence_repair_v3/evaluation/RESULT.json`, 원본과 전체 캡션은 같은 폴더의 `report.html`이다. 각 학습 조건의 `RESULT.json`은 당시 학습 완료 기록으로 보존하며 최신 평가 상태는 상위/평가 `RESULT.json`을 확인한다. `reference_errata.json`은 잘못된 도시 정지 구도 4항목을 제외한 사후 정정 근거다. 이 평가에서는 기존 최종 평가 원본을 새 테스트로 재사용하지 않았다.

활성 준비본은 `outputs/qwen35_evidence_repair_v3/`, 설정은 [qwen35_evidence_repair_v3.json](../configs/qwen35_evidence_repair_v3.json)이다. v1/v2는 준비 과정의 위치 편향 감사 기록으로 보존하며 학습하지 않았다. 인자 없는 실행도 **데이터 준비만** 한다.

```bash
bash scripts/train_qwen35_evidence_repair.sh prepare
bash scripts/train_qwen35_evidence_repair.sh dry-run
bash scripts/train_qwen35_evidence_repair.sh audit-inputs
```

세 준비 명령은 Qwen 가중치를 로딩하거나 학습하지 않는다. 준비 당시 `audit-inputs`는 CPU에서 실제 프로세서·토크나이저를 사용해 492개 입력, 텍스트 경계, 시각 캐시 격자, A/B/C/D·S/C/U 단일 토큰 여부를 검사했다. 최대 968토큰으로 4,096 한도 내였으며 준비 검사 자체의 실제 Qwen forward/backward·optimizer 갱신·새 체크포인트는 모두 0이었다. 이후 사용자 학습의 최대 할당량은 9.51GiB, 조건별 순수 학습 루프는 4.84~11.20분이었다.

아래는 사용자가 실행한 학습 명령이다. 각 조건은 같은 이전 Astra rank 32 체크포인트에서 독립적으로 시작한다. 완료된 현재 조건은 다시 입력해도 검증 후 건너뛰며, 이번 결과 분석에서는 추가 학습하지 않았다.

네 조건을 한 번에 순차 실행하려면 다음 명령을 사용한다.

```bash
bash scripts/train_qwen35_evidence_repair.sh train all
```

`caption → evidence → judgment → preference` 순서로 별도 프로세스를 실행한다. 조건당 2 epoch·60 update이며 네 조건 전체는 240 update다. 각 조건은 같은 출발 모델을 쓰고 결과를 서로 이어 학습하지 않는다. 오류가 나면 즉시 멈추며, 같은 명령을 재실행하면 기존 완료 조건은 검사 후 건너뛰고 미완료 조건은 마지막 완료 epoch에서 재개한다. 각 조건 사이에 모델 프로세스가 종료되어 GPU 메모리를 반환한다. 마지막 `TRAIN ... 2 60` 뒤에도 검증·저장이 남을 수 있으며, 네 조건 모두 정상 종료했을 때 `ALL_CONDITIONS_COMPLETE`가 출력된다. 단일 명령 추가 시 실제 학습은 실행하지 않았다.

조건별로 실행할 때는 기존 명령을 그대로 사용한다.

```bash
# 보강 캡션만 학습
bash scripts/train_qwen35_evidence_repair.sh train caption
# 같은 캡션 + 중간 단계를 포함한 근거 학습
bash scripts/train_qwen35_evidence_repair.sh train evidence
# 위 조건 + 확인됨/반박됨/판단 불가 학습
bash scripts/train_qwen35_evidence_repair.sh train judgment
# 위 조건 + 정확한 캡션과 누락/모순 캡션의 비교 손실
bash scripts/train_qwen35_evidence_repair.sh train preference
```

- 각 조건 120학습 구간·2 epoch·60 update, rank 32/alpha 64, LR 2e-5, 배치 1·누적 4, NF4다. 캡션 CE는 모든 구간에 적용하고 새 보조 손실은 원본을 다시 검수한 학습 17구간에만 적용한다. 근거 문제는 검수된 모든 단계와 인접 단계 쌍을 사용하고 구간 내 문제 수로 평균한다. 기존 미검수 구간을 자동 음성으로 만들지 않는다.
- `judgment`는 매 구간에서 세 판정 종류를 같은 비중으로 사용하고 epoch마다 같은 종류의 대안을 순환한다. 캡션은 항상 기존 4장, 판정 문제는 주석에 명시된 1/2/4장이다. 동일한 문장도 필요한 장면이 빠지면 `판단 불가`가 될 수 있다.
- 근거 손실은 모든 단계의 정답 경로 확률을 합산하고, 캡션 비교는 평균 토큰 CE의 margin을 사용한다. 여러 Qwen 계산 그래프를 동시에 유지하지 않도록 작은 출력의 기울기를 구한 뒤 RNG를 복원해 한 호출씩 역전파한다. CPU 모형에서 일반 역전파와 기울기 일치를 검사했고, 이후 사용자가 실제 Qwen GPU 학습 네 조건을 완료했다. 자유 캡션의 정확도는 별도 개발 비교 결과를 확인해야 한다.
- 모델·선택기·기존 결과를 덮어쓰지 않는다. 학습 결과는 각 조건 하위 폴더에 저장하며 마지막 완료 epoch의 optimizer·RNG부터 재개한다. 완료 조건은 건너뛴다. 준비 코드/데이터 해시가 달라지면 재검사가 필요하고, 학습 시작 이후 조건이 바뀌면 동일 폴더 재개를 거부한다. 열린 파일 한도 8,192와 기존 GPU 잠금을 적용한다.
- 개발 4원본·16구간은 이미 관찰한 자료다. 추가 배정한 TVSum 원본 전체를 이번 학습에서 제외했으며 출발 체크포인트의 학습 원본과도 겹치지 않는다. 기존 최종 평가 영상은 열거나 이동하지 않았다. 새 독립 최종 평가는 아직 준비되지 않았고, 손실·보조 문제 점수만으로 기본 경로를 바꾸지 않는다.

원본·새 정답 검토（로컬 산출물 `outputs/qwen35_evidence_repair_v3/review.html`） · [자료·한계](DATA.md#qwen35-evidence-repair-data) · [구현](../src/semantic_transmission/qwen35_evidence_repair.py)

<a id="qwen35-grounded-training"></a>
## 문장 학습과 근거·순서 연결 학습 비교 — 사용자 실행

두 조건 모두 기존 **Astra 정답·rank 32 체크포인트**에서 출발한다. 동일한 124학습 구간·4키·순서·학습률로 캡션 문장만 학습하는 조건과, 문장에 근거 프레임·전후 순서 손실을 추가하는 조건을 순차 실행한다. Qwen 본체 하나와 언어부 LoRA를 사용하며 시각 인코더·선택 헤드는 고정한다. 새 시간 어댑터나 세 작업 공동학습은 이번 비교 범위가 아니다.

```bash
cd /home/sangukbae/semantic_transmission
bash scripts/train_qwen35_grounded.sh both
```

**2026-10-06 완료:** 사용자 실행으로 `caption`과 `grounded` 모두 2 epoch·62회 갱신을 완료했다. 근거 조건의 WSL 핸들 한도 오류는 실행기가 자식 프로세스의 열린 파일 한도를 최소 8,192개로 높여 복구했다. 중단 전 미저장 갱신 1회는 복원하지 않았고, 학습 코드·목표는 유지했다. 이제 같은 `both` 명령은 완료된 두 조건을 검증 후 건너뛴다. [원인과 재현](RESULTS.md#qwen35-grounded-runtime) · [자유 캡션 비교](RESULTS.md#qwen35-grounded-result)

`bash scripts/train_qwen35_grounded.sh runtime-check`는 초기 검증과 실제 순서의 3개 누적 배치에서 캡션·근거 역전파를 검사한다. 첫 배치 뒤 Adam 모멘트 크기의 메모리를 확보하되 optimizer 갱신·가중치 저장은 하지 않는다. 보고서는 `outputs/qwen35_grounded_v1/runtime_check_시각.json`에 저장한다. 셸의 hard limit이 8,192보다 작으면 실행기가 모델 로딩 전에 원인을 출력한다. 시스템 전체 설정이나 부모 터미널의 한도를 변경하지 않는다.

- `both`: `caption` 완료 후 같은 출발 가중치로 `grounded`를 실행한다. 두 모델을 동시에 GPU에 올리지 않는다. 개별 실행은 마지막 인자를 `caption` 또는 `grounded`로 바꾼다. 각 조건 2 epoch·62 optimizer update, LR 2e-5, 배치 1·누적 4, NF4·rank 32·alpha 64다.
- `caption`의 목적은 평균 토큰 캡션 CE다. `grounded`는 같은 CE에 **0.3 × 근거 선택 손실**을 더한다. 사실 문장을 질문으로 주고 Qwen이 A/B/C/D 관측 위치를 답한다. 사건은 먼저/나중 근거를 각각 예측하고 시간 순서가 맞는 대안 쌍의 확률을 높인다. 정답 프레임 번호는 입력 질문에 넣지 않는다. 한 구간·epoch당 정보가 있는 사실 하나를 순환 사용하며, 상태가 모든 프레임에 보이면 해당 상태의 선택 문제는 건너뛴다.
- 두 조건은 입력·업데이트 수가 같지만 보조 문제 때문에 계산량은 다르다. 학습 출발점은 기존에 학습한 LoRA이며 처음부터 새 어댑터를 학습하는 실험이 아니다. 손실에는 중간 모든 사건 단계, 객체 추적, 방향 반례, 카메라 운동 분류 헤드가 직접 포함되지 않는다.
- 검증 12구간에서 캡션 CE와 근거 선택 정확도를 기록한다. 이는 자유 생성 캡션의 사실·동작 정확도가 아니다. 마지막 2 epoch 모델을 사전 지정하며 테스트 점수로 체크포인트를 고르지 않는다. 학습 후 별도 의미 평가가 필요하다. 새 최종 평가 16구간은 학습·캐시·준비 검사에서 열지 않는다.
- 결과 위치: `outputs/qwen35_grounded_v1/{caption,grounded}/`. `status.json`은 진행률, `RESULT.json`은 학습 완료 상태다. 마지막 완료 epoch의 체크포인트·optimizer·RNG를 복원해 재개한다. 중단된 epoch는 다시 실행한다. 완료 조건은 재학습하지 않는다.
- `verify`는 동결 자료 검사, `cache`는 학습/검증 시각 특징 준비다. `preflight`는 실제 자료의 forward/backward와 Adam 상태 크기의 메모리를 검사하지만 **optimizer.step=0, 가중치 저장 없음**이다. 준비 검사와 코드·자료 해시가 맞지 않으면 학습을 거부한다. 학습하려면 GPU 여유 메모리 11,264MiB 이상이 필요하며 실행 중 다른 GPU 작업과 잠금을 공유한다.

추가 학습 없는 관찰문→캡션 진단은 `bash scripts/run_qwen35_reasoning_probe.sh`로 재확인한다. 완료 출력을 해시 검증해 재사용하며, 원본 대조 검수는 별도 저장된 `review_A/B.json`에 의존한다. 평가 집계 명령은 `PYTHONPATH=src:scripts .local/qwen35_env/bin/python scripts/evaluate_qwen35_reasoning_probe.py score`다.

[준비 자료·한계](DATA.md#qwen35-grounded-data) · [학습 코드](../src/semantic_transmission/qwen35_grounded_training.py) · 준비 검사（로컬 산출물 `outputs/qwen35_grounded_v1/preflight.json`）

### 완료 학습의 같은 입력 캡션 비교

```bash
# 기존 출발 모델과 두 조건의 마지막 epoch를 고정하고 28구간 추론
bash scripts/evaluate_qwen35_grounded.sh prepare
bash scripts/evaluate_qwen35_grounded.sh generate

# 저장된 원본 대조 검수를 검증해 수치와 HTML 보고서 재생성
bash scripts/evaluate_qwen35_grounded.sh score
```

`initial`은 **이번 124구간 추가 학습 전의 Astra 정답 rank 32 모델**이다. 원래 무학습 Qwen과의 비교가 아니다. Qwen 본체 하나에 동결 LoRA 세 개를 올리고 구간마다 실행 순서를 바꾼다. 고정 4장·해상도·캡션 프롬프트·160토큰 한도는 같고, 교사 사실은 추론에 넣지 않는다. optimizer 갱신은 없다. 결과는 `outputs/qwen35_grounded_v1/evaluation/`에 저장한다.

`score`는 의미를 자동 판정하지 않는다. 이번 원본 대조 Astra 검수 `review_A/B.json`을 재사용한다. 새 실험은 출력 열람 전 `reference_packet_A/B.json`의 사실·원본을 반대 작성자가 검수하고 `audit-reference`로 고정한 뒤, 추론 완료 후 `packet`으로 익명 묶음을 만들어 별도 검수를 작성해야 한다. 사실별 실제 문장 인용과 오류의 원본 프레임 근거가 필요하다. 생성·검수·출처 해시나 동일 입력 조건이 달라지면 집계를 거부한다. 두 학습 폴더의 `RESULT.json`은 당시 학습 완료 기록이며, 이후 의미 평가의 최신 상태는 `evaluation/RESULT.json`과 상위 `RESULT.json`을 확인한다.

<a id="qwen35-astra-transfer"></a>
## 동일 입력 Astra 정답 × LoRA rank 대조

기존 학습 76구간의 입력·키를 고정하고 기존/Astra 캡션 × rank 8/32의 네 조건을 순차 학습한다. 각 조건은 새 Qwen LoRA 초기화에서 시작하며 같은 rank끼리 초기 가중치와 학습 순서가 같다. 사건·불확실성 주석도 저장하지만 이번 학습에는 캡션 CE만 사용한다.

```bash
# 동결된 교사 자료 검증 → 네 조건 학습 → 새 24구간 추론
bash scripts/run_qwen35_astra_transfer.sh all

# 개별 조건 또는 완료 epoch부터 재개
bash scripts/run_qwen35_astra_transfer.sh teacher_r8
bash scripts/run_qwen35_astra_transfer.sh teacher_r32
bash scripts/run_qwen35_astra_transfer.sh generate

# 완료 체크포인트·초기화·고정 헤드 감사
bash scripts/run_qwen35_astra_transfer.sh audit

# 이미 작성된 원본 정답과 의미 검수 기록 검증·집계
bash scripts/run_qwen35_astra_transfer.sh freeze-reference
bash scripts/run_qwen35_astra_transfer.sh packet
bash scripts/run_qwen35_astra_transfer.sh score
bash scripts/run_qwen35_astra_transfer.sh analyze
```

- 경로: `outputs/qwen35_astra_transfer_v1/`. 2 epoch·44 update·LR 2e-5·배치 1/누적 4·NF4·alpha/rank=2. 네 조건 모두 사전 지정한 마지막 epoch를 사용하며 테스트나 CE 최솟값으로 고르지 않는다. rank 변경은 어댑터 용량 비교이며 본체 층수 변경이 아니다.
- 4장 정답은 `teacher/labels4_A.json`과 `labels4_B.json`으로 동결한다. 평가의 별도 8장 참조는 작성자를 교차한 `reference8_A/B.json`이다. `freeze-reference`, `packet`, `score`는 **외부 모델을 자동 호출하지 않는다**. 실제 원본 검토로 작성한 주석·판정 JSON이 있어야 진행한다.
- 검수자는 `evaluation/packet_A/B.json`과 표시된 원본만 읽고 `review_A/B.json`을 작성한다. 조건 이름은 `mapping_private.json`에 분리한다. 모든 사실에 boolean 판정, 오류에는 실제 문장 인용·원본 프레임·이유가 필요하다. 4장으로 관측 가능한 사실만 주 집계하고, 8장 전체 사실은 보조 집계한다.
- `analyze`는 완전+부분 언급, 데이터/rank 효과의 쌍별 차이, 원본별 차이를 `paired_analysis.json`에 보조 집계한다. 사전 채택 기준이나 판정 원문을 바꾸지 않는다. `audit`는 교사 시트 124개의 원본 픽셀·프레임/시각 라벨도 대조한다.
- 기존 완료 학습은 체크포인트를 검증 후 재사용한다. 중단된 epoch는 마지막 완료 epoch부터 다시 실행한다. 입력·교사·조건·의존 코드를 바꾼 실험은 새 경로와 프로토콜이 필요하다. GPU는 한 번에 한 조건만 사용한다.

네 조건 학습·144개 학생 캡션 생성·168개 익명 캡션 교차 검수를 완료했다. Astra rank 32는 학습 전보다 개선됐지만 기존 정답 대조군보다 오류가 늘어 채택 기준은 미통과했다. 완료 상태는 `RESULT.json`, 입력·가중치 감사는 `training_audit.json`, 원본 대조는 `report.html`에 남긴다.

[완료 결과](RESULTS.md#qwen35-astra-transfer-result) · [자료·한계](DATA.md#qwen35-astra-readiness) · [학습 코드](../src/semantic_transmission/qwen35_astra_transfer.py)

<a id="qwen35-caption-repair"></a>
## 선택기를 고정한 캡션 보완 대조

새 기본 Qwen LoRA에서 캡션 CE 조건과 동작·순서 QA 보조학습 조건을 각각 실행한다. 같은 데이터·순서·학습률·4키를 사용하고 기존 선택 헤드와 시각 인코더를 고정한다. 이전 공동학습 모델은 비교용으로 보존한다.

```bash
# 준비 검증 → 두 조건 학습 → 같은 입력의 네 캡션 모델 비교
bash scripts/run_qwen35_caption_repair.sh all

# 개별 실행 및 재개: 완료 epoch 체크포인트부터 재개
bash scripts/run_qwen35_caption_repair.sh caption_ce
bash scripts/run_qwen35_caption_repair.sh caption_qa
bash scripts/run_qwen35_caption_repair.sh generate
bash scripts/run_qwen35_caption_repair.sh verify
```

- 각 조건 2 epoch·44 update, LR 2e-5, NF4·LoRA rank 8, 배치 1·누적 4. QA 조건은 근거가 있는 동작 질문 CE를 가중치 0.3으로 더한다. 추가 질문은 학습에만 사용하며 비교 추론 프롬프트는 같다.
- 검증 20구간의 공통 캡션 CE가 가장 낮은 체크포인트를 고른다. 학습 전 epoch 0도 후보로 남겨 악화된 가중치가 자동 선택되지 않게 한다. 실제 정확도는 생성 후 원본 사실·동작·오류 검토로 별도 판정한다.
- 출력: `outputs/qwen35_caption_repair_v1/`. 재실행은 완료 학습·출력을 검증 후 재사용한다. 중단된 epoch는 마지막 완료 epoch부터 다시 실행하며, 미완성 체크포인트는 재개하지 않는다. 원본·설정·의존 코드가 바뀌면 새 실행 경로가 필요하다.
- 평가 준비·집계: 환경을 위 셸과 같이 설정한 뒤 `scripts/evaluate_qwen35_caption_repair.py packet` 및 `score`를 실행한다. `score`는 실제 캡션과 해시로 연결된 사실별 `evaluation/review.json` 검토를 요구한다. 생성 완료를 정확도 개선으로 처리하지 않는다.
- 두 조건 본학습과 96개 캡션 검토를 완료했으나 채택 기준은 미통과했다. [완료 결과·후속 동작 질문 진단](RESULTS.md#qwen35-caption-repair) · [데이터 역할과 한계](DATA.md#qwen35-caption-repair-data). 기본 FC-LGVSC 경로는 교체하지 않았다.
- 추가 동작 질문 진단: `scripts/probe_qwen35_caption_actions.py` 및 `scripts/evaluate_qwen35_caption_actions.py packet/score`. 위 셸과 동일한 환경을 사용한다. 기존 진단 14구간만 사용하며 새 평가·재학습·체크포인트 선택은 하지 않는다. 원본 기준 수동 AI 검토가 필요하고, 이미 제공된 검토 기록은 입력·출력 해시가 일치해야 재사용된다. 학습 전 답변 1개가 160토큰에서 잘렸으며 별도 실패로 남겼다.

<a id="qwen35-oracle-diagnostic"></a>
## 새 사건 영상의 선택·캡션 2×2 진단

8원본·14구간에서 학습/정답 키 × 학습 전/일반 공동학습 캡션의 56출력을 완료했다. 아래 명령은 학습하지 않으며 동결된 참조·원본·일반 공동학습 체크포인트를 사용한다. 기존 완료 기록은 검증 후 건너뛴다.

```bash
bash scripts/run_qwen35_oracle_diagnostic.sh prepare
bash scripts/run_qwen35_oracle_diagnostic.sh generate
bash scripts/run_qwen35_oracle_diagnostic.sh verify

# 기존 source-grounded AI 검수 기록을 검증하고 수치/HTML 보고서 재생성
PYTHONPATH=src:.local/qwen35_training_probe_deps .local/qwen35_env/bin/python \
  scripts/evaluate_qwen35_oracle_diagnostic.py score
```

새 출력 위치는 `prepare`와 `generate`에 동일한 `--run outputs/새진단명`을 준다. 같은 동결 조건을 다시 생성하는 경로이며 데이터가 자동으로 늘어나지는 않는다. 생성 후 평가 스크립트의 `packet --run ...`으로 조건 이름을 가린 검수 묶음을 만들고, 원본을 보고 별도의 `review.json`을 작성해야 한다. `score`는 의미를 자동 판정하지 않는다. 미완료 검수·출력 변경·잘못된 인용은 거부한다. 기존 실험의 동결 구현을 바꾼 뒤 같은 출력 위치에 재개하지 않는다.

출력은 `outputs/qwen35_oracle_diagnostic_v1/`의 `protocol.json`, `reference_freeze.json`, `generation_freeze.json`, `review.json`, `evidence_sensitivity.json`, `RESULT.json`, `report.html`이다. GPU 추론에는 기존 두 작업 잠금을 사용한다. [진단 결과·한계](RESULTS.md#qwen35-oracle-diagnostic)

<a id="qwen35-training"></a>
## 단일 Qwen 단계별·공동학습

2026-10-05 현재 데이터·검수 주석·고정 기준선·Qwen 특징 캐시·PEFT가 준비되어 있으며, **기본 경로의 `staged` 본학습은 완료**했다. 헤드 12 epoch·캡션 3 epoch이며, 이후 주석을 재검수한 공동학습은 [별도 실행기](RUN_GUIDE.md#qwen35-fact-training)를 사용한다. 검증 최적 staged 체크포인트로 32구간을 평가했으나 전반적인 정확도 개선은 확인되지 않았다. [학습 결과·평가 정정](RESULTS.md#qwen35-staged-result). 아래 v1 명령과 결과는 원래 조건의 재현용으로 유지한다.

```bash
cd /home/sangukbae/semantic_transmission

# 4번: 사건·선택 헤드 학습 후 캡션 QLoRA 학습 (기본 경로 완료)
bash scripts/train_qwen35_multitask.sh staged

# 5번: 4번의 검증 최적 체크포인트에서 공동학습 시작
bash scripts/train_qwen35_multitask.sh joint
```

- 기본 설정: [qwen35_multitask_v1.json](../configs/qwen35_multitask_v1.json). 헤드 12 epoch → 캡션 3 epoch → 공동학습 3 epoch, 학습 64구간, 배치 1·gradient accumulation 4, Qwen NF4·언어부 LoRA rank 8, 시각 인코더 고정. 8후보 중 양끝 포함 4키를 사용한다.
- 출력: `outputs/qwen35_multitask_v1/{heads,caption,joint}/`. 각 단계의 `status.json`, `RESULT.json`, `latest.json`, `best.json`에 상태와 검증 손실 기준 최적 체크포인트를 기록한다. 체크포인트는 헤드·adapter·optimizer·RNG·epoch/위치·데이터/설정/코드 해시를 포함한다. 완료 이전 체크포인트로 다음 단계를 시작하지 않는다.
- 중단 후에는 같은 명령 뒤에 `--resume`을 붙인다. 완료한 단계는 검증 후 건너뛴다. 저장 주기는 16 update와 epoch 끝이므로, 예기치 않은 종료 시 마지막 저장 이후 계산은 반복할 수 있다.
- 새 실험은 두 명령에 동일한 `--run outputs/새실험명`을 지정한다. 데이터·설정·코드가 바뀐 실험을 기존 체크포인트에 조용히 이어 붙이지 않는다. 키 수·이미지 조건·캡션 지시문 변경에는 별도 기준선이 필요하다.
- 같은 GPU의 다른 실험이 실행 중이거나 모델 로딩 전 여유 메모리가 12,000 MiB 미만이면 이유를 표시하고 종료한다. 실행 중인 다른 작업을 중단하지 않는다.

```bash
# 원본/분할/주석/해시 검사
bash scripts/train_qwen35_multitask.sh verify

# 단계별 학습의 검증 최적 모델로 평가 (별도 출력; 이번 확인에서 완료)
bash scripts/train_qwen35_multitask.sh evaluate \
  --checkpoint outputs/qwen35_multitask_v1/caption/update-000016-e001-n0000 \
  --run outputs/qwen35_multitask_staged_eval_20261005 --split test

# 기능 재검증은 새 경로에서: 본학습 결과와 분리
bash scripts/train_qwen35_multitask.sh smoke --run outputs/qwen35_smoke_new
```

`evaluate`는 평가 정답을 선택/생성에 입력하지 않고 실제 프레임으로 추론한다. 사건 전후 포착은 자동 계산하지만, **캡션의 사실 재현·잘못된 주장은 학습 전과 동일한 기준으로 추가 검토해야 한다.** 생성 파일 존재나 검증 손실 감소만으로 정확도 향상을 선언하지 않는다. FC-LGVSC 전송·복원은 이 명령의 실행 범위에 포함되지 않는다.

새 환경에서 PEFT만 추가할 때는 고정 추론 환경 설치 후 아래를 사용한다. 현재 워크스페이스에는 이미 설치되어 있다.

```bash
uv pip install --python .local/qwen35_env/bin/python \
  --target .local/qwen35_training_probe_deps --no-deps \
  -r environment/requirements-qwen35-training.txt
```

[데이터 출처·범위](DATA.md#training-readiness) · [학습 구조](MODEL_ARCHITECTURE.md#qwen35-training) · 실행 검증（로컬 산출물 `outputs/qwen35_multitask_verified_v1/smoke/RESULT.json`）

<a id="qwen35-fact-training"></a>
### 재검수 주석으로 일반·사실 보존 공동학습 비교

두 조건 모두 기존 검증 최적 staged 모델에서 시작한다. 수정한 학습/검증 주석 v3를 사용하며, 평가 정답은 학습·체크포인트 선택에 넣지 않는다. GPU 하나에서 아래 명령을 순서대로 실행한다.

```bash
# 일반 공동학습 대조군: 전체 캡션 CE + 기존 근거 손실
bash scripts/train_qwen35_facts.sh control --resume

# 사실 근거 보존: 시간 근거 집합 + 근거 없는 목표 제외 + 사실별 손실
bash scripts/train_qwen35_facts.sh facts --resume

# 두 조건의 검증 최적 모델로 실제 4키 선택·캡션 생성
bash scripts/train_qwen35_facts.sh evaluate
```

- 기본 출력은 `outputs/qwen35_fact_joint_v3/`다. `--resume`은 완료 결과를 해시 검증한 뒤 건너뛰고, 중단 상태는 optimizer·RNG·위치까지 복원한다. 재실험은 두 학습 명령과 평가 명령 모두 새 `--run outputs/새경로`를 지정한다. 현재 코드의 시작 모델은 기존 `outputs/qwen35_multitask_v1/caption/best.json`으로 고정되어 있다.
- [설정](../configs/qwen35_fact_joint_v2.json)은 최대 2 epoch·비개선 1 epoch 조기 종료다. 64개 원래 구간을 모두 유지하고 사건 구간을 반복해 epoch당 92항목·23 update를 수행한다. 두 조건의 데이터·순서·학습률·초기 모델은 같다.
- 검증 사건 포착과 시간 사실 근거를 유지하는 후보 중 동일한 고정 근거 캡션 CE가 가장 낮은 모델을 선택한다. 기준을 통과한 갱신이 없으면 초기 staged 모델로 돌아간다. **학습 완료와 새 가중치 채택은 구분한다.**
- 12 update마다 및 epoch 끝에 저장한다. `--stop-after-updates 1`은 해당 실행에서 1회 갱신 후 정상 저장·종료하는 기능 점검 옵션이다. 다음 호출에서 이 옵션을 빼고 `--resume`으로 이어간다.
- `evaluate`는 학습된 선택기의 실제 4장을 사용한다. 원본·사실·오류의 AI 검토는 별도이며, 생성 완료나 검증 CE 감소만으로 정확도 개선을 판정하지 않는다. [구조](MODEL_ARCHITECTURE.md#qwen35-fact-training) · 정답 검토（로컬 산출물 `outputs/qwen35_fact_joint_v3/data_review.html`）

<a id="setup"></a>
## 환경 준비

- 검증 장비: WSL2 Ubuntu 22.04 · RTX 4080 16GB.
- 필수 도구: Linux Conda · NVIDIA 드라이버 · FFmpeg.
- `lgvsc`: 생성·움직임 추출·키프레임 전송·화질 평가 환경.
- `lgvsc-channel`: 디지털 통신 환경.
- `lgvsc-internvl`: 공식 키프레임 선택 환경.
- `.local/metric_v2_env`: 새 지표 평가 환경.

```bash
# 새 환경 설치; 기존 환경은 activate부터 실행
bash scripts/bootstrap_official.sh
source scripts/activate.sh
semtx doctor
python scripts/probe_environment.py

# 짧은 설치 점검; 새 출력 폴더 사용
semtx smoke --selector skim --skim-keyframes 2 --frames 9 --output outputs/setup_smoke_new
```

<a id="local-extraction"></a>
## 로컬 캡션·전체 원본 추출

- Qwen3-VL 자동 배치의 설치·새 실행·상태 확인: [전용 안내](RUN_GUIDE.md#legacy-local-extraction).
- Qwen3.5 NF4/INT8 환경·단일 캡션·고정 비교·문맥 실험: [전용 안내](RUN_GUIDE.md#qwen-local-environment).
- 전체 원본 보정 상태와 미완료 범위: [데이터 기록](DATA.md#full-extraction). 아래 명령은 중단한 **기본 보정본** 재개용이며 현재 실행하지 않은 명령이다.

```bash
PYTHONPATH=src /home/sangukbae/anaconda3/envs/lgvsc-internvl/bin/python \
  scripts/correct_fc_dataset_extraction.py run
```

- 원자적으로 저장한 쌍별 점수를 재사용하며 진행 중이던 한 쌍은 재계산할 수 있다. Qwen 비교를 위해 중단한 기록과 명령은 `outputs/fc_lgvsc_webvid_tvsum_faithful_20261002/paused_for_qwen_caption_comparison.json`이 기준이다.
- `_v2`는 기본 보정본 102편을 연결하고 WebVid 031만 별도 수정한 경로다. 위 명령만으로 `_v2`의 재선택·변경 캡션 검수·전체 완료가 보장되지 않는다. 인계 파일의 남은 작업을 확인한다.
- 문맥 추출 `RESULT.json`의 `full_78_accuracy_evaluated=false`는 추출 종료 시점 기록이다. 후속 AI 검토는 별도 `_accuracy/RESULT.json`과 [통합 결과](RESULTS.md#qwen-context)에서 확인한다. 기존 기록을 덮어쓰지 않는다.

<a id="qwen35-pairs-caption-comparison"></a>
## 사건 쌍 선택 + 보강 캡션 — 두 조건을 한 명령으로 비교

```bash
bash scripts/run_qwen35_pairs_caption_comparison.sh
```

기존 `tv_low_08` 60초·1,440프레임의 **assistant 후보 + InternVL 79키**와 **Qwen 사건 전후 쌍 보존 79키**를 동결 결과에서 재사용합니다. 두 조건 모두 동일한 보강 캡션 LoRA rank 32로 78개 초안을 새로 작성하고, 어댑터를 끈 본체로 재검토합니다. 원본 구간당 4장 표본 규칙·지시문·토큰 한도를 맞추며, 선택에 따라 구간과 실제 표본은 달라집니다. 후보 생성이나 선택기 학습을 다시 수행하지 않습니다. 현재 Qwen 후보 + InternVL 전체 경로를 새로 실행한 대조군은 아닙니다.

명령은 캡션 생성 → UniMatch 움직임 → AWGN 10 dB 송수신 → T5 FP32·마지막 17프레임 참조·조건 충돌/짧은 구간 수정 Open-Sora → 화질·전송량 → 자동 의미 진단 → 대조 보고서까지 실행합니다. 공통 키의 송신/수신 심볼 일치를 검사하도록 AWGN은 원본 프레임 ID별 seed를 사용합니다. 복원은 같은 seed·단계 수를 쓰지만 구간 길이가 달라 동일한 확산 잡음 텐서 비교라고 주장하지 않습니다. 별도 학습·다운로드·기본 경로 교체는 없습니다.

- 기본 출력: `outputs/qwen35_pairs_caption_comparison_20261007/`. `report.html`, `comparison.mp4`(원본 / InternVL / 사건 쌍), `RESULT.json`, `status.json`, 단계별 로그를 저장합니다.
- 중단 후 **같은 명령**을 다시 입력합니다. 완료 단계와 개별 캡션/검토 응답은 해시·입력·모델 역할을 검사해 재사용하고, 실패한 단계 출력은 별도로 보존합니다. 코드·가중치·입력이 바뀌면 기존 결과를 덮어쓰지 않고 중단합니다. 새 조건은 `--root outputs/새_실험명`으로 구분합니다.
- `--check`는 입력·가중치·수신단 준비 상태만 확인하며 모델 추론·출력 생성이 없습니다. `--stop-after captions` 또는 `--stop-after reconstruct`로 범위를 줄일 수 있고, 나중에 옵션 없이 재개하면 나머지 단계를 실행합니다.
- **의미 평가는 비독립 자동 진단입니다.** 기존 Astra 사건 정의로 선택 사건 포착을 재계산하며, 로컬 Qwen 본체가 초당 원본 4장·복원 4장과 구간 캡션을 보고 사실 재현 및 Added/Missing/Distorted 근거를 기록합니다. 기존 Astra 캡션 점수와 직접 비교하지 않습니다. 검토 원문·근거 이미지·모든 응답을 보존하고 독립 검수는 `PENDING`, 할루시네이션 완화 검증은 `false`로 유지합니다.
- 실측(2026-10-07, RTX 4080, 기존 선택 재사용): 첫 기록 단계부터 자동 진단 종료까지 두 조건 합계 **2시간 38분 44초**. 캡션 단계 26분 46초, 자동 의미 진단 12분 30초입니다. 이전 전체 4~6시간은 실행 전 계획값이며, 실측은 한 번의 실행으로 다른 부하·입력의 소요 시간을 보장하지 않습니다.

사용자 실행은 `COMPLETE_INDEPENDENT_REVIEW_PENDING`으로 정상 종료했습니다. 이는 캡션·송수신·복원·자동 진단 완료이며 독립 의미 검수는 남았다는 뜻입니다. 완료 기록 27개와 영상·지표·출처 해시를 재검증했습니다. 후속 분석은 같은 출력 폴더의 `analysis/review.html`과 [결과 문서](RESULTS.md#qwen35-pairs-caption-launcher)에 있습니다. 자동 진단의 캡션 오류 0개에는 실제 놓친 오류가 확인되어 정확도 근거로 사용하지 않습니다.

**원본 LGVSC 대비 성능을 볼 때는 `analysis_lgvsc/review.html`을 사용합니다.** 원본 InternVL SKEM 110키·PLLaVA 결과와 이번 Qwen 사건 쌍 79키·개선 캡션 결과를 직접 비교한 보고서이며, `analysis_lgvsc/comparison.mp4`의 순서는 원본 / LGVSC / Qwen 조합입니다. 기존 `report.html`·`analysis/review.html`은 GPT-6-Astra 후보 + InternVL 혼합 선택과의 보조 비교입니다. 원본 LGVSC 결과·이번 결과를 재사용해 작성했으므로 이 대조를 보기 위해 모델 추론을 다시 실행할 필요는 없습니다. [원본 LGVSC 대비 개선 폭](RESULTS.md#qwen35-pairs-vs-lgvsc)

<a id="default-method"></a>
<a id="qwen35-fc"></a>
## 새 영상에 Qwen 의미 변화 순차 선택 + 개선 캡션 적용

```bash
bash scripts/run_qwen35_pairs_video.sh
```

기본 입력은 Zea 항구 항공 영상의 **전체 길이** 준비본(`webvid_004_10d5cd04`, 494프레임·24fps·576×320)입니다. 2026-10-09부터 Qwen이 직전 채택 키와 후보를 순차 비교해 의미 변화가 확인될 때 키를 추가합니다. **목표 키 개수와 최대 1초 간격은 없습니다.** 처음·끝을 보존하며, 불확실한 판정은 중간 원본을 추가로 확인하고 해결되지 않으면 중단합니다. 이후 채택된 캡션 LoRA 초안·본체 재검토 → AWGN 10dB 송수신 → FC-LGVSC 복원·화질 평가를 실행합니다. 캡션의 4장 표본 규칙과 기존 수신단 보정은 유지합니다.

- 출력: `outputs/qwen35_semantic_zea_20261009/RESULT.json`, `source_vs_output.mp4`(왼쪽 원본 / 오른쪽 복원), `reconstruction/event_pairs/run/receiver/reconstruction/sample_0000.mp4`. `event_pairs` 디렉터리명은 기존 수신부 연결용으로 유지하며 실제 선택 정책은 `qwen35_sequential_semantic_v1`로 기록합니다.
- `--check`: 입력·모델·코드 출처만 검증하며 추론하지 않습니다. `--prepared <source_prepared.json> --root <새 출력 폴더>`로 다른 전체 길이 24fps 준비본을 지정할 수 있습니다.
- `--candidate-stride 4`는 후보 관찰 빈도이며 키 간격을 강제하지 않습니다. 기존 고정 개수 방식은 `--selection-policy fixed_budget --key-budget N`을 명시한 대조에서만 사용합니다. 의미 선택 기본값에 `--key-budget`을 주면 오류로 중단합니다.
- 순차 의미 판정·캡션·T5 인코딩·생성 구간의 처리 수와 진행률을 터미널에 표시합니다. 수신·복원 전체 표시는 **완료한 단계 수 기준**이며 남은 실행 시간의 비율은 아닙니다.
- 같은 명령으로 재개하면 유효한 Qwen 호출과 검증된 완료 단계를 재사용합니다. 독립 의미 검수·원본 LGVSC 대조 실험은 이 명령에 포함하지 않습니다.

기존 16초 준비본으로 **선택만** 확인하는 명령입니다. `--select-only`를 제거하고 같은 입력·출력 옵션으로 실행하면 판정 호출을 검증·재사용한 뒤 캡션·송수신·복원을 진행합니다. `selection_status.json`의 선택 완료는 복원 완료가 아닙니다. 코드·입력이 바뀌면 기존 동결 계획을 고치지 말고 새 출력 폴더를 사용합니다.

```bash
bash scripts/run_qwen35_pairs_video.sh \
  --prepared outputs/lgvsc_qwen_zea_16s_20261008/common/source_prepared.json \
  --duration-limit 16 --root outputs/qwen35_semantic_selection_20261009/zea \
  --select-only
```

<a id="bsu-zea-three-way"></a>
### 현재 BSU 항구 16초 복원과 기존 두 결과 비교

```bash
bash scripts/run_qwen35_pairs_video.sh \
  --prepared outputs/lgvsc_qwen_zea_16s_20261008/common/source_prepared.json \
  --duration-limit 16 --selection-policy semantic \
  --root outputs/bsu_zea_16s_20261009

# 복원·기본 평가 완료 후 기존 LGVSC 및 고정 22키와 비교
PYTHONPATH=src /home/sangukbae/anaconda3/envs/lgvsc/bin/python scripts/analyze_bsu_zea.py
```

2026-10-09 완료본은 2키·384프레임이다. 비교 명령은 기존 영상과 해시가 일치하는 프레임별 화질 기록을 재사용하고, 공통 381프레임의 움직임을 계산한다. 출력은 `comparison_lgvsc_fixed/report.html`·`comparison.mp4`·`RESULT.json`이다. 자동 수치 분석과 후속 AI 시각 검수를 구분하며, 이메일 자동 발송 기능은 이 명령에 포함하지 않는다. [결과와 비교 한계](RESULTS.md#bsu-zea-three-way)

<a id="lgvsc-qwen-16s-baseline"></a>
## 같은 16초 입력으로 LGVSC → 현재 Qwen 순차 실행

```bash
bash scripts/run_lgvsc_qwen_baseline.sh
```

Zea 원본의 첫 **16초·384프레임·576×320·24fps**를 한 번 준비하고, **InternVL SKEM → PLLaVA → UniMatch → NTSCC/AWGN → 공개 수신부 정책의 Open-Sora**를 완료한 뒤 **Qwen 의미 변화 순차 선택 → 개선 캡션 → 현재 수신부**를 실행한다. 10월 8일 고정 22키 결과와 새 선택기의 결과를 구분한다. 기존 선택·캡션·복원 결과를 가져오지 않는다. 두 경로는 같은 준비 MP4/PNG와 NTSCC·Open-Sora 가중치, SNR 10dB, 생성 seed 2025, 채널 seed 42, 30 생성 단계를 사용한다. 첫 실행 후 같은 명령은 검증된 완료 단계와 저장된 선택·캡션 호출에서 재개한다.

- **새 출력:** `outputs/lgvsc_qwen_semantic_zea_16s_20261009/report.html`, `RESULT.json`. 과거 완료본 `outputs/lgvsc_qwen_zea_16s_20261008/`은 보존한다. 원본 LGVSC 구성 출력은 `lgvsc/run/receiver/reconstruction/sample_0000.mp4`, 현재 모델 출력은 `qwen/reconstruction/event_pairs/run/receiver/reconstruction/sample_0000.mp4`다. 보고서의 두 영상은 원래 생성 결과이며 픽셀 보간·혼합을 하지 않는다.
- **진행률:** 전체 3작업의 완료 수·백분율과 단계별 처리 수·백분율·경과 시간을 터미널에 표시한다. 생성은 RFLOW 반복과 완료 구간을 읽는다. 모델 로딩처럼 총량이 없는 작업에는 임의의 진행률·ETA를 만들지 않는다. 상세 로그·실패 기록·완료 파일 해시를 보존하며 실패하면 다음 경로로 넘어가지 않는다.
- **공식 경로의 범위:** 공개 `get_frame.py`처럼 `int(0.2×24)=4`프레임 간격의 96후보를 실제 InternVL에 넣는다. 2키를 강제하거나 최대 1초 간격을 적용하지 않는다. 마지막 후보는 380번이므로 입력 384장 중 마지막 3장은 이 경로의 복원 범위 밖이다. 공개 연결 규칙의 중복 끝 프레임도 유지한다. 실제 출력 길이와 `output_source_indices`를 `lgvsc/run/output_audit.json`에 기록한다. 384장 전체를 복원하는 현재 모델과 동기 비교할 때에는 이 대응표가 필요하다.
- **원본 수신부 정책:** `align=5`, 이전 생성 구간 전체를 VAE로 인코딩한 뒤 마지막 5 latent 참조, 공개 연결 규칙을 사용한다. 현재 경로의 tail17·조건 충돌·짧은 구간 시간표·scoped RNG 보정을 설치하지 않는다. 첫 생성 구간이 17프레임보다 짧으면 현재 보정을 몰래 적용하지 않고 중단한다.
- **재현 한계:** 공통 FFmpeg 정규화, 로컬 BF16 모델 배치/FP32 CPU T5, 공통 프레임별 AWGN·메타데이터 전송 어댑터를 사용한 **공개 알고리즘 설정 기준선**이다. 논문/GIF와 비트 단위 동일 실행은 아니다. 동일 키의 채널 잡음은 일치하지만 구간 길이와 RNG 정책이 달라 확산 잡음 텐서까지 같은 대조는 아니다. 세 작업 완료 상태는 `COMPLETE_RECONSTRUCTIONS_REVIEW_PENDING`이며 버벅임·의미 보존의 개선 판정은 별도다.
- `--check`는 읽기 전용 준비 검사다. `--prepare-only`는 공통 입력만 만들며 이후 같은 명령에서 옵션을 빼면 이어서 실행한다. `--input <16초 이상 원본.mp4> --output <새 폴더>`를 지정할 수 있다. 설정·코드·입력이 바뀌면 기존 실행에 섞지 않고 새 출력 폴더를 요구한다.

<a id="receiver-joint-windows"></a>
## 인접 구간 공동 생성 대조 실험

```bash
bash scripts/experiment_receiver_joint_windows.sh
```

네 장면에서 같은 공통 시간축의 순차 생성과 공동 생성을 새로 실행하고, 해시 검증한 기존 수신부 결과를 함께 평가한다. 새 두 조건은 같은 수신 키·VAE/T5 조건·초기 잡음·시간표·모델과 조건당 60회 호출을 사용한다. 추가 학습과 출력 픽셀 보간은 없다. 원본 영상은 생성 이후 오프라인 평가에서만 읽는다. 최초 시험은 **비채택**이며 기본 수신부를 바꾸지 않는다.

- 출력: `outputs/receiver_joint_windows_20261008/pilot_<프로토콜 해시 앞 12자리>/report.html`. 최초 완료 실행은 `pilot_8acc98b171d8/`이다. 장면별 `comparison.mp4`는 원본 / 기존 수신부 / 공통 시간축 순차 / 공통 시간축 공동 생성 순서다. 패널 입력 경로는 `panel_sources.json`에 남긴다.
- `--check`는 입력·가중치·과거 참조 산출물을 검증한다. `--stage generate` 또는 `--stage evaluate`로 나눌 수 있다. 기존 네 장면의 완료 수신 데이터와 10월 7일 `receiver_temporal_alignment_20261007/paired_v1`의 기준선 산출물이 필요하다.
- 생성 중 전체/현재 조건의 모델 호출 수·백분율·경과 시간, 조건 완료 시 실측 기반 남은 시간, 평가 시 처리 수·백분율을 터미널에 표시한다. 대기 중 별도 GPU/중간 품질 검사는 필요하지 않다.
- 완료 조건은 산출물 해시를 확인해 재사용한다. 입력·코드가 바뀌면 기본 출력 폴더도 바뀌며, 직접 지정한 `--output`에 다른 프로토콜을 덮어쓰지 않는다. 출력별 실행 잠금과 실패 상태를 기록한다.
- 원본 대비 모든 전환의 움직임 오차·프레임 변화 오차와 비키 프레임 화질을 함께 평가한다. 기존 수신부와 새 시간축은 난수 텐서까지 같은 인과 대조가 아니며, 공동 생성 자체의 효과는 새 순차 대조군과 비교한다. [결과와 한계](RESULTS.md#receiver-joint-windows)

<a id="receiver-temporal-alignment"></a>
## 범용 수신부 개선을 위한 내부 시간 조건 대조 실험

```bash
bash scripts/experiment_receiver_temporal_alignment.sh
```

기존 수신 데이터가 있는 네 장면에서 각각 49프레임·두 구간을 **기존 배치 / 끝 위치만 수정 / VAE 시간 문맥까지 수정**으로 새로 생성하고 원본과 평가한다. 출력 픽셀 보간 없이 실제 Open-Sora 조건을 바꾸는 실험이며, 채택된 기본 경로는 아니다. GPU 생성 단계에서는 원본 프레임을 읽지 않는다. 원본은 이후 오프라인 화질·움직임 평가와 비교 영상 작성에만 사용한다.

- 출력: `outputs/receiver_temporal_alignment_20261007/paired_<프로토콜 해시 앞 12자리>/report.html`, 장면별 `comparison.mp4`와 조건별 `reconstruction.mp4`·저장 전 PNG·난수 및 조건 추적 기록. 비교 영상은 왼쪽부터 원본 / 기존 / 위치만 수정 / 시간 문맥도 수정이다. 최초 파일럿은 `paired_v1/`에 보존한다.
- `--check`: 수신 키·텍스트 cache·가중치 해시를 확인한다. `--stage generate` 또는 `--stage evaluate`로 단계를 나눌 수 있다. 기존 로컬 Zea 및 `etri_latest_short3_20261001`의 완료 수신 실행이 필요하다.
- 진행률은 전체 생성 step 수, 장면·조건·구간의 현재 step, 완료 조건 수, 경과 시간과 완료 조건의 실측 기반 남은 시간을 표시한다. 평가에는 별도 처리 수·진행률이 나온다. 모델 준비 시간의 백분율·ETA는 추정하지 않는다.
- 재실행 시 해시를 검증한 완료 조건은 재사용하고 미완료 조건부터 재생성한다. 기본 출력 폴더는 입력·코드·가중치 변경 시 자동으로 달라진다. 직접 지정한 `--output`의 프로토콜이 다르면 덮어쓰지 않고 중단한다. 실패와 완료 상태는 `status.json`에 남긴다.
- 짧은 창은 첫 선택 키에서 문맥을 다시 시작한다. Zea는 Qwen 캡션, 나머지는 해당 과거 실행의 고정 캡션을 사용한다. 독립 Added/Missing/Distorted 검수와 전체 길이·다중 seed 검증은 포함하지 않는다. 끝 키 일치도나 낮아진 움직임 수치만으로 개선 판정을 내리지 않는다.

<a id="receiver-motion-bridge"></a>
## 완료한 복원 영상의 움직임 보정 — 수신 키프레임 활용

**과거 실험 재현용·비채택.** 사용자 요청으로 이 보간 방식의 개발을 중단했다. 범용 수신부 개선 명령으로 사용하지 않는다. 기존 산출물과 재현 명령만 보존한다.

```bash
bash scripts/repair_receiver_motion.sh
```

기본 입력은 위 Zea 사건 쌍 실행의 완료 수신단입니다. **이미 수신한 28개 키프레임과 기존 Open-Sora PNG만 사용**해 위치 보정 대조군(`geometry.mp4`)과 수신 참조를 함께 활용한 결과(`hybrid.mp4`)를 만듭니다. Qwen·송수신·확산 생성을 다시 실행하지 않으며 원본 영상은 보정 프로그램에서 읽지 않습니다. 이 명령의 보정은 CPU에서 실행됩니다.

- 출력: `outputs/qwen35_pairs_zea_20261007_temporal_repair/`. 기존 영상은 보존합니다. `hybrid.mp4`가 새 결과이며, 수신 참조와 생성 결과를 결합하는 별도 수신단 경로입니다. Open-Sora의 `align=5`나 캡션 자체를 수정한 결과는 아닙니다.
- **진행 표시:** 정합 구간 수 → 처리 프레임 수 → MP4 저장 프레임 수의 단계별 백분율·경과 시간을 터미널과 `status.json`에 표시합니다. 전체 소요 시간 비율은 아닙니다.
- 같은 명령을 다시 입력하면 입력·코드·완료 산출물 해시를 확인해 재사용합니다. 미완료 보정은 처음부터 다시 처리합니다. 기존 결과의 코드·입력·옵션이 달라지면 덮어쓰지 않고 중단하므로 새 `--output`을 지정합니다.
- 다른 완료 실행: `bash scripts/repair_receiver_motion.sh --run outputs/실험/실행폴더 --output outputs/새_보정`. `receiver/decoder_inputs.json`, `receiver/frames/sample/key_frames_received/`, `receiver/reconstruction/sample_0000_frames/`와 양끝 키를 포함한 연속 시간축이 필요합니다. `--mode geometry` 또는 `--mode hybrid`로 한 조건만 실행할 수 있습니다.
- 키 간격 1초 초과, 낮은 영상 정합 점수, 양방향 불일치, 과도한 변형은 보정을 거부합니다. 두 참조가 다르게 보이는 영역은 생성 결과의 비중을 높입니다. 이 검사는 의미 오류 검출기가 아니며 움직이는 물체·가림·급격한 카메라 이동에 대한 일반화 검증은 남아 있습니다.

Zea에서의 자동 화질·시간 평가와 원본 / 이전 / 보정 비교 영상은 별도 오프라인 검증 산출물입니다. 명령 자체는 원본을 사용하는 평가를 실행하지 않습니다. [결과와 범위](RESULTS.md#zea-motion-repair)

## 새 기본 경로 — Qwen3.5 후보·캡션부터 FC-LGVSC 복원까지

공식 Qwen3.5-9B 고정 revision의 **NF4 로컬 추론**입니다. Qwen이 변화 후보와 캡션을 모두 작성하고, 최종 선택은 기존 InternVL SKEM을 사용합니다. 기존 전체 길이 WebVid·TVSum 준비 inventory와 `.local/qwen35_env`가 필요합니다.

2026-10-07부터 새 `init`은 **보강 캡션 LoRA rank 32를 초안에 기본 적용**합니다. 고정 체크포인트는 `outputs/qwen35_evidence_repair_v3/caption/update-000060-e002-n0000/adapter`, PEFT는 기존 `.local/qwen35_training_probe_deps` 설치를 사용합니다. 후보·재검토에는 같은 본체에서 어댑터를 끕니다. 추가 학습 명령이 아닙니다. 가중치가 없거나 바뀌면 중단하며 기본 Qwen으로 조용히 대체하지 않습니다.

```bash
# 새 폴더에 선택한 영상만 등록. --video-id 생략 시 준비 inventory 전체를 등록.
bash scripts/run_fc_qwen35.sh init --root outputs/my_fc_qwen35 \
  --video-id webvid_011_65f0e6cf

# Qwen 후보 → 순차 SKEM → 보강 LoRA 초안 → 본체 재검토. 같은 명령으로 재개.
bash scripts/run_fc_qwen35.sh extract --root outputs/my_fc_qwen35

# 추출 결과·모델 출처·수신단 조건 배치를 검사; 모델 추론 없음.
bash scripts/run_fc_qwen35.sh check --root outputs/my_fc_qwen35 \
  --video-id webvid_011_65f0e6cf

# 캡션 import → UniMatch → AWGN 전송 → T5 FP32 → 수정된 Open-Sora → 평가.
bash scripts/run_fc_qwen35.sh reconstruct --root outputs/my_fc_qwen35 \
  --video-id webvid_011_65f0e6cf
```

- 기본 복원 출력: `<추출폴더>_reconstruction/<video-id>/`; `--output`으로 새 경로 지정 가능.
- `--root` 생략 시 새 기본 폴더는 `outputs/fc_lgvsc_qwen35_caption_repair_20261007`입니다. 이미 생성된 설정·출력에 LoRA를 소급 적용하지 않습니다. 기존 실행을 재현하려면 당시의 동결 코드와 명시적 `--root`가 필요합니다.
- 무학습 본체 캡션을 다시 비교하려면 별도 새 폴더에서 `init --base-caption --root outputs/my_fc_qwen35_base --video-id webvid_011_65f0e6cf`를 사용합니다. 이 옵션은 `init` 전용이며 기존 설정의 덮어쓰기를 허용하지 않습니다.
- 새 캡션의 추론 기록과 수신단 `caption_provenance.json`에 어댑터 ID·두 파일의 SHA256·역할을 저장합니다. 과거 `caption` 대조평가의 평문/160토큰과 현재 기본의 JSON 초안/384토큰+본체 재검토는 서로 다른 실행 조건입니다.
- `--stop-after caption|receive|prepare-text`: 연결 검사만 할 때 사용. 생성 완료로 보고하지 않으며, 옵션 없이 재실행하면 검증된 단계를 재사용합니다.
- 후보 생성과 캡션 추출은 Qwen 환경, SKEM은 기존 InternVL 환경, 전송·복원은 기존 LGVSC 환경을 순차 사용합니다. GPU 공유 잠금을 사용합니다.
- 새 캡션에 대한 T5 CPU FP32 계산 시간을 포함합니다. 과거 assistant 캡션 임베딩을 가져오지 않습니다.
- 원본·선택·샘플·모델 revision·NF4·실제 추론 기록을 검증하고 `caption_provider=local_qwen35`로 기록합니다. 조건을 바꾸면 새 출력 경로를 사용합니다.
- 매우 짧은 첫 구간이 양 끝 키프레임 조건을 담지 못하면 `check`에서 중단합니다. 보호 키프레임을 자동 삭제해 통과시키지 않습니다.
- 과거 Qwen3-VL 배치와 assistant v2 복원 명령은 별도입니다. 아래 명령은 기존 비교 입력을 사용합니다.
- 2026-10-03 연결 검증: `outputs/fc_lgvsc_qwen35_smoke_20261003`에서 7.5초 영상 자동 추출 완료. 복원은 `outputs/fc_lgvsc_qwen35_smoke_20261003_reconstruction/webvid_011_65f0e6cf/RESULT.json`의 `PASS_QWEN35_FC_LGVSC_RECONSTRUCTION`으로 확인합니다. 관련 테스트 122개 통과. [보존한 검증 요약](validation/2026-10-03-fc-qwen35-integration.json)
- 2026-10-07 LoRA 연결 검증: `outputs/fc_lgvsc_qwen35_caption_repair_smoke_20261007`의 7.5초 추출·9키·8캡션과 `reconstruct --stop-after caption`까지 통과했습니다. 관련 테스트 196개, GPU 초안/본체 역할 전환·원문 동일성·가중치 동결 확인. 검증 결과（로컬 산출물 `outputs/fc_lgvsc_caption_adapter_integration_20261007/RESULT.json`）. 최종 복원 완료 기록이 아닙니다.

<a id="caption-v2"></a>
<a id="qwen35-feedback"></a>
## Qwen 단일 모델 상호 피드백 추출·Astra 기준 평가

이 실행기는 후보 관찰·키프레임 선택·캡션을 Qwen3.5-9B NF4 하나로 처리합니다. `run_fc_qwen35.sh`의 InternVL 선택 경로와 별도입니다. 현재 대상은 기존 Qwen 캡션 실험과 동일한 `tv_low_08` 전체 60초이며, 한 번의 공통 1차 출력을 피드백 전후 비교에 재사용합니다. 수신단 복원은 이 실험 범위에 포함하지 않습니다.

```bash
# 새 출력 경로에 준비된 24 FPS baseline을 등록
bash scripts/run_qwen35_feedback.sh prepare --root outputs/my_qwen35_feedback \
  --baseline outputs/etri_60s_tv_low_08_42057b2ee8ed/baseline --video-id tv_low_08

# 모델 한 번 로딩 → 관찰 → 선택 → 캡션 → 추가 원본 검토 → 수정
bash scripts/run_qwen35_feedback.sh run --root outputs/my_qwen35_feedback

# 원본·모델·호출 원문·구간 경계·순차 선택 재현 검사, GPU 추론 없음
bash scripts/run_qwen35_feedback.sh check --root outputs/my_qwen35_feedback

# 검증된 선택 PNG와 모델 출처를 별도 conditions/ 폴더로 내보내기
PYTHONPATH=src .local/qwen35_env/bin/python -m semantic_transmission.qwen35_feedback_export \
  --root outputs/my_qwen35_feedback
```

`plan.json`은 원본·코드·프롬프트·수정 예산을 고정합니다. `ledger.json`에는 원본 근거를 포함한 관찰, `forward.json`과 `feedback.json`에는 각 방식의 키프레임·구간 캡션·순차 선택 이력, `feedback_requests.json`에는 수정·보류 요청을 저장합니다. `calls/`와 `failed_attempts/`에 실제 모델 응답·시간·토큰·VRAM과 모든 실패를 보존합니다. 같은 입력의 완료 호출을 재개할 때는 모델을 다시 실행하지 않습니다.

피드백 요청은 개별 검증합니다. 제공하지 않은 프레임이나 다른 캡션 구간을 근거로 삼은 요청은 이유와 함께 제외합니다. 전체 JSON 응답이 세 번 연속 유효하지 않으면 해당 창을 `AUDIT_FAILED_UNAPPLIED`로 기록합니다. `feedback_trace_audit.json`은 실제 응답에서 적용된 사건·후보·캡션 요청까지 다시 검사한 결과입니다.

`conditions/forward/`와 `conditions/feedback/` 내보내기는 선택 원본 PNG, 키프레임 인덱스, 캡션과 출처를 제공합니다. 이 실험은 표본 정책과 선택기가 달라 기존 네 표본·InternVL용 복원 importer에 자동 연결하지 않습니다.

Astra 평가는 자동으로 모델명을 붙인 자체 점수가 아닙니다. 사용자 요청으로 GPT6-Astra가 원본만 보고 기준을 작성·고정한 후, 방법 이름을 가린 캡션을 실제 원본과 대조합니다. `semantic_transmission.qwen35_feedback_eval`은 기준 해시와 평가 입력을 동결하고, 모든 구간·정확한 오류 인용·사실 ID를 검증해 집계합니다. 추출기에는 이 기준·평가 결과를 전달하지 않습니다.

- 실행 결과 위치: `outputs/qwen35_feedback_20261003/`.
- 독립 원본 AI 기준: `astra_reference/reference.json`, `astra_reference/freeze.json`.
- 평가 준비·집계: `evaluation/protocol.json`, `evaluation/packets/`, `evaluation/judgments/`.
- 완료 시 결과와 원본 대조 화면: `evaluation/RESULT.json`, `evaluation/report.html`.
- 캡션 구간은 반개구간 `[시작, 다음 키프레임)`입니다. 마지막 원본 프레임은 종단 키프레임으로 보존합니다.
- 피드백 전후는 원본·모델·1차 결과가 같지만 추가 이미지와 계산량은 다릅니다. 이전 Qwen 문맥 실험은 프롬프트·표본·구간도 다르며 기존 키프레임을 사용했으므로, 피드백만의 효과로 해석하지 않습니다.
- 2026-10-04 완료: 피드백 전후 오류 구간 23→27, 사실 재현 76→77/123, 사건 포착 동일·추론 시간 증가. [평가 결과와 채택 보류 근거](RESULTS.md#qwen35-feedback-result)

<a id="qwen35-state-delta"></a>
## 상태·변화 캡션 / 예산 선택 / 제한적 대응 검사

기존 `outputs/qwen35_feedback_20261003/`의 원본과 152키 비교 조건을 읽고 별도 출력에 네 조건을 실행합니다. 모델 하나를 로딩해 관찰과 캡션을 순서대로 처리합니다. 선택은 CPU 계산이며 추가 신경망을 사용하지 않습니다.

```bash
# 새 출력에 계획 고정; 완료된 같은 경로를 덮어쓰지 않음
bash scripts/run_qwen35_state_delta.sh prepare --root outputs/my_state_delta
bash scripts/run_qwen35_state_delta.sh run --root outputs/my_state_delta
bash scripts/run_qwen35_state_delta.sh check --root outputs/my_state_delta

# --arm fixed_state|budget_state|uniform_state|budget_checked 로 개별 재개 가능
# 익명 평가 패킷 준비: 완료한 10초 구간부터 동결, GPU 추론 없음
PYTHONPATH=src .local/qwen35_env/bin/python -m semantic_transmission.qwen35_state_delta_eval \
  prepare --root outputs/my_state_delta

# 실제 GPT6-Astra 평가 JSON이 judgments/에 작성된 뒤 집계
PYTHONPATH=src .local/qwen35_env/bin/python -m semantic_transmission.qwen35_state_delta_eval \
  report --root outputs/my_state_delta
```

`fixed_state.json`, `budget_state.json`, `uniform_state.json`, `budget_checked.json`에 키프레임·캡션·실제 추론 연결을 저장합니다. `selection.json`은 점수와 제약 선택의 재현 자료입니다. `calls/`와 `failed_attempts/`는 성공·실패 원문과 비용을 모두 보존합니다. `check`는 의미 정확도 평가가 아닌 출처·규칙·출력 일치 검사입니다.

현재 실험 경로는 `outputs/qwen35_state_delta_20261004/`입니다. 평가는 기존 Astra 원본 기준의 해시를 유지합니다. 같은 문장·같은 구간만 실제 이전 Astra 판정을 재사용하며, 변경된 문장은 새 판정을 요구합니다. 키프레임 수를 제한한 추출 실험이므로 송수신·복원 품질과 실제 전송량은 별도 검증 사항입니다. [구조와 논문 적용 범위](MODEL_ARCHITECTURE.md#qwen35-state-delta)

래퍼는 원문의 문장·근거·불확실성 값을 유지하는 JSON 중첩 정규화와 모든 문장 ID를 명시하는 대응 검사 스키마를 적용합니다. 각 확장은 `syntax_recovery_extension_v3.json`, `correspondence_schema_extension.json`에 코드 해시로 선언됩니다. 원 계획·성공 원문·실패 원문·실행 단계는 보존하며, 동결된 코드가 달라지면 검증을 거부합니다. 기존 실행 중 코드 변경 후 조용히 이어 쓰지 말고 별도 출력 계획을 사용합니다.

2026-10-04 네 조건 검증 완료: 예산 79키는 균등 79키보다 사건 포착 7→12/18, 오류 구간 20→17/60, 사실 재현 77→75/123입니다. 제한적 검사는 출력 변화 없이 추가 생성 55.9초가 들었습니다. [결과·해석](RESULTS.md#qwen35-state-delta-result) · 원본 대조 HTML（로컬 산출물 `outputs/qwen35_state_delta_20261004/evaluation/report.html`）

<a id="qwen35-event-pairs"></a>
## 사건 전후 쌍 보존 / 겹친 관찰

기존 `outputs/qwen35_state_delta_20261004/`의 원본·관찰·캡션 규칙을 고정하고, 사건 쌍 선택과 겹친 관찰 추가의 두 조건을 별도 출력에 실행합니다. SciPy 1.15.3의 CPU HiGHS 정수 최적화를 사용하며 GPU 모델은 같은 Qwen3.5-9B NF4 하나입니다.

```bash
bash scripts/run_qwen35_event_pairs.sh prepare --root outputs/my_event_pairs
bash scripts/run_qwen35_event_pairs.sh run --root outputs/my_event_pairs
bash scripts/run_qwen35_event_pairs.sh check --root outputs/my_event_pairs

PYTHONPATH=src .local/qwen35_env/bin/python -m semantic_transmission.qwen35_event_pairs_eval \
  prepare --root outputs/my_event_pairs
# pending.json에 나열된 변경 구간만 실제 Astra 원본 대조 판정 후 집계
PYTHONPATH=src .local/qwen35_env/bin/python -m semantic_transmission.qwen35_event_pairs_eval \
  report --root outputs/my_event_pairs
```

실험 경로는 `outputs/qwen35_event_pairs_20261005/`입니다. `pairs_selection.json`, `overlap_pairs_selection.json`에 실제 근거 쌍·선택·최적화 결과, `overlap_ledger.json`에 29개 추가 관찰 원문 연결을 저장합니다. 두 조건의 캡션은 기존 메모만 입력받습니다. 정확히 동일한 이미지·프롬프트 호출만 재사용하며, Astra 판정도 캡션 원문과 시간 구간이 같은 경우만 출처를 검증해 재사용합니다. `diagnosis.json`의 정답 기반 가능성 계산은 추출기 입력이 아닙니다.

2026-10-05 검증 완료: 쌍 보존으로 사건 포착 12→14/18·오류 구간 17→15/60, 사실 재현 75→67/123. 겹친 관찰 추가는 사건 12/18·사실 63/123으로 전반 개선이 없었습니다. 기본 경로는 유지합니다. [원인·결과·비용](RESULTS.md#qwen35-event-pairs-result)

## 기존 assistant v2 복원 — FC-LGVSC + 두 가지 구현 오류 수정

```bash
# 기본: tv_low_08 60초
bash scripts/reconstruct_fc_lgvsc.sh

# 보행 10초
bash scripts/reconstruct_fc_lgvsc.sh --video person_walk

# 준비 검사만; 모델 실행 없음
bash scripts/reconstruct_fc_lgvsc.sh --check
```

- **기본 구성:** 혼합 키프레임·캡션 v2·마지막 17프레임 참조·T5 FP32 저장값 재사용.
- **추가 수정:** 시작·끝 키프레임 조건 충돌 방지 + 2~16프레임 구간의 생성 시간표 정상화.
- **다른 영상:** `--video single_subject`(밀밭 인물), `--video candle_flowers`(등갓). 준비된 4편 지원.
- **실행:** 기존 FP32 수신 자료로 한 번 복원 → 검사 → 화질 평가 → 원본·수정 전·기본 방식 비교. 선택·캡션·전송·T5 계산 재실행 없음.
- **통제:** 기존 VAE·초기·반복 생성 잡음 해시 일치 검사. 생성기 BF16·30단계 유지.
- **표시:** 진행률·GPU 전체 사용률을 한 줄로 갱신. 상세 로그는 `.local/reconstruction_console/fc-lgvsc-*/console.log`.
- **출력:** 기존 FP32 결과와 나란히 `*_fc_lgvsc_default_v1/review.html`·`RESULT.json` 저장. `--output 새경로` 지정 가능.
- **재개:** 같은 명령 사용. 완료 단계는 검증 후 재사용; 중단된 생성 단계는 처음부터 실행.
- **채택 근거:** 보행 10초에서 두 수정 결합 검증 완료. 초반 모자이크 해소·평균 개선, 후반 일부 왜곡 잔존. [결과](RESULTS.md#schedule-fix-person)
- **검증 범위:** 새 명령으로 4편 전체 복원·화질 평가 완료. 평균 개선과 새 왜곡 공존; 독립 할루시네이션 완화는 미검증. [결과](RESULTS.md#fc-lgvsc-four)
- **이전 명령:** `reconstruct.sh` 등은 완료 실험의 해시 검증 때문에 수정 전 비교용으로 보존. 앞으로는 위 명령 사용.

<a id="fc-lgvsc-all"></a>
### 현재 FC-LGVSC로 4편 일괄 복원

```bash
bash scripts/reconstruct_fc_lgvsc_all.sh

# 4편 준비 검사만; 모델 실행 없음
bash scripts/reconstruct_fc_lgvsc_all.sh --check
```

- **순서:** `tv_low_08`(60초) → `single_subject`(약 14초) → `candle_flowers`(약 7.5초) → `person_walk`(10초·기존 이름 `01_person_walk`).
- **방식:** 위 기본 실행기로 한 편씩 복원·검사·평가. 캡션 v2·T5 FP32·마지막 17프레임 참조·두 구현 오류 수정 모두 적용.
- **재사용:** 준비된 키프레임·캡션·수신 자료·T5 저장값 사용. 네 편의 입력을 모두 검사한 뒤 첫 복원 시작.
- **화면:** 전체 진행률·GPU 사용률·현재 영상 한 줄 표시. 진행률은 생성 반복 수 기준이며 남은 시간 비율은 아님.
- **출력:** 개별 기본 명령과 같은 `*_fc_lgvsc_default_v1/` 폴더. 완료 시 4개의 `review.html` 경로 표시.
- **재개:** 같은 명령 사용. 완료 단계는 해시 검증 후 재사용; 실패 시 다음 영상으로 넘어가지 않음. 상세 로그는 `.local/reconstruction_console/fc-lgvsc-all-*/<영상>.log`.
- **별도 저장:** `--output-root outputs/fc_lgvsc_four_new`를 붙이면 그 아래 영상별 새 폴더 사용. 기존 결과는 보존.
- **실행 완료:** 사용자 일괄 실행으로 4편 복원 완료. 생성 합계 56분 53초, 첫 단계부터 마지막 비교 완료까지 약 63분 51초; T5 저장값 재사용. 기존 LGVSC 비교（로컬 산출물 `outputs/fc_lgvsc_four_comparison_20261001/review.html`）

<a id="legacy-bf16"></a>
## 이전 BF16 비교용 — 마지막 17프레임 참조 + T5 재사용

```bash
bash scripts/reconstruct.sh

# 준비 검사만; 모델 실행 없음
bash scripts/reconstruct.sh --check

# 캡션의 T5 변환·저장까지만; 영상 생성 없음
bash scripts/reconstruct.sh --prepare-text-only
```

- 당시 구성: 혼합 키프레임 + AI 캡션 v2 → 매 구간 마지막 17프레임을 참조하는 LGVSC 복원. 두 구현 오류 수정은 미포함.
- 터미널: `복원 진행률:  12.34% | GPU 사용률:  87%` 한 줄 갱신. 생성 78구간×30단계 기준; 준비 중 0%·평가/비교 중 99.99%·정상 종료 시 100%.
- GPU: 사용 중인 GPU 전체 사용률을 1초마다 조회. 다른 프로그램 부하도 포함하며 조회 불가 시 `--%`; CPU에서 T5 준비 중에는 GPU 사용률이 낮을 수 있음.
- 상세 로그: 단계별 `logs/`와 `.local/reconstruction_console/run-*/console.log`에 보관. 실패·Ctrl+C 때 로그 위치 표시; `--check`·`--prepare-text-only`는 기존 안내 출력 유지.
- 현재 대상: `tv_low_08` 60초·79장·78캡션. 다른 영상은 같은 기준으로 선택·캡션을 새로 준비해야 함.
- 사전 자료: 기존 원본·모델·선택·v1/v2 완료 결과 필요; 현재 WSL에 준비됨.
- 실행 내용: 기존 수신 자료 → T5 GPU 사전 계산·저장 → T5 메모리 해제 → 복원 → 평가·비교. 선택·캡션 생성·전송 재실행 없음.
- T5 재사용: 같은 모델·문장·설정이면 저장한 조건 사용; 반복 실행에서 T5 모델 로딩 생략. 저장 위치 `.local/cache/t5_embeddings/`.
- 정밀도: T5만 기존 CPU FP32 → GPU BF16. 문장·움직임 점수·생성 30단계는 유지; 수치·복원 화질의 동일성은 미검증.
- 재개: 같은 명령 입력; 완료 단계는 검증 후 재사용, 중단된 생성 단계는 재실행.
- 시간: 이번 복원 42분 28초, T5 캐시 준비·검사 28초, 평가·비교 포함 단계 합계 46분 8초. 최초 T5 사전 계산 48.7초는 이전에 수행한 별도 비용.
- 상태: 60초 복원 완료, T5 저장값 78개 재사용. 이전 v2 대비 PSNR 개선·다른 지표 4개 소폭 악화; 할루시네이션 완화 입증은 미완료. [결과](RESULTS.md#tail17-t5-result)
- 결과 폴더: `outputs/etri_hybrid_keys_tv_low_08_v1/reconstruction_assistant_captions_v2_tail17_t5gpu_console/`; 복원 완료 후 `review.html`·`RESULT.json` 생성.
- 이전 기록 보존: 표시 실행기가 바뀌어 새 결과 폴더 사용. 기존 `_t5gpu/`의 준비 기록·검증 자료는 보존하며 T5 저장값은 공유.
- 새 폴더 지정: `bash scripts/reconstruct.sh --output outputs/etri_hybrid_keys_tv_low_08_v1/reconstruction_tail17_new`; 기존 결과와 같은 상위 폴더의 새 경로 사용.
- T5 변경 전 비교용: `bash scripts/reconstruct_tail17.sh` → `_v2_tail17/`. 참조 변경 전: `bash scripts/reconstruct_hybrid_v2.sh`.
- 처리 기록: `run/receiver/text_embeddings.json`에 계산·재사용 수, 소요 시간, 모델·조건 해시 저장. 복원 시 `text_embedding_trace.json`으로 전체 구간 사용 검사.
- 이전 결과: 기존 LGVSC / 변경 전 v2（로컬 산출물 `outputs/etri_hybrid_keys_tv_low_08_v1/reconstruction_assistant_captions_v2/analysis_lgvsc/review.html`） · v1/v2 비교（로컬 산출물 `outputs/etri_hybrid_keys_tv_low_08_v1/reconstruction_assistant_captions_v2/caption_comparison.html`）.
- 비교 한계: 기존 v2 대비 참조 방식과 T5 정밀도가 함께 변경됨. 초기 생성 잡음 완전 일치는 미보장; 첫 구간 키프레임 조건 충돌도 별도 문제.
- 기준·한계: [선택·캡션 작성 규칙](../README.md#scope). 한 개발 영상의 결과이며 캡션 단독 효과·일반화는 미확정.

<a id="short3"></a>
## 수정 전 세 영상 FP32 비교용 복원

```bash
# 세 영상을 순서대로 복원
bash scripts/reconstruct_short3.sh

# 준비 검사만; 모델 실행 없음
bash scripts/reconstruct_short3.sh --check

# 한 영상만 복원
bash scripts/reconstruct_short3.sh --video single_subject
```

- 대상: `single_subject` 14.04초 / `candle_flowers` 7.54초 / `person_walk` 10초. 기존 복원과 같은 입력 구간.
- 영상 내용: 밀밭·스카프·인물 / 노란 등갓의 초점 변화 / 벽 앞 보행·방향 전환. `candle-flowers`는 실제 장면과 다른 기존 파일명.
- 복원 완료: 순서대로 키프레임 **20 / 9 / 16장**, 캡션 v2 **19 / 8 / 15개**. 키프레임·캡션 기록（로컬 산출물 `outputs/etri_latest_short3_20261001/review.html`） · 기존 LGVSC 비교（로컬 산출물 `outputs/etri_latest_short3_20261001/analysis_lgvsc/review.html`）.
- 시간: T5+생성은 **12분 / 6분 21초 / 9분 31초**, 세 영상의 명령 전체는 약 **33분 2초**. 미리 수행한 선택·AI 캡션 작성 제외.
- 결과: 세 영상의 PSNR·SSIM·LPIPS·DISTS 평균 개선, 밀밭·보행 CLIP 소폭 악화. 전송량 +39.6% / +192.0% / +443.4%; 보행 0.33초 화면 붕괴와 등갓 4.13초 새 무늬 등은 미해결.
- 구성: 미리 선택한 혼합 키프레임·AI 캡션 v2 → 움직임 추출·AWGN 10dB 전송 → 마지막 17프레임 참조·T5 CPU FP32 저장·생성 30단계.
- 화면: 전체 진행률·GPU 사용률·현재 영상 한 줄 표시. 자세한 로그는 `.local/reconstruction_console/short3-*/`에 저장.
- 재개: 같은 명령 사용. 완료 단계를 해시 검증 후 재사용하며, 실패한 단계만 다시 실행. 선택·캡션 작성은 반복하지 않음.
- 자료: `outputs/etri_latest_short3_20261001/<영상>/selected_frames/`, `assistant_captions/captions_bundle.json`.
- 결과: 각 영상의 `reconstruction_fp32/review.html`. 원본·기존 LGVSC·수정 전 FP32 결과를 같은 시각으로 비교하며 기존 출력의 중복 경계 프레임은 비교용 사본에서 제거.
- 비교 범위: 공통 키프레임의 수신 심벌 재사용, 새 키프레임에는 고정 AWGN. 과거 생성 잡음과 동일한 비교는 아님.

<a id="t5-fp32"></a>
## T5 FP32 복원 — 잡음 고정·기록

```bash
bash scripts/reconstruct_fp32.sh

# 준비 검사만; 모델 실행 없음
bash scripts/reconstruct_fp32.sh --check
```

- 대상: 같은 `tv_low_08` 60초·79장·78캡션; 마지막 17프레임 참조·영상 생성 BF16·30단계 유지.
- T5: 기존처럼 **CPU FP32로 새 계산** 후 저장·재사용. 4.7B 모델의 FP32 가중치는 16GB GPU보다 큼; 최초 준비는 GPU BF16보다 느릴 수 있음.
- 잡음: 수신 키프레임·이전 영상의 VAE 표본, 초기 생성 잡음, 생성 30단계의 추가 잡음을 각각 고정. 시드·모든 잡음 해시를 `run/receiver/generation_noise.json`에 기록.
- 화면: 진행률·GPU 사용률을 한 줄로 표시; 상세 로그는 기존 위치에 보관.
- 결과: `outputs/etri_hybrid_keys_tv_low_08_v1/reconstruction_assistant_captions_v2_tail17_t5fp32_fixednoise/review.html`.
- 재개: 같은 명령 사용. 완료 단계·캡션 캐시는 검증 후 재사용; 중단된 영상 생성은 처음부터 재실행.
- 비교 한계: 과거 실행에는 잡음 기록이 없어 과거 BF16과 동일 잡음 비교는 불가. 새 잡음 기준의 FP32 결과만으로 T5 원인을 확정하지 않음.
- 상태: FP32 준비 7분 22초·영상 생성 37분 54초·평가 완료. 비교 페이지의 영상 크기 목록 표시 오류를 2026-10-01 수정해 전체 완료; 모델·평가 재실행 없이 기존 결과 재사용.
- 분석: 이전 BF16 대비 최종 평균 지표 5개 소폭 개선, 일부 큰 오류 개선과 새 오류 공존. 준비·평가 포함 48분 13초; T5만의 효과는 미확정. [상세 비교](RESULTS.md#t5-fp32-result)

FP32 완료 후 **같은 잡음의 BF16 비교**가 필요할 때만 아래 명령 실행. 두 새 실행의 잡음 해시가 다르면 중단하며, FP32는 CPU·BF16은 GPU여서 계산 장치 차이도 포함됨.

```bash
bash scripts/reconstruct_fp32.sh --t5-precision bf16 \
  --noise-reference outputs/etri_hybrid_keys_tv_low_08_v1/reconstruction_assistant_captions_v2_tail17_t5fp32_fixednoise/run/receiver/generation_noise.json
```

- BF16 결과는 별도 `_tail17_t5bf16_fixednoise/`에 저장. 기존 결과·기존 캐시는 보존.
- 새 FP32/BF16 캐시: `.local/cache/t5_precision_embeddings/`; 정밀도별 별도 저장, 기존 BF16 값을 FP32로 형변환해 대체하지 않음.

<a id="condition-fix"></a>
## 조건 충돌 수정 단독 비교 — 과거 실험 재현용

```bash
# 보행 영상 10초: 같은 입력·잡음으로 충돌 수정안 복원
bash scripts/reconstruct_condition_fix.sh --video person_walk

# tv_low_08 60초
bash scripts/reconstruct_condition_fix.sh --video tv_low_08

# 준비 검사만; 모델 실행 없음
bash scripts/reconstruct_condition_fix.sh --video person_walk --check
```

- **수정:** 조건이 겹치는 구간만 끝 키프레임 위치 변경. 각 영상의 첫 구간 1곳이 대상.
- **재사용:** 기존 수신 키프레임·캡션·T5 FP32 저장값. 키프레임 선택·설명 생성·전송 재실행 없음.
- **비교:** 이전 FP32의 VAE·초기·반복 생성 잡음 해시 일치 검사; 조건 위치 변경 효과 비교용.
- **출력:** 기존 FP32 결과의 옆 폴더 `*_condition_fix/review.html`; 원본·수정 전·수정 후 영상 및 지표 비교.
- **진행:** 진행률·GPU 사용률 한 줄 표시. 같은 명령으로 완료 단계 재사용; 중단된 생성은 처음부터 실행.
- **다른 영상:** `--video single_subject` 또는 `--video candle_flowers`. 현재 두 영상은 조건 충돌이 없어 변경 대상 0개.
- **지원 범위:** 준비된 4편 검사 완료. 압축 칸이 하나뿐인 첫 구간(4프레임 이하)은 실행 전 중단.
- **상태:** 보행 복원·비교 완료. 0.33초 개선과 잔여 깨짐·후반 악화 공존. 현재 기본에는 아래 시간표 수정도 함께 포함. [결과](RESULTS.md#condition-fix-person) · [처리](MODEL_ARCHITECTURE.md#condition-fix)
- **후속 진단:** 남은 초반 모자이크는 생성 시간표 오류로 확인. 아래 별도 명령으로 시간표까지 수정해 비교. [검사](RESULTS.md#vae-short-schedule)

<a id="schedule-fix"></a>
### 시간표 수정 추가 비교 — 과거 실험 재현용

```bash
bash scripts/reconstruct_schedule_fix.sh --video person_walk
```

- **대상:** 완료된 보행 조건 충돌 수정 결과의 수신 자료·캡션·T5·난수 재사용.
- **수정:** 생성기에 들어가는 2~16프레임 구간만 시간표 수정; 현재 보행에서는 첫 9프레임 1곳.
- **검사:** 모든 구간의 생성 시각·고정 조건 보존·잡음 갱신 검사 후 전체 화질 평가.
- **표시:** 기존과 같은 한 줄 진행률·GPU 사용률. `--check`는 모델 실행 없이 준비 검사.
- **출력:** `outputs/etri_latest_short3_20261001/person_walk/reconstruction_fp32_condition_fix_schedule_fix/`.
- **결과:** 보행 10초 완료, 생성 7분 40초. 초반 개선·후반 일부 악화; 분석·비교 영상（로컬 산출물 `outputs/etri_latest_short3_20261001/person_walk/reconstruction_fp32_condition_fix_schedule_fix/analysis/review.html`）.
- **재개:** 같은 명령 재실행 시 검증된 완료 단계 재사용. 새 복원은 [기본 명령](RUN_GUIDE.md#default-method)에서 두 수정을 한 번에 적용.

<a id="long-video"></a>
## 비교용 기존 SKEM·PLLaVA — 60초 실행

- 현재 결과: `tv_low_08` 전체 복원 완료; 의미 오류 검수·완화 효과는 미완료.
- 비교 조건: 576×320 · 24fps · AWGN 10 dB · 생성 30 steps.
- 실행 시간: 기존 사례의 키프레임 선택 약 23시간, 이후 처리 약 112분.

```bash
# 입력 검사 + 짧은 모델 점검
bash scripts/check_etri_60s.sh

# 실제 60초 전체 복원
bash scripts/run_etri_60s.sh

# 기존 tv_low_08 선택 결과를 재사용해 복구
bash scripts/recover_etri_60s.sh
bash scripts/recover_etri_60s.sh --status

# 다른 개발용 영상
bash scripts/run_etri_60s.sh --video tv_medium_02
```

- 사전 확인: `run`·`recover` 명령에 `--dry-run`을 붙이면 추론 없이 검사.
- 재개 범위: SKEM은 후보별, 캡션은 구간별 저장; 나머지는 완료 단계만 재사용.
- 생성 중단: 생성 구간 내부 체크포인트가 없어 미완료 생성 단계는 재실행.
- 실행 관리: WSL·터미널 유지 또는 tmux 사용; GPU 실험은 하나씩 실행.
- 결과 폴더: `outputs/etri_60s_tv_low_08_42057b2ee8ed/`.
- 결과 파일: `RESULT.json`은 판정·수치, `comparison.mp4`는 원본·복원 비교.

<a id="experiments"></a>
## 다른 실험 명령

- 선택 속도 비교: `bash scripts/run_skem_speed_benchmark.sh`
- 17프레임 참조: `bash scripts/check_etri_tail_reference.sh`
- 두 개선안 결합: `bash scripts/check_etri_combined_reference.sh`
- 기존 ETRI 배치: `bash scripts/run_etri_remaining.sh`
- WebVid 비교: `bash scripts/run_webvid_ablation.sh` / `bash scripts/run_webvid5.sh`
- 품질 비교: `bash scripts/run_quality_validation.sh`
- 지표 평가: `bash scripts/run_metric_validation.sh` / `bash scripts/run_metric_revision.sh`

### 키프레임 선택 속도 비교

- 조건: 기존 BF16 / 8비트 / 설명 간략화 / 초당 4장 후보 검사.
- 설명 간략화: 키프레임 선택용 설명만 축약; 전송용 캡션 설정은 동일.
- 범위: `tv_low_08`의 사람·차량·문, 각 25프레임; 출력은 24fps 유지.
- 검사: 선택 시간·실제 복원·전송량; 완료 결과는 재사용.
- 장면전환: 개발 영상 3편에서 후보 포함 여부만 별도 검사; 복원 평가는 별개.
- 결과: `outputs/skem_speed_20260929/review.html`과 `SUMMARY.json`.
- 시간 기준: 세 구간 기준선은 검증한 과거 선택 로그; 부하 변동 때문에 전체 가속 배수는 미확정.
- 별도 재측정: 게임 종료 후 같은 프레임 쌍을 반복; `timing_recheck/`에 측정·실패 이력 저장.
- 8비트 준비: 기존 환경을 바꾸지 않고 의존성을 별도 폴더에 설치.

```bash
"$(python3 -c 'import json; print(json.load(open(".local/settings.json"))["internvl_python"])')" \
  -m pip install --no-deps --target .local/selector_speed_deps bitsandbytes==0.43.3
bash scripts/run_skem_speed_benchmark.sh
```

- 주의: 짧은 개발 실험이며, 60초 검증·기본 선택기 교체를 뜻하지 않음.
- 판정: 12개 복원 완료; 새 오류로 세 방법 모두 채택 보류. [결과](RESULTS.md#selector-speed)

### AI 직접 선택 결과 재사용

- 선택 목록·이유: assistant_selection.json（로컬 산출물 `outputs/etri_visual_keys_20260929/assistant_selection.json`）.
- 원본 키프레임 85장: `outputs/etri_visual_keys_20260929/selected_frames/`.
- 60초 비교·판정: review.html（로컬 산출물 `outputs/etri_visual_keys_20260929/review.html`）; 새 오류로 기본값 대체 보류.
- 선택 목록은 같은 원본·프레임 번호에 한해 재사용; 선택 이후 설명 생성·복원 비용은 별도.

<a id="hybrid-selection"></a>
### 비교용 혼합 키프레임 + PLLaVA / 캡션 v1

- 대상: `tv_low_08` 60초; AI 후보·필수 사건 + SKEM 중복 판단 + 최대 1초 간격.
- 현재 상태: 79장 + AI 캡션 v1/v2로 각각 60초 복원 완료; 같은 79장의 PLLaVA 조건은 미실행.
- 소요 시간: AI 캡션 조건의 후속 처리 약 53분(영상 생성 약 48분); 사전 선택·설명 작성 시간 제외.
- 복원 명령: 저장된 선택을 재사용해 설명·움직임 추출 → 전송 → 복원 → 평가·비교 영상 생성.

```bash
bash scripts/reconstruct_hybrid.sh

# AI가 미리 작성한 78개 설명 사용; PLLaVA 실행 생략
bash scripts/reconstruct_hybrid.sh --captions
```

- 준비 검사만: `bash scripts/reconstruct_hybrid.sh --check`; 모델 실행 없이 입력·경로 검사.
- AI 캡션 준비 검사: `bash scripts/reconstruct_hybrid.sh --captions --check`.
- AI 캡션: 구간별 설명·표본（로컬 산출물 `outputs/etri_hybrid_keys_tv_low_08_v1/assistant_captions/review.html`）; 기존과 같은 4개 프레임 위치를 보고 작성, 복원 성능 우위는 미검증.
- 선택만 추출: `bash scripts/extract_hybrid_keyframes.sh`; 완료된 선택은 검증 후 재사용.
- 재개: 같은 복원 명령 입력; 완료 단계는 재사용, 캡션은 구간별 재개, 미완료 생성 단계는 다시 실행.
- 진행 표시: 단계 시작·완료만 출력; 상세 로그는 아래 결과 폴더의 `logs/`.
- 키프레임: 목록·미리보기（로컬 산출물 `outputs/etri_hybrid_keys_tv_low_08_v1/selection.html`）, `outputs/etri_hybrid_keys_tv_low_08_v1/selected_frames/`.
- 복원 결과: `outputs/etri_hybrid_keys_tv_low_08_v1/reconstruction/` 아래 `review.html`·`RESULT.json`.
- AI 캡션 결과: `--captions` 사용 시 같은 상위 폴더의 `reconstruction_assistant_captions/`에 별도 저장.
- 최신 분석: 수치·오류 표본（로컬 산출물 `outputs/etri_hybrid_keys_tv_low_08_v1/reconstruction_assistant_captions/analysis/review.html`）; 전송량 감소, 새 왜곡으로 대체 보류.
- 복원 영상: 같은 폴더 아래 `run/receiver/reconstruction/sample_0000.mp4`.
- 실행 환경: 기존 WSL·모델·원본·기준선 결과 필요; 별도 환경 활성화 없이 명령 실행.
- 비교 한계: 키프레임이 달라지면 구간·캡션·생성 잡음도 달라짐; 화질·새 오류는 복원 후 별도 검수.
- PLLaVA 비교: 같은 79장으로 위 명령의 옵션 없음·`--captions`를 각각 실행해야 PLLaVA/v1 차이를 비교할 수 있음. 현재 기본 v2 명령은 [위 안내](RUN_GUIDE.md#default-method) 참고.

<a id="migration"></a>
## 다른 PC로 이전

- 복사: 데이터·실험 결과·모델 캐시를 별도 보존.
- 캐시: Hugging Face의 `snapshots` 링크와 실제 `blobs`를 함께 복사.
- 환경: 새 PC에서 Conda·모델 경로 재설정; 가상환경·드라이버 폴더 덮어쓰기 금지.
- 경로: 코드·캐시는 WSL Linux 파일시스템에 배치.
- 지표: 이전 결과·교정값·관측 캐시도 함께 준비.
- 기록: 과거 실험의 선언·경로·해시를 일괄 치환하지 않음.

<a id="verification"></a>
## 구현 검사

```bash
source scripts/activate.sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q --disable-warnings
python scripts/probe_environment.py
```

- 판정: 구현 검사 통과와 모델 복원·완화 성능은 각각 확인.
- 상세 안내: [설치·이전·재개 조건·결과 경로](https://github.com/SangukBae/semantic_transmission/blob/4f566feabe213b870e5c1e441ef9b369b348d589/docs/RUN_GUIDE.md).

<a id="qwen-local-environment"></a>
## 독립 Qwen 환경·입력 크기 실험

Ubuntu 22.04 / RTX 4080 16GB용 별도 환경이다. 기존 LGVSC·PLLaVA·InternVL·Qwen3-VL 환경과 분리되어 있다. 기본 실행은 **공식 Qwen3.5-9B 가중치의 NF4 4비트 추론**이다. 모델 전체를 BF16으로 GPU에 올리는 구성이 아니다.

2026-10-03부터 **FC-LGVSC 후보·캡션 작성용 통합 경로**는 `bash scripts/run_fc_qwen35.sh`를 사용한다. 순차 InternVL SKEM과 기존 수신단을 유지하며, 새 Qwen 결과의 출처 검증·전송·복원을 연결했다. [명령·범위](RUN_GUIDE.md#qwen35-fc). 아래 단일 캡션·문맥 비교 실행기는 별도로 유지한다.

## RTX 4080 학습 가능 범위 — 2026-10-05 실측

같은 날 이후 **실제 AI 검수 주석을 쓰는 단계별·공동학습 실행기**도 구현했다. 시각 특징 캐시, 실제 4키 선택, 캡션 손실→후보·선택 헤드 역전파, 각 단계 2회 갱신과 체크포인트 재개, 저장 모델 추론을 통과했다. 캡션/공동학습 기능 검증의 최대 PyTorch 할당은 약 **8.52/8.55 GiB**였다. 이후 **헤드 12 epoch·캡션 3 epoch의 단계별 본학습도 완료**했으며 캡션 최대 할당은 **8.53 GiB**다. 장치 전체 사용량과 구분한다. 32구간 평가에서는 오류 감소와 사실 누락이 함께 나타나 전반적인 정확도 개선은 확인되지 않았다. 이후 수정 주석 v3로 일반/사실 보존 공동학습을 각각 2 epoch·46 update 완료했으며 두 조건 최대 PyTorch 할당은 **8.56 GiB**였다. 검증 최적 첫 epoch에서 64개 캡션을 실제 생성했다. 일반 공동학습은 일부 사실을 회복했으나 추가 사실 손실의 이득은 없었다. [최신 결과](RESULTS.md#qwen35-joint-result) · [공동학습 실행](RUN_GUIDE.md#qwen35-fact-training). [학습 결과](RESULTS.md#qwen35-staged-result) · [준비된 데이터](DATA.md#training-readiness) · [실행 명령](RUN_GUIDE.md#qwen35-training) · 최초 기능 검증（로컬 산출물 `outputs/qwen35_multitask_verified_v1/smoke/RESULT.json`）

아래 표는 그 이전의 입력 크기별 용량 시험이다.

**제한된 영상 QLoRA의 순전파·역전파·AdamW 갱신을 실제 통과했다.** Qwen3.5-9B NF4, 언어부 rank 8 LoRA(21,639,168개 학습 파라미터), 배치 1, gradient checkpointing, 시각 인코더 고정 조건이다. 양자화하지 않은 고정 가중치는 BF16, norm과 어댑터는 FP32로 유지했다. PEFT 기본 준비 함수의 모든 비양자화 가중치 FP32 변환을 사용한 결과는 아니다. PEFT 0.21.2는 기존 추론 환경 패키지를 바꾸지 않고 `.local/qwen35_training_probe_deps`에 별도 설치했다.

| 입력 | 총 입력 토큰 / 정답 토큰 | 갱신 단계 | 최대 PyTorch 할당 | 장치 전체 표본 최대 |
|---|---:|---:|---:|---:|
| 원본 해상도 4장, 기본 계산 | 906 / 75 | 2, 통과 | 8.86 GiB | 12,994 MiB |
| 원본 해상도 8장, 캡션 위치만 logits 계산 | 1,644 / 75 | 2, 통과 | 9.07 GiB | 13,217 MiB |
| 원본 해상도 16장, 캡션 위치만 logits 계산 | 3,130 / 75 | 2, 통과 | 10.28 GiB | 14,533 MiB |

- 기본 계산 경로의 16장은 역전파에서 **1 GiB 여유를 남긴 프로세스 상한**을 초과했다. 마스킹된 이미지·프롬프트 위치의 어휘 logits 생성을 생략하고 같은 캡션 토큰 교차엔트로피를 계산하자 통과했다. 물리 VRAM의 절대 한계를 확정한 시험은 아니다.
- 추가로 Qwen에서 추출한 실제 영상 특징에 1,049,603개 파라미터 헤드를 붙인 3단계 갱신을 통과했다. 헤드 정답은 용량 시험용 합성 값이며 사건·선택 헤드의 의미 학습을 검증한 것이 아니다.
- WebVid 한 구간의 기존 AI 캡션을 사용한 **용량 시험**이다. 어댑터·체크포인트는 저장하지 않았다. 이 용량 시험만으로 장기 수렴·시각 인코더 학습·긴 정답·공동학습·GRPO·정확도 개선을 검증한 것은 아니다. 이후 공동학습 실행은 위의 별도 결과를 따른다. 전체 BF16 가중치의 GPU 상주 학습과 구분한다.
- 기본 측정（로컬 산출물 `outputs/qwen35_training_readiness_20261005/training_probe.json`） · 상한 초과 기록（로컬 산출물 `outputs/qwen35_training_readiness_20261005/training_probe_extended.json`） · 최적화 후 측정（로컬 산출물 `outputs/qwen35_training_readiness_20261005/training_probe_target_logits.json`） · 재현 스크립트（로컬 산출물 `outputs/qwen35_training_readiness_20261005/probe_training.py`）. 전용 Python으로 스크립트의 `--suite base`, `extended`, `target_logits`를 실행한다. 기존 측정 파일을 덮어쓸 수 있으므로 재시험 전 별도 출력 경로로 복사한다.
- 학습 입력·정답 준비 상태는 [데이터 감사](DATA.md#training-readiness)를 따른다.

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
- 원본 `tv_low_08`의 프레임 `[1, 4, 7, 9]`를 사용한 **CPU NF4 전체 모델 추론 통과**. 결과: cpu_caption_smoke.json（로컬 산출물 `outputs/qwen35_setup/cpu_caption_smoke.json`）.
- CPU 2 threads, 이미지당 픽셀 상한 65,536에서 로딩 약 115초, 생성 약 344초, 출력 27 tokens, `truncated=false`. 이는 CPU 검증 시간이며 GPU 속도 추정값이 아니다.
- 이후 사용자의 명시적 요청으로 SKEM 작업을 재개 가능한 상태로 중단하고 **RTX 4080 GPU에서 78개 캡션 생성 완료**. 출력 잘림·재시도 0개. CPU 표본은 별도의 환경 동작 확인 기록이다.
- GPU 입력·시간·캡션 오류 비교는 [실험 결과](RESULTS.md#qwen-caption), 환경 검증은 GPU_VALIDATION.json（로컬 산출물 `outputs/qwen35_setup/GPU_VALIDATION.json`）에서 확인한다. 독립 정답·복원 품질 검증은 미완료다.

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

이 지시문과 형식 검사는 새로운 일반 실행에 적용한다. 기존 78개 캡션 비교와 48개 상향 설정 시험은 당시 지시문·실행 파일을 보존한다. 새 지시문만의 정확도 효과는 분리하지 않았다. 문맥 전략의 대응 비교는 별도 [실험 결과](RESULTS.md#qwen-context)를 따른다.

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

개발 기본 후보는 **NF4·최대 16장 `detail`**, INT8·최대 16장 `max`는 비교 실험용이다. 6설정·48개 진단 캡션의 시간·메모리·오류와 한계는 [상향 설정 결과](RESULTS.md#qwen-capacity)에 기록한다. 576×320 원본의 표본 검사이며 다른 해상도·긴 입력의 최대 용량을 보장하지 않는다.

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

이 단일 캡션 실행기의 출력은 독립 JSON이며 기존 caption bundle과 다른 형식이다. 새 FC-LGVSC 통합 실행기는 같은 로더로 후보·캡션을 새로 생성하고 전용 importer를 사용한다. 과거 78개 비교 결과가 자동으로 이 경로의 캡션이나 성능 근거가 되지는 않는다. 모델 자체의 우위와 4비트 양자화의 품질 영향은 동일 프롬프트의 PLLaVA 재추론, BF16 대조 및 독립 정답 평가가 필요하다.

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

현재 구간만/직전 최대 2초 문맥 적용의 고정 78구간 실행이다. 두 조건의 이미지 수·토큰 예산을 맞추고 기존 원본 4개 샘플을 유지한다. [처리 구조](MODEL_ARCHITECTURE.md#caption-providers) · [생성·후속 오류 평가 결과](RESULTS.md#qwen-context)

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

GPU 공유 잠금 후 30쌍을 실제 추론하며 `RESULT.json`에 쌍별 시간과 전체 시간 추정을 구분해 저장한다. 전체 영상 선택·품질 평가는 실행하지 않는다. [10월 3일 측정](RESULTS.md#qwen-selector-timing)

공식 참고: [Qwen 모델](https://huggingface.co/Qwen/Qwen3.5-9B) · [Transformers 구현](https://huggingface.co/docs/transformers/en/model_doc/qwen3_5) · [bitsandbytes 양자화](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes).

<a id="legacy-local-extraction"></a>
## 이전 Qwen3-VL 배치와 자동 추출

이 구현은 대화형 assistant가 관찰해서 작성하던 **후보 프레임과 캡션 생성**을 로컬 공개 가중치 VLM으로 대체한다. 기존 FC-LGVSC의 순차 SKEM 판정과 최대 24프레임 간격은 유지한다. 원본 복원, 채널 전송, ETRI Added/Missing/Distorted 평가까지 완료한 결과는 아니다. 대화형 assistant와 동등한 품질도 아직 입증하지 않았다.

<a id="status"></a>
## 이전 자동 추출 경로의 상태 기록 — 2026-10-07

아래는 당시 InternVL 혼합 경로와 Qwen3-VL 배치의 기록이다. 10월 9일 이후 현재 BSU의 기본 선택·최근 완료 상태는 [현재 영상 경로](#qwen35-fc)와 [진행 현황](RESULTS.md#bsu-progress)을 따른다.

- **10월 7일 캡션 변경:** 당시 Qwen3.5 혼합 경로는 보강 LoRA rank 32를 초안에 기본 적용하고, 후보·시각 재검토에는 같은 본체에서 어댑터를 껐습니다. 당시 최종 선택은 InternVL SKEM이며, 현재 BSU의 Qwen 의미 선택과 구분합니다. 새 어댑터의 복원 품질 개선은 별도 검증 대상입니다. [채택·검증 범위](RESULTS.md#qwen35-caption-adoption)
- **10월 3일 연결 기록 — 어댑터 없음:** Qwen3.5-9B NF4가 후보와 캡션을 모두 작성하고 기존 순차 SKEM을 유지하는 경로를 추가했습니다. 출처 검증 importer를 통해 FC-LGVSC 전송·복원에 연결합니다. 7.5초 표본에서 키프레임 9개·캡션 8개 추출, AWGN 송수신·180프레임 복원·화질 평가 완료. [현재 실행 명령과 구분](RUN_GUIDE.md#qwen35-fc) · [검증 기록](validation/2026-10-03-fc-qwen35-integration.json)
- 새 경로의 기존 assistant v2 대비 품질 동등성·독립 할루시네이션 완화·103편 전체 처리는 미검증입니다. 아래 Qwen3-VL 기록은 과거 배치입니다.
- 기존 103편 배치는 Qwen3-VL-8B INT8을 사용했으며 **7편 완료·1편 실패 후 중단** 상태다. 이후 전체 실행 완료 근거는 없다. 배치 상태（로컬 산출물 `outputs/fc_lgvsc_auto_20261002/status.json`） · 중단 기록（로컬 산출물 `outputs/fc_lgvsc_auto_20261002/paused_for_faithful_correction.json`）
- assistant 작성 전체 원본과 순차 SKEM 보정본은 별도 경로이며 현재 미완료다. [데이터별 상태](DATA.md#full-extraction)
- Qwen3.5-9B의 캡션·문맥 비교는 [별도 실행기](RUN_GUIDE.md#qwen-local-environment)로 진행했다. 이 Qwen3-VL 배치를 교체하거나 재개한 결과가 아니다.

## 기존 Qwen3-VL 배치의 처리 범위와 구조

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

기존 수동 `assistant_captions` 묶음으로 가장하거나 author 필드를 바꾸지 않는다. 새 Qwen3.5 경로는 [전용 importer](../src/semantic_transmission/qwen35_conditions.py)로 모델·샘플 출처를 보존한다. 이 importer가 과거 Qwen3-VL 결과나 독립 캡션 JSON도 자동으로 가져온다는 뜻은 아니다.

`keyframe_png_bytes`와 `caption_utf8_bytes`는 산출물 크기다. NTSCC 채널 사용량이나 실제 CBR을 측정한 값이 아니다. 최대 1초 간격과 보호 프레임 증가에 따른 전송 비용도 후속 복원/전송 평가에서 별도 비교해야 한다.

ETRI에는 실제 `run_config.json`과 완료 상태에 근거하여 “로컬 Qwen3-VL 기반 시각 후보·구간 캡션 생성, TransNetV2 장면전환 후보, InternVL2-8B 순차 SKEM을 결합한 자동 전처리”로 기술할 수 있다. 기존 대화형 assistant 결과는 별도 참고 자료이고 자동 모델의 성능 정답이나 학습 데이터로 사용하지 않았다. 표본을 보면서 설정을 정했으므로 해당 표본에서의 확인은 개발 검증이며 독립적인 성능 평가가 아니다. 프롬프트만으로 객체 누락/추가/왜곡이 해결되었다고 보고해서는 안 된다.

검사 명령:

```bash
PYTHONPATH=src .local/fc_auto_env/bin/python -m pytest \
  tests/test_auto_extraction.py tests/test_hybrid_selection.py tests/test_assisted_captions.py -q
```

<a id="legacy-environments"></a>
## 원본 LGVSC 환경 대응

The pipeline spans several large models with mutually incompatible dependencies, so
each stage runs in its own environment. Specs were exported with
`conda env export --no-builds` from the machines we ran on.

| File | Stage(s) | Notes |
|---|---|---|
| `llava.yml` | 01 data prep, 05 baselines, 06 eval | also hosts the captioning/eval helpers |
| `internvl.yml` | 02 SKEM / PSSS keyframes | needs InternVL2-8B weights |
| `pllava.yml` | 02 captioning | needs `pllava-7b` weights |
| `ntscc.yml` | 03 keyframe transmission | needs the `*_psnr.pth` checkpoints |
| `opensora.yml` | 04 decoder | exported from the GPU/decoder server |
| `timesformer.yml` | 07 action recognition | Kinetics-400 |
| `depth.yml` | 07 depth estimation | Depth-Anything-V2 |
| `video_summary.yml` | 07 video captioning eval | BERTScore/BLEU/ROUGE |

```bash
conda env create -f environment/<name>.yml
```

## Caveats for reproduction
- The exported `channels:` may include a Tsinghua (TUNA) mirror — replace with your
  preferred channels if needed.
- `--no-builds` keeps specs portable but not bit-exact; pin exact builds only if you
  hit version conflicts.
- CUDA: these envs target the CUDA toolkit available on our boxes (Open-Sora v1.2 +
  ColossalAI on the decoder server). Adjust `pytorch`/`cudatoolkit` to your driver.

## Large weights (not in git)
| Weight | Size | Where |
|---|---|---|
| `OpenGVLab/InternVL2-8B` | ~16 GB | HuggingFace (or local `$INTERNVL_MODEL`) |
| `pllava-7b` | ~15 GB | HuggingFace `ermu2001/pllava-7b` |
| `ntscc_hyperprior_quality_{1..4}_psnr.pth` | ~132 MB each | our trained checkpoints → `$NTSCC_CKPT` |
| Open-Sora v1.2 (STDiT3 + VAE + T5) | ~20 GB | HuggingFace `hpcai-tech/OpenSora-STDiT-v3` etc. |
| UniMatch `gmflow-scale2-regrefine6-mixdata...pth` | ~30 MB | auto-downloaded by `after_extract.sh` |
| Depth-Anything-V2 | ~1.3 GB | HuggingFace `depth-anything/Depth-Anything-V2-Large` |

## 원본 LGVSC 단계별 명령 참조

이 절은 원본 구현의 실행 경로를 보존합니다. 원본 문서의 학습·논문 결과 언급은 업스트림의 설명이며 이 FC-LGVSC 연구에서 새로 달성한 결과를 뜻하지 않습니다. 기본 FC 실행에는 위 명령을 사용합니다. 각 단계의 상대 명령은 해당 단계 디렉터리 기준입니다.

<a id="legacy-01"></a>
## 01_data_prep

**Env:** `llava`  ·  **Input:** a folder of raw `*.mp4`  ·  **Output:** normalized
videos + per-frame PNGs + `videos.csv`.

```bash
conda activate llava
./get_prepare.sh "$DATA_ROOT" 16x24
```

`16x24` means "≤16 s, 24 fps". The driver runs, in order:

| step | script | result |
|---|---|---|
| 1 | `get_csv.py --root $DATA_ROOT --subname ""` | `videos.csv` of raw clips |
| 2 | `edit_movie.py` | re-encode to 24 fps, ≤16 s into `$DATA_ROOT/16x24/` |
| 3 | `resize_videoframe.sh` | resize to **576×320** |
| 4 | `get_csv.py` (subname `16x24`) | csv repointed to processed clips |
| 5 | `get_frame.py` | dump frames to `$DATA_ROOT/16x24/frames/<video>/` |

> `get_csv.py` and `edit_movie.py` live in the upstream LLaVA working dir in the
> original setup; copies are provided here. They take `--root` / `--subname` and have
> no hardcoded paths.

<a id="legacy-02"></a>
## 02_semantic_encoder

Splits each video into semantic segments and encodes them as `{I_text, I_frame, I_side}`.

## 2a. Keyframe selection (`I_frame`)

### SKIM — fixed interval (`skim/`)  ·  env `internvl`
```bash
python skim/baseline-keyframe.py --csv-path "$DATA_ROOT/16x24/videos.csv" --num-keyframes 4
```
3–4 keyframes per clip, evenly spaced (first and last frame always kept).

### SKEM — semantic-guided via PSSS (`skem/`)  ·  env `internvl`
```bash
# The scripts import InternVL's `internvl_chat` package — put the InternVL repo on
# PYTHONPATH (just `cd`-ing there is NOT enough when the script is given by abs path):
export PYTHONPATH="$INTERNVL_DIR:$PYTHONPATH"
#   --method: pass WITHOUT "key_frames" — the code prepends it (see note below)
#   --threshold: PSSS divergence eta_th (keep consistent with the value in --method)
#   --model_path: defaults to OpenGVLab/InternVL2-8B
python skem/MLM-keyframe-internvl.py \
    --csv-path "$DATA_ROOT/16x24/videos.csv" \
    --method    internvl_diff_0.35 \
    --threshold 0.35 \
    --model_path "$INTERNVL_MODEL"
```
> **Method naming:** the code writes keyframes to `frames/<v>/key_frames<method>/`, i.e.
> it **prepends `key_frames`**. So pass `--method internvl_diff_0.35` to get the folder
> `key_framesinternvl_diff_0.35`, which is the `--method` value the NTSCC and decoder
> stages then expect. (`eta_th` defaults to `0.35` and is set with `--threshold`; the value
embedded in `--method` is only a label and is **not** parsed — keep the two consistent.)
Uses the **relative** score `S_rel = P("No") − P("Yes")` (the paper's choice).
`skem/run.sh` is a small batch runner — set `--method` for your run.

The prompts (`--q1`, `--q2`) default to the PSSS templates: ask InternVL to describe
each frame, then "Determine whether they are similar from the perspective of
<Semantic Focus>, use yes or no to answer."

**Output:** keyframe PNGs in `frames/<video>/<method>/` and `<method>_video_paths.csv`.

## 2b. Caption (`I_text`) + optical flow (`I_side`) (`caption/`)

```bash
conda activate llava        # the driver switches envs internally (llava -> pllava)
./caption/after_extract.sh "$DATA_ROOT" 16x24 frames key_framesinternvl_diff_0.35
```
1. `get_key_video.py` cuts clips at keyframe boundaries → `<method>_video_paths.csv`
2. **PLLaVA** captions each clip (env `pllava`, model `pllava-7b`) →
   `<method>_video_paths_text.csv`
3. **UniMatch** optical flow via `OPENSORA_DIR/tools/scoring/optical_flow/inference.py`
   → `<method>_video_paths_text_flow.csv`

> `after_extract.sh` resolves the PLLaVA / UniMatch tools from `$OPENSORA_DIR` (set in
> `env.sh`) — no manual path edits needed; just `source env.sh` first.

## Patches
None — the InternVL library is used unmodified; only the scripts above are ours.

<a id="legacy-03"></a>
## 03_jscc_transmission

Two channels: keyframes go through the learned **NTSCC** codec; captions and
optical-flow side information go through classical **LDPC + modulation** (Sionna).

## NTSCC — keyframe transmission (`ntscc/`)  ·  env `ntscc`

Built on [wsxtyrdd/NTSCC_JSAC22](https://github.com/wsxtyrdd/NTSCC_JSAC22).

### Setup
```bash
git clone https://github.com/wsxtyrdd/NTSCC_JSAC22.git "$NTSCC_DIR"
cd "$NTSCC_DIR"
git apply /path/to/LGVSC/03_jscc_transmission/ntscc/patches/ntscc_modifications.patch
cp /path/to/LGVSC/03_jscc_transmission/ntscc/main_save.py .
```
The patch covers `config.py`, `main.py`, `net/NTSCC_Hyperior.py`, `requirements.txt`.
`main_save.py` is our inference/saving entry point (new file).

### Weights — what ships vs. what doesn't

> **`SNR=10` is the runnable default.** Only its weight is publicly available; the
> sub-10 dB points require **author-provided weights** that are *not* part of this release.

We trained on 100k frames (lr 1e-4, batch 64, 100 epochs). The checkpoints are **not** in
git (~132 MB each). The `--model` value selects the weight (`model_dict` in `main_save.py`):

| `--model` | Weight | Availability | Resolved from |
|---|---|---|---|
| `10` | `ntscc_hyperprior_quality_4_psnr.pth` | **Released** (NTSCC upstream quality-4) | `$NTSCC_CKPT` |
| `8` / `6` / `4` / `2` / `0` | `SNR=<n>dB_2_0.2_train/.../best_loss_<n>dB.model` | **Not released** (separately trained) | `$NTSCC_MAYU` |

To run a sub-10 dB point you must supply the matching weight yourself and point
`$NTSCC_MAYU` at it (or edit `model_dict`). Out of the box, run with `--model 10`.
See `environment/README.md` for where to place `$NTSCC_CKPT`, and `docs/MODEL_ARCHITECTURE.md`
for the provenance note.

### Run
```bash
#   --model: SNR dB in {0,2,4,6,8,10} -> selects the matching checkpoint
#   --save_frames: bare flag; include to write frames, omit to only compute CBR
python main_save.py -p test \
    --test_path "$DATA_ROOT/frames" \
    --method    key_framesinternvl_diff_0.35 \
    --model     10 \
    --save_frames
```
Writes corrupted-then-reconstructed keyframes to `frames/<video>/<method>_<snr>/`
and logs per-video CBR to `<method>.csv`.

> `config.py` (via the patch) points `train_data_dir`/`test_data_dir` at absolute
> paths — edit them, or override, for your machine. Only needed for (re)training.

## LDPC — text / side-info transmission (`ldpc_sionna/`)  ·  Sionna 0.19.2

`SionnaPlus_sora.py` runs the bitstream of `I_text` + `I_side` through LDPC coding and
modulation over AWGN, inside the Sionna Docker image. No source patch to Sionna is required:

```bash
pip install sionna==0.19.2   # or use the official Sionna Docker image
source env.sh                # exports DATA_ROOT (the --seq_list default)
python SionnaPlus_sora.py --snr 10 --seq_list "$DATA_ROOT/H264_test_10"
```

<a id="legacy-04"></a>
## 04_semantic_decoder

The world model is **Open-Sora v1.2**, used **unmodified**. All LGVSC logic (SFA/DSA
adapters, the masked-generation driver) lives in the standalone scripts in `scripts/`,
which you drop into the Open-Sora `scripts/` directory.

> In our setup this stage ran on a separate GPU server (env `opensora`). The frames
> and the `<method>_video_paths_text_flow.csv` from stages 02–03 must be copied there
> first (e.g. with `rsync`/`scp`).

## Setup
```bash
git clone https://github.com/hpcaitech/Open-Sora.git "$OPENSORA_DIR"
cd "$OPENSORA_DIR"
git checkout bf4d6673
# install per Open-Sora instructions into the `opensora` env (environment/opensora.yml)
cp /path/to/LGVSC/04_semantic_decoder/scripts/* scripts/
```

## Run (SKEM + DSA — the main scheme)
```bash
conda activate opensora
cd "$OPENSORA_DIR/scripts"
./run.bash  <csv_name>  <save_dir>  <method>  <root_dir>
# e.g.
./run.bash key_framesinternvl_diff_0.35_10_video_paths_text_flow \
           key_framesinternvl_diff_0.35_10_save_dir_individual \
           key_framesinternvl_diff_0.35_10 \
           "$DATA_ROOT"
```
`run.bash` splits the master CSV into one file per video, then calls
`mydemo_new_align_sh.py` per video with the v1.2 sampling config
(`configs/opensora-v1-2/inference/sample.py`, `--num-frames 4s --resolution 576
--aspect-ratio 5:9 --aes 6.5`). Output mp4s land in `<root_dir>/<save_dir>/`.

## Scripts
| script | role |
|---|---|
| `mydemo_new_align_sh.py` | the decoder; **DSA** = per-segment `num_frames_ls` from real keyframe spacing |
| `config_utils.py` | config parsing for the above (adds `root_path`/`csv_path`/`method`) |
| `run.bash` | per-video driver (one scheme/SNR) |
| `run_text.bash` | **Text-Only** baseline (caption-conditioned, no keyframe refs) |

**SFA (static) vs DSA (dynamic):** DSA derives each segment's length from the keyframe
indices, so variable-length segments are regenerated at their true length and
concatenated. The static path (fixed segment length, the world model's inherent
behavior) corresponds to the SKIM+SFA configuration.

<a id="legacy-05"></a>
## 05_baselines

Source-code the video with a standard codec, then transmit the bitstream through the
LDPC channel of `03_jscc_transmission/ldpc_sionna`. Compared against LGVSC at matched
SNR in the paper.

## H.264 (`h264/`)  ·  env `llava`
```bash
source env.sh                   # exports VIDEO_FOLDER (defaults to $DATA_ROOT/16x24)
python h264/h264compress.py     # reads $VIDEO_FOLDER; no manual path edits
```
`preprocessing.py` and `video_bgr2rgb.py` are color/format helpers.

## H.265 / HEVC (`h265/`)  ·  env `llava`
```bash
source env.sh
python h265/h265compress.py     # reads $VIDEO_FOLDER; no manual path edits
python h265/convert_yuv_to_mp4.py
```

## Codec parameters (from the paper)

| SNR (dB) | Codec | QP | GOP | LDPC rate | Modulation |
|---|---|---|---|---|---|
| 6  | H.264 / H.265 | 51 / 51 | 400 / 400 | 3/4 | 16-QAM / 4-QAM |
| 8  | H.264 / H.265 | 51 / 51 | 80 / 50  | 1/2 | 16-QAM |
| 10 | H.264 / H.265 | 51 / 50 | 20 / 13  | 2/3 | 16-QAM |

A cliff effect occurs below 6 dB. After compression, transmit with
`03_jscc_transmission/ldpc_sionna/SionnaPlus_sora.py`.

## DVST
The DVST baseline reported in the paper used a collaborator's implementation and is
**not** included or redistributed here. To reproduce it, request the code from the
**original DVST authors** — Wang et al., "Wireless Deep Video Semantic Transmission,"
*IEEE JSAC* 2023 ([paper](https://arxiv.org/abs/2205.13129)); we used `eta = 0.2`,
`lambda = 128` (minimum-CBR operating point).

<a id="legacy-06"></a>
## 06_evaluation

**Env:** `llava` (metrics need CLIP/LPIPS/DISTS).

## Video-quality metrics
```bash
python final_score.py \
    --reference_video_folder "$DATA_ROOT/16x24" \
    --result_video_folder    "$DATA_ROOT/<ntscc_recon_dir>" \
    --generated_video_folder "$DATA_ROOT/<method>_save_dir_individual" \
    --output_csv comparison_scores.csv
```
Computes **CLIP, PSNR, SSIM, LPIPS, DISTS** between the original, the NTSCC-reconstructed
keyframes, and the Open-Sora generated video. Writes `comparison_scores.csv` into the
generated folder.

`final_score_calc_100.py` is the 100-frame aggregation variant used for the paper's main
table; it takes the same folder arguments.

<a id="legacy-07"></a>
## 07_downstream

Run the **reconstructed** videos through off-the-shelf task models and compare against
the originals. Demonstrates that LGVSC preserves task-relevant semantics with no
task-specific fine-tuning.

## Action recognition — TimeSformer (`timesformer/`)  ·  env `timesformer`

Built on [facebookresearch/TimeSformer](https://github.com/facebookresearch/TimeSformer),
evaluated on Kinetics-400 (we used 14 clips).

```bash
git clone https://github.com/facebookresearch/TimeSformer.git "$TIMESFORMER_DIR"
cd "$TIMESFORMER_DIR"
git apply /path/to/LGVSC/07_downstream/timesformer/patches/timesformer_modifications.patch
cp /path/to/LGVSC/07_downstream/timesformer/get_video_lable.py .

python tools/run_net.py --cfg configs/Kinetics/TimeSformer_divST_8x32_224_TEST.yaml
```
The provided `TimeSformer_divST_8x32_224_TEST.yaml` is our test config; the patch
adapts the Kinetics dataloader / test loop. `get_video_lable.py` builds the label list.

## Depth estimation — Depth-Anything-V2 (`depth_anything_v2/`)  ·  env `depth`

Library is used **unmodified**; only `score.py` is ours.
```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2.git "$DEPTH_DIR"
cp /path/to/LGVSC/07_downstream/depth_anything_v2/score.py "$DEPTH_DIR"/
cd "$DEPTH_DIR"
python score.py <origin_csv> <method_csv>   # compares depth maps: origin vs reconstructed
```

## Video captioning  ·  env `video_summary`

Caption originals and reconstructions with PLLaVA, then score with BERTScore / BLEU /
ROUGE. Drivers: `video_summary.sh` (captioning) and `video_depth.sh` (depth pipeline
wrapper). Both take `$DATA_ROOT`/method as arguments and resolve tools from
`$OPENSORA_DIR` / `$DEPTH_DIR` — `source env.sh` first; no manual path edits needed.

### NTSCC 원본 가중치 출처

원본 NTSCC README의 사전학습 모델은 [업스트림 가중치 저장소](https://drive.google.com/drive/folders/1qNRu_08-O5-lkqo3Sht48FCLITqqK6t-?usp=sharing)를 안내합니다. 이는 업스트림 모델의 출처 기록이며 본 연구에서 학습한 가중치가 아닙니다. 현재 통합 경로의 모델 준비·SNR 대응은 위 Stage 03 및 `scripts/download_models.py`의 고정 설정을 따릅니다. 원본 코드의 재구성으로 논문과 결과가 다를 수 있다는 업스트림 설명도 유지합니다.
