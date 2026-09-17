# Windows 11 / RTX 4080 연구 환경 이전

점검일: 2026-09-17. 대상은 Windows 11과 기존 컴퓨터와 같은 GPU이다.
현재 검증된 실제 모델 실행은 Linux / RTX 4080 16GB이다. 새 Windows 컴퓨터의
WSL GPU 실행을 완료했다는 뜻은 아니다. Bash, `fcntl`, Linux CUDA 확장 모듈을
사용하므로 WSL2 Ubuntu에서 환경을 다시 구성한다.

## GitHub에서 가져오는 것과 따로 옮기는 것

코드·프로필·설치 스크립트·테스트·프로토콜·요약 검증 기록은 이 저장소에 있다.
`.gitignore`가 제외하는 원본 데이터·가중치·전체 실험 결과는 Git clone으로 복구되지 않는다.

| 자료 | 현재 위치 | 이전 방법 |
|---|---|---|
| 연구 코드와 기록 | 이 저장소 | Git clone |
| 데이터·주석·다운로드/전처리 manifest | `/legend/semantic_transmission/datasets/`, 약 33 GiB | 디렉터리 전체 복사 후 `data` 링크 재생성 |
| 전체 실험·평가·복원·동결 선언·캐시 | `outputs/`, 점검 당시 약 95 GiB | 기존 결과를 계속 분석/검증하려면 전체 복사 |
| 핵심 생성/인식 모델 6개 | `configs/models.json`의 고정 revision | `download_models.py`로 재다운로드 또는 캐시 복사. 현재 선택 snapshot 합계 약 52.35 GiB |
| NTSCC / UniMatch | `.local/checkpoints/`, 약 154 MiB | 복사 또는 `download_auxiliary.py`로 해시 검증 다운로드 |
| SAM2 / DINOv2 | `.local/metric_v2_models/` | `prepare_metric_v2_assets.py`로 소스·가중치 재구축 |
| 평가 모델 캐시 | `~/.cache/torch/hub/checkpoints/`, `~/.cache/clip/` | 선택적으로 복사; 없으면 해당 평가 모델에서 다운로드 |
| 실험 이력·추가 진단 | `.local/etri_continue.json`, `.local/webvid5_history.json`, `.local/validation/` | 별도 보존. 새 경로/환경으로 과거 실행을 자동 재사용할 수 있다고 가정하지 않음 |
| Python 환경·외부 소스 | `.local/settings.json`, `.local/model_paths.json`, `.local/vendor/`, Conda 환경 | 새 컴퓨터에서 설치 스크립트로 재구축 |
| 지표 평가 환경 | `.local/metric_v2_env/` + 기존 `semantic-diffusers` Conda 환경 | 아래 환경 복구 항목 참고. venv 폴더만 복사하면 작동하지 않음 |
| SGD-JSCC 비교 기준선 | 형제 `sgdjscc_lab/` 저장소·가중치·`ptest` 환경 | 해당 비교 실험에만 필요. 이 저장소에 포함되지 않음 |

33/95 GiB는 당시 할당 크기를 반올림한 값이다. 진행 중인 실험으로 출력 크기가 늘어난다.
전체 Hugging Face 캐시는 다른 연구 모델까지 포함해 약 251 GiB였으므로 전부 필수는 아니다.
모델 cache를 복사할 때 `snapshots`의 심볼릭 링크만 복사하면 끊어진다. 해당 모델의
`blobs`까지 함께 보존하거나 다운로드 스크립트를 사용한다.

WebVid 원본은 53/55편만 확보돼 있다. 나머지 2편을 GitHub나 이전 과정에서 새로
확보한 것은 아니다. ETRI 영상·주석과 기존 실험 산출물은 특히 별도 복사가 필요하다.
이전 PC에서 실험이 실행 중이면 먼저 복사할 수 있지만, 종료 후 마지막 동기화와
checksum 비교를 완료하기 전에는 완전한 실험 백업으로 보지 않는다.

## WSL2와 저장 경로

Windows 관리자 PowerShell에서 실행하고 필요한 경우 재부팅한다.

```powershell
wsl --install -d Ubuntu-22.04
wsl --update
wsl --list --verbose
```

Ubuntu가 WSL **2**인지 확인한다. NVIDIA 드라이버는 Windows에 설치하고 WSL 안에
Linux 디스플레이 드라이버를 설치하지 않는다. WSL의 CUDA 드라이버는 Windows에서
제공된다. 기존 `.local/metric_v2_driver/`의 Linux `libcuda`를 WSL에 적용하지 않는다.
[Microsoft WSL 설치](https://learn.microsoft.com/en-us/windows/wsl/install),
[NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

코드·캐시·대량 프레임은 `/mnt/c`보다 WSL Linux 파일 시스템에 두는 것을 권한다.
예: `~/research/semantic_transmission`, `~/datasets/semantic_transmission`.
[Microsoft 파일 저장 지침](https://learn.microsoft.com/en-us/windows/wsl/filesystems).
원본 데이터·출력·선택 모델만 약 180 GiB이며, 환경·새 결과·다운로드 임시 공간은 별도다.
CPU 오프로딩도 사용하므로 새 PC의 RAM 및 WSL 메모리 할당을 함께 확인한다.

## 기본 모델 환경 재구축

아래 명령은 WSL Ubuntu에서 실행한다. Linux용 Conda를 먼저 설치한다.
`CONDA_BASE`는 실제 Linux Conda 설치 위치로 지정한다. Windows Conda 경로를 쓰지 않는다.

```bash
sudo apt-get update
sudo apt-get install -y git ffmpeg build-essential rsync
mkdir -p ~/research ~/datasets/semantic_transmission
cd ~/research
git clone https://github.com/SangukBae/semantic_transmission.git
cd semantic_transmission
export CONDA_BASE="$HOME/miniconda3"  # 실제 설치 경로
nvidia-smi
bash scripts/bootstrap_official.sh
source scripts/activate.sh
semtx doctor
python scripts/probe_environment.py
```

`nvidia-smi`가 PATH에 없으면 `/usr/lib/wsl/lib/nvidia-smi`를 확인한다.
설치 스크립트는 `lgvsc`, `lgvsc-channel`, `lgvsc-internvl`을 구성하며 고정된
소스와 모델을 받는다. Apex 빌드는 현재 같은 GPU의 SM 8.9를 대상으로 한다.
FlashAttention/Apex GPU probe 통과 여부를 새 PC에서 확인해야 한다.
`semtx doctor`의 경로 확인만으로 GPU 생성 성공을 대신하지 않는다.

## 데이터와 결과 복사

새 WSL에서, `OLD_HOST`는 기존 Linux PC의 실제 SSH 주소로 바꾼다.
공유 디스크로 옮기는 경우에도 같은 디렉터리 구조와 파일 바이트를 보존한다.

```bash
rsync -a --info=progress2 sangukbae@OLD_HOST:/legend/semantic_transmission/datasets/ \
  "$HOME/datasets/semantic_transmission/"
# 새 clone에 data가 없을 때만 링크를 만든다. 기존 경로를 덮어쓰지 않는다.
ln -s "$HOME/datasets/semantic_transmission" data
mkdir -p outputs
rsync -a --info=progress2 \
  sangukbae@OLD_HOST:/home/sangukbae/ETRI/Semantic/semantic_transmission/outputs/ outputs/
python scripts/prepare_lgvsc_datasets.py audit --root "$HOME/datasets/semantic_transmission"
```

완료 후 같은 rsync 명령에 `--checksum --dry-run`을 추가해 바뀐 파일이 없는지
확인한다. 위 명령은 `--delete`를 사용하지 않는다. `.local/validation/`과 이력 JSON은
별도의 백업 폴더에 보존하고, `.local/settings.json`과 `model_paths.json`은 새 설치가
생성한 값을 사용한다. 이전 Python 가상환경·CUDA 드라이버·빌드 폴더를 통째로 덮어쓰지 않는다.

## 새 실행과 과거 결과의 재사용

WebVid5 입력 확인은 다음과 같이 한다.

```bash
bash scripts/run_webvid5.sh --dry-run
```

ETRI는 새 설정 파일 없이도 명시적 입력 경로로 확인할 수 있다.

```bash
python -m semantic_transmission.research --profile configs/etri_official.json \
  --input-dir data/etri_video_eval/processed --output outputs/etri_on_new_pc --dry-run
```

실제 모델 smoke test는 GPU가 비어 있을 때 새 출력 폴더로 수행한다.

```bash
semtx smoke --selector skim --skim-keyframes 2 --frames 9 \
  --output outputs/wsl_first_smoke
```

이 smoke test는 축소된 로컬 프로필이다. 공식 설정 전체 및 지표 관측 모델 검증은 별도다.
새 환경에서 전체 실험을 시작하기 전 위 dry-run과 필요한 GPU 모델 검사를 수행한다.

과거 `outputs/*/protocol.json`, `campaign.json`, 실행 manifest에는 절대경로와
코드·환경·가중치 식별자가 들어 있다. 원본 선언을 일괄 문자열 치환하거나 해시 검사를
우회하지 않는다. 이전 결과는 보존하고 새 환경 실험은 새 출력 루트에서 시작한다.
WebVid5의 장면 선택 중간 상태는 재개되지 않으며, 완료 영상도 동일 조건 검증이 필요하다.
자료의 복사 완료와 기존 실험의 실행 재개 가능 여부는 별도다.

## 지표 연구 환경과 동결된 실험

현재 `.local/metric_v2_env/bin/python`은 저장소 밖 `semantic-diffusers` 환경을
참조하고 `include-system-site-packages=true`이다. 이 의존 관계는 Git clone이나
venv 폴더 복사로 재현되지 않는다. 관측한 Python/전체 패키지 버전은
[metric-v2-observed.json](../environment/locks/metric-v2-observed.json)에 기록했다.
이 파일은 설치 lock이나 WSL 검증 완료 기록이 아니다.

생성용 `lgvsc`와 별도 Python 3.10 평가 환경을 만들고, 관측 버전 및 SAM2의
고정 소스를 기준으로 패키지를 설치·검증한다. 핵심 관측값은 PyTorch 2.12.0,
torchvision 0.27.0, NumPy 2.2.6이다. GPU 드라이버와 wheel 호환성을 새 PC에서 확인한다.
SAM2/DINO 소스·가중치 및 DAVIS 입력은 아래 스크립트로 복구한다.

```bash
python scripts/prepare_metric_v2_assets.py
```

SAM2는 출력 안내대로 별도 평가 환경에 설치한다. 평가 Python의 기본 위치는
`.local/metric_v2_env/bin/python`이며 v2~v5 shell 실행기는 `LGVSC_METRIC_PYTHON`으로
덮어쓸 수 있다. 생성/픽셀 환경은 `.local/settings.json`의 `python`을 따르고
`LGVSC_PYTHON`으로 덮어쓸 수 있다. 사용자 이름과 Conda 절대경로를 코드에 넣지 않는다.

WSL의 후속 campaign은 [metric_validation_wsl.json](../configs/metric_validation_wsl.json)을
별도 사용한다. `driver_library`는 `/usr/lib/wsl/lib`이다. 평가 환경 위치가 다르면
새 로컬 config의 `evaluation_python`도 지정한다.

```bash
bash scripts/run_metric_validation.sh --config configs/metric_validation_wsl.json \
  --output outputs/metric_validation_wsl_new --dry-run
```

이 명령은 이전 v5/ERE 결과와 calibration/cache도 요구한다. **Git clone과 모델 다운로드만으로
기존 지표 campaign이 준비되지는 않는다.** `outputs/`를 빠짐없이 복사하고, 역사적 선언의
경로·해시 일치 여부를 확인한다. 경로/환경 변경으로 동결 검증이 거부되면 새 선언과
독립적인 재검증이 필요하다. 기존 결과를 새 환경의 결과로 바꿔 기록하지 않는다.

이전 코드 전체의 이관 전 snapshot은 `66fc8fd4`이다. 그 뒤의 이관 보완은 shell 환경
선택과 CI/문서에 한정한다. 기존 동결된 실행에서 wrapper 해시를 요구하면 해당 버전의
별도 checkout을 사용하고, 원래 결과/선언은 수정하지 않는다.

SGD-JSCC 비교를 계속할 경우 `sgdjscc_lab`도 별도 이전해야 한다. 점검 당시 그 저장소에
미커밋 변경이 있었으며 이 작업에서 수정·커밋·푸시하지 않았다. 비교 실행에는 그 저장소의
설정, JSCC/diffusion/ControlNet/MUGE 가중치와 실행 환경이 필요하다.
`semantic_transmission.sgd_bridge --sgd-repo ... --python ...`으로 새 위치를 지정한다.

## GitHub 코드만으로 확인하는 CPU 검사

아래는 GPU 모델·데이터·과거 결과 없이 실행하는 CI와 같은 검사다.

```bash
python3.10 -m venv .local/cpu-tests
.local/cpu-tests/bin/python -m pip install --upgrade pip
.local/cpu-tests/bin/python -m pip install torch==2.2.2 torchvision==0.17.2 \
  --index-url https://download.pytorch.org/whl/cpu
.local/cpu-tests/bin/python -m pip install -r environment/requirements-test.txt
.local/cpu-tests/bin/python -m pip install -e . --no-deps
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 \
  .local/cpu-tests/bin/python -m pytest -q --disable-warnings
```

이 검사는 코드 이관·CPU 회귀 확인이며 실제 GPU 품질이나 WSL 전체 파이프라인의 성공을 뜻하지 않는다.
