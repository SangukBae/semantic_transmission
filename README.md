# semantic_transmission

[LGVSC 공식 구현](https://github.com/TT2TER/LGVSC)을 기반으로 한 영상 의미 통신 연구 저장소입니다.
공식 저장소의 이력을 보존하고, **RTX 4080 16GB에서 실제 모델을 실행하는 환경과 검증 가능한 파이프라인**을 추가했습니다.
이 저장소는 SangukBae의 연구용 파생 저장소이며, 논문 저자의 공식 저장소는 위 링크입니다.

## 시작하기

Linux, NVIDIA GPU/드라이버, Conda, ffmpeg가 필요합니다. 기본 Conda 경로는 `~/anaconda3`이며
다르면 `CONDA_BASE`를 지정합니다. 모델 가중치와 환경 설치에는 수십 GB의 디스크가 필요합니다.

```bash
git clone https://github.com/SangukBae/semantic_transmission.git
cd semantic_transmission
bash scripts/bootstrap.sh
source scripts/activate.sh
semtx doctor
semtx smoke --output outputs/my_first_skem
```

이미 구축된 이 컴퓨터에서는 `source scripts/activate.sh`부터 실행하면 됩니다.
`lgvsc`는 PyTorch/CUDA 모델 실행 환경, `lgvsc-channel`은 Sionna/LDPC 전송 환경입니다.
설치 스크립트는 소스와 가중치 버전을 고정하고, 재실행 시 다운로드 캐시를 재사용합니다.

실행 순서는 **영상 준비 → InternVL SKEM → PLLaVA 자막 → UniMatch 광류 → NTSCC 키프레임 전송
→ LDPC 자막·광류 전송 → Open-Sora 영상 복원 → 프레임 수·화질 검증**입니다.
각 모델을 별도 프로세스에서 실행하며, 단계가 실패하면 중단하고 로그와 실패 상태를 남깁니다.
출력 폴더는 실행마다 새 이름을 사용해야 합니다.

```bash
# 자신의 영상
semtx smoke --input /path/to/video.mp4 --frames 33 --output outputs/custom_skem
# 균일한 키프레임 세 개로 두 구간의 연결 확인
semtx smoke --selector skim --skim-keyframes 3 --frames 33 --output outputs/skim_two_segments
```

## ETRI 전체 영상 실행

이 컴퓨터에서 완료된 영상을 재사용하고 나머지를 생성하려면, 상위 `Semantic` 폴더에서
다음 명령 하나를 실행하면 됩니다. Conda 활성화는 스크립트가 처리합니다.

```bash
bash semantic_transmission/scripts/run_etri_remaining.sh
```

현재 명령은 [공식 공개 코드 설정](configs/etri_official.json)을 사용합니다.
576×320·24fps 전처리, 전체 프레임 SKEM, 공식 구간 캡션/광류, Open-Sora 30단계·시드 42를
적용합니다. 자세한 일치 범위와 공개 코드의 제약은 [공식 설정 실행 기록](docs/ETRI_OFFICIAL_PROTOCOL.md)에 있습니다.

2026-09-11 첫 영상은 전체 239개 SKEM 비교와 송수신·생성·평가를 완료했습니다.
결과는 `outputs/etri01_official_20260911_v2/01_person_walk/receiver/reconstruction/sample_0000.mp4`이며,
오른쪽 이동 후 중앙 복귀 동선이 관찰됐습니다. 보행 자세와 동작 시점까지 원본과 같지는 않습니다.

로컬 `.local/etri_continue.json`에 입력 경로와 **프로필별** 실행 이력을 기록합니다.
같은 새 설정으로 검증이 끝난 `01_person_walk`를 재사용하고 2~10번을 생성합니다. 매번 새 결과 폴더를
출력하고, 완료된 영상은 원본·설정·복원 MP4·전송 파일을 검증한 뒤 복사해 한 배치로 모읍니다.
중단 후 같은 명령을 다시 실행하면 검증이 끝난 영상은 재사용하고, 미완료 영상은 처음부터
생성합니다. 두 명령의 동시 실행은 잠금으로 방지합니다. `--dry-run`을 붙이면 모델을
실행하지 않고 재사용/생성 목록만 확인합니다.

이전 100프레임·512×256·10fps 원본을 그대로 사용하는 HQ 프로필도 별도 실행기로 보존합니다.
InternVL BF16과 Open-Sora 50 sampling steps를 사용하며, 16GB GPU에서는 모델을 순서대로
올리고 CPU 메모리를 함께 사용합니다. 모든 프레임의 의미를 비교하므로 시간이 오래 걸립니다.

```bash
bash scripts/bootstrap_hq.sh
source scripts/activate.sh
python -m semantic_transmission.research \
  --input-dir ../sgdjscc_lab/data/etri_video_eval/processed \
  --output outputs/etri10_hq_new
```

이 실행기는 NTSCC 연속 심벌과 캡션·광류·rate-index·정규화 정보의 실제 송수신 파일을
분리하고 전송량을 기록합니다. 기존 결과의 패킷 검증 및 공통 화질 평가 방법은
[ETRI 실험 프로토콜](docs/ETRI_HQ_PROTOCOL.md)에 있습니다.

공식 설정의 환경을 새 컴퓨터에 구성하려면 `bash scripts/bootstrap_official.sh`를 실행합니다.
GPU 실행 검증 결과와 환경 차이는 위 프로토콜 문서에 기록합니다.

## 연구 코드 구조

```text
src/semantic_transmission/  실행기, 실제 모델 단계, 패킷·채널, 결과 기록
configs/                   RTX 4080 설정, 외부 소스·모델 버전
scripts/                   환경 설치, 다운로드, 활성화, GPU 점검
patches/                   외부 라이브러리의 작은 호환성 수정
tests/                     패킷, CSV, 시간축, 실패 처리 회귀 검사
environment/               의존성 및 검증 환경 기록
docs/                      연구 범위, 실행 방법, 검증 결과
01_data_prep/ ... 07_downstream/  공식 구현의 단계별 코드
.local/                    외부 소스·로컬 설정·가중치 링크 (Git 제외)
outputs/                   실행별 로그·측정값·복원 영상 (Git 제외)
```

새로운 모델 모듈과 실험 설정은 원본 기준선과 구분하여 추가하고, 실행 manifest와 측정값을 함께 보관합니다.

## 검증과 범위

```bash
python -m unittest discover -s tests -v
python scripts/probe_environment.py
```

자세한 환경·변경점·제약은 [로컬 연구 안내](docs/LOCAL_RESEARCH.md)를 참고하세요.
실제 GPU에서 완료한 세 가지 실행과 회귀 검사 결과는 [검증 기록](docs/VALIDATION.md)에 있습니다.
원 논문의 설치·실험 설명은 [원본 README](README_UPSTREAM.md)에 보존했습니다.

기본 실행은 **17프레임, 256×256, 10 sampling steps의 실제 모델 실행 확인용**입니다.
논문 성능 재현이나 ETRI 목표 달성을 의미하지 않습니다. InternVL int8, PLLaVA CPU offload 등
16GB GPU를 위한 설정 차이를 기록했습니다. 현재 실행기는 공개된 NTSCC **10dB 체크포인트**를 지원하며,
0–8dB 저자 체크포인트와 DVST는 제공되지 않았습니다. 기본 `smoke` 실행의 전송량에는
rate-index가 빠져 있습니다. 위 ETRI 실행기는 이를 포함한 모든 영상별 모델 입력을 계측하며,
연속 JSCC 심벌의 복소수 파일 크기와 실제 무선 bit 수를 구분합니다.

## 원저작물

LGVSC: A Large-Model-Driven Generative Video Semantic Communication Framework.
저자·논문 인용 정보는 [CITATION.cff](CITATION.cff)와 [원본 README](README_UPSTREAM.md),
라이선스는 [LICENSE](LICENSE)를 참조하세요. 외부 모델과 데이터에는 각각의 라이선스가 적용됩니다.
