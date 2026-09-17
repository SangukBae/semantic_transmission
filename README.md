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

LGVSC 평가·학습 데이터의 SSD 위치와 확보/검증 명령은
[로컬 데이터셋 안내](docs/LOCAL_DATASETS.md)에 정리되어 있습니다.

이 컴퓨터에서 완료된 영상을 재사용하고 나머지를 생성하려면, 상위 `Semantic` 폴더에서
다음 명령 하나를 실행하면 됩니다. Conda 활성화는 스크립트가 처리합니다.

```bash
bash semantic_transmission/scripts/run_etri_remaining.sh
```

현재 명령은 [공식 공개 코드 설정](configs/etri_official.json)을 사용합니다.
2026-09-17부터 v2 프로필은 공식 5개 화질 지표와 항목별 전송량 집계를 사용합니다.
경계 중복 제거는 생성 설정과 독립된 옵션입니다. 기존 결과 재평가와 WebVid 실행 방법은
[정합성 수정 안내](docs/LGVSC_ALIGNMENT.md)에 있습니다. v1 완료 결과는 보존되며,
v2 이어하기에서 새 평가까지 완료된 결과로 재사용하지 않습니다.
576×320·24fps 전처리, 전체 프레임 SKEM, 공식 구간 캡션/광류, Open-Sora 30단계·시드 42를
적용합니다. 자세한 일치 범위와 공개 코드의 제약은 [공식 설정 실행 기록](docs/ETRI_OFFICIAL_PROTOCOL.md)에 있습니다.

2026-09-11 첫 영상은 전체 239개 SKEM 비교와 송수신·생성·평가를 완료했습니다.
결과는 `outputs/etri01_official_20260911_v2/01_person_walk/receiver/reconstruction/sample_0000.mp4`이며,
오른쪽 이동 후 중앙 복귀 동선이 관찰됐습니다. 보행 자세와 동작 시점까지 원본과 같지는 않습니다.

로컬 `.local/etri_continue.json`에 입력 경로와 **프로필별** 실행 이력을 기록합니다.
현재 v2 전체 파이프라인 완료 이력은 없으므로 첫 실행은 1~10번을 생성합니다.
이후에는 같은 v2 설정으로 검증이 끝난 영상만 재사용합니다. 매번 새 결과 폴더를
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
  --input-dir data/etri_video_eval/processed \
  --output outputs/etri10_hq_new
```

이 실행기는 NTSCC 연속 심벌과 캡션·광류·rate-index·정규화 정보의 실제 송수신 파일을
분리하고 전송량을 기록합니다. 기존 결과의 패킷 검증 및 공통 화질 평가 방법은
[ETRI 실험 프로토콜](docs/ETRI_HQ_PROTOCOL.md)에 있습니다.

공식 설정의 환경을 새 컴퓨터에 구성하려면 `bash scripts/bootstrap_official.sh`를 실행합니다.
GPU 실행 검증 결과와 환경 차이는 위 프로토콜 문서에 기록합니다.

## WebVid 서로 다른 유형 5편 검증

이 컴퓨터에서는 상위 `Semantic` 폴더에서 다음 명령 하나로 실행합니다.
Conda 활성화와 영상 선택은 자동으로 처리합니다.

```bash
bash semantic_transmission/scripts/run_webvid5.sh
```

저동작 장면·한 사람의 이동·빠른 이동·카메라 이동·여러 객체와 가림에 해당하는 5편을
고정해, 총 1,670프레임의 전송·복원과 공식 5개 화질 지표 평가를 순차 실행합니다.
RTX 4080 16GB용 메모리 보완 v2 설정의 잠정 예상은 약 35시간이며 장면에 따라 달라집니다.
`--dry-run`은 모델 실행 없이 입력·환경·재사용 가능 여부만 확인합니다.
중단 후 같은 명령을 다시 입력하면 검증이 끝난 영상은 재사용하고, 미완료 영상은 처음부터 실행합니다.

결과는 `outputs/webvid5_날짜_시간_식별자/`의 `summary.json`, `per_video_metrics.csv`,
원본/복원 비교용 `report.html`에 저장합니다. 눈에 보이는 의미 오류는 `manual_review.csv`에
별도로 기록합니다. **5편은 개발용 초기 검증이며 WebVid 전체 성능의 입증이 아닙니다.**
선정 목록·실행 조건·재사용 범위는 [WebVid5 실행 안내](docs/WEBVID5_VALIDATION.md)를 참고하세요.

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

## 연구 문서

- [모델 구조](docs/MODEL_ARCHITECTURE.md): 현재 SKEM+DSA 송신단·채널·수신단의 Mermaid 블록다이어그램
- [ETRI 개발 계획](docs/ETRI_DEVELOPMENT_PLAN.md): 네 가지 연구 목표, 새 평가 지표 1~2개 개발, 여섯 단계의 작업과 검증 기준
- [잔존 오류·사건 지연·STA 검증 결과](docs/EVENT_DURATION_STA_RESULTS.md): 새 합성 원본 24개·720건 평가.
  잔존 오류 후보만 이번 통제 기준을 통과했으며 자연 영상의 유효성·논문 신규성은 미입증

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

## 사람 검수 없는 지표 개발

[ERE·STA 본 평가 결과와 분석](docs/ERE_STA_FORMAL_RESULTS.md): 독립 평가를 완료했으며 두 후보의
주 채택 기준은 미달이다. ERE 시간 오류 검출률은 11.1%, STA 주 원인 판정 정확도는 52.6%였다.
STA에 동일한 개발 분류기를 붙인 보조 비교는 99.2%였지만, 주 규칙은 정상 외형 변화에서
41.6%를 오탐했다. 실제 복원 6쌍 진단과 [선행 연구·신규성 검토](docs/ERE_STA_NOVELTY_REVIEW.md)도 정리했다.

[ERE·STA v3.1 수정·재감사 결과](docs/ERE_STA_REAUDIT_RESULTS.md): 존재·움직임 분리와 내용 의존
근거 판정을 적용했다. 합성 개발 사건 재현율 95.8%·정밀도 92.0%, 방향 전환 16/16,
STA 규칙 6/6으로 필수 관문이 통과했다. 공개 영상은 진단 전용이며 이 재감사 시점에는 본 성능
비교·영상 귀착 실험을 실행하지 않았다. [1회차 관문 실패 기록](docs/ERE_STA_RESULTS.md)도 보존한다.

[MTE·OTF 2차 구현·검증 결과](docs/MTE_OTF_RESULTS.md)와
[수식·자동 정답·재현 방법](docs/MTE_OTF_PROTOCOL.md)을 정리했다.
새 공개 영상 24개와 합성 장면 24개에서 파생한 1,176개 사례를 평가하고,
기존 복원 6쌍에도 계산했다. 두 후보는 채택 기준 미달이며 신규성도 입증되지 않았다.

[1차 작업 결과](docs/AUTOMATIC_METRIC_RESULTS.md)를 확인할 수 있습니다.
[자동 정답·후보 수식·실행 방법](docs/AUTOMATIC_METRIC_PROTOCOL.md)에 1차 실험과 한계를 정리했다.
자동 평가 기반을 구현했으며 첫 두 지표 후보는 채택 기준 미달이다. 실제 복원의 의미 오류가
확정되었다는 뜻은 아니다.

[FSO·EOI·UEP 후속 검증 일괄 실행](docs/METRIC_VALIDATION_CAMPAIGN.md):
`bash scripts/run_metric_validation.sh`로 오류×외형 교차 → 구성 요소 비교 → 자연 영상·실제 LGVSC 복원을
순차 실행한다. 동일 명령으로 재개하며 `--dry-run`으로 입력과 예상 시간을 확인한다.
실제 복원의 독립 정답이 없으면 최종 정확도 판정을 보류한다.
