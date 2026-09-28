# ETRI 영상 의미 통신 연구

**적은 전송량으로 영상의 객체·동작·시간 흐름을 보존하고, 생성 복원의 할루시네이션을 줄이는 과제입니다.**
[LGVSC 공개 구현](https://github.com/TT2TER/LGVSC)을 기반으로 키프레임과 장면 설명을 전송하고,
수신단의 생성 모델로 영상을 복원합니다. 이 저장소는 SangukBae의 연구용 파생 저장소입니다.

## 처음 보는 분께

[**과제 목표와 현재 현황**](docs/README.md) → [개발·평가 계획](docs/ETRI_DEVELOPMENT_PLAN.md) →
[실험 결과](docs/ETRI_COMBINED_REFERENCE_DIAGNOSIS.md) 순서로 읽으면 됩니다.
문서는 [핵심 8개](docs/README.md#documents)로 통합했습니다. 재현용 동결 원문 8개는 별도로 유지합니다.

기존 네 목표는 **E1 시간축 신뢰성 · E2 할루시네이션 검출·완화 · E3 평가 지표 신뢰성 · E4 전송량 절감**입니다.
[ETRI 후속 메일](docs/ETRI_FOLLOWUP_EMAIL_SUMMARY.md)에 따라 AWGN 채널에서 검출·완화에 집중하고,
실제 연속 60초 이상·장면전환 저/중/고 영상으로 성공과 실패를 확인합니다.

## 현재 상태 — 2026-09-28

- **입력 준비:** 60초 원본 60개(유형별 20개), 120초 확장 6개. 개발/교정/평가를 분리했습니다.
- **실제 복원:** 저전환 개발 영상 한 편의 60초·1,440프레임 복원을 완료했습니다.
- **개선 실험:** 같은 영상의 짧은 세 구간에서 생성 조건 변경을 비교했습니다. 일부 개선과 새 왜곡이 함께 있어 기본값 채택을 보류했습니다.
- **남은 검증:** 60초 전체 완화 비교, 중·고전환 평가, 독립 오류 정답, 씬 체인지 누락 대안과 동적 의미정보 보강입니다.

복원 실행 성공은 할루시네이션 완화 성공을 뜻하지 않습니다.
수치·판정·산출물은 [모델 실험](docs/ETRI_COMBINED_REFERENCE_DIAGNOSIS.md)과 [지표 연구](docs/METRICS.md)에서 확인합니다.

## 실행 시작

Linux/WSL2, NVIDIA GPU, Conda, FFmpeg가 필요합니다. 로컬 검증 장비는 RTX 4080 16GB입니다.
**데이터·모델 가중치·복원 영상은 Git에 포함되지 않으므로 별도 준비가 필요합니다.**

```bash
git clone https://github.com/SangukBae/semantic_transmission.git
cd semantic_transmission
bash scripts/bootstrap.sh
source scripts/activate.sh
semtx doctor
semtx smoke --output outputs/my_first_skem
```

기본 `smoke`는 짧은 설치 점검입니다. 60초 복원은 [실행 안내](docs/RUN_GUIDE.md#long-video)를 따릅니다.
이미 구축된 환경의 명령, 실패 복구, 과거 실험 실행은 [실행 명령 모음](docs/RUN_GUIDE.md)에 모았습니다.
Windows/WSL 이전과 이 PC의 경로도 [설치·복구 안내](docs/RUN_GUIDE.md)에 통합했습니다.

## 코드와 문서

| 위치 | 역할 |
|---|---|
| [src/semantic_transmission/](src/semantic_transmission/) | 실행기, 송수신, 평가·결과 기록 |
| [configs/](configs/) · [scripts/](scripts/) | 실험 설정, 설치·실행 명령 |
| [tests/](tests/) | 시간축·패킷·재개·오류 처리 검사 |
| [docs/](docs/README.md) | 과제 현황, 개발 계획, 실행·실험 근거 |
| `01_data_prep/` ~ `07_downstream/` | LGVSC 공개 구현의 단계별 코드 |
| `data/` · `.local/` · `outputs/` | 로컬 데이터, 모델·환경, 실험 산출물; Git 제외 |

[모델 구조](docs/MODEL_ARCHITECTURE.md) · [논문·구현 일치 범위](docs/MODEL_ARCHITECTURE.md#reproduction) ·
[환경과 개발 안내](docs/RUN_GUIDE.md#setup)

## 원저작물

원 논문은 *LGVSC: A Large-Model-Driven Generative Video Semantic Communication Framework*입니다.
저자·인용은 [CITATION.cff](CITATION.cff), 원본 설명은 [README_UPSTREAM.md](README_UPSTREAM.md),
라이선스는 [LICENSE](LICENSE)와 [외부 저작물 안내](THIRD_PARTY_NOTICES.md)를 확인하세요.
