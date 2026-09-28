# ETRI 영상 의미 통신 연구

**영상의 핵심 정보만 보내고 AI로 복원해, 전송량과 내용 왜곡을 줄이는 연구입니다.**

- 기반 모델: [LGVSC](https://github.com/TT2TER/LGVSC)의 연구용 파생 구현.
- 연구 목표: 시간 흐름 보존 · 할루시네이션 완화 · 새 평가 지표 · 전송량 절감.
- 현재 상태: 60초 영상 한 편 복원 완료, 할루시네이션 완화 효과는 미검증.
- 다음 작업: 생성기에 전달한 키프레임이 올바른 시간 위치에 반영되는지 확인.

## 문서 읽기

- [과제 현황](docs/README.md): 무엇을 연구하고 어디까지 했는지.
- [데이터셋](docs/DATA.md): 어떤 영상으로 실험하는지.
- [모델 구조](docs/MODEL_ARCHITECTURE.md): 각 모델이 무엇을 하는지.
- [개발 계획](docs/ETRI_DEVELOPMENT_PLAN.md): 다음에 무엇을 검증할지.
- [실험 결과](docs/ETRI_COMBINED_REFERENCE_DIAGNOSIS.md): 개선된 점과 남은 오류.
- [전체 문서](docs/README.md#documents): 핵심 안내 8개와 상세 기록.

## 실행 시작

- 필요 환경: Linux/WSL2 · NVIDIA GPU · Conda · FFmpeg.
- 검증 장비: RTX 4080 16GB.
- 별도 준비: 데이터·모델 가중치·실험 결과는 Git에 미포함.

```bash
bash scripts/bootstrap.sh
source scripts/activate.sh
semtx doctor
semtx smoke --output outputs/my_first_skem
```

- 위 명령: 저장소 루트에서 실행하는 짧은 설치 점검.
- [60초 실행·복구](docs/RUN_GUIDE.md): 전체 복원과 환경 이전 안내.

## 저장소 구성

- [src/](src/semantic_transmission/): 실행·송수신·평가 코드.
- [configs/](configs/) · [scripts/](scripts/): 설정과 실행 명령.
- [tests/](tests/): 구현 검사.
- `01_data_prep/` ~ `07_downstream/`: LGVSC 단계별 코드.
- `data/` · `.local/` · `outputs/`: 로컬 데이터·모델·결과.

원 논문·인용: [CITATION.cff](CITATION.cff) · [공개 구현 설명](README_UPSTREAM.md) · [라이선스](LICENSE) · [외부 저작물](THIRD_PARTY_NOTICES.md).
