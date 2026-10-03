# ETRI 영상 의미 통신 연구

**영상의 핵심 정보만 보내고 AI로 복원해, 전송량과 내용 왜곡을 줄이는 연구입니다.**

- 기반 모델: [LGVSC](https://github.com/TT2TER/LGVSC)의 연구용 파생 구현.
- 연구 목표: 시간 흐름 보존 · 할루시네이션 완화 · 새 평가 지표 · 전송량 절감.
- 현재 기본 모델: **FC-LGVSC** — 혼합 키프레임·캡션 v2·마지막 17프레임 참조·T5 FP32 재사용. **조건 충돌 방지와 짧은 구간 시간표 수정 포함**(2026-10-01). [간단한 설명](docs/MODEL_ARCHITECTURE.md#fc-lgvsc) · [기준](docs/ETRI_DEVELOPMENT_PLAN.md#default-method)
- 최신 비교: FC-LGVSC 4편 복원 완료. 기존 LGVSC 대비 PSNR·SSIM·LPIPS·DISTS 평균 개선, 복원은 3편 단축·보행 증가. T5 저장값 재사용 기준이며 새 얼굴·무늬·위치 오류는 남음. [결과](docs/ETRI_COMBINED_REFERENCE_DIAGNOSIS.md#fc-lgvsc-four)
- 10월 3일 후속: Qwen3.5 캡션·문맥 비교와 30쌍 선택 시간 측정 완료. 자동 모델의 기본 경로 채택·복원 품질 검증은 미완료이며, 전체 원본 추출 보정은 중단 상태. [현재 상태](docs/README.md#status)

## 문서 읽기

- [과제 현황](docs/README.md): 무엇을 연구하고 어디까지 했는지.
- [데이터셋](docs/DATA.md): 어떤 영상으로 실험하는지.
- [모델 구조](docs/MODEL_ARCHITECTURE.md): 각 모델이 무엇을 하는지.
- [개발 계획](docs/ETRI_DEVELOPMENT_PLAN.md): 다음에 무엇을 검증할지.
- [실험 결과](docs/ETRI_COMBINED_REFERENCE_DIAGNOSIS.md): 개선된 점과 남은 오류.
- [로컬 자동 추출](docs/FC_LGVSC_LOCAL_EXTRACTION.md): WebVid·TVSum 전체 원본의 공개 모델 키프레임·캡션 추출과 재시작.
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
- 기본 복원: `bash scripts/reconstruct_fc_lgvsc.sh`; `tv_low_08` 60초에 두 수정 적용, 진행률·GPU 사용률 표시.
- 4편 일괄 복원: `bash scripts/reconstruct_fc_lgvsc_all.sh`; TVSum·WebVid 2편·보행을 순서대로 복원·평가. [안내](docs/RUN_GUIDE.md#fc-lgvsc-all)
- 보행 복원: `bash scripts/reconstruct_fc_lgvsc.sh --video person_walk`; 기존 입력·캡션·T5 FP32·잡음 재사용.
- 준비 검사: `bash scripts/reconstruct_fc_lgvsc.sh --check`; 모델 실행 없음.
- 이전 `reconstruct.sh`·`reconstruct_fp32.sh`는 수정 전 비교용으로 보존.
- [60초 실행·복구](docs/RUN_GUIDE.md#default-method): 필요한 기존 자료·결과 경로·환경 이전 안내.

## 저장소 구성

- [src/](src/semantic_transmission/): 실행·송수신·평가 코드.
- [configs/](configs/) · [scripts/](scripts/): 설정과 실행 명령.
- [tests/](tests/): 구현 검사.
- `01_data_prep/` ~ `07_downstream/`: LGVSC 단계별 코드.
- `data/` · `.local/` · `outputs/`: 로컬 데이터·모델·결과.

원 논문·인용: [CITATION.cff](CITATION.cff) · [공개 구현 설명](README_UPSTREAM.md) · [라이선스](LICENSE) · [외부 저작물](THIRD_PARTY_NOTICES.md).
