# LGVSC 로컬 데이터셋

## 2026-09-24 WSL 장시간 입력 준비

최신 입력은 [ETRI 평가용 v1](ETRI_BENCHMARK_V1.md)이다. 별도 경로
`data/etri_benchmark_v1_20260924/`에 **60초 원본별 입력 60개 + 120초 확장 6개**를 생성·검증했다.
본 세트는 저·중·고 각 20개, TVSum/ClipShots 균형 구성, 개발 18·교정 12·평가 30개다.
ClipShots 확보 원본은 257개로 늘었다. 독립 정답과 장시간 LGVSC 성능평가는 미완료다.
아래 6개는 보존된 초기 파일럿이다.

현재 WSL의 `data/`는 `/home/sangukbae/datasets/semantic_transmission`을 가리킨다.
새 `data/etri_long_video_20260924/`에 저·중·고 전환별 개발/평가 1개씩,
**60초·24 FPS·1,440프레임 입력 6개**를 준비하고 전체 디코딩을 확인했다.
[확보·검수 결과와 제한](ETRI_LONG_VIDEO_INPUTS.md),
[실제 입력 목록](../data/etri_long_video_20260924/manifest.csv).
경계는 AI 시각 검토 수준이며 독립 사람 정답은 아니다. LGVSC 복원·완화 비교는 미실행이다.

## 기존 legend 구축 기록

저장 위치는 `/legend/semantic_transmission/datasets`이다. 저장소의 `data/`는 이
디렉터리를 가리키는 로컬 심볼릭 링크이며 Git에는 포함하지 않는다.
원본 영상, 학습 이미지와 실행 로그는 SSD에 두고, 공식 평가 목록은
`docs/eval_manifest_webvid.csv`, `docs/eval_manifest_kinetics.csv`로 유지한다.

## 2026-09-11 구축 결과

| 항목 | 확보·검증 결과 |
|---|---|
| WebVid | **53/55개** 원본 SHA-256 일치, 53개 전처리 및 전체 영상 디코딩 통과 |
| Kinetics-400 | **14/14개** 원본 SHA-256·정답 라벨 확인, 14개 전처리 및 전체 영상 디코딩 통과 |
| OpenImages | **100,000/100,000장**, 고유 ID·파일 해시·전체 이미지 디코딩 통과, 실패 0건 |
| ETRI | 개발 영상 10개와 기존 주석 보존, 새 경로에서 기존 완료 영상 재사용 dry-run 통과 |

전체 구축 상태는 `PARTIAL_MISSING_TWO_WEBVID_SOURCES`이다. 다음 WebVid 영상은
원출처가 HTTP 403을 반환했고, 동일 해시의 공개 보관본을 확보하지 못했다.
다른 영상으로 대체하지 않았다.

- ID `4440995`: `stock-footage-norway-pulpit-rock-falls.mp4`
- ID `1018898026`: `stock-footage-in-a-park-a-small-white-dog-is-waiting-on-a-leash-next-to-its-owner-who-is-sitting-on-a-bench-it.mp4`

Kinetics의 대상 영상은 공식 `val/part_1.tar.gz`에서 4개,
`val/part_11.tar.gz`에서 10개를 확보했다. 검사한 코드 테스트는 23개 통과했다.
2026-09-11 구축 확인 당시 실행 중인 다운로드나 학습 작업은 없었다. 고정된 검증 결과는
[구축 기록](validation/2026-09-11-legend-datasets.json), SSD의 최신 상태는
`data/reports/provisioning_summary.json`에서 확인한다.

## 데이터 구성

| 경로 (`data/` 기준) | 목적 | 동일성 기준 |
|---|---|---|
| `webvid55/raw/` | 논문의 WebVid 평가 영상 55개 | 공식 목록의 파일명과 SHA-256 모두 일치 |
| `kinetics400_14/raw/` | 논문의 행동 분류 평가 영상 14개 | 공식 목록의 SHA-256과 Kinetics 라벨 확인 |
| 각 영상 세트의 `processed/` | 공식 전처리를 적용한 평가 입력 | 원본 해시, 출력 해시, 프레임 수, 전체 디코딩 검증 |
| `openimages_ntscc/train/` | NTSCC 학습용 OpenImages 100,000장 | 고정 선택 목록, 다운로드 해시, 이미지 디코딩 검증 |
| `openimages_ntscc/metadata/` | 학습 표본의 ID·원출처·라이선스·선택 규칙 | 원본 CSV와 선택 CSV의 해시 보관 |
| `etri_video_eval/` | 기존 ETRI 개발 영상 10개와 주석 | 기존 디렉터리를 같은 SSD 안에서 이동 |
| `reports/` | 다운로드·전처리·이전 검증 기록 | 실제 파일을 확인한 결과 |

`raw/`에는 공식 해시와 일치한 영상만 들어간다. 다운로드에 성공해도 해시가
다르면 `candidates/`에 분리한다. 누락 영상이나 후보 영상을 공식 평가 영상 수에
포함하지 않는다. 실제 확보 수는 `reports/evaluation_audit.json`을 확인한다.

## 출처와 재현 범위

- [LGVSC 데이터 안내](https://github.com/TT2TER/LGVSC/blob/main/docs/DATA.md)
  및 저장소에 포함된 55개/14개 고정 목록을 사용한다.
- WebVid URL은 공개 메타데이터 보관본
  [`TempoFunk/webvid-10M`](https://huggingface.co/datasets/TempoFunk/webvid-10M/tree/461f7da9a310b67c7fa05fcbd6e312ddbffc8ba5/data/val/partitions)의
  검증 세트에서 파일명이 정확히 일치하는 항목을 찾는다. 실제 영상은 원출처의
  `contentUrl`에서 받는다. 55개 모두의 URL이 이 목록에 있다. WebVid 공식
  배포는 중단되었으므로 URL을 알아도 접근하지 못하는 영상이 있을 수 있다.
- Kinetics 영상은 [CVDF 공식 검증 세트](https://github.com/cvdfoundation/kinetics-dataset)에서
  필요한 14개만 추출한다. 원본 아카이브를 스트리밍으로 읽고 다른 영상은 저장하지 않는다.
  14개 모두 Kinetics-400 `val` 주석과 대응된다.
- OpenImages는 [공식 boxable 이미지 목록](https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv)에서
  seed `20260911`의 reservoir sampling으로 100,000장을 선택하고 CVDF의 공개
  `open-images-dataset` S3 버킷에서 받는다. 각 이미지의 원출처와 라이선스는
  `train_100k_selection.csv`에 남긴다.

**LGVSC의 정확한 OpenImages 학습 ID 목록은 공개되어 있지 않다.** 이 로컬
표본은 같은 데이터셋과 규모를 사용하는 재현 가능한 별도 학습 표본이며,
논문과 동일한 학습 데이터라고 주장하지 않는다. 모델 학습은 실행하지 않았다.

## 실행과 검증

저장소 루트에서 실행한다. 다운로드 명령은 Python 3, `requests`, Pillow가
필요하다. 전처리는 MoviePy가 설치된 로컬 `lgvsc` 환경을 사용한다.

```bash
python3 scripts/prepare_lgvsc_datasets.py webvid --workers 8
python3 scripts/prepare_lgvsc_datasets.py kinetics --workers 6
python3 scripts/prepare_lgvsc_datasets.py openimages --workers 32
/home/sangukbae/anaconda3/envs/lgvsc/bin/python scripts/prepare_lgvsc_datasets.py preprocess --workers 2
python3 scripts/prepare_lgvsc_datasets.py audit
python3 scripts/prepare_lgvsc_datasets.py verify-images --workers 16
```

명령은 기존 완료 파일을 재사용한다. 전처리는 원본을 보존하고 공개 코드와 같은
순서로 MoviePy H.264 재인코딩(24 fps, 최대 16초), FFmpeg H.264 재인코딩
(576×320, CRF 18, preset medium)을 수행한다. 영상별 길이와 프레임 수는
`preprocess_records/`와 `reports/preprocessing.json`에 기록한다.
`processed/videos.csv`는 준비된 영상 경로만 포함한다. 모델 단계에 필요한
프레임 추출과 의미 정보 생성은 이후 실행 단계에서 수행한다.

기존 `etri_official.json`과 ETRI 배치 실행기는 10초 ETRI 입력에 맞춰 고정되어 있다.
서로 길이가 다른 WebVid/Kinetics 전체를 이 ETRI 프로필로 바로 실행하면 안 된다.
공식 Stage 01 이후 입력 경로로 `processed/videos.csv`를 사용하거나, 후속 연구에서
영상별 프레임 수를 반영한 별도 실행 프로필을 구성한다.

## 후속 연구용 장시간 영상 확보 기준 — 2026-09-21

[ETRI 후속 메일](ETRI_FOLLOWUP_EMAIL_SUMMARY.md)에 따라, 기존 WebVid·ETRI 단편과 별도로
실제 연속 구간 60초 이상 확보를 추진한다. 이 문서 개정은 새 자료 다운로드·디코딩·복원 완료 기록이 아니다.

- 장면전환이 거의 없음 / 일반적인 빈도 / 빈번하고 객체·배경 변화가 많음의 세 유형을 구성한다.
  각 원본의 실제 경계·객체 변화 근거와 유형별 선정 기준을 결과 확인 전에 남긴다.
- 원본 한 영상의 연속 시간 구간을 사용한다. 짧은 영상 반복·무관한 클립 연결·재생 속도 변경이나
  여러 영상의 총 길이로 목표를 충족하지 않는다. 자연스러운 장면전환을 포함한 원본은 사용할 수 있다.
- 기존 고정 평가 목록을 바꾸지 않고 별도 manifest에 원출처·영상 ID·해시·추출 시작/끝 시각,
  실제 길이·FPS·프레임 수·전처리·시간 대응·유형·개발/평가 분할을 기록한다.
  유튜브 등 플랫폼 자료를 확보할 때도 출처를 남기며 원본은 내부 검증용으로 관리한다.
- 기존 MoviePy 최대 16초 절단과 `lgvsc_variable.json`의 `max_frames=384`를 그대로 적용하지 않는다.
  별도 전처리·실행 프로필의 길이 보존과 구간 간 상태 전달을 점검한다.
- 확보 완료, 전처리 검증, 완화 전 복원, 완화 후 복원, 오류 검수 상태를 각각 기록한다.
  원본 확보를 복원 완료로 간주하지 않는다. 실제 검증 길이가 60초 미만이면 사유·병목을 남긴다.

유형별 편수·장면전환 빈도 기준·평가 산출물은 [후속 프로토콜](ETRI_FOLLOWUP_PROTOCOL.md)을 따른다.
OpenImages 이미지 확보는 NTSCC용 데이터 준비 기록이며, 전체 영상 모델의 학습 이력이나
이번 후속 연구의 동영상 재학습 완료를 뜻하지 않는다.

## 기존 SSD 데이터 정리

2026-09-11 `/legend/sgdjscc_workspace/datasets`의 기존 데이터셋을 삭제했다.
삭제 전 할당 크기는 719,625,375,744 bytes이며, 이 중 현재 패키지도 사용하는
ETRI 자료 216,473,600 bytes는 `data/etri_video_eval/`로 이동했다.
ImageNet, COCO, CC3M, Journey, CelebA, DAVIS, YouTube-VOS, OVIS 등
기존 전용 자료를 삭제했으며, 체크포인트와 실험 출력은 삭제하지 않았다.
개별 삭제 경로와 완료 기록은 `reports/migration.json`에 있다.

`.local/etri_continue.json`과 기존 SGD 결과 감사 스크립트의 원본 입력 경로를
새 위치로 갱신했다. 완료 영상 재사용은 원본 SHA-256을 비교하고, 이동 전
실행 기록을 보존하면서 새 복사본에 경로 변경을 기록한다. 기존 `sgdjscc_lab`의
삭제된 데이터셋을 가리키는 링크는 더 이상 사용할 수 없다.
