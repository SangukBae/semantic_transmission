# LGVSC 로컬 데이터셋

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
현재 실행 중인 다운로드나 학습 작업은 없다. 고정된 검증 결과는
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
