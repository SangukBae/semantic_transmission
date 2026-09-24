# ETRI 60초 연속 입력영상 준비 결과

> 이 문서는 보존된 **초기 6개 파일럿**의 기록이다. 이후 성능평가 준비를 위해
> [60개 본 세트와 120초 확장 6개](ETRI_BENCHMARK_V1.md)를 별도 구성했다.
> 파일럿의 원본·처리본·기존 분할은 덮어쓰지 않았으며 본 평가의 표본 수로 중복 계산하지 않는다.

- 실행일: 2026-09-24
- 상태: **3유형 × 개발/평가 각 1개 = 6개 입력 준비·전체 디코딩 검증 완료**.
- 저장 위치: `/home/sangukbae/datasets/semantic_transmission/etri_long_video_20260924/`
  (저장소에서는 `data/etri_long_video_20260924/`).
- [재생·검토 페이지](../data/etri_long_video_20260924/index.html),
  [CSV 목록](../data/etri_long_video_20260924/manifest.csv),
  [상세 JSON](../data/etri_long_video_20260924/manifest.json).
- LGVSC 복원, 완화 전후 비교, Added/Missing/Distorted 오류 주석은 **미실행**.

## 준비한 입력

모든 출력은 **60.000초, 24 FPS, 1,440프레임, 576×320**이다. 원본 한 영상의 연속 구간이며
여러 클립을 잇거나 감속하지 않았다. 비율을 유지해 크기를 조절하고 여백을 넣었다.

| ID | 유형·분할 | 원본 | 선택 구간(원본 상대 초) | 시각 검토로 지지된 전환 |
| --- | --- | --- | --- | --- |
| `low_dev` | 저전환·개발 | TUM `freiburg2_desk_with_person` | 35–95 | 0 |
| `low_eval` | 저전환·평가 | TUM `freiburg3_long_office_household` | 10–70 | 0 |
| `medium_dev` | 일반·개발 | TVSum `37rzWOQsNIw`, 샌드위치 가게·광장 | 10–70 | 4 |
| `medium_eval` | 일반·평가 | TVSum `AwmHb44_ouw`, 타이어·차량·작업장 | 10–70 | 6 |
| `high_dev` | 고전환·개발 | ClipShots `-96lap9jYibdZbCgpB8pBw__`, 모터사이클 | 10–70 | 31, 추가 모호 후보 4 |
| `high_eval` | 고전환·평가 | ClipShots `4255081509`, 방송 무대·관객·장치 | 10–70 | 28, 화면 분할 변경 1 별도 |

전환 횟수는 모델 출력과 독립적으로 **원본의 자동 후보와 앞뒤 프레임을 AI가 시각 검토한 값**이다.
전 프레임을 사람이 검수한 정답이 아니다. 카메라 이동 오탐 5개를 제외하고 모호 후보는 유지했다.
shot cut과 의미적인 scene 전환은 서로 다르므로 이 숫자를 의미적 scene 수로 사용하지 않는다.
`high_dev`의 원본에 있던 검은 전환 프레임도 보존했다.

첫 기준선 실행에는 `processed/low_dev.mp4`를 사용한다. 사람의 등장·착석·전화기 사용과
카메라 이동에 따른 가림이 있어 긴 구간에서 사람·물체·동작 보존을 확인할 수 있다.

## 확보 및 검증 근거

- TUM: 원본 TGZ 두 개에서 RGB 이미지와 촬영 타임스탬프를 추출했다.
  실제 RGB 시간 범위는 각각 약 **141.639초와 87.140초**다.
- TUM의 배포 AVI 미리보기는 약 135.567초로 원본 촬영 시간과 달랐다.
  따라서 미리보기를 입력 원본으로 사용하지 않고 촬영 타임스탬프에서 24 FPS를 선택했다.
  원본 PNG의 순서를 엄격히 증가시키며 같은 PNG를 반복하지 않는다.
  선택한 1,440개 프레임별 대응과 최대 시간 오차를 JSON에 기록했다.
- 첫 TUM 원본에는 약 131.29–133.88초에 2.59초의 촬영 공백이 있다.
  선택한 35–95초 구간에는 이 공백이 포함되지 않는다.
- TVSum: 공식 약 672 MB 배포본에서 원본 영상 **50개**를 확보했다.
- ClipShots: 공개 Google Drive의 첫 압축 조각 앞 **512 MiB**만 받아 완전한 MP4 **54개**를
  추출했다. 부분 압축파일을 완전한 아카이브로 취급하지 않는다. 끝에서 잘린 영상은 선정하지 않았다.
- ClipShots 추출본은 `only_gradual` 분할이다. 이 분할의 경계 주석은 cut 전체를 망라하지 않으므로
  별도 후보 검토를 수행했다. 개발/평가는 **이번 과제의 원본 단위 분할**이며 공식 ClipShots test가 아니다.
- 선택한 MP4 원본 네 개와 처리본 여섯 개는 FFmpeg 전체 디코딩 오류 0건이다.
  TUM 선택 PNG들은 인코딩 과정에서 전체 디코딩했다.
- 원본/압축파일/처리본 SHA-256과 취득 주소를 기록했다. 로컬 해시이며 제공자의 공식 체크섬과
  비교했다는 뜻은 아니다.

## 목록과 주석

| 파일 (`data/etri_long_video_20260924/` 기준) | 내용 |
| --- | --- |
| `metadata/selection_protocol.json` | 복원 전 기록한 전환 분류·원본 분할·시간축 규칙 |
| `metadata/selections.json` | 원본 ID·출처·구간·분할·이용 조건·선정 이유 |
| `manifest.json`, `manifest.csv` | 실제 경로·해시·길이·FPS·프레임 수·단계별 상태 |
| `annotations/source_boundaries.json` | 전환 후보 시각, 지지/기각/모호 판정, 검수 근거 |
| `annotations/source_events.json` | 객체 및 대략적인 동작·사건 구간; 정밀 행동 정답은 아님 |
| `annotations/reconstruction_errors.template.csv` | 향후 복원 오류 주석 양식. 비어 있는 것을 오류 0건으로 해석하지 않음 |
| `metadata/*_time_mapping.json` | TUM 출력 프레임과 원본 PNG·촬영 시각 대응 |
| `reports/source_validation.json` | 원본 전체 디코딩 또는 TUM 프레임 대응 검증 |
| `reports/source_inventory.json` | 확보한 MP4 104개의 실제 길이·해시. 60초 이상은 88개; TUM은 별도 RGB 시퀀스 |
| `reports/review/` | 원본 개요와 경계 앞뒤 검토 이미지 |
| `freeze_manifest.json` | 이번 입력·선정·주석 파일과 준비 코드의 해시 |

저/중/고 분류는 이 예비 실험에서 각각 0–1, 2–8, 9개 이상 전환/분을 기준으로 삼고
고전환에는 객체·배경 변화도 확인했다. 세 유형에 서로 다른 데이터셋을 사용했으므로 유형 효과와
데이터셋 효과가 섞일 수 있다. 각 유형 2개는 실행 가능성·오류 탐색용이며 일반화 성능 입증 규모가 아니다.
모델 재학습이나 SKEM teacher label 생성은 수행하지 않았다.

## 출처·이용 안내

- [TUM 제공자](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)는 CC BY 4.0을 명시한다.
- [TVSum 제공자](https://github.com/yalesong/tvsum)는 CC BY 3.0과 2019년 직접 배포 전환을
  안내한다. 압축파일의 과거 Yahoo 조건과 차이가 있어 두 안내문을 모두 보존하고
  `license_status`에 미확정 상태를 남겼다.
- [ClipShots 제공자](https://github.com/Tangshitao/ClipShots)의 공개 배포 경로를 사용했다.
  저장소 MIT 표기를 원본 영상 전체에 적용하지 않았으며, 원본 영상 이용 조건은 별도 확인 대상으로 기록했다.
- 영상은 현재 내부 연구용 로컬 자료다. 공개 업로드·재배포·접근 신청·메일 발송은 수행하지 않았다.

## 재실행과 다음 단계

기존 데이터와 고정 평가 목록은 보존했다. 새 전처리는
[`prepare_etri_long_videos.py`](../scripts/prepare_etri_long_videos.py)에서 실행한다.
원본과 선정 JSON이 준비된 상태에서 다음 명령으로 처리본을 다시 만들 수 있다.

```bash
/home/sangukbae/anaconda3/envs/lgvsc/bin/python scripts/prepare_etri_long_videos.py build \
  --selections data/etri_long_video_20260924/metadata/selections.json
```

전처리 재실행 시 처리본·manifest가 갱신되므로 이후 실험을 진행했다면 별도 데이터 디렉터리를 사용하고
입력 동결 해시를 새로 기록한다. 기존 16초 전처리와 `max_frames=384`는 사용하지 않았다.

다음 작업은 `low_dev`의 시간축을 유지하는 장시간 실행 경로를 확인하고 AWGN 기준선 복원을
완료하는 것이다. 이번 입력 준비는 60초 복원 성공이나 할루시네이션 완화 효과의 근거가 아니다.
