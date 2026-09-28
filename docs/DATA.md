# 데이터와 주석

갱신: **2026-09-28**. [과제 현황](README.md) · [평가 기준](ETRI_DEVELOPMENT_PLAN.md#evaluation)
원본은 내부 성능검증용이다. 원본·가중치·전체 결과는 Git에 포함하지 않는다.
현재 `data/`는 `/home/sangukbae/datasets/semantic_transmission`을 가리킨다.

<a id="inputs"></a>
## 장시간 평가 입력

| 구성 | 내용 |
|---|---|
| 본 세트 | 서로 다른 원본 60개의 연속 60초; 저/중/고 전환 후보 각 20개 |
| 출처 | 각 유형 TVSum 10개 + ClipShots 10개 |
| 분할 | 개발 18 · 교정 12 · 평가 30개; 관련 콘텐츠 54그룹, 평가 27그룹 |
| 확장 | 평가 원본 6개의 연속 120초; 부모와 같은 분할, 독립 표본 수를 늘리지 않음 |
| 형식 | 576×320·24fps; 60초 1,440프레임, 120초 2,880프레임 |
| 실제 검증 | 처리본 66개·103,680프레임 전수 디코딩과 시간 대응 검사 |
| 실행 계획 | 66입력 × 3채널 시드 × 기준선/완화 = 396개 **예정** 작업 |

초기 6개 파일럿은 `data/etri_long_video_20260924/`, 본 세트는 `data/etri_benchmark_v1_20260924/`다.
유형 분류는 예비 경계 후보와 AI 검토에 기반한다. 60초의 후보 수 0–1 / 2–8 / 9개 이상을 사용하며,
shot 편집 경계와 의미적 scene 변화는 구분한다. 확정 독립 정답은 아직 없다.

원본 하나의 연속 시간축을 유지했다. 다른 영상을 연결하거나 루프·감속·보간하지 않았다.
23.976→24fps 정규화로 전체 66파일에서 20프레임이 재사용됐고 최대 표집 오차는 30.083ms다.
최대 원본 인접 프레임 간격은 41.709ms다. 기준 입력은 리사이즈·시간 정규화 후 CRF 0으로 저장했다.
원본과 픽셀 동일 영상이라는 뜻은 아니다. 여백·원본 자막을 기록하고 유효 영상 영역의 지표도 별도로 본다.

확보한 원본은 TVSum 50개·ClipShots 257개이며 248개가 60초 이상이다.
ClipShots는 `only_gradual` 아카이브 앞부분 표본으로 전체 데이터셋의 무작위 표본이 아니다.
동일 파일·긴 정렬 중복 후보는 0개였으나 짧은 중복·재크롭·사전학습 노출까지 배제한 것은 아니다.
선정·분할에는 복원 점수를 사용하지 않았다.

<a id="review"></a>
## 검수와 현재 실행 범위

66개 입력의 2초 간격 및 마지막 프레임 **2,226장**을 AI 한 명이 검토했다. 의심 구간은 인접 프레임·4fps로 보충했다.
전 시간대의 표본 검토이며 전 프레임 시각 검수·독립 두 명 검수·연속 사건 정답은 아니다.
`tv_low_01_120s`에는 자동 검출이 놓친 와이프가 있어 low 분류 재검토가 남아 있다.
AI의 성긴 표집이 짧은 삽입 컷을 놓친 사례도 있으므로 표본 일치를 정답 완전성으로 보지 않는다.

- 원본 주석: 독립 검수·불일치 조정 후 객체 출입·가림·동작·scene·관찰 불가 구간을 확정한다.
- 복원 주석: 방법 이름을 가리고 원본/기준선/완화를 비교해 Added/Missing/Distorted 시각·영역·잔존·새 오류를 기록한다.
- 오류가 없어도 전 구간 검수와 오류 0을 명시한다. 빈 CSV와 AI 표본은 완료 정답이 아니다.
- **실제 모델 완료:** 저전환 개발용 `tv_low_08` 한 편의 60초 기준선. 중·고전환·120초 및 전체 완화 평가는 남았다.

| 로컬 경로 | 자료 |
|---|---|
| `etri_benchmark_v1_20260924/manifest.json`, `manifest.csv`, `index.html` | 입력·분할·재생/검수 페이지 |
| `metadata/time_mapping/`, `annotations/`, `plans/` | 원본 시각 대응·예비 주석·예정 대응 작업 |
| `reports/readiness_audit.json`, `freeze_manifest.json` | 준비 판정·동결 해시 |
| `etri_ai_source_review_20260925/reviews.json`, `index.html`, `audit.json` | AI 관찰·검토 범위·검사 |

입력 준비의 판정은 `PASS_INPUT_PREPARATION`이며 독립 정답·성능평가 인증은 `false`다.
[동결 입력 설계 원문](https://github.com/SangukBae/semantic_transmission/blob/f41bc8885e8d8409e59b162fe7be63c0d11ee737/docs/ETRI_BENCHMARK_V1.md) ·
[AI 검수 원문](https://github.com/SangukBae/semantic_transmission/blob/f41bc8885e8d8409e59b162fe7be63c0d11ee737/docs/ETRI_AI_SOURCE_REVIEW.md)

<a id="legacy"></a>
## 기존 데이터와 학습의 구분

| 자료 | 확보 기록과 용도 |
|---|---|
| WebVid | 53/55개 원본 해시 일치. 미확보 ID `4440995`, `1018898026`; 다른 영상으로 대체하지 않음 |
| Kinetics-400 | 14/14개 원본·라벨 확인; 행동 분류 평가 자료 |
| OpenImages | 100,000장 확보·해시·디코딩 검사; **보유가 로컬 모델 학습 완료를 뜻하지 않음** |
| 기존 ETRI | 짧은 개발 영상 10개와 주석 보존 |
| TUM RGB | 두 시퀀스는 파일럿 보조 진단용; 본 세트의 유형별 평균에 섞지 않음 |

WebVid·Kinetics 목록은 [WebVid CSV](eval_manifest_webvid.csv), [Kinetics CSV](eval_manifest_kinetics.csv)에 있다.
원본은 `webvid55/raw/`, `kinetics400_14/raw/`, 학습 후보 이미지는 `openimages_ntscc/train/`에 둔다.
최신 확보 상태는 `data/reports/provisioning_summary.json`, `evaluation_audit.json`으로 확인한다.

TVSum·ActivityNet·SumMe 등 영상 요약 자료의 사람 중요도는 LGVSC 키프레임 정답과 다르다.
학습형 선택기에는 SKEM의 상태 의존 비교·선택/생략 궤적을 별도로 확보하고 원본별로 분할해야 한다.
[선택기 개발 후보](ETRI_DEVELOPMENT_PLAN.md#candidates)

## 재검증과 보존

```bash
source scripts/activate.sh
python scripts/prepare_lgvsc_datasets.py audit --root "$HOME/datasets/semantic_transmission"
python scripts/etri_benchmark_package.py audit
python scripts/etri_ai_source_review.py audit
```

입력 동결본을 덮어쓰지 않고 변경은 새 버전으로 남긴다. 원본 영상 이용 조건과 모델/코드 라이선스는 별개다.
공개 다운로드 가능성을 재배포 허가로 확대하지 않는다. 기존 조건 검토·획득 명령·출처는
[데이터 출처 원문](https://github.com/SangukBae/semantic_transmission/blob/f41bc8885e8d8409e59b162fe7be63c0d11ee737/docs/DATA.md),
[로컬 구축 원문](https://github.com/SangukBae/semantic_transmission/blob/f41bc8885e8d8409e59b162fe7be63c0d11ee737/docs/LOCAL_DATASETS.md),
[장시간 선정 조사](https://github.com/SangukBae/semantic_transmission/blob/f41bc8885e8d8409e59b162fe7be63c0d11ee737/docs/research/ETRI_LONG_VIDEO_DATASET_REVIEW_2026-09-24.md)에 보존한다.
