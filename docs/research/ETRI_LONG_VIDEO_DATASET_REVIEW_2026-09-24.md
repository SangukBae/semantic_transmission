# ETRI 60초 이상 평가영상용 공개 데이터셋 조사

> 최신 후속 결과: ClipShots 원본을 257개로 확대하고 TVSum과 교차하여
> [60초 평가 입력 60개·120초 확장 6개](../ETRI_BENCHMARK_V1.md)를 준비했다.
> 기존 파일럿과 아래 취득 전 조사 기록은 당시 상태로 보존한다.

> 같은 날 후속 실행: TUM 2개 시퀀스, TVSum 50개, ClipShots 부분 확보 54개에서
> 60초 입력 6개를 준비·검증했다. [실제 확보 결과](../ETRI_LONG_VIDEO_INPUTS.md)를 확인한다.
> 아래 내용은 확보 전에 수행한 조사 기록이며 당시의 미확인 상태를 보존한다.

- 조사일: 2026-09-24
- 요구사항: [ETRI 후속 메일](../ETRI_FOLLOWUP_EMAIL_SUMMARY.md)의 실제 연속 60초 이상, 장면전환 저·중·고, 객체 출현·퇴장 및 동작 변화 평가.
- 범위: 제공자 문서·논문, 배포 주소, 일부 메타데이터 확인. 전체 영상 다운로드, 영상별 재생 검수, 장면 경계 주석, LGVSC 실행은 수행하지 않았다.
- 결론: 저전환 시험은 **TUM RGB-D**, 일반 영상은 **TVSum**, 장면전환 스트레스 시험은 **ClipShots**가 유력하다. TVSum 배포물의 상충하는 이용 안내와 ClipShots 원본 접근은 미해결이다. 세 데이터셋을 모두 확보했다고 해석하면 안 된다.

## 후보와 실제 확인 수준

| 후보 | 확인한 원본·주석 특성 | ETRI에서 맡길 역할 | 접근·이용 조건 및 한계 |
| --- | --- | --- | --- |
| TUM RGB-D | 실제 연속 촬영 RGB 이미지와 타임스탬프, 깊이·카메라 궤적. 60초 초과 시퀀스 존재 | 저전환에서 객체 외형 유지, 시점 변화, drift·깜빡임 시험 | 공식 데이터 CC BY 4.0. 실내 중심이므로 일반 영상과 병행. RGB 원본과 타임스탬프를 사용하고 깊이·카메라 정답은 LGVSC 입력에 추가하지 않음 |
| TVSum | 50개, 10개 범주. 배포 메타데이터상 98–647초, 모두 60초 이상. 원본 URL·제목·길이 및 중요도 점수 | 다양한 일반 영상에서 중·고전환 구간을 우선 선별 | 약 672 MB 압축파일 응답 확인. 공개 README의 CC BY 3.0 및 직접 배포 안내와 압축파일 내부의 과거 Yahoo 계약·비상업 연구 조건이 공존. 적용 조건은 미확정 |
| ClipShots | 영상 1–20분, cut 및 점진적 전환의 프레임 구간 주석. test JSON 500개 항목 확인 | 씬 전환 검출 실패, 이전 장면 잔존, 키프레임·packet 갱신 시험 | 공식 Google Drive/Baidu 링크 존재하나 원본 다운로드 미확인. 저장소 MIT 표기를 원본 영상의 포괄적 이용허락으로 해석하지 않음 |
| ActivityNet Captions | 긴 영상의 사건별 시작·종료 시각과 설명. 제공자 기준 약 2만 영상, 849시간 | 동작 순서·행동 변화 및 구간 caption 평가의 후속 확장 | caption/feature 파일만으로는 LGVSC 입력 불가. 실제 원본을 별도로 확보해야 하며 이번 조사에서 원본 다운로드는 확인하지 못함 |
| 50 Salads | 25명 × 2회 조리, 총 4시간 이상 RGB-D와 상세 행동 주석 | 저전환에서 손·도구·물체 상호작용 및 행동 순서 시험 | 공식 CC BY-NC-SA 4.0. 대학 메타데이터 페이지는 열리지만 연결된 기존 원본 서버는 이번 환경에서 DNS 오류 |

공식 근거:

- TUM: [개요·라이선스](https://cvg.cit.tum.de/data/datasets/rgbd-dataset), [시퀀스별 설명·길이·배포](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download), [원본 파일 형식](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats).
- TVSum: [저자 저장소](https://github.com/yalesong/tvsum), [저자 배포 서버](https://people.csail.mit.edu/yalesong/tvsum/), [배포 압축파일](https://people.csail.mit.edu/yalesong/tvsum/tvsum50_ver_1_1.tgz).
- ClipShots: [저자 저장소 및 다운로드 안내](https://github.com/Tangshitao/ClipShots), [test 경계 주석](https://github.com/Tangshitao/ClipShots/blob/master/annotations/test.json).
- ActivityNet Captions: [저자 프로젝트 페이지](https://cs.stanford.edu/people/ranjaykrishna/densevid/).
- 50 Salads: [제공 대학 데이터 페이지](https://discovery.dundee.ac.uk/en/datasets/50-salads/).

## 바로 시작할 저전환 후보

아래 길이는 공식 시퀀스 설명의 값이며, 실제 RGB 파일의 재생 길이를 이번에 측정한 값은 아니다.

| TUM 시퀀스 | 공식 길이 | 선정 이유 |
| --- | --- | --- |
| `freiburg2_desk_with_person` | 142.08초 | 사람이 책상 주변에서 물체와 상호작용. 첫 60초 실행 및 동작·객체 유지 진단 후보 |
| `freiburg3_long_office_household` | 87.09초 | 사무·가정 물체를 연속 이동 촬영. 시점 변화와 객체 외형 유지 진단 후보 |
| `freiburg2_dishes` | 100.55초 | 식기·도구를 여러 시점에서 촬영. 작은 객체 누락과 외형 왜곡 진단 후보 |

TUM은 원래 SLAM 평가 데이터셋이다. 이를 ETRI의 저전환 입력으로 활용하자는 것은 본 조사의 실험 설계 제안이며, 제공자가 semantic transmission 평가 적합성을 보증했다는 뜻은 아니다. 영상 검수 후 최종 채택한다. 제공되는 2 Hz 파생 자료나 미리보기 영상보다 원본 RGB와 타임스탬프를 우선한다.

배포 파일 HEAD 확인: `freiburg2_desk_with_person.tgz` 2,552,157,258바이트, 대응 RGB AVI 40,863,846바이트, `freiburg3_long_office_household.tgz` 1,483,556,251바이트 모두 HTTP 200이었다. 공식 링크는 `webshare.cvg.cit.tum.de`로 이동했다. AVI의 실제 길이·프레임률·원본과의 대응은 아직 검사하지 않았다. 응답 기록은 조사 자료의 `tum_access.json`에 있다.

## TVSum 실제 메타데이터 확인

- 압축파일 HEAD: HTTP 200, `Content-Length=671779858`.
- 앞 262,144바이트 범위 요청: HTTP 206. 전체 영상 파일을 받지 않고 포함된 README와 작은 metadata ZIP을 확인했다.
- `ydata-tvsum50-info.tsv`에서 헤더 제외 50개 원본 ID·URL·제목·길이를 확인했다.
- 최소 98초, 최대 647초, 60초 이상 50개. 길이는 제공자 메타데이터 값이며 ffprobe 측정값이 아니다.
- 제공자가 부르는 shot importance는 고정 2초 구간의 중요도다. 실제 장면전환 경계 정답으로 사용하면 안 된다.
- 공개 안내와 압축파일 내부 조건 차이는 공급자 문서의 상태 차이로 기록한다. 옛 조건이 폐기됐다고 단정하거나 사용 가능 여부를 확정하지 않았다.
- 원본 URL이 있으므로 후속 확보 때 영상별 출처와 이용 조건도 함께 기록할 수 있다.

원시 조사 자료는 `outputs/research/etri_long_video_datasets_20260924/`에 저장했다. 여기의 TVSum README는 조사용으로 확인한 배포물 안내문이며 로컬 프로젝트의 실행 지침이 아니다.

## 선별·평가 방법 제안

1. **최초 개발 영상 1개:** TUM `freiburg2_desk_with_person`에서 사람이 물체와 상호작용하는 실제 연속 60–90초를 재생 확인 후 선택한다. 시간 유지, 입력·출력 길이 일치, 긴 구간 연결을 먼저 점검한다.
2. **작은 예비 세트:** 저·중·고 전환별 개발용 1개와 평가용 1개, 총 6개 독립 원본으로 시작한다. 이는 실행·문제 발견용 최소 제안이며 충분한 통계 검증 규모를 뜻하지 않는다.
3. **전환 유형은 영상별로 판정:** 단위 시간당 검수된 cut/점진적 전환 수, 가장 긴 무전환 구간, 객체 출현·퇴장 및 동작 변화를 함께 기록한다. 데이터셋 이름이나 카테고리만으로 난도를 부여하지 않는다. 저전환이 항상 쉬운 것도 아니다.
4. **연속성 유지:** 한 원본의 연속 구간만 취하고 반복·이어붙이기·감속은 하지 않는다. 기존 편집 전환은 그대로 보존한다. 원본에 없는 연결은 만들지 않는다.
5. **분할은 원본 단위:** 같은 원본의 다른 시간 구간이나 같은 촬영 세션을 개발용·평가용에 나눠 넣지 않는다. 여러 데이터셋을 섞으면 데이터셋별 결과도 별도 보고하여 도메인과 전환 난도를 혼동하지 않는다.
6. **원본 목록 고정 후 복원:** 동일 원본·구간·AWGN 조건에서 완화 전후를 비교한다. 독립 평가영상은 결과를 보고 유리한 구간으로 바꾸지 않는다.

원본 목록의 최소 필드:

```text
dataset, source_video_id, source_url, download_url, retrieved_at,
license_source, license_status, source_sha256,
duration_sec, fps_or_timestamps, width, height,
clip_start_sec, clip_end_sec, split, scene_transition_level,
boundaries_sec_and_type, boundary_review_status, key_object_action_events
```

현재 모델에 맞춘 24 FPS 변환이 필요해도 타임스탬프와 총 재생시간을 보존한다. 기존 16초 제한 전처리를 그대로 적용하면 이번 데이터 확보 목적을 충족하지 못한다.

## 기존 주석으로 대체할 수 없는 것

- TVSum의 요약 중요도는 객체 누락·추가·왜곡의 정답이 아니다.
- ClipShots의 shot 경계는 의미적 scene 경계와 완전히 같지 않다. 동일 장소에서 카메라만 바뀌는 cut도 있으므로 의미적 변경 여부를 보완한다. `only_gradual` 분할은 cut 주석이 완전한 분할로 취급하지 않는다.
- 행동 주석과 caption은 일부 사건 설명을 제공하지만 모든 작은 객체·동작을 망라하지 않는다.
- **Added/Missing/Distorted는 우리의 복원 출력에 대해 원본과 비교하여 추가 주석해야 한다.** 이번에 조사한 원본 데이터셋 어느 것도 LGVSC의 완화 효과를 이미 검증해 주지 않는다.

## 후순위 후보

- **SumMe:** 저전환 보완 가능성이 있으나 기존 [저자 페이지](https://gyglim.github.io/me/vsum/index.html)는 이번 확인에서 HTTP 404였다. 제3자 미러의 이용 표기를 원본 제공자의 조건으로 대신하지 않는다.
- **V3C1:** [제공자 측 분석 저장소](https://github.com/klschoef/V3C1Analysis)에서 데이터 이용 동의 및 접근 요청 경로를 안내한다. 광범위한 웹 영상 확장용으로 검토할 수 있지만 최초 60초 실험에는 확보 비용이 크다.

접근 확인과 이용 안내는 조사일의 상태다. 파일 응답 성공이 전체 다운로드·무결성·영상별 길이·장면 유형 검증을 뜻하지 않는다.
