# 데이터셋

[과제 현황](README.md) · [평가 기준](ETRI_DEVELOPMENT_PLAN.md#evaluation)

## 60초 복원 평가에 쓰는 데이터

- **TVSum:** 영상 요약용 데이터셋; 다양한 일반 영상의 복원 평가에 사용.
- **ClipShots:** 장면전환 검출용 데이터셋; 장면이 바뀔 때의 복원 오류 평가에 사용.

<a id="inputs"></a>
## 평가 영상 구성

- 원본: TVSum 30개 + ClipShots 30개 = 서로 다른 영상 60개.
- 길이: 원본별 연속 60초; 그중 6개는 120초 구간도 준비해 총 66파일.
- 난도: 장면전환 저·중·고 후보 각 20개; 독립 검수로 확정 필요.
- 분할: 개발 18개 · 교정 12개 · 평가 30개; 확장본은 원본과 같은 분할.
- 형식: 576×320, 초당 24프레임; 반복·다른 영상 연결로 길이를 늘리지 않음.
- 예정 작업: 66입력 × 채널 시드 3개 × 기준선·완화 = 396개; 실행 완료 수 아님.
- 입력 위치: `data/etri_benchmark_v1_20260924/`.

<a id="review"></a>
## 준비·검수 상태

- 파일 검사: 66개 전체의 디코딩·시간 대응 검사 완료.
- AI 검토: 표본 프레임 2,226장 확인; 독립 정답·전 프레임 검수는 미완료.
- 실제 복원: 개발용 `tv_low_08` 한 편, 기존 110장·AI 선택 85장·혼합 79장(캡션 v1/v2)으로 60초 완료. [비교](ETRI_COMBINED_REFERENCE_DIAGNOSIS.md#caption-v2)
- 검토: v1/v2 78시점·최종 MP4 12시점, 기존 LGVSC/v2 35시점·최종 MP4 8시점 확인. 표본 검토이며 독립 정답은 미확보. [기존 방식 비교](../outputs/etri_hybrid_keys_tv_low_08_v1/reconstruction_assistant_captions_v2/analysis_lgvsc/review.html)
- 현재 기본 입력: `tv_low_08`의 혼합 키프레임 79장·수정 캡션 v2 78개.
- 추가 짧은 입력: `single_subject` / `candle_flowers` / `person_walk`, 키프레임 20·9·16장·캡션 v2 19·8·15개로 FP32 복원 완료. 같은 시간축의 전 프레임 화질 지표와 36시점 AI 표본 비교 완료; 독립 오류 정답은 미확보. 60초 평가 세트와 별도. [비교](../outputs/etri_latest_short3_20261001/analysis_lgvsc/review.html)
- 최신 기본: 위 4편을 두 구현 오류 수정이 포함된 FC-LGVSC로 복원 완료. 최종 MP4 29시점 AI 검토; 독립 오류 정답은 미확보. [기존 LGVSC 비교](../outputs/fc_lgvsc_four_comparison_20261001/review.html)
- 남은 주석: 객체 출입·동작·장면 경계와 복원 오류의 시각·영역 확정.
- 검수 방식: 독립 검수자 2명, 방법 이름을 가린 비교, 불일치 조정.
- 선정 한계: ClipShots 일부에서 확보한 표본이며 전체 데이터셋의 무작위 표본은 아님.

<a id="full-extraction"></a>
## WebVid·TVSum 전체 원본 추출 — 2026-10-03 상태

60초 ETRI 평가 세트와 별개로 WebVid 53편·TVSum 50편을 전체 길이·24 FPS·576×320으로 준비했다. 반복·길이 제한 없이 총 324,690프레임이며 복원 성능 평가 완료 수가 아니다.

| 산출물 | 확인한 상태·해석 |
|---|---|
| `fc_lgvsc_webvid_tvsum_full_20261001` | 103편 직접 선택·캡션 산출물 생성 기록: 키프레임 15,660개·캡션 15,557개. 102편의 SKEM 비교가 0회여서 원래 순차 혼합 선택을 모두 수행한 결과로 해석하지 않음. |
| `fc_lgvsc_webvid_tvsum_faithful_20261002` | 필수/선택 후보 재분류 103편, 순차 선택 64편(WebVid 53·TVSum 11) 완료 기록. Qwen 비교를 위해 중단; 쌍별 점수·재개 명령 보존. |
| 같은 이름의 `_v2` 보정본 | WebVid 031의 124번 프레임 손 재등장을 보호하도록 별도 수정. 해당 재선택·변경 TVSum 캡션·전체 감사는 미완료. 인계 기록상 유효 WebVid 캡션 52편. |
| `fc_lgvsc_auto_20261002` | 별도 Qwen3-VL 자동 배치: 7편 완료·1편 실패 후 `STOPPED`. 위 assistant 작성 결과와 합산하지 않음. |

- 캡션 재사용은 구간·표본 인덱스·원본 해시가 모두 같은 경우만 허용. 선택 변경 구간에는 새 관찰·캡션 검증 필요.
- 이전 `extractions_complete=103`은 이전 직접 선택 산출물 상태다. 보정본 완료나 103편 복원·독립 의미 오류 검증 완료를 뜻하지 않는다.
- 문맥 캡션의 78구간·대응 PLLaVA 65구간은 동일 개발 영상 안의 구간이다. 독립 영상 78편/65편으로 세거나 미사용 평가 세트로 취급하지 않는다.
- 근거: [원본 준비](../outputs/fc_lgvsc_webvid_tvsum_full_20261001/preparation_status.json) · [직접 선택 기록](../outputs/fc_lgvsc_webvid_tvsum_full_20261001/extraction_status.json) · [중단·재개 기록](../outputs/fc_lgvsc_webvid_tvsum_faithful_20261002/paused_for_qwen_caption_comparison.json) · [보정 인계](../outputs/fc_lgvsc_webvid_tvsum_faithful_20261002_v2/handoff_status.json).

<a id="public-keyframes"></a>
## 공개 요약·키프레임의 ETRI 적합성 — 2026-09-29 검증

**원본 평가 영상은 TVSum+ClipShots 유지. 공개 요약으로 SKEM을 대체했을 때의 복원 품질은 미검증.**

| 데이터셋·제공 정보 | 실제 확인 | 사용 판단 |
|---|---|---|
| [TVSum](https://github.com/yalesong/tvsum): 2초 구간별 중요도 점수 | 원본 50개 모두 60초 이상; 기존 60초 입력 30개 전체 디코딩 통과 | **현재 평가 원본으로 적합**; 점수로 키프레임을 고르는 방법은 별도 검증 |
| [SumMe](https://data.vision.ee.ethz.ch/cvl/SumMe/SumMe.zip): 사람이 선택한 요약 구간 | 주석 25개 중 23개가 60초 이상; Jumps·Fire Domino는 제외 | **보조 원본 후보**; 원본 WebM과 주석 시간축 확인 후 사용 |
| [VSUMM](https://github.com/sandraavila/vsumm): 사람이 고른 키프레임 JPEG | Open Video 50개에 요약 250묶음·JPEG 2,162장; 영상 표본 3개에서 디코딩 오류 | **직접 재사용 보류**; 일부 60초 미만, JPEG 번호와 원본 대응도 확인 필요 |

- **시간 대응:** TVSum 2개는 주석이 영상보다 1프레임 짧음; 변환·자르기 후 시각 대응 검사 필요.
- **파일 형식:** SumMe의 Cooking은 WebM 1,286프레임으로 주석과 일치; MP4는 1,280프레임만 디코딩됨. 전체 영상의 호환성을 확인한 것은 아님.
- **E1 시간축:** 원본의 연속 60초를 사용; 요약 장면 연결·키프레임 슬라이드쇼로 대체 불가. 저·중·고 전환 분류는 독립 확인 필요.
- **E2 오류·E3 지표:** 세 자료의 요약 주석에는 복원 오류의 추가·누락·왜곡 정답이 없음; 시각·영역별 독립 주석 필요.
- **E4 전송량:** 키프레임 감소만으로 절감 성공 판정 불가; 복원 품질·새 오류와 실제 전송 비용을 함께 비교.
- **메일 요구:** AWGN 전후 비교, 장면전환·패킷 갱신·누락 대안, 동작 정보·학습 이력은 별도 시스템 검증 대상.
- **SKEM 생략:** 사람의 요약 주석을 활용한 비교 실험으로 표시; 미지 영상의 자동 선택 성능이나 품질 유지 근거로 사용 불가.
- **검증 범위:** TVSum 전체 주석·원본 헤더와 기존 60초 입력 30개, SumMe 전체 주석·2개 영상의 두 형식, VSUMM 요약 목록·영상 3개. 새 GPU 복원·독립 오류 검수는 미실시.
- **근거:** [검사 결과](../outputs/etri_public_summary_fit_20260929/RESULT.json) · [저장 자료 재검사 코드](../scripts/audit_public_summary_fit.py). 기존 평가 입력·분할은 유지.

<a id="legacy"></a>
## 기타 사용 자료

- **DAVIS:** 영상 객체 분할용 데이터셋; 의미 지표 검증에 사용.
- **WebVid:** 설명 문장이 붙은 웹 영상; 기존 짧은 영상 평가용, 55개 중 53개 확보.
- **Kinetics-400:** 사람 행동 분류용 영상; 기존 행동 평가용 14개 확보.
- **OpenImages:** 객체가 담긴 이미지 모음; 학습 후보 10만 장 확보, 로컬 학습 완료 근거는 없음.
- **기존 ETRI 영상:** 초기 개발에 사용한 짧은 시험영상 10개.
- **TUM RGB:** 연속 촬영 RGB 영상; 초기 보조 진단용 시퀀스 2개.

## 후속 후보 — 현재 평가 세트에는 미포함

- **ActivityNet:** 다양한 사람 행동이 담긴 영상 데이터셋.
- **SumMe:** 사람이 요약 구간을 표시한 영상; 길이·시간축 확인 후 보조 평가 후보.

## 자료 찾기

- 데이터 루트: `data/` → `/home/sangukbae/datasets/semantic_transmission`.
- 입력·분할: `etri_benchmark_v1_20260924/manifest.json` · `index.html`.
- 검수 기록: `etri_ai_source_review_20260925/`.
- 기존 목록: [WebVid](eval_manifest_webvid.csv) · [Kinetics-400](eval_manifest_kinetics.csv).
- 이용 범위: 원본은 내부 성능검증용; 데이터셋별 이용 조건 확인, Git에 원본 미포함.
- 상세 근거: [선정·정규화·검수·재검증 명령](https://github.com/SangukBae/semantic_transmission/blob/4f566feabe213b870e5c1e441ef9b369b348d589/docs/DATA.md).
