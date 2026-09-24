# 단기 개선에 사용할 코드와 검증 상태

개정일: 2026-09-21. [단기 실험안](ETRI_LGVSC_HALLUCINATION_EXPERIMENT_PLAN_2026-09-21.md)의 구현 재사용 범위를 정리했다. 이번에는 로컬 소스·기존 산출물을 확인했으며 새로운 모델 실행·설치·장시간 검증은 하지 않았다.

**키프레임 보강은 기존 코드를 재사용하고, 씬 경계 처리에 개발을 집중한다. 저해상도 보정은 기존 후처리로 짧게 확인한다.**

## 1. 재사용과 추가 구현

| 우선순위 | 현재 코드 | 확인한 동작 | 남은 작업 |
|---|---|---|---|
| 1: 최대 1초 키프레임 | [ablation_transport.py](../../src/semantic_transmission/ablation_transport.py)의 `densify`, `preserve_transport`; [webvid_ablation.py](../../src/semantic_transmission/webvid_ablation.py)의 `initialize_variant`, `worker` | 기존 SKEM 프레임 보존, 최대 간격 보충, 공통 시각 심벌 재사용, 추가 프레임 실제 송수신 | 60초 입력과 별도 실행 프로필에 연결. 영상 전체 길이·비용 확인 |
| 1: 기존 정렬 recipe | [mydemo_new_align_sh.py](../../04_semantic_decoder/scripts/mydemo_new_align_sh.py) | 끝점 조건과 overlap·출력 연결 정책을 별도로 제어 | 생성된 `receiver/decoder_config.py`에 실제 전달됐는지 확인. 단독 화질 개선으로 표시하지 않음 |
| 2: 씬 경계·주기 갱신 | [workers.py](../../src/semantic_transmission/workers.py)의 caption/flow; [codec_transport.py](../../src/semantic_transmission/codec_transport.py); 위 decoder | 구간 caption·광류·키프레임 전달 및 이전 생성 참조 경로 존재 | 자동 shot detector, 경계 직전/직후 키프레임, 구간 분리, 새 caption/flow 전송, `scene_id`·유효범위, 경계에서 이전 overlap 제거 연결 |
| 3: 저해상도 보강 | [quality_validation_worker.py](../../src/semantic_transmission/quality_validation_worker.py)의 `encode_side`, `side_channel`; [quality_methods.py](../../src/semantic_transmission/quality_methods.py)의 `low_resolution_constraint` | 72×40 영상 인코딩, 실제 LDPC/16QAM/AWGN·무결성 확인, 받은 영상으로 저주파 후처리 | 선택한 생성물에 연결하고 보정 전후 의미 오류 검수. 새 장시간 경로의 전송·메모리 확인 |
| 공통: 60초 지원 | [edit_movie.py](../../01_data_prep/edit_movie.py), [lgvsc_variable.json](../../configs/lgvsc_variable.json), 기존 순차 decoder | 기존 전처리는 16초 절단, 가변 설정은 `max_frames=384` | 별도 긴 입력 전처리·설정, PTS·1,440프레임, 구간 간 상태·재개 확인 |

**주의할 두 동작:** `subdivide_metadata`는 기존 caption·flow를 복사한다. 키프레임을 추가했다고 동적 설명도 갱신되는 것은 아니다. 또한 기존 생성 참조는 다음 구간에 전달되므로, shot 경계 키프레임만 추가해도 이전 장면의 영향이 자동으로 끊어진다고 가정하면 안 된다.

모델·텍스트 캐시는 기존 [중복 제거 구현](../EXACT_REUSE_VALIDATION.md)을 유지해 실행 비용을 줄인다. 그 검증은 작은 입력에서의 출력 일치이며, 이번 개선법의 의미 품질이나 60초 실행을 보장하지 않는다.

## 2. 실행 근거의 경계

| 산출물 | 확인한 범위 | 미확인 범위 |
|---|---|---|
| [ETRI01 결과](../../outputs/diagnostics/etri01_ablation_20260917/REPORT.md) · [측정 JSON](../../outputs/diagnostics/etri01_ablation_20260917/comparison_summary.json) | 10초·1개 seed의 정렬/키프레임/수동 caption 비교. 1초 보강의 큰 화질 개선과 4.09배 전송량 | 다른 영상·장시간 일반화, 정량적인 Added/Missing/Distorted 감소 |
| [WebVid 결과](../../outputs/webvid1_ablation_single_subject_611ed735e038/REPORT.md) · [측정 JSON](../../outputs/webvid1_ablation_single_subject_611ed735e038/comparison_summary.json) | 14초·1개 seed의 네 조건 실행. 키프레임 보강 후 지표별 개선/악화가 섞임 | 동작 검수와 장시간 검증 |
| [보정·보조영상 결과](../../outputs/quality_validation_e8babaef576c/REPORT.md) · [상태 JSON](../../outputs/quality_validation_e8babaef576c/summary.json) | `completed_validations=[1,2]`. 기존 짧은 두 영상의 보정·균일/적응형/보조영상 비교 | 검증 3의 별도 5편 결과 없음. `hallucination_review=PENDING` |

`execution=COMPLETED`만 보고 세 검증 전체가 끝났다고 해석하지 않는다. 기존 [quality_validation 안내](../QUALITY_VALIDATION.md)는 실행기·당시 구현 확인을 설명하므로, 이번 우선순위는 위 실제 결과의 완료 목록과 수치를 함께 사용했다.

## 3. 최소 확인 후 긴 실행으로 이동

1. **전송:** 원본을 수신단이 읽지 않는지, 기존 시각 심벌/수신 PNG 재사용, 추가 프레임·scene·caption·보조영상·패딩 비용을 확인한다.
2. **경계:** 두 실제 장면 사이에서 caption과 생성 overlap이 모두 갱신되는지, 경계 누락 시 주기 갱신이 남는지, 정상 연속 구간에서는 참조가 유지되는지 확인한다.
3. **시간축:** 짧은 구간과 경계 주변에서 중복·누락·길이 오류를 확인한 뒤 20초→60초로 늘린다. 실제 경계 처리 검증에는 자동 검출과 독립 정답을 구분한다.
4. **재사용:** 송신·기준선·완료된 복원은 해시 확인 후 재사용하고 GPU 생성과 평가를 순차 실행한다. 후처리 비교 때문에 생성기를 다시 실행하지 않는다.

기존 실행 스크립트는 짧은 영상용이다. 그 명령을 그대로 실행하면 단기 9회 계획이나 씬 대응·60초 검증이 자동 수행되는 상태가 아니다. 이번 개정은 이를 구현 완료로 바꾸지 않는다.

## 4. 외부 코드의 사용 범위

[TransNet V2](https://github.com/soCzech/TransNetV2)는 씬 경계 검출 후보이며 새로 선정한 실행 경로·가중치 버전을 구현 시 고정한다. 기존 9개 저장소 정적 감사에 포함된 것으로 세지 않는다.

SparkVSR·STCDiT의 참조 활용, DynamicDPS의 관측 제약, HalluGen·sFRC의 평가 설계는 참고한다. 외부 복원 모델·새 검출기·재학습은 첫 제출물 이후에 검토한다. 외부 논문의 성능을 로컬 LGVSC 개선 수치로 사용하지 않는다.

9개 저장소·35개 선택 소스·24개 참조 위치의 전체 기록은 [개정 전 코드 감사](archive/2026-09-21-before-prioritization/ETRI_HALLUCINATION_CODE_AUDIT_2026-09-21.md)에 보존했다. [기존 evidence JSON](ETRI_HALLUCINATION_EVIDENCE_2026-09-21.json)은 당시 외부 코드·논문·로컬 파일의 스냅샷이며, 이번에 확인한 성능 수치는 위 결과 JSON에서 추적한다. 정적 감사는 가중치 실행·논문 재현·라이선스 전체 검증을 뜻하지 않는다.
