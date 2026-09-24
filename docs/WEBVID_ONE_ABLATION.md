# WebVid 한 편: 네 조건 자동 비교

2026-09-21 역할: 기존 **14.04초 개발 영상**의 조건별 원인 분석·후보 비교를 안내한다.
[ETRI 후속 요청](ETRI_FOLLOWUP_EMAIL_SUMMARY.md)에 대한 장시간 평가 기준은
[후속 프로토콜](ETRI_FOLLOWUP_PROTOCOL.md)에 있다. 원본/기준선/후보 비교 도구는 활용할 수 있지만,
`clean_keys` 진단이나 화질 점수 상승을 동일 채널의 할루시네이션 완화 입증으로 간주하지 않는다.
아래 구현 검증·미실행 표현은 당시 기록이며, 이번 문서 개정은 복원 실행 상태를 다시 판정한 작업이 아니다.

이 PC에서는 아래 명령 하나로 영상 선택, SKEM, 자동 캡션·광류, 송수신, 네 조건 복원,
공식 화질 지표 계산, 비교 영상과 보고서 생성을 순서대로 실행한다. Conda를 따로 활성화할 필요가 없다.

```bash
bash /home/sangukbae/semantic_transmission/scripts/run_webvid_ablation.sh
```

저장소 안에서는 `bash scripts/run_webvid_ablation.sh`도 같다.
터미널에 실행 단계, 경과 시간, 상세 로그 경로를 표시한다. 실행 중에는 WSL을 종료하지 않는다.

## 선택과 비교 조건

기본값은 [기존 선정 목록](../configs/webvid5_manifest.json)의 `single_subject` 영상이다.
들판에서 스카프를 들고 이동·회전하는 사람이 나오는 337프레임·14.04초 영상이며,
복원 품질을 본 뒤 영상을 선택하지 않는다. 해당 영상의 raw/processed SHA-256과
전체 프레임 수·해상도·fps를 확인한다. 다른 네 영상 파일은 기본 실행에 필요하지 않다.

| 조건 | 입력과 목적 |
|---|---|
| `baseline` | 기존 공개 코드 생성 설정, SKEM 키프레임, 자동 캡션·광류, 실제 송수신 |
| `aligned` | 같은 수신 PNG·캡션·광류에 끝점 조건·마지막 17프레임 overlap·중복 경계 제거 적용 |
| `clean_keys` | 정렬 보정 상태에서 원본 키프레임을 직접 입력하는 원인 분리용 진단 |
| `dense_1s` | 원래 SKEM 키프레임을 모두 보존하고 최대 간격을 24프레임으로 제한한 뒤 정렬 보정 |

576×320·24fps·30 steps·seed 42·10dB, 전체 영상, SKEM stride 1을 사용한다.
프로필은 [WebVid5 메모리 보완 설정](../configs/webvid5.json)을 그대로 사용한다.
키프레임 개수는 선택 결과에 따라 달라지며, ETRI의 12장으로 고정하지 않는다.
추가 구간에는 해당 원래 구간의 자동 캡션과 광류 scalar를 반복한다. 수동 캡션은 쓰지 않는다.
학습이나 모델 가중치 변경은 없다.

추가 키프레임은 실제 NTSCC 송수신을 수행한다. 기존 키프레임의 송수신 심벌과 수신 PNG는
바이트 단위로 보존하고, 추가 키프레임에는 `channel_seed + 100000 + frame_index`의
독립 PCG64 AWGN을 적용한다. 전체 메타데이터를 LDPC/16QAM 채널로 다시 보내고
시각·디지털 복소 채널 사용량을 합산한다. 원본 키프레임 진단에는 전송 성능 수치를 부여하지 않는다.

## 결과

설정·소스·모델 파일 정보·실행 환경으로 정한 식별자가 붙은 폴더에 저장한다.

```text
outputs/webvid1_ablation_single_subject_<식별자>/
  comparison.html                 원본 / 기존 / 선택한 조건 동기 재생
  comparison_four_panel.mp4        원본 / 기존 / 정렬 / 정렬+1초 간격
  comparison_oracle.mp4            원본 / 정렬 / 원본 키프레임 진단
  REPORT.md                       비교 표·전송량·해석 범위
  comparison_summary.json         PNG 및 MP4의 5개 지표, 채널 사용량
  protocol.json                   선택·전체 설정·실행 환경
  AUDIT.json                      입력·시간축·전송·지표 검증
  COMPLETE.json                   전체 자동 처리 완료 시에만 기록
  status.json                     현재 실행 상태
  evaluations/<조건>/             지표 JSON 및 프레임별 CSV
  stages/                         단계별 완료 상태와 파일 해시
  logs/                           단계별 상세 로그
  failed_attempts/                실패한 실행의 부분 파일 보존
```

평가는 PSNR·SSIM·LPIPS-VGG·DISTS·CLIP을 PNG와 MP4에 각각 계산한다.
기존 설정의 중복 경계 프레임을 제거한 **동일한 337개 원본 시점**의 평균으로 비교한다.
비교용 MP4에서도 중복 경계를 제거하지만 그 재인코딩 영상으로 화질 점수를 계산하지 않는다.
HTML은 결과 폴더 안의 상대 경로 영상을 사용하므로 브라우저에서 로컬로 열 수 있다.

의미적 동작 검수는 `PENDING`으로 남긴다. 사람 수·이동 방향·회전·스카프 동작 순서와
배경 변형을 비교 영상으로 직접 확인해야 한다. ETRI 전용 남색 코트 추적 지표는 적용하지 않는다.
한 개발 영상·한 시드의 예비 검증이며, 동일 전송량 우월성이나 일반화의 입증은 아니다.

## 재실행과 옵션

**같은 명령을 다시 실행하면 완료 단계의 파일 해시를 확인하고 이어서 실행한다.**
예를 들어 SKEM 선택이 완료된 후 복원이 중단됐다면 선택 결과를 재사용한다.
SKEM 선택 도중 중단됐다면 선택 단계 전체를 다시 실행한다. 프레임 비교 도중 재개는 지원하지 않는다.
실패 단계의 부분 파일과 로그는 보존하고 해당 단계만 처음부터 다시 시작한다.
변조·누락된 완료 파일은 성공 결과로 받아들이지 않는다. 동일 명령의 동시 실행도 차단한다.

코드·설정·입력·환경 또는 모델 파일 정보가 바뀌면 기본 출력 식별자가 달라진다.
다른 실행 조건의 완료 단계를 섞지 않는다. 큰 모델 가중치는 경로·크기·수정 시각을 기록하며,
실행할 때마다 전체 가중치를 다시 SHA-256으로 읽는 방식은 아니다.

```bash
# 모델 추론·결과 파일 생성 없이 기본 영상과 환경 확인
bash scripts/run_webvid_ablation.sh --dry-run

# 결과 폴더를 지정. 이후 같은 명령으로 이어서 실행
bash scripts/run_webvid_ablation.sh --output outputs/my_webvid_comparison

# 사전에 고정된 다른 유형 한 편 선택
bash scripts/run_webvid_ablation.sh --category fast_motion
```

유형은 `single_subject`, `low_motion`, `fast_motion`, `camera_motion`, `multi_object_occlusion`이다.
기본 영상의 첫 전체 실행은 약 8~10시간으로 예상하며 보장 시간은 아니다.
가장 오래 걸리는 SKEM을 포함한 완료 단계 재사용 시 추가 비교 시간은 줄어든다.
네 조건과 평가는 메모리 사용이 겹치지 않도록 별도 프로세스로 순차 실행한다.

## 구현 검증

2026-09-18: 합성 CPU 데이터로 네 조건 실행·실제 FFmpeg 비교 영상 생성·공통 시간축·
완료 단계 재사용·중단 파일 보존·변조 차단·원본 심벌 보존·진단 전송량 제외를 확인했다.
관련 회귀 검사 44개 통과, 별도 GPU parity 선택 검사 1개 생략.
실제 로컬 기본 WebVid 파일과 환경의 `--dry-run`은 통과했다.
이 명령을 통한 337프레임 실제 모델 비교는 아직 실행하지 않았으므로 품질 결과를 주장하지 않는다.

실행기: [webvid_ablation.py](../src/semantic_transmission/webvid_ablation.py)
· [전송 제어](../src/semantic_transmission/ablation_transport.py)
· [평가 검증·보고서](../src/semantic_transmission/webvid_ablation_report.py)
· [검사](../tests/test_webvid_ablation.py)
