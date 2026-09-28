# 모델 구조

[과제 현황](README.md) · [개선 실험](ETRI_COMBINED_REFERENCE_DIAGNOSIS.md#decoder)

## 전체 흐름

**영상 → 대표 프레임·설명·움직임 추출 → 전송 → AI 영상 생성 → 시간순 연결**

## 모델별 역할

- **SKEM / InternVL:** 앞서 선택한 프레임과 비교해 새 키프레임을 선택.
- **PLLaVA:** 구간의 표본 프레임 4장으로 내용 설명 생성.
- **UniMatch:** 프레임 사이 움직임을 추정해 구간당 숫자 하나로 요약.
- **NTSCC:** 키프레임을 통신 신호로 바꿔 보내고 수신 이미지로 복원.
- **LDPC / 16-QAM:** 설명·위치 등 보조정보의 오류 정정과 신호 변환.
- **AWGN:** 잡음이 섞이는 통신 채널; 현재 주 조건 10 dB.
- **DSA:** 구간 길이에 맞춰 생성 길이와 조건을 구성.
- **Open-Sora:** 수신 키프레임·설명·움직임·이전 구간을 이용해 영상 생성.
- **VAE:** 영상과 압축 표현을 변환; 키프레임의 시간 위치를 현재 점검 중.

## 현재 한계

- 선택 속도: SKEM 계산 비용이 큼; 60초 사례에서 약 23시간 소요.
- 동적 정보: 객체별 속도·방향·등장 시각을 충분히 전달하는지 미검증.
- 생성 품질: 중간 프레임의 형태·동작 오류와 이전 구간 오류 전파 가능.
- 장면전환: 검출·패킷 갱신·검출 실패 대안의 효과 검증이 남음.

<a id="implementation"></a>
## 주요 코드

- [workers.py](../src/semantic_transmission/workers.py): 전처리·선택·설명·움직임 추출 연결.
- [SKEM](../02_semantic_encoder/skem/MLM-keyframe-internvl.py): 키프레임 비교·선택.
- [codec_transport.py](../src/semantic_transmission/codec_transport.py): 부호화·채널·전송량 계산.
- [생성기](../04_semantic_decoder/scripts/mydemo_new_align_sh.py): 조건 주입·이전 구간 참조·영상 생성.
- [temporal.py](../src/semantic_transmission/temporal.py): 구간 경계의 중복 프레임 제거.
- [etri_60s.py](../src/semantic_transmission/etri_60s.py): 장시간 실행·재개·전체 복원 검사.
- [official_quality.py](../src/semantic_transmission/official_quality.py): 화질 평가.
- [models.json](../configs/models.json): 모델·가중치 설정.

<a id="reproduction"></a>
## 논문 재현·학습 범위

- 알고리즘: LGVSC의 주요 구성 사용; 논문 전체 실험 재현은 미완료.
- 가중치: 공개 사전학습 모델 사용; NTSCC는 quality-4, 논문 가중치와 동일성 미확인.
- 로컬 변경: 메모리 절약·결과 재사용·시간축 연결·실패 재개 지원.
- 경계 연결: `endpoint_exact`로 중복 제거; 생성 조건의 시간 위치 정렬은 별도 문제.
- 전송량: 시각·메타데이터를 합산하며 모든 물리 통신 비용을 포함하지는 않음.
- 추가 학습: 최근 실험은 학습 없이 생성 조건만 변경.
- 남은 확인: 모듈별 이미지·동영상 학습 자료와 사용 체크포인트의 이력.

[상세 구조·학습 범위](https://github.com/SangukBae/semantic_transmission/blob/4f566feabe213b870e5c1e441ef9b369b348d589/docs/MODEL_ARCHITECTURE.md) · [논문 대응 감사](https://github.com/SangukBae/semantic_transmission/blob/f41bc8885e8d8409e59b162fe7be63c0d11ee737/docs/LGVSC_PAPER_IMPLEMENTATION_AUDIT.md)
