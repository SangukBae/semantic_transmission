# 모델 실험 결과

실험 기준: **2026-09-28** · [과제 현황](README.md) · [다음 작업](ETRI_DEVELOPMENT_PLAN.md)

<a id="baseline"></a>
## 60초 복원 — 실행 완료

- 대상: 저전환 개발 영상 `tv_low_08` 한 편, AWGN 10 dB.
- 출력: 60초·1,440프레임, 576×320, 초당 24프레임.
- 구성: 키프레임 110개 · 생성 구간 109개 · 구간 간 참조 전달 108회.
- 화질: PSNR 14.85 dB · SSIM 0.648 · LPIPS 0.388.
- 취약점: 키프레임보다 중간 생성 프레임에서 품질 저하가 큼.
- 시간: 키프레임 선택 23시간 19분; 선택 결과 재사용 후 나머지 112.44분.
- 판정: `PASS_60S_RECONSTRUCTION`; 오류 검수 `PENDING`, 완화 검증 `false`.
- 근거: [완료 결과](../outputs/etri_60s_tv_low_08_42057b2ee8ed/RESULT.json) · [비교 영상](../outputs/etri_60s_tv_low_08_42057b2ee8ed/comparison.mp4).

## 발견한 오류

- 표본 검토: 원본·복원 프레임 76쌍에서 AI가 오류·후보 11건 기록.
- 관찰 내용: 사람 형태 왜곡, 개 누락, 차량 추가·위치·문 구조 왜곡.
- 해석 범위: 독립 정답·전 구간 오류 발생률은 미확보.
- 원인 조사: 캡션·움직임 109구간의 대응은 정상; 설명 내용과 조건 시간 위치는 추가 확인 필요.
- 근거: [오류·조건 검토 페이지](../outputs/etri_conditioning_review_20260928/review.html).

<a id="decoder"></a>
## 개선안 비교 — 기본값 채택 보류

- 비교 범위: 같은 영상의 사람·차량·차량 문, 짧은 세 구간.
- 공통 조건: 같은 수신 데이터·초기 생성 잡음, 한 시드; 추가 전송량 0.
- 기준선: 이전 생성 구간 전체를 참조하고 조건 위치를 반올림.

| 변경안 | 확인된 결과 | 판단 |
|---|---|---|
| 위치 반올림 해제 | 차량 문 개선, 사람 확대 왜곡 발생 | 일괄 적용 보류 |
| 마지막 17프레임만 참조 | 차량 일부 개선, 형태·문 오류 잔존 | 개발 후보 유지 |
| 두 변경 결합 | 사람의 강한 흐림, 일관된 우위 없음 | 60초 확대 보류 |

- 개선 수치: 반올림 해제 시 차량 문 LPIPS 0.537→0.311; 낮을수록 양호.
- 악화 수치: 결합안의 사람 중간 프레임 LPIPS 0.383→0.522.
- 한계: 짧게 재시작한 구간이며 60초 전체의 누적 오류를 재현한 실험은 아님.
- 다음 검사: 키프레임 압축 표현과 VAE 시간 위치의 대응.
- 근거: [네 조건 비교](../outputs/etri_combined_reference_20260928/review.html) · [수치](../outputs/etri_combined_reference_20260928/RESULT.json) · [조건 검증](../outputs/etri_combined_reference_20260928/verification.json).

<a id="legacy"></a>
## 이전 짧은 영상 실험

- ETRI 10초: 정렬·키프레임 보강으로 LPIPS 0.355→0.296 개선, 전송량 4.091배.
- WebVid: 같은 변경으로 LPIPS 0.284→0.306 악화, 전송량 1.419배.
- 품질 검증: 1·2차 실행 완료; 별도 5편·3시드 전체 검증과 의미 오류 검수는 미완료.
- 해석: 일부 화질 개선 근거이며 장시간 할루시네이션 감소의 증거는 아님.
- 근거: [품질 검증 기록](../outputs/quality_validation_e8babaef576c/REPORT.md).

<a id="reuse"></a>
## 계산 재사용 실험

- 방법: 같은 프레임의 전처리 결과와 생성 모델을 재사용.
- 결과: 제한된 GPU 비교에서 출력 35프레임 일치, 일부 처리 시간 단축.
- 범위: 작은 해상도의 부품 검사; 전체 실행 가속률은 미확정.
- 근거: [측정 기록](../outputs/exact_reuse_validation_20260920/comparison.json).

[세부 수치·조건·실패 이력](https://github.com/SangukBae/semantic_transmission/blob/4f566feabe213b870e5c1e441ef9b369b348d589/docs/ETRI_COMBINED_REFERENCE_DIAGNOSIS.md) · [실행 명령](RUN_GUIDE.md)
