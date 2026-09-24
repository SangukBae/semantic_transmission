# 중복 처리 제거: 적용 전후 최소 GPU 검증

2026-09-20, RTX 4080 16GB / PyTorch 2.2.2+cu121.

## 적용 내용

- SKEM: 같은 기준 프레임의 이미지 로딩·전처리·BF16 CUDA 전송 결과를
  최대 두 프레임까지 보관한다. 영상별로 초기화하며 영상 특징, 캡션,
  PSSS 계산, 임계값, 후보 프레임 수는 변경하지 않는다.
- 디코더: pretrained CUDA STDiT3-XL/2를 구간마다 다시 구성하지 않고
  실행 중 한 번 구성해 재사용한다. 고정 upstream STDiT3의 forward는 실제
  입력에서 시간·공간 크기를 계산한다. 구간마다 latent 크기, scheduler,
  텍스트 조건, 키프레임 mask와 이전 구간 참조는 계속 새로 처리한다.
- CPU, 다른 모델 종류, pretrained 경로가 없는 모델은 기존 구성 경로를 유지한다.
- 기존 작업 중이던 T5 캐시·endpoint 정렬 수정은 보존했다.

기본 적용이며, A/B용으로 SKEM의 `--no-frame-cache`와 디코더 설정의
`reuse_diffusion_model = False`를 사용하면 기존 경로로 실행할 수 있다.
입력 파일은 실행 중 변경하지 않아야 하며 모델 설정·가중치도 실행 중 고정한다.

## 가장 빠른 검증 경계

전체 송수신을 반복하는 대신 변경 부품의 입력·출력을 직접 비교했다.

1. 실제 기존 smoke PNG로 기준 프레임 유지·교체를 포함하는 6쌍을 처리한다.
   baseline은 수정 전 파일에서 추출한 원래 전처리 함수다. 1회 준비 실행 후
   before/after 순서를 교대하며 3회 측정한다. 모델에 전달되는 결합 텐서를
   `torch.equal`로 비교한다. InternVL 추론은 동일 입력 경계 이후이므로 생략했다.
2. 실제 로컬 STDiT3와 OpenSora VAE 가중치로 128×128, 9→17→9프레임을
   전후 각각 복원한다. 양쪽 모두 BF16, FlashAttention, RFLOW **30 steps**,
   CFG 7.0, 같은 초기 CUDA RNG 상태, 끝점 mask와 이전 생성 구간 참조를 사용한다.
   가변 길이 변경 후 이전 길이로 돌아가는 경우도 포함한다.
3. 느린 T5 로딩·추론을 생략하고 양쪽에 동일한 **합성 텍스트 임베딩**을 넣었다.
   따라서 의미 품질 평가용 자연 영상 복원이나 전체 production 실행은 아니다.
4. 모델 구성 직전·직후 CUDA RNG가 같은지 검사하고, 최종 latent·VAE 출력
   부동소수 텐서·PNG 직전 uint8 픽셀을 각각 완전 일치 비교했다.

## 결과

| 비교 항목 | 적용 전 | 적용 후 | 판정 |
|---|---:|---:|---|
| 전처리 6쌍 중앙값 | 46.82 ms | 28.55 ms | 약 1.64배, 시간 39.0% 감소 |
| 전처리 실제 로딩 횟수 | 12 | 7 | 캐시 적중 5회 |
| 모델 구성 횟수 / 3구간 | 3 | 1 | 중복 구성 2회 제거 |
| 두 번째·세 번째 구간의 측정 시간 합계 | 9.377 s | 6.962 s | 약 1.35배, 시간 25.7% 감소 |
| 전처리 결합 텐서 | 기준 | 완전 일치 | 6쌍 × 3회 통과 |
| 생성 latent·VAE 출력·uint8 픽셀 | 기준 | 완전 일치 | 35프레임, 최대 차이 0 |
| 관련 회귀 테스트 | — | 25 passed / 0.92 s | 통과 |

디코더 시간은 모델 구성 + 확산 sampling + VAE decode의 합계다.
양쪽 첫 구간은 baseline의 초기 체크포인트·커널 준비 비용으로 인한 순서 편향을
줄이기 위해 속도 비교에서 제외했다. 이후 구간에서도 단일 A/B 실행의 변동은 남는다.
공통 VAE 적재, 참조 프레임 인코딩, 파일 저장, T5, 송신·채널 시간은 포함하지 않는다.
전체 송수신 속도가 1.35배 빨라졌다는 결론으로 확대하면 안 된다.

검증한 입력에서는 출력 자체가 동일하므로 최적화 때문에 추가된 복원 오류나
할루시네이션은 없다. 기존 모델의 오류가 해결되었다는 뜻은 아니며, 전체 해상도·
긴 영상·모든 프롬프트에서의 출력 동일성을 입증한 것도 아니다.

## 근거 및 재현

- [비교 JSON](../outputs/exact_reuse_validation_20260920/comparison.json)
- [GPU 실행 로그](../outputs/exact_reuse_validation_20260920/validation.log)
- [적용 전 코드와 프레임](../outputs/exact_reuse_validation_20260920/before/)
- [적용 후 프레임](../outputs/exact_reuse_validation_20260920/after/)
- [검증 스크립트](../scripts/validate_exact_reuse.py)
- [공통 재사용 코드](../src/semantic_transmission/exact_reuse.py)

원래 코드·현재 코드·입력 이미지 해시와 실제 가중치 경로는 비교 JSON에 기록했다.
스크립트 실행 후 GPU 재실행 없이 첫 구간 제외 통계와 provenance를 보충했다.

```bash
source scripts/activate.sh
# 기존 결과를 덮어쓰지 않도록 새 폴더에 원래 코드 스냅샷을 복사한다.
mkdir -p outputs/exact_reuse_recheck/before
cp outputs/exact_reuse_validation_20260920/before/*.py outputs/exact_reuse_recheck/before/
python scripts/validate_exact_reuse.py --output outputs/exact_reuse_recheck
python -m pytest -q tests/test_exact_reuse.py tests/test_decoder_conditioning.py \
  tests/test_internvl_memory.py tests/test_research_runtime.py
```
