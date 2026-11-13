# DPO: SFT ep4에서 시작 (선호 답 출처 섞기)

- 시작 모델: SFT(일반 풀이, vic_era_2000) ep4 병합 모델 (MMMU-val 510/900 = 56.67%) → models/sft_general_vic2000_e5/
- 학습 문항: vic_era_2000 1,997문항 (SFT와 같은 프롬프트·이미지·해상도)
- 샘플링: ep4·베이스 모델 각 문항당 8개, 평가와 같은 샘플링 설정, 프롬프트+응답 최대 8192토큰
- 채점: 평가와 같은 파서 + 로컬 Qwen3-8B 저지 (판정 불가 응답은 제외)
- 쌍: rejected = ep4 오답, chosen = 베이스·ep4 정답 섞기 (문항당 최대 2쌍, 첫 쌍은 베이스 정답 우선)
- DPO: beta 0.1, lr 5e-6, 1 epoch, LoRA r16/alpha 32/dropout 0.05, grad accum 16, AdamW wd 0.01, clip 1.0,
  cosine + warmup 3%, BF16, seed 3407. reference = 시작 모델(LoRA 끈 상태)

## 결과
- DPO(ep4 시작): 503/900 = 55.89% (베이스라인 576/900 = 64.00%, 차이 -73문항 / -8.11%p)
- 비교: 베이스라인 576/900 = 64.00%, 시작점 SFT ep4 510/900 = 56.67%

쌍 통계: data_report/pairs.report.json, 학습 로그: train/, 문항별 출력: eval/
