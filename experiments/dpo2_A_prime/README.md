# DPO2-A_prime: 베이스에서 순수 DPO (자기 정답 vs 자기 오답)

베이스 모델(Qwen3-VL-4B-Instruct @ ebb281ec)에서 시작하는 DPO ablation 4개 중 하나 (A′ → A → B → C).
- 학습 문항: vic_era_2000 1,997문항. 학습·평가 프롬프트는 베이스라인과 동일 (ERA 지시는 B·C의 답안 샘플링 때만 사용)
- 공통: beta 0.1, 1 epoch, LoRA r16/alpha 32/dropout 0.05, grad accum 16, AdamW wd 0.01, clip 1.0, cosine + warmup 3%, BF16, seed 3407
- 학습률·SFT 항 가중치 λ: 실제 기울기 측정으로 정한 값을 4개 실험에 똑같이 사용 → {"lr":4.23e-06,"lambda":1.0,"lr_raw":4.2319103257144005e-06,"lambda_raw":10.472696765099725,"clipped_to":{"lr":[1e-06,0.0001],"lambda":[0.05,1.0]}}
- 쌍 통계: data_report/pairs_A.report.json

## 결과
- DPO2-A_prime: 573/900 = 63.67% (베이스라인 576/900 = 64.00%, 차이 -3문항 / -0.33%p)
- 답변 길이 중앙값 270.0토큰, 출력 한도 초과 63개(베이스라인 81), 정상 추출 문항 정답률 69.4%(베이스라인 71.0%)

문항별 출력: eval/, 학습 로그: train/
