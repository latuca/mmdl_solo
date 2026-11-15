# DPO2-D: C + 루프 쌍은 반복 구간 512토큰만 벌하기 + 8/8 문항 자기 정답 보존 데이터

베이스 모델(Qwen3-VL-4B-Instruct @ ebb281ec)에서 시작하는 DPO ablation 4개 중 하나 (A′ → A → B → C).
- 학습 문항: vic_era_2000 1,997문항. 학습·평가 프롬프트는 베이스라인과 동일 (ERA 지시는 B·C의 답안 샘플링 때만 사용)
- 공통: beta 0.1, 1 epoch, LoRA r16/alpha 32/dropout 0.05, grad accum 16, AdamW wd 0.01, clip 1.0, cosine + warmup 3%, BF16, seed 3407
- 학습률·SFT 항 가중치 λ: 학습률 실측 시험(2e-5/5e-5/1e-4 각 30스텝, lr_sweep_summary.txt)으로 정한 값을 4개 실험에 똑같이 사용 → {"lr":5e-05,"lambda":1.0,"source":"lrsweep(2e-5/5e-5/1e-4,30stepseach,pairsA,held-outper-stepmetrics)-seelr_sweep/summary.txt","old_lr_estimate_discarded":4.23e-06}
- 쌍 통계: data_report/pairs_D.report.json
- 신호 시험(B·C 설정 각 30스텝, 처음 보는 쌍의 유형별 정답 선호)으로 효과 없는 요소를 뺐다: gate_signal_test.json → 유지한 쌍 조합 ['base>truncated', 'era>wrong', 'era>truncated'] / A′·A 실행 False / C 이미지 조건 손실 True

## 결과
- DPO2-D: 598/900 = 66.44% (베이스라인 576/900 = 64.00%, 차이 +22문항 / +2.44%p)
- 답변 길이 중앙값 347.0토큰, 출력 한도 초과 14개(베이스라인 81), 정상 추출 문항 정답률 69.3%(베이스라인 71.0%)

문항별 출력: eval/, 학습 로그: train/
