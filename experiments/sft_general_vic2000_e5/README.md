# SFT: 일반 풀이만 학습 (vic_era_2000, 5 epoch)

- 베이스 모델: Qwen/Qwen3-VL-4B-Instruct @ ebb281ec70b05090aa6165b016eac8ec08e71b17
- 데이터: vic_era_2000의 `explanation`(일반 풀이) 1,997문항 (V 398 / I 600 / C 999, 풀이 없는 3문항 제외)
- LoRA r16 / alpha 32 / dropout 0.05 (q,k,v,o,gate,up,down), AdamW lr 1e-4, wd 0.01, grad clip 1.0
- cosine + warmup 3%, micro batch 1 × grad accum 16, BF16, gradient checkpointing, packing 없음, seed 3407
- 해상도: min_pixels 1280×28×28, max_pixels 5120×28×28 (평가와 동일), 학습·추론 프롬프트 동일, `<think>` 제거
- 평가: LoRA 병합 모델, MMMU-val 900문항, mmdl_test 평가 코드 (버전: config/eval_code_version.txt)

## 결과

| 체크포인트 | MMMU-val | 베이스라인 대비 |
|---|---:|---:|
| baseline | 576/900 = 64.00% | - |
| epoch 2 | 497/900 = 55.22% | -79문항 / -8.78%p |
| epoch 4 | 510/900 = 56.67% | -66문항 / -7.33%p |
| epoch 5 | (평가 대기/진행 중) | - |

체크포인트별 과목 표: `ckpt_epN/summary_vs_baseline.md`, 문항별 출력: `ckpt_epN/eval/`

## 폴더
- code/: 데이터 변환·학습·병합·평가 파이프라인 스크립트
- data_report/: 학습 데이터 변환 통계, 이미지 복구 리포트 (원본 데이터·이미지는 미포함)
- train/: 학습 로그 (loss 등)
- ckpt_epN/adapter/: 해당 epoch LoRA 어댑터
