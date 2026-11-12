# SFT 모델 보관본: 일반 풀이만 학습 (vic_era_2000, 5 epoch)

이후 강화학습(DPO/KTO 등)의 시작점으로 쓰려고 보관한다.
- 베이스 모델: Qwen/Qwen3-VL-4B-Instruct @ ebb281ec70b05090aa6165b016eac8ec08e71b17
- 학습 설정과 코드: experiments/sft_general_vic2000_e5/

| 체크포인트 | MMMU-val | 베이스라인 64.00% 대비 |
|---|---:|---:|
| ep2 | 497/900 = 55.22% | -8.78%p |
| ep4 | 510/900 = 56.67% | -7.33%p |
| ep5 | 505/900 = 56.11% | -7.89%p |

## 구성 (체크포인트마다)
- adapter/: LoRA 어댑터 (r16, alpha 32). adapter_model.safetensors는 132MB
- merged/: 베이스에 병합한 BF16 전체 모델 (약 8.3GB, vLLM 평가에 바로 사용)
- SHA256SUMS: 가중치 파일 체크섬
- summary_vs_baseline.md: 과목별 비교표

## 보관 위치
- 서버 AIX-408-UBUNTU-97: /home/latuca/MMDL_solo/models/sft_general_vic2000_e5/
- GitHub 레포에는 가중치(.safetensors)를 올리지 않았다(파일당 100MB 제한). 설정 파일과 체크섬만 올렸다.

## merged 복원 (어댑터만 있을 때)
python MMDL_solo/scripts/merge_adapter.py --baseline MMDL_solo/config/training_settings_only.json \
  --adapter ep4/adapter --out ep4/merged
