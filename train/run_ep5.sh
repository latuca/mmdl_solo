#!/usr/bin/env bash
# era_lora_ep5: G, E를 5 epoch 학습 -> merge -> 첫 토큰 확률 -> held-out 생성 (HF, vLLM)
cd "$(dirname "$0")/.."
PY=~/miniconda3/envs/mmdl_train/bin/python
EXP=era_lora_ep5
F='warn|^INFO|^\(EngineCore|Loading checkpoint|it/s\]'
for c in G E; do
  echo "######## train $c $(date +%T)"
  $PY train/train_lora.py --cond $c --exp $EXP --epochs 5 2>&1 | grep -v -iE "$F" | grep -E "built|optimizer steps|\(1\)|\(2\)|\(3\)|\"step\": (1|10|20|30|40|50|51|52),|summary|adapter|rror"
done
for c in G E; do
  echo "######## merge $c $(date +%T)"
  $PY train/merge_lora.py --cond $c --exp $EXP 2>&1 | grep -v -iE "$F" | grep -E "logits|visual_bitwise|llm_tensors_changed|render_identical|tokenization_identical|IDENTICAL|DIFFERENT|rror"
done
mkdir -p outputs/$EXP/stage5 && cp outputs/era_lora_pilot/stage5/base_*.jsonl outputs/$EXP/stage5/   # base는 동일 모델/설정이라 재사용
echo "######## first-token $(date +%T)"
$PY train/first_token_prob.py --exp $EXP --models base G E 2>&1 | grep -v -iE "$F"
for m in E G; do for e in hf vllm; do
  echo "######## gen $m/$e $(date +%T)"
  $PY train/gen_heldout.py --engine $e --model $m --exp $EXP 2>&1 | grep -v -iE "$F" | grep -E "^done|rror"
done; done
echo "######## ALL DONE $(date +%T)"
