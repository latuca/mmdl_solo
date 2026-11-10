#!/usr/bin/env bash
# 엔진/모델마다 별도 프로세스로 순서대로 실행 (vLLM이 GPU 메모리 90%를 점유하므로 동시 실행 금지)
cd "$(dirname "$0")/.."
PY=~/miniconda3/envs/mmdl_train/bin/python
for m in E G base; do
  for e in hf vllm; do
    echo "######## $m / $e  $(date +%T)"
    $PY train/gen_heldout.py --engine $e --model $m 2>&1 | grep -v -iE "warn|^INFO|^\(EngineCore|Loading checkpoint|it/s\]"
  done
done
echo "######## ALL DONE $(date +%T)"
