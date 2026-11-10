#!/usr/bin/env bash
cd "$(dirname "$0")/.."
PY=~/miniconda3/envs/mmdl_train/bin/python
D=outputs/era_lora_ep5/trunc111
F='warn|^INFO|^\(EngineCore|Loading checkpoint|it/s\]|Processed prompts'
for m in G E; do
  echo "######## gen $m $(date +%T)"
  $PY train/eval_subset.py --exp era_lora_ep5 --model $m --ids $D/ids_111.json --out-dir $D 2>&1 | grep -v -iE "$F" | grep -E "prepared|seed|rror|Traceback"
done
cd scripts
for m in G E; do for s in 3407 3408 3409; do
  echo "######## score $m seed$s $(date +%T)"
  $PY score.py --pred ../$D/${m}_seed$s.jsonl --out ../$D/${m}_seed${s}_results.json 2>&1 | grep -v -iE "$F" | tail -3
done; done
echo "######## ALL DONE $(date +%T)"
