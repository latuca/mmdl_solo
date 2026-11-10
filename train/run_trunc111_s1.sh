#!/usr/bin/env bash
# seed 3407만: G(이어받기) -> E -> 채점. 이후 확장 여부는 PREREGISTERED.md의 순차 규칙에 따른다.
cd "$(dirname "$0")/.."
PY=~/miniconda3/envs/mmdl_train/bin/python
D=outputs/era_lora_ep5/trunc111
F='warn|^INFO|^\(EngineCore|Loading checkpoint|it/s\]|Processed prompts'
for m in G E; do
  if [ -f $D/${m}_seed3407.jsonl ] && [ "$(wc -l < $D/${m}_seed3407.jsonl)" -ge 111 ]; then echo "######## skip gen $m (111/111 완료)"; continue; fi
  echo "######## gen $m seed3407 $(date +%T)"
  $PY train/eval_subset.py --exp era_lora_ep5 --model $m --ids $D/ids_111.json --out-dir $D --seeds 3407 2>&1 | grep -v -iE "$F" | grep -E "prepared|seed|rror|Traceback"
done
cd scripts
for m in G E; do
  echo "######## score $m seed3407 $(date +%T)"
  $PY score.py --pred ../$D/${m}_seed3407.jsonl --out ../$D/${m}_seed3407_results.json 2>&1 | grep -v -iE "$F" | tail -3
done
echo "######## ALL DONE $(date +%T)"
