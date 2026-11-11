#!/usr/bin/env bash
# 일반 풀이(explanation)만 학습 -> epoch 2/4/5 체크포인트 merge -> MMMU-val 900 평가.
# 재실행하면 끝난 단계는 건너뛰고 이어서 진행한다 (평가는 같은 run-name으로 재개).
set -euo pipefail

ROOT=/home/latuca/MMDL_solo
EXP=$ROOT/runs/sft_general_vic2000
TRAIN_PY=/home/latuca/venvs/train/bin/python
EVAL_DIR=/home/latuca/mmdl_test
EVAL_PY=$EVAL_DIR/.venv/bin/python
BASECFG=$ROOT/config/training_settings_only.json
ATTN=${ATTN:-flash_attention_2}
EPOCHS_TO_EVAL="2 4 5"
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export PYTHONUNBUFFERED=1

mkdir -p "$EXP"
cd "$ROOT"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$EXP/pipeline.log"; }

wait_gpu(){
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "$used" -lt 1500 ]; then return; fi
    log "GPU 사용 중 (${used}MiB) - 5분 후 재확인"
    sleep 300
  done
}

# 0) 평가용 모델(4B, 저지 8B)·MMMU 준비 (네트워크만 사용, 이미 있으면 건너뜀)
if [ ! -f "$EVAL_DIR/outputs/preparation.json" ]; then
  log "평가 자산 준비 시작"
  (cd "$EVAL_DIR" && $EVAL_PY src/prepare.py --existing-env) 2>&1 | tee -a "$EXP/prepare_eval.log"
fi

# 1) 학습 데이터 변환 + 8192 토큰 초과 제외
DATA=$EXP/data/train.jsonl
if [ ! -f "$DATA" ]; then
  log "학습 데이터 변환"
  $TRAIN_PY scripts/prepare_sft_general.py --data-dir data/vic_era_2000 --eval-src "$EVAL_DIR/src" \
      --out "$DATA" --max-tokens 8192 2>&1 | tee -a "$EXP/prepare_data.log"
fi

# 2) 학습 (5 epoch, epoch 2/4/5 끝에 어댑터 저장)
if [ ! -f "$EXP/train/adapter_ep5/adapter_model.safetensors" ]; then
  if [ -d "$EXP/train" ]; then mv "$EXP/train" "$EXP/train_incomplete_$(date +%Y%m%d_%H%M%S)"; fi
  wait_gpu
  log "학습 시작"
  $TRAIN_PY scripts/train_sft.py --data "$DATA" --images data/vic_era_2000 --baseline "$BASECFG" \
      --diag scripts --out "$EXP/train" --epochs 5 --grad-accum 16 --lr 1e-4 --warmup-ratio 0.03 \
      --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 --weight-decay 0.01 --max-grad-norm 1.0 \
      --max-tokens 8192 --save-every 1000000 --save-epochs 2,4,5 --seed 3407 --workers 2 \
      --attn "$ATTN" --loss-mode token_mean --no-merge 2>&1 | tee -a "$EXP/train.log"
  log "학습 종료"
fi

# 3) 체크포인트별 merge -> 평가 (결과는 $EXP/ep{N}/eval/ 에 따로 보관)
for ep in $EPOCHS_TO_EVAL; do
  D=$EXP/ep$ep
  mkdir -p "$D"
  if [ ! -f "$D/merged/config.json" ]; then
    log "ep$ep merge"
    $TRAIN_PY scripts/merge_adapter.py --adapter "$EXP/train/adapter_ep$ep" --out "$D/merged" \
        --baseline "$BASECFG" 2>&1 | tee -a "$D/merge.log"
  fi
  if [ ! -f "$D/eval/summary.json" ]; then
    wait_gpu
    log "ep$ep 평가 시작"
    (cd "$EVAL_DIR" && $EVAL_PY src/evaluate.py --config src/config.json --checkpoint "$D/merged" \
        --run-name eval --output-root "$D") 2>&1 | tee -a "$D/eval.log"
  fi
  acc=$($EVAL_PY -c "import json;s=json.load(open('$D/eval/summary.json'));print(f\"{s['correct']}/900 = {s['accuracy_percent']:.2f}%\")")
  log "ep$ep 결과: $acc (baseline 576/900 = 64.00%)"
done

# 4) 결과 표
{
  echo "| checkpoint | correct | accuracy | Δ vs baseline 64.00 |"
  echo "|---|---:|---:|---:|"
  echo "| baseline | 576 | 64.00% | - |"
  for ep in $EPOCHS_TO_EVAL; do
    $EVAL_PY -c "import json;s=json.load(open('$EXP/ep$ep/eval/summary.json'));print(f\"| ep$ep | {s['correct']} | {s['accuracy_percent']:.2f}% | {s['accuracy_percent']-64:+.2f}%p |\")"
  done
} > "$EXP/results.md"
log "완료: $EXP/results.md"
cat "$EXP/results.md"
