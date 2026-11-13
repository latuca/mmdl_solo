#!/usr/bin/env bash
# DPO: SFT ep4 모델에서 시작. 샘플링(ep4·베이스 각 8개) → 채점 → 쌍 생성 → DPO 학습 → 병합 → MMMU-val 평가 → 푸시
# 다시 실행하면 완료 표시(.done)가 있는 단계는 건너뛴다. 샘플링은 끝난 문항부터 이어서 한다.
set -euo pipefail

ROOT=/home/latuca/MMDL_solo
EVAL=/home/latuca/mmdl_test
TRAIN_PY=/home/latuca/venvs/train/bin/python
EVAL_PY=$EVAL/.venv/bin/python
SFT_RUN=$ROOT/runs/sft_general_vic2000_e5
START=$ROOT/models/sft_general_vic2000_e5/ep4/merged
BASE_REPO=Qwen/Qwen3-VL-4B-Instruct
BASE_REV=ebb281ec70b05090aa6165b016eac8ec08e71b17
RUN=$ROOT/runs/dpo_ep4_mix_b0.1_lr5e-6
RUN_NAME=dpo_ep4_mix_b0.1_lr5e-6
mkdir -p "$RUN/data"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$RUN/pipeline.log"; }
done_mark() { touch "$RUN/.$1.done"; }
is_done() { [ -f "$RUN/.$1.done" ]; }
wait_gpu() {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2000 ] && return 0
    log "GPU 사용 중(${used}MiB) - 5분 후 다시 확인"; sleep 300
  done
}
export HF_HUB_OFFLINE=1

log "===== DPO 파이프라인 시작 ====="
COMMON="--data $SFT_RUN/data/train.jsonl --images $ROOT/data/vic_era_2000 --eval-src $EVAL/src --n 8 --max-total 8192 --chunk 128"

if ! is_done sample_ep4; then
  wait_gpu; log "[1] ep4 모델 응답 샘플링 (2,000문항 × 8)"
  (cd "$EVAL" && $EVAL_PY $ROOT/scripts/dpo_sample.py --model "$START" --tag ep4 $COMMON \
    --out "$RUN/data/samples_ep4.jsonl") >> "$RUN/sample_ep4.log" 2>&1
  done_mark sample_ep4
fi

if ! is_done sample_base; then
  wait_gpu; log "[2] 베이스 모델 응답 샘플링 (2,000문항 × 8)"
  (cd "$EVAL" && $EVAL_PY $ROOT/scripts/dpo_sample.py --model $BASE_REPO --revision $BASE_REV --tag base $COMMON \
    --out "$RUN/data/samples_base.jsonl") >> "$RUN/sample_base.log" 2>&1
  done_mark sample_base
fi

if ! is_done grade; then
  wait_gpu; log "[3] 샘플 채점 (평가 파서 + Qwen3-8B 저지)"
  (cd "$EVAL" && $EVAL_PY $ROOT/scripts/dpo_grade.py --samples "$RUN/data/samples_ep4.jsonl" "$RUN/data/samples_base.jsonl" \
    --source $ROOT/data/vic_era_2000/data.json --eval-src $EVAL/src --out "$RUN/data/labels.jsonl") >> "$RUN/grade.log" 2>&1
  done_mark grade
fi

if ! is_done pairs; then
  log "[4] DPO 쌍 생성 (rejected = ep4 오답, chosen = 베이스·ep4 정답 섞기, 문항당 최대 2쌍)"
  $TRAIN_PY $ROOT/scripts/dpo_pairs.py --train $SFT_RUN/data/train.jsonl \
    --samples "$RUN/data/samples_ep4.jsonl" "$RUN/data/samples_base.jsonl" --labels "$RUN/data/labels.jsonl" \
    --policy-tag ep4 --pairs-per-q 2 --seed 3407 --out "$RUN/data/pairs.jsonl" >> "$RUN/pairs.log" 2>&1
  log "쌍 리포트: $(cat "$RUN/data/pairs.report.json" | tr -d '\n ')"
  done_mark pairs
fi

if ! is_done train; then
  wait_gpu; log "[5] DPO 학습 (beta 0.1, lr 5e-6, 1 epoch, LoRA r16/a32, grad accum 16)"
  rm -rf "$RUN/train"
  $TRAIN_PY -u $ROOT/scripts/train_dpo.py --pairs "$RUN/data/pairs.jsonl" --images $ROOT/data/vic_era_2000 \
    --start-model "$START" --vision-config $ROOT/config/training_settings_only.json --diag $ROOT/scripts \
    --out "$RUN/train" --beta 0.1 --lr 5e-6 --epochs 1 --grad-accum 16 --warmup-ratio 0.03 \
    --weight-decay 0.01 --max-grad-norm 1.0 --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
    --max-tokens 8400 --seed 3407 --workers 2 > "$RUN/train.log" 2>&1
  done_mark train
fi

if ! is_done eval; then
  wait_gpu; log "[6] MMMU-val 평가 (병합 모델)"
  (cd "$EVAL" && $EVAL_PY src/evaluate.py --config src/config.json --checkpoint "$RUN/train/merged" \
    --run-name $RUN_NAME) > "$RUN/eval.log" 2>&1
  rm -rf "$RUN/eval"; cp -r "$EVAL/outputs/$RUN_NAME" "$RUN/eval"
  $EVAL_PY $ROOT/scripts/summarize_eval.py --eval-dir "$RUN/eval" --out "$RUN/summary_vs_baseline.md" \
    --label "DPO(ep4 시작)" >> "$RUN/eval.log" 2>&1 || true
  done_mark eval
  log "평가 완료: $(sed -n 3p "$RUN/summary_vs_baseline.md" 2>/dev/null) / 시작점 ep4는 510/900 = 56.67%"
fi

if ! is_done publish; then
  log "[7] 개인 레포 푸시"
  until PATH=$HOME/.local/bin:$PATH bash $ROOT/scripts/publish_dpo.sh >> "$RUN/publish.log" 2>&1; do
    log "푸시 실패 - 10분 후 재시도"; sleep 600; done
  done_mark publish
fi
log "===== DPO 파이프라인 전체 완료 ====="
