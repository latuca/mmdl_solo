#!/usr/bin/env bash
# 실험 E: 친구 루프 쌍 326개(×4) + D의 ERA 정답>오답 683쌍 + 보존 300개. lr 1e-5(친구 설정), 그 밖의 설정은 D와 같다.
# 원래 맞히던 문항을 덜 잃도록 lr을 5e-5 → 1e-5로 낮춘다. 친구 데이터는 서버에만 두고 레포에 올리지 않는다.
set -euo pipefail
ROOT=/home/latuca/MMDL_solo
EVAL=/home/latuca/mmdl_test
TRAIN_PY=/home/latuca/venvs/train/bin/python
EVAL_PY=$EVAL/.venv/bin/python
SFT_DATA=$ROOT/runs/sft_general_vic2000_e5/data/train.jsonl
OLD=$ROOT/runs/dpo_ep4_mix_b0.1_lr5e-6/data
BASE=/home/latuca/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17
IMAGES=$ROOT/data/vic_era_2000
VCFG=$ROOT/config/training_settings_only.json
RUN=$ROOT/runs/dpo2
D=$RUN/data
X=E
log() { echo "[$(date '+%F %T')] $*" | tee -a "$RUN/pipeline.log"; }
done_mark() { touch "$RUN/.$1.done"; }
is_done() { [ -f "$RUN/.$1.done" ]; }
wait_gpu() {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2000 ] && return 0
    sleep 300
  done
}
export HF_HUB_OFFLINE=1
FR=/home/latuca/friend_dpo_r4/x

if ! is_done pairs_E; then
  $TRAIN_PY $ROOT/scripts/dpo_pairs_e.py --friend $FR/datasets/dpo_r4_loop/train.jsonl --friend-root $FR \
    --pairs-d "$D/pairs_D.jsonl" --tokenizer $BASE --out "$D/pairs_E.jsonl" > "$D/pairs_E.log" 2>&1
  log "쌍 E: $(cat "$D/pairs_E.report.json" | tr -d '\n ')"
  done_mark pairs_E
fi
LR=1e-5
LAM=$(python3 -c "import json;print(json.load(open('$RUN/hparams.json'))['lambda'])")
IMG=""; [ "$(python3 -c "import json;print(json.load(open('$RUN/gate.json'))['img_pref'])")" = "True" ] && IMG="--img-pref-types V,I"
if ! is_done train_$X; then
  wait_gpu; log "[$X] 학습 시작 (lr $LR, λ $LAM, 친구 루프 쌍×4 + ERA>오답 + 보존 300, 시각 옵션 D와 같음 $IMG)"
  rm -rf "$RUN/$X/train"; mkdir -p "$RUN/$X"
  $TRAIN_PY -u $ROOT/scripts/train_dpo_d.py --pairs "$D/pairs_E.jsonl" --images $IMAGES --start-model $BASE \
    --vision-config $VCFG --diag $ROOT/scripts --out "$RUN/$X/train" --beta 0.1 --lr "$LR" --epochs 1 \
    --grad-accum 16 --warmup-ratio 0.03 --weight-decay 0.01 --max-grad-norm 1.0 \
    --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 --max-tokens 8400 --seed 3407 --workers 2 \
    --rpo-alpha "$LAM" --vision-lora --gates --gate-lr 1e-3 $IMG --replay-weight 1.0 > "$RUN/$X/train.log" 2>&1
  done_mark train_$X
fi
if ! is_done eval_$X; then
  wait_gpu; log "[$X] MMMU-val 평가 시작"
  (cd "$EVAL" && $EVAL_PY src/evaluate.py --config src/config.json --checkpoint "$RUN/$X/train/merged" \
    --run-name dpo2_$X) > "$RUN/$X/eval.log" 2>&1
  rm -rf "$RUN/$X/eval"; cp -r "$EVAL/outputs/dpo2_$X" "$RUN/$X/eval"
  $EVAL_PY $ROOT/scripts/summarize_eval.py --eval-dir "$RUN/$X/eval" --out "$RUN/$X/summary_vs_baseline.md" \
    --label "DPO2-$X" >> "$RUN/$X/eval.log" 2>&1 || true
  python3 $ROOT/scripts/eval_extra.py --eval-dir "$RUN/$X/eval" --out "$RUN/$X/extra_stats.json" >> "$RUN/$X/eval.log" 2>&1 || true
  done_mark eval_$X
  log "[$X] 평가 완료: $(sed -n 3p "$RUN/$X/summary_vs_baseline.md" 2>/dev/null) / $(cat "$RUN/$X/extra_stats.json" 2>/dev/null | tr -d '\n ')"
fi
if ! is_done publish_$X; then
  until PATH=$HOME/.local/bin:$PATH bash $ROOT/scripts/publish_dpo2.sh "$X" >> "$RUN/publish.log" 2>&1; do
    log "[$X] 푸시 실패 - 10분 후 재시도"; sleep 600; done
  done_mark publish_$X; log "[$X] 개인 레포 푸시 완료"
fi
log "===== 실험 E 완료 ====="
