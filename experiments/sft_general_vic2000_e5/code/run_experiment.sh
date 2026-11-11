#!/usr/bin/env bash
# 일반 풀이(explanation)만 학습 → epoch 2/4/5 체크포인트 병합 → MMMU-val 평가
# 다시 실행하면 완료 표시(.done)가 있는 단계는 건너뛴다.
set -euo pipefail

ROOT=/home/latuca/MMDL_solo
EVAL=/home/latuca/mmdl_test
TRAIN_PY=/home/latuca/venvs/train/bin/python
EVAL_PY=$EVAL/.venv/bin/python
RUN=$ROOT/runs/sft_general_vic2000_e5
BASECFG=$ROOT/config/training_settings_only.json
EPOCHS_TO_EVAL="2 4 5"
mkdir -p "$RUN"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$RUN/pipeline.log"; }
done_mark() { touch "$RUN/.$1.done"; }
is_done() { [ -f "$RUN/.$1.done" ]; }

# GPU에 다른 프로세스가 있으면 비워질 때까지 대기
wait_gpu() {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2000 ] && return 0
    log "GPU 사용 중(${used}MiB) - 5분 후 다시 확인"; sleep 300
  done
}

# xet 전송이 이 회선에서 ~90KB/s로 막혀서 일반 HTTP 사용 (~1MB/s)
export HF_HUB_DISABLE_XET=1

log "===== 파이프라인 시작 ====="

if ! is_done base_model; then
  log "[0a] 학습용 베이스 모델(Qwen3-VL-4B, 고정 revision) 다운로드"
  until $TRAIN_PY -c "
from huggingface_hub import snapshot_download
print(snapshot_download('Qwen/Qwen3-VL-4B-Instruct', revision='ebb281ec70b05090aa6165b016eac8ec08e71b17',
  allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja','merges.txt','vocab.json']))" >> "$RUN/download_base.log" 2>&1
  do log "다운로드 실패 - 1분 후 재시도"; sleep 60; done
  done_mark base_model
fi

# 저지 모델(Qwen3-8B)·MMMU는 학습과 동시에 백그라운드로 준비 (평가 직전에 완료 대기)
ASSETS_PID=""
if ! is_done assets && pgrep -f "src/prepare.py --existing-env" >/dev/null; then
  log "[0b] 저지 모델·MMMU 준비가 이미 백그라운드에서 진행 중 - 새로 시작하지 않음"
elif ! is_done assets; then
  log "[0b] 저지 모델·MMMU 데이터 준비를 백그라운드로 시작"
  ( until (cd "$EVAL" && $EVAL_PY src/prepare.py --existing-env) >> "$RUN/prepare_assets.log" 2>&1
    do echo "[$(date '+%F %T')] prepare 실패 - 1분 후 재시도" >> "$RUN/prepare_assets.log"; sleep 60; done
    done_mark assets ) &
  ASSETS_PID=$!
fi

if ! is_done data; then
  log "[1] 학습 데이터 변환 (일반 풀이, 8192토큰 초과 제외)"
  mkdir -p "$RUN/data"
  $TRAIN_PY $ROOT/scripts/prepare_sft_general.py --data-dir $ROOT/data/vic_era_2000 \
    --eval-src $EVAL/src --out "$RUN/data/train.jsonl" --max-tokens 8192 >> "$RUN/data/prepare.log" 2>&1
  done_mark data
fi

if ! is_done train; then
  wait_gpu
  log "[2] 학습 시작 (5 epoch, epoch 2/4/5에 어댑터 저장)"
  rm -rf "$RUN/train"
  $TRAIN_PY -u $ROOT/scripts/train_sft.py --data "$RUN/data/train.jsonl" --images $ROOT/data/vic_era_2000 \
    --baseline $BASECFG --diag $ROOT/scripts --out "$RUN/train" \
    --epochs 5 --lr 1e-4 --weight-decay 0.01 --max-grad-norm 1.0 --warmup-ratio 0.03 \
    --grad-accum 16 --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
    --max-tokens 8192 --seed 3407 --workers 2 --save-every 1000000 --save-epochs 2,4,5 \
    --loss-mode token_mean --no-merge > "$RUN/train.log" 2>&1
  done_mark train
fi

for EP in $EPOCHS_TO_EVAL; do
  CK=$RUN/ckpt_ep$EP
  if ! is_done merge_ep$EP; then
    log "[3] epoch $EP 어댑터 병합 → $CK/merged"
    mkdir -p "$CK"; rm -rf "$CK/merged"
    $TRAIN_PY $ROOT/scripts/merge_adapter.py --baseline $BASECFG --adapter "$RUN/train/adapter_ep$EP" \
      --out "$CK/merged" > "$CK/merge.log" 2>&1
    done_mark merge_ep$EP
  fi
  if ! is_done eval_ep$EP; then
    if ! is_done assets; then
      log "평가 전: 저지 모델·MMMU 다운로드 완료 대기"
      [ -n "$ASSETS_PID" ] && wait "$ASSETS_PID" || true
      until is_done assets; do sleep 60; done
    fi
    wait_gpu
    log "[4] epoch $EP 평가 시작 (MMMU-val 900문항)"
    (cd "$EVAL" && $EVAL_PY src/evaluate.py --config src/config.json --checkpoint "$CK/merged" \
      --run-name sft_general_e5_ep$EP) > "$CK/eval.log" 2>&1
    rm -rf "$CK/eval"; cp -r "$EVAL/outputs/sft_general_e5_ep$EP" "$CK/eval"
    $EVAL_PY $ROOT/scripts/summarize_eval.py --eval-dir "$CK/eval" --out "$CK/summary_vs_baseline.md" \
      --label "epoch $EP" >> "$CK/eval.log" 2>&1 || true
    done_mark eval_ep$EP
    log "epoch $EP 평가 완료: $(head -3 "$CK/summary_vs_baseline.md" 2>/dev/null | tail -1)"
  fi
done
log "===== 전체 완료 ====="
