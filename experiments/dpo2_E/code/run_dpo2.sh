#!/usr/bin/env bash
# 베이스 모델에서 시작하는 DPO ablation 4개: A′(순수 DPO) / A(DPO+SFT 항) / B(A+ERA 향) / C(B+시각 향)
# 순서: 쌍A → (학습률: lr_sweep으로 정함) → ERA 샘플링·채점 → 쌍B → B·C 신호 시험(각 30스텝) → 효과 없는 요소 제외
#       → A′·A 학습·평가(오답 쌍에 신호 있을 때만) → B 학습·평가 → C 학습·평가
# 실험마다 runs/dpo2/<실험>/ 에 따로 저장하고, 평가가 끝날 때마다 개인 레포에 따로 푸시한다.
# 다시 실행하면 완료 표시(.done)가 있는 단계는 건너뛴다.
set -euo pipefail

ROOT=/home/latuca/MMDL_solo
EVAL=/home/latuca/mmdl_test
TRAIN_PY=/home/latuca/venvs/train/bin/python
EVAL_PY=$EVAL/.venv/bin/python
SFT_DATA=$ROOT/runs/sft_general_vic2000_e5/data/train.jsonl
OLD=$ROOT/runs/dpo_ep4_mix_b0.1_lr5e-6/data          # 이전 실험에서 받은 베이스 샘플·채점 재사용
BASE=/home/latuca/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17
IMAGES=$ROOT/data/vic_era_2000
VCFG=$ROOT/config/training_settings_only.json
RUN=$ROOT/runs/dpo2
D=$RUN/data
mkdir -p "$D"
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
ERA_INSTR=$'\nStructure your reasoning in three labeled parts before the final line:\nEvidence: what the image shows that matters for the question (read values, labels, and axes carefully).\nRule: the knowledge, formula, or principle that applies.\nApplication: apply the rule to the evidence step by step, including any calculations.'

train_exp() {   # $1=실험 이름, $2=쌍 파일, 나머지=추가 옵션
  local X=$1 P=$2; shift 2
  local LR LAM
  LR=$(python3 -c "import json;print(json.load(open('$RUN/hparams.json'))['lr'])")
  LAM=$(python3 -c "import json;print(json.load(open('$RUN/hparams.json'))['lambda'])")
  if ! is_done train_$X; then
    wait_gpu; log "[$X] 학습 시작 (lr $LR, beta 0.1, 1 epoch, 옵션: $*)"
    rm -rf "$RUN/$X/train"; mkdir -p "$RUN/$X"
    $TRAIN_PY -u $ROOT/scripts/train_dpo.py --pairs "$P" --images $IMAGES --start-model $BASE \
      --vision-config $VCFG --diag $ROOT/scripts --out "$RUN/$X/train" --beta 0.1 --lr "$LR" --epochs 1 \
      --grad-accum 16 --warmup-ratio 0.03 --weight-decay 0.01 --max-grad-norm 1.0 \
      --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 --max-tokens 8400 --seed 3407 --workers 2 \
      "${@//__LAMBDA__/$LAM}" > "$RUN/$X/train.log" 2>&1
    done_mark train_$X
  fi
}

eval_exp() {    # $1=실험 이름
  local X=$1
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
}

log "===== DPO2 파이프라인 시작 ====="

if ! is_done pairs_A; then
  log "[쌍 A] 베이스 자기 정답 vs 자기 오답 (기존 베이스 샘플 재사용)"
  python3 $ROOT/scripts/dpo_pairs2.py --train $SFT_DATA --samples $OLD/samples_base.jsonl --labels $OLD/labels.jsonl \
    --mode self --out "$D/pairs_A.jsonl" > "$D/pairs_A.log" 2>&1
  log "쌍 A: $(cat "$D/pairs_A.report.json" | tr -d '\n ')"
  done_mark pairs_A
fi

if ! is_done measure; then
  wait_gpu; log "[측정] 학습률·λ 1차 근사 측정 (쌍 A 64개)"
  NA=$(wc -l < "$D/pairs_A.jsonl")
  $TRAIN_PY -u $ROOT/scripts/dpo_measure.py --pairs "$D/pairs_A.jsonl" --images $IMAGES --start-model $BASE \
    --vision-config $VCFG --diag $ROOT/scripts --out "$RUN/measure.json" --beta 0.1 --grad-accum 16 --batches 4 \
    --z-target 2.0 --n-pairs-total "$NA" > "$RUN/measure.log" 2>&1
  # 4개 실험 모두 같은 lr·λ (ablation 비교를 깨끗하게). 극단값 방지를 위해 범위 제한.
  python3 - "$RUN/measure.json" "$RUN/hparams.json" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
lr = min(max(m["lr_adam_estimate"], 1e-6), 1e-4)
lam = min(max(m["lambda_balanced"], 0.05), 1.0)
json.dump({"lr": float(f"{lr:.2e}"), "lambda": round(lam, 3), "lr_raw": m["lr_adam_estimate"],
           "lambda_raw": m["lambda_balanced"], "clipped_to": {"lr": [1e-6, 1e-4], "lambda": [0.05, 1.0]}},
          open(sys.argv[2], "w"), indent=1)
EOF
  log "측정 결과: $(cat "$RUN/hparams.json" | tr -d '\n ')"
  done_mark measure
fi

if ! is_done sample_era; then
  wait_gpu; log "[ERA] ERA 지시로 베이스 응답 샘플링 (2,000문항 × 8, 지시는 샘플링 때만)"
  (cd "$EVAL" && $EVAL_PY $ROOT/scripts/dpo_sample.py --model $BASE --tag base_era --data $SFT_DATA --images $IMAGES \
    --eval-src $EVAL/src --n 8 --max-total 8192 --chunk 128 --gen-instruction "$ERA_INSTR" \
    --out "$D/samples_base_era.jsonl") >> "$RUN/sample_era.log" 2>&1
  done_mark sample_era
fi
if ! is_done grade_era; then
  wait_gpu; log "[ERA] 채점 (평가 파서 + Qwen3-8B 저지)"
  (cd "$EVAL" && $EVAL_PY $ROOT/scripts/dpo_grade.py --samples "$D/samples_base_era.jsonl" \
    --source $ROOT/data/vic_era_2000/data.json --eval-src $EVAL/src --out "$D/labels_era.jsonl") >> "$RUN/grade_era.log" 2>&1
  done_mark grade_era
fi
if ! is_done pairs_B; then
  log "[쌍 B/C] chosen = ERA 정답·베이스 정답, rejected = 끊긴 응답·베이스 오답"
  python3 $ROOT/scripts/dpo_pairs2.py --train $SFT_DATA --samples $OLD/samples_base.jsonl "$D/samples_base_era.jsonl" \
    --labels $OLD/labels.jsonl "$D/labels_era.jsonl" --mode era --out "$D/pairs_B.jsonl" > "$D/pairs_B.log" 2>&1
  log "쌍 B/C: $(cat "$D/pairs_B.report.json" | tr -d '\n ')"
  done_mark pairs_B
fi


# ---- 신호 시험: B·C 설정으로 각 30스텝, 처음 보는 쌍에서 유형별 정답 선호를 측정해 효과 없는 요소를 뺀다 ----
gate_test() {   # $1=시험 이름, 나머지=추가 옵션
  local X=$1; shift
  local LR LAM
  LR=$(python3 -c "import json;print(json.load(open('$RUN/hparams.json'))['lr'])")
  LAM=$(python3 -c "import json;print(json.load(open('$RUN/hparams.json'))['lambda'])")
  if ! is_done gtest_$X; then
    wait_gpu; log "[신호 시험 $X] 30스텝 (lr $LR, 옵션: $*)"
    rm -rf "$RUN/gate_test/$X"; mkdir -p "$RUN/gate_test"
    $TRAIN_PY -u $ROOT/scripts/train_dpo.py --pairs "$D/pairs_B.jsonl" --images $IMAGES --start-model $BASE \
      --vision-config $VCFG --diag $ROOT/scripts --out "$RUN/gate_test/$X" --beta 0.1 --lr "$LR" --max-steps 30 \
      --grad-accum 16 --warmup-ratio 0.03 --weight-decay 0.01 --max-grad-norm 1.0 \
      --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 --max-tokens 8400 --seed 3407 --workers 2 --no-merge \
      "${@//__LAMBDA__/$LAM}" > "$RUN/gate_test/$X.log" 2>&1
    done_mark gtest_$X
  fi
}
gate_test B --rpo-alpha __LAMBDA__
gate_test C --rpo-alpha __LAMBDA__ --vision-lora --gates --gate-lr 1e-3 --img-pref-types V,I
if ! is_done gate; then
  python3 $ROOT/scripts/dpo2_gate.py --b-log "$RUN/gate_test/B/pair_log.jsonl" --c-log "$RUN/gate_test/C/pair_log.jsonl" \
    --pairs "$D/pairs_B.jsonl" --pairs-out "$D/pairs_B_final.jsonl" --out "$RUN/gate.json" > "$RUN/gate.log" 2>&1
  log "[신호 시험 판정] $(python3 -c "import json;g=json.load(open('$RUN/gate.json'));print('조합별',g['B_test_combo'],'/ 이미지조건',g['C_test_img'],'→ 유지',g['keep_combos'],'A실행',g['run_A'],'이미지손실',g['img_pref'],'최종쌍',g['final_pairs']['pairs'])")"
  done_mark gate
fi
G() { python3 -c "import json;print(json.load(open('$RUN/gate.json'))['$1'])"; }
if [ "$(G stop)" = "True" ]; then
  log "===== 오답쌍·끊긴응답쌍 모두 신호 없음 → 중단 (쌍 구성 재설계 필요) ====="; exit 3
fi

if [ "$(G run_A)" = "True" ]; then
  train_exp A_prime "$D/pairs_A.jsonl" --rpo-alpha 0
  eval_exp A_prime
  train_exp A "$D/pairs_A.jsonl" --rpo-alpha __LAMBDA__
  eval_exp A
else
  log "[A′·A] 자기 오답 쌍에 신호 없음 → 생략 (근거: lr_sweep/summary.txt, gate.json)"
fi

train_exp B "$D/pairs_B_final.jsonl" --rpo-alpha __LAMBDA__
eval_exp B
if [ "$(G img_pref)" = "True" ]; then
  train_exp C "$D/pairs_B_final.jsonl" --rpo-alpha __LAMBDA__ --vision-lora --gates --gate-lr 1e-3 --img-pref-types V,I
else
  log "[C] 이미지 조건 손실에 신호 없음 → 시각 LoRA·게이트만 유지"
  train_exp C "$D/pairs_B_final.jsonl" --rpo-alpha __LAMBDA__ --vision-lora --gates --gate-lr 1e-3
fi
eval_exp C
log "===== DPO2 파이프라인 전체 완료 ====="
