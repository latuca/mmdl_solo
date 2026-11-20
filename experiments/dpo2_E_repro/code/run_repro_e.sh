#!/usr/bin/env bash
# DPO2-E 재현 파이프라인 (실습실 노드용). 베이스 샘플링부터 E 학습·MMMU 평가까지.
# 원본: experiments/dpo_ep4_mix_b0.1_lr5e-6/code/run_dpo.sh (베이스 샘플·채점),
#       experiments/dpo2_E/code/run_dpo2.sh (ERA·쌍 B), run_dpo2_d.sh (쌍 D), run_dpo2_e.sh (쌍 E·학습·평가)
# 바뀐 점: 경로를 /shared_dir 기준으로, ep4 샘플링 생략(쌍 B·D는 tag=base 라벨만 씀),
#          측정·lr 시험·신호 시험 생략(hparams.json·gate_signal_test.json 고정값), publish 생략,
#          단계마다 원래 리포트와 숫자를 비교해 크게 어긋나면 멈춘다(exit 4).
# 다시 실행하면 완료 표시(.done)가 있는 단계는 건너뛴다. 샘플링은 끝난 문항부터 이어서 한다.
# 실행: nohup bash /shared_dir/MMDL_solo/repro/run_repro_e.sh >> /shared_dir/logs/repro_e.out 2>&1 &
#   STOP_AFTER=<단계이름> 을 주면 그 단계까지만 하고 멈춘다.
set -euo pipefail
source /shared_dir/MMDL_solo/repro/env.sh
mkdir -p "$D" "$OLD" "$(dirname "$SFT_DATA")" "$ROOT/config" "$ROOT/scripts"
X=E
log() { echo "[$(date '+%F %T')] $*" | tee -a "$RUN/pipeline.log"; }
done_mark() { touch "$RUN/.$1.done"; }
is_done() { [ -f "$RUN/.$1.done" ]; }
stop_after() { if [ "${STOP_AFTER:-}" = "$1" ]; then log "STOP_AFTER=$1 → 여기서 멈춤"; exit 0; fi; }
wait_gpu() {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2000 ] && return 0
    log "GPU 사용 중(${used}MiB) - 5분 후 다시 확인"; sleep 300
  done
}
# 이어서 할 수 없는 긴 작업(학습·평가) 전: 수업 차단 시간에 걸리면 차단이 끝날 때까지 기다린다.
# 차단 시간은 repro/blocked_times.txt (예: "Mon 09:00-12:00", 한 줄에 하나). 파일이 없으면 검사하지 않는다.
guard_window() {   # $1=필요한 분, $2=작업 이름
  local f=$ROOT/repro/blocked_times.txt
  [ -f "$f" ] || return 0
  while true; do
    w=$(python3 "$ROOT/repro/blocked_window.py" "$f" "$1")
    [ "$w" = "0" ] && return 0
    log "[$2] ${1}분 안에 수업 차단 시간이 시작됨 → ${w}초 기다림"; sleep "$w"
  done
}
# 원래 리포트와 비교: $1=이름 $2=재현값 $3=원래값 $4=허용 상대오차
check() {
  python3 - "$@" <<'EOF' || { log "[확인 실패] $1: 재현 $2 / 원래 $3 (허용 ±$4) → 멈춤, 원인 확인 필요"; exit 4; }
import sys
name, got, ref, tol = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
sys.exit(0 if abs(got - ref) <= tol * abs(ref) else 1)
EOF
  log "[확인] $1: 재현 $2 / 원래 $3 (허용 ±$4) OK"
}
J() { python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(eval(sys.argv[2],{},{'d':d}))" "$1" "$2"; }
export HF_HUB_OFFLINE=1
ERA_INSTR=$'\nStructure your reasoning in three labeled parts before the final line:\nEvidence: what the image shows that matters for the question (read values, labels, and axes carefully).\nRule: the knowledge, formula, or principle that applies.\nApplication: apply the rule to the evidence step by step, including any calculations.'

# ---- 0) 코드·설정 배치 (원본은 그대로 두고 $ROOT/scripts, $ROOT/config, $RUN 에 복사) ----
if ! is_done setup; then
  for f in "$ROOT"/experiments/dpo2_E/code/*.py "$ROOT"/experiments/sft_general_vic2000_e5/code/prepare_sft_general.py; do
    dst=$ROOT/scripts/$(basename "$f")
    if [ -f "$dst" ] && ! cmp -s "$f" "$dst"; then echo "충돌: $dst 가 이미 있고 내용이 다름" >&2; exit 2; fi
    cp "$f" "$dst"
  done
  cp "$ROOT/experiments/sft_general_vic2000_e5/config/training_settings_only.json" "$VCFG"
  cp "$EXP_E/hparams.json" "$RUN/hparams.json"
  log "===== DPO2-E 재현 시작 (코드 배치 완료) ====="
  done_mark setup
fi
for p in "$IMAGES/data.json" "$EVAL_PY" "$BASE/config.json"; do
  [ -e "$p" ] || { log "필요한 파일 없음: $p"; exit 2; }
done

# ---- 1) SFT 학습 데이터 (1,997문항: V 398 / I 600 / C 999) ----
if ! is_done sft_data; then
  log "[SFT 데이터] prepare_sft_general.py (max-tokens 8192)"
  $TRAIN_PY $ROOT/scripts/prepare_sft_general.py --data-dir $IMAGES --eval-src $EVAL/src \
    --out "$SFT_DATA" --max-tokens 8192 > "$RUN/prepare_sft.log" 2>&1
  n=$(wc -l < "$SFT_DATA"); log "SFT 데이터: $n문항 / $(tail -1 "$RUN/prepare_sft.log" | cut -c1-300)"
  check sft_rows "$n" 1997 0
  done_mark sft_data
fi
stop_after sft_data

# ---- 2) 베이스 샘플링 (2,000문항 × 8, 예전 서버 약 7.8시간, 이어서 하기 가능) ----
COMMON="--data $SFT_DATA --images $IMAGES --eval-src $EVAL/src --n 8 --max-total 8192 --chunk 128"
if ! is_done sample_base; then
  wait_gpu; log "[베이스] 응답 샘플링 (문항당 8개)"
  (cd "$EVAL" && $EVAL_PY $ROOT/scripts/dpo_sample.py --model $BASE_REPO --revision $BASE_REV --tag base $COMMON \
    --out "$OLD/samples_base.jsonl") >> "$RUN/sample_base.log" 2>&1
  check base_sample_rows "$(wc -l < "$OLD/samples_base.jsonl")" 1997 0
  done_mark sample_base
fi
if ! is_done grade_base; then
  wait_gpu; log "[베이스] 채점 (평가 파서 + Qwen3-8B 저지)"
  (cd "$EVAL" && $EVAL_PY $ROOT/scripts/dpo_grade.py --samples "$OLD/samples_base.jsonl" \
    --source $IMAGES/data.json --eval-src $EVAL/src --out "$OLD/labels.jsonl") >> "$RUN/grade_base.log" 2>&1
  acc=$(python3 -c "import json;L=[json.loads(l) for l in open('$OLD/labels.jsonl')];print(round(sum(r['label'] is True for r in L)/len(L),4))")
  log "[베이스] 정답 비율 $acc (원래 9367/15976 = 0.5863)"
  check base_acc "$acc" 0.5863 0.06
  done_mark grade_base
fi
stop_after grade_base

# ---- 3) ERA 샘플링 (예전 서버 약 6시간, 이어서 하기 가능) + 채점 (약 20분) ----
if ! is_done sample_era; then
  wait_gpu; log "[ERA] ERA 지시로 베이스 응답 샘플링 (2,000문항 × 8, 지시는 샘플링 때만)"
  (cd "$EVAL" && $EVAL_PY $ROOT/scripts/dpo_sample.py --model $BASE --tag base_era --data $SFT_DATA --images $IMAGES \
    --eval-src $EVAL/src --n 8 --max-total 8192 --chunk 128 --gen-instruction "$ERA_INSTR" \
    --out "$D/samples_base_era.jsonl") >> "$RUN/sample_era.log" 2>&1
  check era_sample_rows "$(wc -l < "$D/samples_base_era.jsonl")" 1997 0
  done_mark sample_era
fi
if ! is_done grade_era; then
  wait_gpu; log "[ERA] 채점 (평가 파서 + Qwen3-8B 저지)"
  (cd "$EVAL" && $EVAL_PY $ROOT/scripts/dpo_grade.py --samples "$D/samples_base_era.jsonl" \
    --source $IMAGES/data.json --eval-src $EVAL/src --out "$D/labels_era.jsonl") >> "$RUN/grade_era.log" 2>&1
  done_mark grade_era
fi
stop_after grade_era

# ---- 4) 쌍 B (원래 1810쌍: chosen_era 984 / base 826, rejected wrong 1359 / truncated 451) ----
if ! is_done pairs_B; then
  log "[쌍 B] chosen = ERA 정답·베이스 정답, rejected = 끊긴 응답·베이스 오답"
  python3 $ROOT/scripts/dpo_pairs2.py --train $SFT_DATA --samples $OLD/samples_base.jsonl "$D/samples_base_era.jsonl" \
    --labels $OLD/labels.jsonl "$D/labels_era.jsonl" --mode era --out "$D/pairs_B.jsonl" > "$D/pairs_B.log" 2>&1
  log "쌍 B: $(tr -d '\n ' < "$D/pairs_B.report.json")"
  check pairs_B "$(J "$D/pairs_B.report.json" "d['pairs']")" 1810 0.15
  check pairs_B_chosen_era "$(J "$D/pairs_B.report.json" "d['stats'].get('chosen_era',0)")" 984 0.2
  done_mark pairs_B
fi

# ---- 5) 쌍 B_final + gate.json (신호 시험 생략, 원래 판정 고정. 원래 1134쌍) ----
if ! is_done gate; then
  python3 $ROOT/repro/repro_gate.py --orig-gate "$EXP_E/gate_signal_test.json" --pairs "$D/pairs_B.jsonl" \
    --pairs-out "$D/pairs_B_final.jsonl" --out "$RUN/gate.json" > "$RUN/gate.log" 2>&1
  log "[쌍 B_final] $(tr -d '\n ' < "$D/pairs_B_final.report.json")"
  check pairs_B_final "$(J "$D/pairs_B_final.report.json" "d['pairs']")" 1134 0.15
  done_mark gate
fi

# ---- 6) 쌍 D (원래 931쌍, era_wrong 683, replay 300) ----
if ! is_done pairs_D; then
  python3 $ROOT/scripts/dpo_pairs_d.py --train $SFT_DATA --samples $OLD/samples_base.jsonl "$D/samples_base_era.jsonl" \
    --labels $OLD/labels.jsonl "$D/labels_era.jsonl" --b-final "$D/pairs_B_final.jsonl" --out "$D/pairs_D.jsonl" > "$D/pairs_D.log" 2>&1
  log "쌍 D: $(tr -d '\n ' < "$D/pairs_D.report.json")"
  check pairs_D "$(J "$D/pairs_D.report.json" "d['pairs']")" 931 0.15
  check pairs_D_era_wrong "$(J "$D/pairs_D.report.json" "d['stats']['era_wrong_pairs']")" 683 0.15
  done_mark pairs_D
fi
stop_after pairs_D

# ---- 7) 쌍 E (원래 rows 2287, friend_loop_pairs 326 ×4, era_wrong 683, replay 300) ----
if ! is_done pairs_E; then
  [ -f "$FR/datasets/dpo_r4_loop/train.jsonl" ] || { log "팀원 루프 쌍 없음: $FR/datasets/dpo_r4_loop/train.jsonl"; exit 2; }
  $TRAIN_PY $ROOT/scripts/dpo_pairs_e.py --friend $FR/datasets/dpo_r4_loop/train.jsonl --friend-root $FR \
    --pairs-d "$D/pairs_D.jsonl" --tokenizer $BASE --out "$D/pairs_E.jsonl" > "$D/pairs_E.log" 2>&1
  log "쌍 E: $(tr -d '\n ' < "$D/pairs_E.report.json")"
  check pairs_E_friend "$(J "$D/pairs_E.report.json" "d['friend_loop_pairs']")" 326 0
  check pairs_E_rows "$(J "$D/pairs_E.report.json" "d['rows']")" 2287 0.1
  done_mark pairs_E
fi
stop_after pairs_E

# ---- 8) E 학습 (예전 서버 약 2시간, 143스텝, 이어서 하기 불가) ----
LR=1e-5
LAM=$(J "$RUN/hparams.json" "d['lambda']")
IMG=""; [ "$(J "$RUN/gate.json" "d['img_pref']")" = "True" ] && IMG="--img-pref-types V,I"
if ! is_done train_$X; then
  guard_window 150 "train_$X"; wait_gpu
  log "[$X] 학습 시작 (lr $LR, λ $LAM, 팀원 루프 쌍×4 + ERA>오답 + 보존 300, $IMG)"
  rm -rf "$RUN/$X/train"; mkdir -p "$RUN/$X"
  $TRAIN_PY -u $ROOT/scripts/train_dpo_d.py --pairs "$D/pairs_E.jsonl" --images $IMAGES --start-model $BASE \
    --vision-config $VCFG --diag $ROOT/scripts --out "$RUN/$X/train" --beta 0.1 --lr "$LR" --epochs 1 \
    --grad-accum 16 --warmup-ratio 0.03 --weight-decay 0.01 --max-grad-norm 1.0 \
    --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 --max-tokens 8400 --seed 3407 --workers 2 \
    --rpo-alpha "$LAM" --vision-lora --gates --gate-lr 1e-3 $IMG --replay-weight 1.0 > "$RUN/$X/train.log" 2>&1
  [ -f "$RUN/$X/train/merged/config.json" ] || { log "[$X] merged 없음 → 학습 로그 확인"; exit 5; }
  log "[$X] 학습 완료: $(tail -1 "$RUN/$X/train/train_log.jsonl" | cut -c1-300)"
  done_mark train_$X
fi
stop_after train_$X

# ---- 9) MMMU-val 평가 (예전 서버 약 5.5시간, 이어서 하기 불가) ----
if ! is_done eval_$X; then
  guard_window 360 "eval_$X"; wait_gpu; log "[$X] MMMU-val 평가 시작"
  rm -rf "$EVAL/outputs/dpo2_$X"
  (cd "$EVAL" && $EVAL_PY src/evaluate.py --config src/config.json --checkpoint "$RUN/$X/train/merged" \
    --run-name dpo2_$X) > "$RUN/$X/eval.log" 2>&1
  rm -rf "$RUN/$X/eval"; cp -r "$EVAL/outputs/dpo2_$X" "$RUN/$X/eval"
  $EVAL_PY $ROOT/scripts/summarize_eval.py --eval-dir "$RUN/$X/eval" --out "$RUN/$X/summary_vs_baseline.md" \
    --label "DPO2-$X (재현)" >> "$RUN/$X/eval.log" 2>&1 || true
  python3 $ROOT/scripts/eval_extra.py --eval-dir "$RUN/$X/eval" --out "$RUN/$X/extra_stats.json" >> "$RUN/$X/eval.log" 2>&1 || true
  done_mark eval_$X
  log "[$X] 평가 완료: $(sed -n 3p "$RUN/$X/summary_vs_baseline.md" 2>/dev/null) / $(tr -d '\n ' < "$RUN/$X/extra_stats.json" 2>/dev/null) (원래 608/900 = 67.56%)"
fi
log "===== DPO2-E 재현 완료 (다음: rclone 회수) ====="
