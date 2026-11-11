#!/usr/bin/env bash
# 실험 코드·결과를 github.com/latuca/mmdl_solo 의 experiments/sft_general_vic2000_e5/ 에 올린다.
# 기존 레포 파일은 건드리지 않는다. 대용량(병합 모델·이미지·학습 데이터)은 올리지 않는다.
# 평가 코드(yunheeseong1/mmdl_test)는 비공개 레포라 복사하지 않고 커밋 해시만 기록한다.
set -euo pipefail
ROOT=/home/latuca/MMDL_solo
RUN=$ROOT/runs/sft_general_vic2000_e5
REPO=$ROOT/_publish
DST=$REPO/experiments/sft_general_vic2000_e5
MSG=${1:-"sft_general_vic2000_e5: 결과 업데이트"}

[ -d "$REPO/.git" ] || git clone -q git@github.com:latuca/mmdl_solo.git "$REPO"
cd "$REPO" && git pull -q --rebase origin main

rm -rf "$DST"; mkdir -p "$DST/code" "$DST/config" "$DST/data_report" "$DST/train"
cp $ROOT/scripts/*.py $ROOT/scripts/*.sh "$DST/code/"
cp $ROOT/config/training_settings_only.json "$DST/config/"
cp $RUN/data/train.report.json $RUN/data/prepare.log "$DST/data_report/" 2>/dev/null || true
cp $ROOT/data/vic_era_2000/recover_report.json "$DST/data_report/" 2>/dev/null || true
cp $RUN/pipeline.log "$DST/" 2>/dev/null || true
cp $RUN/train.log "$DST/train/" 2>/dev/null || true
cp $RUN/train/train_info.json $RUN/train/train_log.jsonl "$DST/train/" 2>/dev/null || true
echo "mmdl_test commit: $(git -C /home/latuca/mmdl_test rev-parse HEAD)" > "$DST/config/eval_code_version.txt"

RES="| 체크포인트 | MMMU-val | 베이스라인 대비 |\n|---|---:|---:|\n| baseline | 576/900 = 64.00% | - |\n"
for EP in 2 4 5; do
  CK=$RUN/ckpt_ep$EP; D=$DST/ckpt_ep$EP
  A=$RUN/train/adapter_ep$EP
  if [ -d "$A" ]; then
    mkdir -p "$D/adapter"
    # GitHub 단일 파일 100MB 제한 - 95MB 넘는 파일은 제외
    find "$A" -maxdepth 1 -type f -size -95M -exec cp {} "$D/adapter/" \;
  fi
  if [ -f "$RUN/.eval_ep$EP.done" ]; then
    mkdir -p "$D"
    cp "$CK/summary_vs_baseline.md" "$D/" 2>/dev/null || true
    [ -d "$CK/eval" ] && cp -r "$CK/eval" "$D/eval"
    find "$D/eval" -type f -size +95M -delete 2>/dev/null || true
    tail -200 "$CK/eval.log" > "$D/eval_log_tail.txt" 2>/dev/null || true
    LINE=$(sed -n 3p "$CK/summary_vs_baseline.md" 2>/dev/null | sed 's/\*\*//g')
    ACC=$(echo "$LINE" | sed -E 's/^[^:]*: ([^ ]+ = [^ ]+) .*/\1/')
    DIFF=$(echo "$LINE" | sed -E 's/.*차이 ([^)]*)\).*/\1/')
    RES+="| epoch $EP | $ACC | $DIFF |\n"
  else
    RES+="| epoch $EP | (평가 대기/진행 중) | - |\n"
  fi
done

cat > "$DST/README.md" <<EOF
# SFT: 일반 풀이만 학습 (vic_era_2000, 5 epoch)

- 베이스 모델: Qwen/Qwen3-VL-4B-Instruct @ ebb281ec70b05090aa6165b016eac8ec08e71b17
- 데이터: vic_era_2000의 \`explanation\`(일반 풀이) 1,997문항 (V 398 / I 600 / C 999, 풀이 없는 3문항 제외)
- LoRA r16 / alpha 32 / dropout 0.05 (q,k,v,o,gate,up,down), AdamW lr 1e-4, wd 0.01, grad clip 1.0
- cosine + warmup 3%, micro batch 1 × grad accum 16, BF16, gradient checkpointing, packing 없음, seed 3407
- 해상도: min_pixels 1280×28×28, max_pixels 5120×28×28 (평가와 동일), 학습·추론 프롬프트 동일, \`<think>\` 제거
- 평가: LoRA 병합 모델, MMMU-val 900문항, mmdl_test 평가 코드 (버전: config/eval_code_version.txt)

## 결과

$(printf "%b" "$RES")

체크포인트별 과목 표: \`ckpt_epN/summary_vs_baseline.md\`, 문항별 출력: \`ckpt_epN/eval/\`

## 폴더
- code/: 데이터 변환·학습·병합·평가 파이프라인 스크립트
- data_report/: 학습 데이터 변환 통계, 이미지 복구 리포트 (원본 데이터·이미지는 미포함)
- train/: 학습 로그 (loss 등)
- ckpt_epN/adapter/: 해당 epoch LoRA 어댑터
EOF

cd "$REPO"
git add -A experiments/sft_general_vic2000_e5
if [ -n "${NOPUSH:-}" ]; then git status --short | head -40; exit 0; fi
if git diff --cached --quiet; then echo "변경 없음"; exit 0; fi
git commit -q -m "$MSG

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push -q origin HEAD:main
echo "푸시 완료: $(git rev-parse --short HEAD)"
