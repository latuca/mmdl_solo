#!/usr/bin/env bash
# DPO2 실험 하나($1 = A_prime | A | B | C)의 코드·결과를 개인 레포 experiments/dpo2_<실험>/ 에 올린다.
# 가중치·샘플 원문·쌍 데이터는 올리지 않는다.
set -euo pipefail
X=$1
ROOT=/home/latuca/MMDL_solo
RUN=$ROOT/runs/dpo2
REPO=$ROOT/_publish
DST=$REPO/experiments/dpo2_$X
declare -A DESC=(
  [A_prime]="베이스에서 순수 DPO (자기 정답 vs 자기 오답)"
  [A]="A′ + SFT 항(RPO, chosen NLL)"
  [B]="A + ERA 향 (ERA 지시로 만든 자기 정답을 chosen에 섞기, 끝없이 이어진 응답을 rejected에 추가)"
  [C]="B + 시각 향 (시각 연결부 merger·DeepStack LoRA, DeepStack 게이트, V·I 유형 이미지 조건 선호 손실)")
PAIRS=pairs_B; [[ $X == A* ]] && PAIRS=pairs_A

cd "$REPO" && git pull -q --rebase origin main
rm -rf "$DST"; mkdir -p "$DST/code" "$DST/data_report" "$DST/train"
cp $ROOT/scripts/{dpo_sample.py,dpo_grade.py,dpo_pairs2.py,dpo_measure.py,train_dpo.py,train_sft.py,common.py,summarize_eval.py,eval_extra.py,run_dpo2.sh,publish_dpo2.sh} "$DST/code/"
cp $RUN/pipeline.log $RUN/hparams.json $RUN/measure.json "$DST/" 2>/dev/null || true
cp $RUN/data/$PAIRS.report.json "$DST/data_report/" 2>/dev/null || true
cp $RUN/$X/train.log "$DST/train/"; cp $RUN/$X/train/train_info.json $RUN/$X/train/train_log.jsonl "$DST/train/" 2>/dev/null || true
cp $RUN/$X/train/adapter/gates.json "$DST/train/" 2>/dev/null || true
cp -r "$RUN/$X/eval" "$DST/eval"; find "$DST/eval" -type f -size +95M -delete
cp "$RUN/$X/summary_vs_baseline.md" "$RUN/$X/extra_stats.json" "$DST/" 2>/dev/null || true
LINE=$(sed -n 3p "$RUN/$X/summary_vs_baseline.md" 2>/dev/null | sed 's/\*\*//g')
EXTRA=$(python3 -c "import json;e=json.load(open('$RUN/$X/extra_stats.json'));print(f\"답변 길이 중앙값 {e['output_tokens_median']}토큰, 출력 한도 초과 {e['output_limit_truncated']}개(베이스라인 81), 정상 추출 문항 정답률 {e['explicit_parser_acc']*100:.1f}%(베이스라인 71.0%)\")" 2>/dev/null || echo "")
HP=$(cat $RUN/hparams.json 2>/dev/null | tr -d '\n ')
cat > "$DST/README.md" <<EOF
# DPO2-$X: ${DESC[$X]}

베이스 모델(Qwen3-VL-4B-Instruct @ ebb281ec)에서 시작하는 DPO ablation 4개 중 하나 (A′ → A → B → C).
- 학습 문항: vic_era_2000 1,997문항. 학습·평가 프롬프트는 베이스라인과 동일 (ERA 지시는 B·C의 답안 샘플링 때만 사용)
- 공통: beta 0.1, 1 epoch, LoRA r16/alpha 32/dropout 0.05, grad accum 16, AdamW wd 0.01, clip 1.0, cosine + warmup 3%, BF16, seed 3407
- 학습률·SFT 항 가중치 λ: 실제 기울기 측정으로 정한 값을 4개 실험에 똑같이 사용 → $HP
- 쌍 통계: data_report/$PAIRS.report.json

## 결과
- ${LINE:-평가 전}
- $EXTRA

문항별 출력: eval/, 학습 로그: train/
EOF
git add -A "$DST"
if git diff --cached --quiet; then echo "변경 없음"; exit 0; fi
git commit -q -m "dpo2_$X: ${DESC[$X]} 결과

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push -q origin HEAD:main
echo "푸시 완료: $(git rev-parse --short HEAD)"
