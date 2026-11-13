#!/usr/bin/env bash
# DPO 실험 코드·결과를 github.com/latuca/mmdl_solo 의 experiments/dpo_ep4_mix_b0.1_lr5e-6/ 에 올린다.
# 가중치·샘플 원문·쌍 데이터는 올리지 않는다 (리포트·로그·평가 결과만).
set -euo pipefail
ROOT=/home/latuca/MMDL_solo
RUN=$ROOT/runs/dpo_ep4_mix_b0.1_lr5e-6
REPO=$ROOT/_publish
DST=$REPO/experiments/dpo_ep4_mix_b0.1_lr5e-6

cd "$REPO" && git pull -q --rebase origin main
rm -rf "$DST"; mkdir -p "$DST/code" "$DST/data_report" "$DST/train"
cp $ROOT/scripts/{dpo_sample.py,dpo_grade.py,dpo_pairs.py,train_dpo.py,train_sft.py,common.py,summarize_eval.py,run_dpo.sh,publish_dpo.sh} "$DST/code/"
cp $RUN/pipeline.log "$DST/"
cp $RUN/data/pairs.report.json "$DST/data_report/" 2>/dev/null || true
for f in grade.log pairs.log; do tail -50 $RUN/$f > "$DST/data_report/$f" 2>/dev/null || true; done
python3 - "$RUN/data/labels.jsonl" > "$DST/data_report/label_stats.json" <<'EOF' || true
import json, sys
from collections import Counter
c = Counter()
for r in map(json.loads, open(sys.argv[1])):
    c[f'{r["tag"]}|{r["vic_type"]}|{r["label"]}'] += 1
print(json.dumps(dict(sorted(c.items())), ensure_ascii=False, indent=1))
EOF
cp $RUN/train.log "$DST/train/"; cp $RUN/train/train_info.json $RUN/train/train_log.jsonl "$DST/train/" 2>/dev/null || true
if [ -d "$RUN/eval" ]; then
  cp -r "$RUN/eval" "$DST/eval"; find "$DST/eval" -type f -size +95M -delete
  cp "$RUN/summary_vs_baseline.md" "$DST/"
fi
LINE=$(sed -n 3p "$RUN/summary_vs_baseline.md" 2>/dev/null | sed 's/\*\*//g')
cat > "$DST/README.md" <<EOF
# DPO: SFT ep4에서 시작 (선호 답 출처 섞기)

- 시작 모델: SFT(일반 풀이, vic_era_2000) ep4 병합 모델 (MMMU-val 510/900 = 56.67%) → models/sft_general_vic2000_e5/
- 학습 문항: vic_era_2000 1,997문항 (SFT와 같은 프롬프트·이미지·해상도)
- 샘플링: ep4·베이스 모델 각 문항당 8개, 평가와 같은 샘플링 설정, 프롬프트+응답 최대 8192토큰
- 채점: 평가와 같은 파서 + 로컬 Qwen3-8B 저지 (판정 불가 응답은 제외)
- 쌍: rejected = ep4 오답, chosen = 베이스·ep4 정답 섞기 (문항당 최대 2쌍, 첫 쌍은 베이스 정답 우선)
- DPO: beta 0.1, lr 5e-6, 1 epoch, LoRA r16/alpha 32/dropout 0.05, grad accum 16, AdamW wd 0.01, clip 1.0,
  cosine + warmup 3%, BF16, seed 3407. reference = 시작 모델(LoRA 끈 상태)

## 결과
- ${LINE:-평가 전}
- 비교: 베이스라인 576/900 = 64.00%, 시작점 SFT ep4 510/900 = 56.67%

쌍 통계: data_report/pairs.report.json, 학습 로그: train/, 문항별 출력: eval/
EOF
git add -A "$DST"
if git diff --cached --quiet; then echo "변경 없음"; exit 0; fi
git commit -q -m "dpo_ep4_mix_b0.1_lr5e-6: 결과

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push -q origin HEAD:main
echo "푸시 완료: $(git rev-parse --short HEAD)"
