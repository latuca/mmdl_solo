#!/usr/bin/env bash
# 사용법: ./run_eval.sh <실행이름>      예) ./run_eval.sh run2
# 결과는 ../results/<실행이름>/ 에 저장된다. 같은 이름의 결과가 있으면 중단한다.
# (중단된 실행을 이어서 하려면 RESUME=1 ./run_eval.sh <같은이름>)
set -euo pipefail
cd "$(dirname "$0")"

NAME="${1:-run_$(date +%Y%m%d_%H%M%S)}"
OUT="../results/$NAME"

if [ -e "$OUT/predictions.jsonl" ] && [ "${RESUME:-0}" != "1" ]; then
  echo "이미 결과가 있습니다: $OUT  (다른 이름을 쓰거나, 이어서 하려면 RESUME=1)"; exit 1
fi
mkdir -p "$OUT"

# 이 실행에 쓴 설정을 함께 보관
cp config.py "$OUT/config_snapshot.py"
echo "MMDL_SEED=${MMDL_SEED:-3407}" > "$OUT/env.txt"

python infer.py  --out "$OUT/predictions.jsonl" 2>&1 | tee -a "$OUT/infer.log"
python score.py  --pred "$OUT/predictions.jsonl" --out "$OUT/results.json" 2>&1 | tee "$OUT/score.log"
python export_csv.py --pred "$OUT/predictions.jsonl" --res "$OUT/results.json" --out "$OUT/mmmu_answers.csv"

echo "완료 -> $OUT"
