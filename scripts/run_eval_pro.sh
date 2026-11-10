#!/usr/bin/env bash
# 사용법: ./run_eval_pro.sh <실행이름> <데이터셋 revision>
# 결과는 ../results/<실행이름>/ 에 저장. 같은 이름이 있으면 중단 (이어서 하려면 RESUME=1)
set -euo pipefail
cd "$(dirname "$0")"

NAME="${1:?실행이름 필요 (예: pro_run1)}"
REV="${2:?데이터셋 revision 필요}"
OUT="../results/$NAME"

if [ -e "$OUT/predictions.jsonl" ] && [ "${RESUME:-0}" != "1" ]; then
  echo "이미 결과가 있습니다: $OUT  (다른 이름을 쓰거나, 이어서 하려면 RESUME=1)"; exit 1
fi
mkdir -p "$OUT"

cp config.py "$OUT/config_snapshot.py"
{ echo "MMDL_SEED=${MMDL_SEED:-3407}"; echo "PRO_DATASET_REVISION=$REV"; } > "$OUT/env.txt"
python -m pip freeze > "$OUT/requirements.freeze.txt"
nvidia-smi > "$OUT/gpu.txt" 2>&1 || true

python infer_pro.py --revision "$REV" --out "$OUT/predictions.jsonl" 2>&1 | tee -a "$OUT/infer.log"
python score.py  --pred "$OUT/predictions.jsonl" --out "$OUT/results.json" 2>&1 | tee "$OUT/score.log"
python export_csv.py --pred "$OUT/predictions.jsonl" --res "$OUT/results.json" --out "$OUT/mmmu_answers.csv"
echo "완료 -> $OUT"
