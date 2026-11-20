"""신호 시험(B·C 각 30스텝)을 생략하고, 원래 판정 결과(gate_signal_test.json)를 그대로 써서
pairs_B → pairs_B_final 을 만든다.

dpo2_gate.py에서 pairs_B_final은 pairs_B를 keep_combos로 순서를 유지한 채 거르기만 한다
(신호 시험 로그는 keep_combos·run_A·img_pref·stop을 정하는 데만 쓰인다). 그래서 keep_combos와
img_pref를 원래 판정값으로 고정하면, 같은 pairs_B에서는 dpo2_gate.py와 같은 결과가 나온다.
아래 필터·리포트 코드는 dpo2_gate.py 57-72행과 같다.
"""
import argparse, json
from collections import Counter
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--orig-gate", required=True, help="experiments/dpo2_E/gate_signal_test.json")
ap.add_argument("--pairs", required=True)
ap.add_argument("--pairs-out", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

res = json.load(open(a.orig_gate))
keep = res["keep_combos"]
assert keep == ["base>truncated", "era>wrong", "era>truncated"] and res["img_pref"] is True, res

rows = [json.loads(l) for l in open(a.pairs)]
kept = [r for r in rows if f"{r['chosen_src']}>{r['rejected_src']}" in keep]
with open(a.pairs_out, "w", encoding="utf-8") as f:
    for r in kept:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
n = max(1, len(kept))
rep = {"source": Path(a.pairs).name, "kept_combos": keep, "pairs": len(kept),
       "stats": dict(Counter(f"chosen_{r['chosen_src']}" for r in kept) + Counter(f"rejected_{r['rejected_src']}" for r in kept)),
       "by_vic_type": dict(Counter(r["vic_type"] for r in kept)),
       "mean_chosen_tokens": round(sum(r["chosen_tokens"] for r in kept) / n),
       "mean_rejected_tokens": round(sum(r["rejected_tokens"] for r in kept) / n)}
json.dump(rep, open(a.pairs_out.replace(".jsonl", ".report.json"), "w"), ensure_ascii=False, indent=1)
res["final_pairs_original"] = res.get("final_pairs")
res["final_pairs"] = rep
res["note"] = "재현: 신호 시험 생략, keep_combos·img_pref·run_A·stop은 원래 gate_signal_test.json 값 고정"
json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
print(json.dumps(rep, ensure_ascii=False))
