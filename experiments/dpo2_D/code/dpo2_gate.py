"""B·C 쌍 신호 시험(각 30스텝) 결과로, 효과 없는 구성 요소를 빼고 나머지 실험을 정한다.

각 쌍의 지표는 그 쌍으로 업데이트하기 전에 계산되므로 = 처음 보는 쌍에 대한 성능.
스텝 1-10은 모델이 아직 거의 안 움직여서 빼고, 스텝 11-30만 본다.
신호 있음 = 해당 유형 쌍이 40개 이상이고 정답 선호 비율 >= 0.58 (0.5에서 표준오차 약 2배).
쌍이 40개 미만이면 판단 불가 → 원래 계획대로 유지.

- (chosen 출처 × rejected 출처) 조합별 판정. 신호 없는 조합은 B·C 쌍에서 제외
- 베이스 정답 > 베이스 오답 조합(= A′·A의 쌍) 신호 없음 → A′·A 생략
- 모든 조합 신호 없음 → 중단
- 이미지 조건 선호(C 시험의 zi) 신호 없음 → C에서 이미지 조건 손실만 빼고 시각 LoRA·게이트는 유지
"""
import argparse, json
from collections import Counter
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--b-log", required=True)
ap.add_argument("--c-log", required=True)
ap.add_argument("--pairs", required=True)
ap.add_argument("--pairs-out", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--from-step", type=int, default=11)
ap.add_argument("--min-n", type=int, default=40)
ap.add_argument("--thr", type=float, default=0.58)
a = ap.parse_args()


def load(p):
    return [r for r in map(json.loads, open(p)) if r["step"] >= a.from_step]


def judge(zs):
    n = len(zs)
    acc = sum(z > 0 for z in zs) / n if n else None
    verdict = "undetermined" if n < a.min_n else ("signal" if acc >= a.thr else "no_signal")
    return {"n": n, "acc": None if acc is None else round(acc, 3),
            "mean_z": None if not n else round(sum(zs) / n, 4), "verdict": verdict}


B, C = load(a.b_log), load(a.c_log)
res = {"rule": f"steps>={a.from_step}, n>={a.min_n}, acc>={a.thr}",
       "B_test": {src: judge([r["z"] for r in B if r["rejected_src"] == src]) for src in ("wrong", "truncated")},
       "B_test_by_chosen": {src: judge([r["z"] for r in B if r["chosen_src"] == src]) for src in ("base", "era")},
       "B_test_all": judge([r["z"] for r in B]),
       "C_test_img": judge([r["zi"] for r in C if r["zi"] is not None]),
       "C_test_all": judge([r["z"] for r in C])}

# 판정은 (chosen 출처, rejected 출처) 조합별로 한다. rejected 출처만으로 묶으면 ERA chosen 쌍이 섞여
# "베이스 정답 vs 베이스 오답"(= A′·A 쌍)에 신호가 없어도 있는 것처럼 보인다 (11/14 첫 판정에서 실제로 발생).
COMBOS = [(c, j) for c in ("base", "era") for j in ("wrong", "truncated")]
res["B_test_combo"] = {f"{c}>{j}": judge([r["z"] for r in B if r["chosen_src"] == c and r["rejected_src"] == j])
                       for c, j in COMBOS}
keep = [k for k, v in res["B_test_combo"].items() if v["verdict"] != "no_signal"]
res["keep_combos"] = keep
res["keep_rejected_src"] = sorted({k.split(">")[1] for k in keep})
res["run_A"] = res["B_test_combo"]["base>wrong"]["verdict"] != "no_signal"
res["img_pref"] = res["C_test_img"]["verdict"] != "no_signal"
res["stop"] = not keep

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
res["final_pairs"] = rep
json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False))
