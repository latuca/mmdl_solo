"""베이스 모델 시작 DPO용 쌍 생성 (on-policy).

mode=self : chosen = 베이스 정답(정상 종료), rejected = 베이스 오답(정상 종료)          → 실험 A′, A
mode=era  : chosen = ERA 지시로 생성한 베이스 정답(ERA 형식 지킨 것) 과 베이스 정답을 번갈아,
            rejected = 베이스가 길이 한도까지 이어간 응답(끝맺지 못함) 과 베이스 오답을 번갈아  → 실험 B, C
- 학습 입력은 평가와 같은 프롬프트(input_prompt)다. ERA 지시는 샘플링 때만 쓰였다.
- 판정 불가(label=None) 응답은 쓰지 않는다. 문항당 최대 --pairs-per-q 쌍, 같은 응답은 한 문항에서 한 번만 쓴다.
- 길이 한도로 끊긴 rejected는 rejected_truncated=true (학습 때 <|im_end|>를 붙이지 않음).
"""
import argparse, json, random, re
from collections import Counter, defaultdict

ERA_RE = [re.compile(rf"^\W*{k}\W*:", re.I | re.M) for k in ("Evidence", "Rule", "Application")]

ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True)
ap.add_argument("--samples", nargs="+", required=True)
ap.add_argument("--labels", nargs="+", required=True)
ap.add_argument("--mode", choices=["self", "era"], required=True)
ap.add_argument("--pairs-per-q", type=int, default=2)
ap.add_argument("--seed", type=int, default=3407)
ap.add_argument("--out", required=True)
a = ap.parse_args()

rng = random.Random(a.seed)
train = {r["id"]: r for r in map(json.loads, open(a.train, encoding="utf-8"))}
text = {}
for p in a.samples:
    for q in map(json.loads, open(p, encoding="utf-8")):
        for k, s in enumerate(q["samples"]):
            text[(q["id"], q["tag"], k)] = s["raw"]
by_q = defaultdict(list)
for p in a.labels:
    for r in map(json.loads, open(p, encoding="utf-8")):
        if (r["id"], r["tag"], r["k"]) in text:
            by_q[r["id"]].append(r)


def is_era(r):
    t = text[(r["id"], r["tag"], r["k"])]
    return all(x.search(t) for x in ERA_RE)


pairs, stat = [], Counter()
for qid in sorted(by_q):
    rs = by_q[qid]
    base_ok = [r for r in rs if r["tag"] == "base" and r["label"] is True and r["finish_reason"] == "stop"]
    base_bad = [r for r in rs if r["tag"] == "base" and r["label"] is False and r["finish_reason"] == "stop"]
    if a.mode == "self":
        chosen_pools = [("base", base_ok)]
        rej_pools = [("wrong", base_bad)]
    else:
        era_ok = [r for r in rs if r["tag"] == "base_era" and r["label"] is True and r["finish_reason"] == "stop"
                  and is_era(r)]
        trunc = [r for r in rs if r["tag"] == "base" and r["finish_reason"] == "length"]
        chosen_pools = [("era", era_ok), ("base", base_ok)]
        rej_pools = [("truncated", trunc), ("wrong", base_bad)]
    for _, v in chosen_pools + rej_pools:
        rng.shuffle(v)
    made = 0
    for i in range(a.pairs_per_q):
        c_order = chosen_pools[i % len(chosen_pools):] + chosen_pools[:i % len(chosen_pools)]
        r_order = rej_pools[i % len(rej_pools):] + rej_pools[:i % len(rej_pools)]
        cs = next(((n, v) for n, v in c_order if v), None)
        rj = next(((n, v) for n, v in r_order if v), None)
        if not cs or not rj:
            break
        ch, rr = cs[1].pop(), rj[1].pop()
        t = train[qid]
        pairs.append({"id": qid, "vic_type": t["vic_type"], "image": t["image"], "input_prompt": t["input_prompt"],
                      "chosen": text[(qid, ch["tag"], ch["k"])], "rejected": text[(qid, rr["tag"], rr["k"])],
                      "rejected_truncated": rr["finish_reason"] == "length",
                      "chosen_src": cs[0], "rejected_src": rj[0],
                      "chosen_tokens": ch["output_tokens"], "rejected_tokens": rr["output_tokens"]})
        stat[f"chosen_{cs[0]}"] += 1; stat[f"rejected_{rj[0]}"] += 1; made += 1
    stat["questions_with_pairs"] += made > 0
rng.shuffle(pairs)
with open(a.out, "w", encoding="utf-8") as f:
    for p in pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
n = max(1, len(pairs))
rep = {"mode": a.mode, "pairs": len(pairs), "stats": dict(stat),
       "by_vic_type": dict(Counter(p["vic_type"] for p in pairs)),
       "mean_chosen_tokens": round(sum(p["chosen_tokens"] for p in pairs) / n),
       "mean_rejected_tokens": round(sum(p["rejected_tokens"] for p in pairs) / n)}
json.dump(rep, open(a.out.replace(".jsonl", ".report.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(rep, ensure_ascii=False))
