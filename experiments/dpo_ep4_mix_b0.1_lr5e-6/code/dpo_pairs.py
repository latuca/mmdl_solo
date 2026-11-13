"""채점된 샘플로 DPO 쌍을 만든다.

- rejected: 시작 모델(ep4)의 오답 응답 (정상 종료한 것만, on-policy)
- chosen  : 정답 응답. 출처는 ep4 + 베이스 모델(긴 추론)을 섞는다.
- 문항당 최대 --pairs-per-q 쌍. 첫 쌍은 베이스 정답을 우선, 둘째 쌍은 ep4 정답을 우선 사용해 출처를 섞는다.
  같은 응답은 한 문항 안에서 두 번 쓰지 않는다.
- 출력 길이 초과(finish_reason=length)·판정 불가(label=None) 응답은 쓰지 않는다.
"""
import argparse, json, random
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True, help="SFT train.jsonl (id, image, input_prompt)")
ap.add_argument("--samples", nargs="+", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--policy-tag", default="ep4")
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
for r in map(json.loads, open(a.labels, encoding="utf-8")):
    if r["label"] is not None and r["finish_reason"] == "stop":
        by_q[r["id"]].append(r)

pairs, stat = [], Counter()
for qid in sorted(by_q):
    rs = by_q[qid]
    rej = [r for r in rs if r["tag"] == a.policy_tag and r["label"] is False]
    good = {t: [r for r in rs if r["tag"] == t and r["label"] is True] for t in {r["tag"] for r in rs}}
    for v in list(good.values()) + [rej]:
        rng.shuffle(v)
    if not rej:
        stat["no_rejected"] += 1; continue
    if not any(good.values()):
        stat["no_chosen"] += 1; continue
    order = [t for t in good if t != a.policy_tag] + [a.policy_tag]   # 베이스 우선, 다음 ep4
    made = 0
    for i in range(a.pairs_per_q):
        if not rej:
            break
        pref = order[i % len(order):] + order[:i % len(order)]
        src = next((t for t in pref if good.get(t)), None)
        if src is None:
            break
        ch, rj = good[src].pop(), rej.pop()
        t = train[qid]
        pairs.append({"id": qid, "vic_type": ch["vic_type"], "image": t["image"], "input_prompt": t["input_prompt"],
                      "chosen": text[(qid, ch["tag"], ch["k"])], "rejected": text[(qid, rj["tag"], rj["k"])],
                      "chosen_src": src, "chosen_tokens": ch["output_tokens"], "rejected_tokens": rj["output_tokens"]})
        stat[f"chosen_from_{src}"] += 1; made += 1
    stat["questions_with_pairs"] += made > 0
rng.shuffle(pairs)
with open(a.out, "w", encoding="utf-8") as f:
    for p in pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
rep = {"pairs": len(pairs), "stats": dict(stat), "by_vic_type": dict(Counter(p["vic_type"] for p in pairs)),
       "mean_chosen_tokens": round(sum(p["chosen_tokens"] for p in pairs) / max(1, len(pairs))),
       "mean_rejected_tokens": round(sum(p["rejected_tokens"] for p in pairs) / max(1, len(pairs)))}
json.dump(rep, open(a.out.replace(".jsonl", ".report.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(rep, ensure_ascii=False))
