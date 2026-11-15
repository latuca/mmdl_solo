"""실험 D용 학습 데이터: C의 쌍을 바탕으로 두 가지를 바꾼다.

1) 루프 쌍 (친구 방식): rejected = 같은 문장/줄을 5번 이상 되풀이하다 길이 한도에서 끊긴 베이스 응답.
   루프 시작 위치(rej_span_char = 반복 단위의 두 번째 등장 위치)부터 512토큰만 벌한다(train_dpo_d.py).
   반복이 보이지 않는 끊긴 응답(끝없이 계산·재확인)은 정상 풀이일 수도 있어서 쓰지 않는다.
   chosen = 같은 문항의 정답(ERA 형식 정답과 베이스 정답을 번갈아). 문항당 최대 2쌍.
2) ERA 정답 > 베이스 오답 쌍: 신호 시험에서 신호가 있던 조합이라 pairs_B_final에서 그대로 가져온다.
   (베이스 정답 > 베이스 오답 조합은 신호가 없어 B·C처럼 쓰지 않는다.)
3) 보존 데이터(replay): 베이스가 8번 다 맞힌 문항에서 자기 정답 1개씩 --n-replay개.
   ERA 형식 정답이 있으면 그것을, 없으면 베이스 정답을 쓴다 (B·C가 가르치는 형식과 맞추기 위해).
   이 행은 쌍이 아니라 NLL만 학습한다 = "원래 잘 풀던 문제는 그대로 풀어라"는 닻.
"""
import argparse, json, random, re
from collections import Counter, defaultdict

ERA_RE = [re.compile(rf"^\W*{k}\W*:", re.I | re.M) for k in ("Evidence", "Rule", "Application")]

ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True)
ap.add_argument("--samples", nargs="+", required=True)
ap.add_argument("--labels", nargs="+", required=True)
ap.add_argument("--b-final", required=True, help="pairs_B_final.jsonl (era>wrong 쌍을 가져온다)")
ap.add_argument("--loop-pairs-per-q", type=int, default=2)
ap.add_argument("--span-tokens", type=int, default=512)
ap.add_argument("--min-repeat", type=int, default=5)
ap.add_argument("--n-replay", type=int, default=300)
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
T = lambda r: text[(r["id"], r["tag"], r["k"])]


def loop_start(t):
    """같은 줄(15자 이상) 또는 문장(20자 이상)이 min_repeat번 이상 나오면 루프로 보고,
    그중 가장 먼저 나온 단위의 두 번째 등장 위치(문자)를 루프 시작으로 돌려준다. 루프가 아니면 None."""
    lines = Counter(l.strip() for l in t.split("\n") if len(l.strip()) >= 15)
    sents = Counter(x.strip() for x in t.replace("\n", " ").split(". ") if len(x.strip()) >= 20)
    rep = [u for u, c in list(lines.items()) + list(sents.items()) if c >= a.min_repeat]
    if not rep:
        return None
    u = min(rep, key=t.find)
    return t.find(u, t.find(u) + 1)


def is_era(r):
    return all(x.search(T(r)) for x in ERA_RE)


rows, stat = [], Counter()
for qid in sorted(by_q):
    rs, t = by_q[qid], train[qid]
    base = [r for r in rs if r["tag"] == "base"]
    base_ok = [r for r in base if r["label"] is True and r["finish_reason"] == "stop"]
    era_ok = [r for r in rs if r["tag"] == "base_era" and r["label"] is True and r["finish_reason"] == "stop" and is_era(r)]
    loops = []
    for r in base:
        if r["finish_reason"] == "length":
            p = loop_start(T(r))
            if p is not None:
                loops.append((r, p))
            else:
                stat["truncated_no_repeat_skipped"] += 1
    stat["loop_responses"] += len(loops)
    common = {"id": qid, "vic_type": t["vic_type"], "image": t["image"], "input_prompt": t["input_prompt"]}
    # 1) 루프 쌍
    pools = [("era", era_ok[:]), ("base", base_ok[:])]
    for _, v in pools:
        rng.shuffle(v)
    rng.shuffle(loops)
    for i in range(min(a.loop_pairs_per_q, len(loops))):
        order = pools[i % 2:] + pools[:i % 2]
        cs = next(((n, v) for n, v in order if v), None)
        if not cs:
            break
        ch = cs[1].pop(); rr, p = loops[i]
        rows.append({**common, "chosen": T(ch), "rejected": T(rr), "rejected_truncated": True,
                     "rej_span_char": p, "rej_span_tokens": a.span_tokens,
                     "chosen_src": cs[0], "rejected_src": "loop",
                     "chosen_tokens": ch["output_tokens"], "rejected_tokens": rr["output_tokens"]})
        stat[f"loop_pair_chosen_{cs[0]}"] += 1
    # 3) 보존 후보
    if len(base) == 8 and all(r["label"] is True and r["finish_reason"] == "stop" for r in base):
        stat["all_correct_questions"] += 1

# 2) ERA 정답 > 베이스 오답 (신호 있던 조합)
ew = [r for r in map(json.loads, open(a.b_final, encoding="utf-8")) if r["chosen_src"] == "era" and r["rejected_src"] == "wrong"]
rows += ew
stat["era_wrong_pairs"] = len(ew)

# 3) 보존 데이터
cands = []
for qid in sorted(by_q):
    rs = by_q[qid]; base = [r for r in rs if r["tag"] == "base"]
    if len(base) == 8 and all(r["label"] is True and r["finish_reason"] == "stop" for r in base):
        era_ok = [r for r in rs if r["tag"] == "base_era" and r["label"] is True and r["finish_reason"] == "stop" and is_era(r)]
        ch, src = (rng.choice(era_ok), "era") if era_ok else (rng.choice(base), "base")
        t = train[qid]
        cands.append({"id": qid, "vic_type": t["vic_type"], "image": t["image"], "input_prompt": t["input_prompt"],
                      "kind": "replay", "chosen": T(ch), "rejected": None, "chosen_src": src, "rejected_src": None,
                      "chosen_tokens": ch["output_tokens"], "rejected_tokens": 0})
rng.shuffle(cands)
rep = cands[:a.n_replay]
rows += rep
stat["replay"] = len(rep); stat["replay_era"] = sum(r["chosen_src"] == "era" for r in rep)

rng.shuffle(rows)
with open(a.out, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
pairs = [r for r in rows if r.get("kind") != "replay"]
n = max(1, len(pairs))
report = {"rows": len(rows), "pairs": len(pairs), "stats": dict(stat),
          "by_vic_type": dict(Counter(r["vic_type"] for r in pairs)),
          "mean_chosen_tokens": round(sum(r["chosen_tokens"] for r in pairs) / n),
          "mean_rejected_tokens": round(sum(r["rejected_tokens"] for r in pairs) / n)}
json.dump(report, open(a.out.replace(".jsonl", ".report.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(report, ensure_ascii=False))
