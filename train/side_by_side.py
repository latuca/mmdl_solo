"""held-out 20문항에서 base/G/E의 실제 응답을 문항별로 나란히 정리한다 (vLLM 응답 전문 + HF 최종 답 요약)."""
import json, os, sys
sys.path.insert(0, "scripts")
from score import explicit_answer, mc_letter
exp = sys.argv[1] if len(sys.argv) > 1 else "era_lora_ep5"
D = f"outputs/{exp}/stage5"
recs = {json.loads(l)["sample_id"]: json.loads(l) for l in open("data/era_dataset_v1.1_partial_187.jsonl")}
load = lambda m, e: {r["sample_id"]: r for r in map(json.loads, open(f"{D}/{m}_{e}.jsonl"))}
V = {m: load(m, "vllm") for m in ("base", "G", "E")}
H = {m: load(m, "hf") for m in ("base", "G", "E")}
def fin(raw):
    a = explicit_answer(raw); return (mc_letter(a) or a) if a else None
mark = lambda a, g: "✓" if a == g else ("∅" if a is None else "✗")
ids = sorted(V["E"])
out = [f"# held-out 20문항 응답 비교 ({exp})\n", "표기: ✓ 정답, ✗ 오답, ∅ Final answer 추출 실패. 표의 값은 추출된 최종 답.\n",
       "| 문항 | 정답 | base(vLLM) | G(vLLM) | E(vLLM) | base(HF) | G(HF) | E(HF) |", "|---|---|---|---|---|---|---|---|"]
tot = {k: 0 for k in ["bv","gv","ev","bh","gh","eh"]}
for i in ids:
    g = str(recs[i]["original_answer"]).strip()
    cells = []
    for key, d, m in [("bv",V,"base"),("gv",V,"G"),("ev",V,"E"),("bh",H,"base"),("gh",H,"G"),("eh",H,"E")]:
        a = fin(d[m][i]["raw"]); tot[key] += (a == g); cells.append(f"{a} {mark(a, g)}")
    out.append(f"| {i} | {g} | " + " | ".join(cells) + " |")
out.append("| **정답 수** | | " + " | ".join(f"**{tot[k]}/20**" for k in ["bv","gv","ev","bh","gh","eh"]) + " |\n")
out.append("---\n## 문항별 응답 전문 (vLLM)\n")
for i in ids:
    r = recs[i]; g = str(r["original_answer"]).strip()
    out.append(f"### {i}  (owner {r['owner']}, 정답 {g})\n\n**질문**\n```\n{r['prompt'].split(chr(10)+'On the last line')[0]}\n```\n")
    for m in ("base", "G", "E"):
        x = V[m][i]; out.append(f"**{m}** — {x['n_tokens']}토큰, 최종 답 {fin(x['raw'])} {mark(fin(x['raw']), g)}\n```\n{x['raw'].strip()}\n```\n")
open(f"{D}/side_by_side.md", "w").write("\n".join(out))
print("\n".join(out[:out.index("---\n## 문항별 응답 전문 (vLLM)\n")]))
