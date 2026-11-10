"""5단계 비교: held-out 20문항 생성 결과를 모델(base/G/E) x 엔진(hf/vllm)별로 요약한다.

- ERA 형식: Evidence:/Rule:/Application:/Final answer: 네 줄이 순서대로 모두 있는가
- 최종 답: 평가용 파서(scripts/score.py의 explicit_answer, mc_letter)로 추출, 정답은 original_answer
  (20문항 sanity 용도이며 성능 지표로 해석하지 않는다)
- HF vs vLLM: 같은 모델의 최종 답 일치율, 형식 준수율, 길이
"""
import json
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from score import explicit_answer, mc_letter  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = sys.argv[1] if len(sys.argv) > 1 else "era_lora_pilot"
D = os.path.join(ROOT, "outputs", EXP, "stage5")
ERA_RE = re.compile(r"(?s)\A\s*Evidence:.*?\nRule:.*?\nApplication:.*?\nFinal answer:\s*\S.*\Z")

gold = {}
for l in open(os.path.join(ROOT, "data", "era_dataset_v1.1_partial_187.jsonl")):
    r = json.loads(l)
    gold[r["sample_id"]] = str(r["original_answer"]).strip()


def load(model, engine):
    p = os.path.join(D, f"{model}_{engine}.jsonl")
    return {r["sample_id"]: r for r in map(json.loads, open(p))} if os.path.exists(p) else None


def final(raw):
    a = explicit_answer(raw)
    return (mc_letter(a) or a) if a else None


summary, answers = {}, {}
for model in ("base", "G", "E"):
    for engine in ("hf", "vllm"):
        rows = load(model, engine)
        if rows is None:
            continue
        v = list(rows.values())
        toks = [r["n_tokens"] for r in v]
        fa = {k: final(r["raw"]) for k, r in rows.items()}
        answers[(model, engine)] = fa
        summary[f"{model}/{engine}"] = {
            "n": len(v),
            "era_format": sum(bool(ERA_RE.match(r["raw"])) for r in v),
            "has_final_answer": sum(a is not None for a in fa.values()),
            "correct": sum(a is not None and a == gold[k] for k, a in fa.items()),
            "truncated(length)": sum(r["finish_reason"] == "length" for r in v),
            "tokens_mean": round(st.mean(toks), 1), "tokens_median": st.median(toks), "tokens_max": max(toks),
        }
print(f"{'model/engine':14s} " + " ".join(f"{k:>17s}" for k in next(iter(summary.values()))))
for k, s in summary.items():
    print(f"{k:14s} " + " ".join(f"{str(x):>17s}" for x in s.values()))

print("\nHF vs vLLM (같은 모델, 같은 20문항)")
for model in ("base", "G", "E"):
    if (model, "hf") in answers and (model, "vllm") in answers:
        a, b = answers[(model, "hf")], answers[(model, "vllm")]
        same = sum(a[k] == b[k] for k in a)
        both = sum(a[k] == b[k] == gold[k] for k in a)
        print(f"  {model}: 최종 답 일치 {same}/{len(a)}, 둘 다 정답 {both}")

json.dump({"summary": summary}, open(os.path.join(D, "summary.json"), "w"), indent=1, ensure_ascii=False)

# 샘플 출력 (사람이 확인용)
for model in ("E", "G", "base"):
    rows = load(model, "vllm") or load(model, "hf")
    if rows:
        k = sorted(rows)[0]
        print(f"\n--- {model} 응답 예시 ({k}, gold={gold[k]}) ---\n{rows[k]['raw'][:700]}")
