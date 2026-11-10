"""채점: 길이 한도 → 명시 Final answer 파서 → 로컬 Qwen3-8B 저지 순서.
추론이 끝나고 GPU가 비어 있을 때 실행한다 (저지 모델을 따로 올리므로).

실행: python score.py --pred results/predictions.jsonl --out results/results.json
"""
import argparse
import collections
import json
import os
import re
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from config import JUDGE_REPO, JUDGE_REV, MODEL_REPO, MODEL_REV, MAX_NEW_TOKENS, SEED

FINAL_RE = re.compile(r"(?im)^\W*final answer\W*:\s*(.+?)\s*$")

JUDGE_TEMPLATE = (
    "You are an AI assistant who will help me to match an answer with several options of a single-choice question. "
    "You are provided with a question, several options, and an answer, and you need to find which option is most "
    "similar to the answer. If the meaning of all options are significantly different from the answer, output Z. "
    "Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z. \n"
    "Example 1: \nQuestion: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n"
    "Answer: a cute teddy bear\nYour output: A\n"
    "Example 2: \nQuestion: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n"
    "Answer: Spider\nYour output: Z\n"
    "Example 3: \nQuestion: {}?\nOptions: {}\nAnswer: {}\nYour output: "
)


def processed(raw):
    # 마지막 </think> 이후만 사용 (Qwen3 계열 대비; Instruct 모델이면 보통 그대로 전체)
    return raw.rsplit("</think>", 1)[-1]


def clean(s):
    return re.sub(r"[*_`#>\s]+", " ", s).strip(" .:;")


def explicit_answer(text):
    """Final answer 줄을 모두 찾아 한 값으로 모이면 그 값, 여러 값이면 None(모호 → 저지)."""
    hits = [clean(m) for m in FINAL_RE.findall(text)]
    hits = [h for h in hits if h]
    if not hits:
        return None
    if len(set(hits)) == 1:
        return hits[0]
    return None


def mc_letter(value):
    m = re.match(r"^\(?([A-L])\)?(?:[\s.):]|$)", value)
    return m.group(1) if m else None


def parse_judge(out):
    m = re.search(r"Your output:\s*([A-LZ])", out)
    if m:
        return m.group(1)
    letters = re.findall(r"\b([A-LZ])\b", out.strip())
    return letters[-1] if letters else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="results/predictions.jsonl")
    ap.add_argument("--out", default="results/results.json")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.pred, encoding="utf-8") if l.strip()]
    t_score0 = time.time()
    judge_seconds = 0.0
    graded, need_judge = [], []

    for r in recs:
        g = {"id": r["id"], "subject": r["subject"], "question_type": r["question_type"],
             "reference_answer": r["reference_answer"], "finish_reason": r["finish_reason"]}
        if r["finish_reason"] == "length":
            g.update(method="output_token_limit", correct=False)
            graded.append(g)
            continue
        text = processed(r["raw"])
        val = explicit_answer(text)
        if val is not None and r["question_type"] == "multiple-choice":
            letter = mc_letter(val)
            if letter is not None:
                g.update(method="explicit_final_parser", extracted=letter,
                         correct=(letter == str(r["reference_answer"]).strip().upper()[:1]))
                graded.append(g)
                continue
        # 명시 필드가 없거나 모호 → 저지 (MC는 옵션 매칭, open은 A=참조답/B=Other)
        need_judge.append((g, r, val if val is not None else text))

    if need_judge:
        print(f"judge items: {len(need_judge)}", flush=True)
        t_j0 = time.time()
        tok = AutoTokenizer.from_pretrained(JUDGE_REPO, revision=JUDGE_REV)
        llm = LLM(model=JUDGE_REPO, revision=JUDGE_REV, dtype="bfloat16", max_model_len=32768,
                  gpu_memory_utilization=0.9, max_num_seqs=1, max_num_batched_tokens=2048,
                  enforce_eager=True, seed=SEED)
        sp = SamplingParams(temperature=0, top_p=1, top_k=-1, repetition_penalty=1,
                            presence_penalty=0, max_tokens=4096, seed=SEED)
        for g, r, answer in need_judge:
            if r["question_type"] == "multiple-choice":
                opts = " ".join(f"{k}. {v}" for k, v in r["options"].items())
                prompt_body = JUDGE_TEMPLATE.format(r["question"], opts, answer)
                msgs = [{"role": "user", "content": prompt_body}]
                text_in = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)
                out = llm.generate([text_in], sp)[0].outputs[0].text
                dec = parse_judge(out)
                if dec is None or dec == "Z":
                    g.update(method="local_llm_judge", judge_output=out, judge_decision=dec,
                             correct=False, extracted=None)
                else:
                    g.update(method="local_llm_judge", judge_output=out, judge_decision=dec,
                             extracted=dec, correct=(dec == str(r["reference_answer"]).strip().upper()[:1]))
            else:
                opts = "A. {}\nB. Other Answers".format(r["reference_answer"])
                prompt_body = JUDGE_TEMPLATE.format(r["question"], opts, answer)
                text_in = tok.apply_chat_template([{"role": "user", "content": prompt_body}], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)
                out = llm.generate([text_in], sp)[0].outputs[0].text
                dec = parse_judge(out)
                g.update(method="local_llm_judge", judge_output=out, judge_decision=dec,
                         correct=(dec == "A"))
            graded.append(g)

    if need_judge:
        judge_seconds = time.time() - t_j0   # 저지 모델 로딩 + 판정 시간
    total = len(graded)
    correct = sum(1 for g in graded if g["correct"])
    methods = collections.Counter(g["method"] for g in graded)
    by_subj = collections.defaultdict(lambda: {"total": 0, "correct": 0})
    for g in graded:
        by_subj[g["subject"]]["total"] += 1
        by_subj[g["subject"]]["correct"] += int(g["correct"])
    for s in by_subj.values():
        s["accuracy"] = s["correct"] / s["total"] if s["total"] else 0.0

    summary = {"total": total, "correct": correct, "accuracy": correct / total if total else 0.0,
               "method_counts": dict(methods), "by_subject": dict(by_subj)}
    runs_path = args.pred.replace(".jsonl", ".runs.jsonl")
    runs = [json.loads(l) for l in open(runs_path, encoding="utf-8") if l.strip()] if os.path.exists(runs_path) else []
    elapsed = [r["elapsed_seconds"] for r in recs if "elapsed_seconds" in r]
    full = len(elapsed) == len(recs) and len(recs) > 0
    meta = {
        "generation_model": {"repo": MODEL_REPO, "revision": MODEL_REV, "dtype": "bfloat16",
                             "max_new_tokens": MAX_NEW_TOKENS, "temperature": 0.7, "top_p": 0.8,
                             "top_k": 20, "presence_penalty": 1.5, "seed": SEED},
        "judge_model": {"repo": JUDGE_REPO, "revision": JUDGE_REV, "temperature": 0,
                        "enable_thinking": False, "max_new_tokens": 4096},
        "judge_items": len(need_judge),
        "timing": {
            "generation_seconds_sum_of_items": round(sum(elapsed), 1) if full else None,
            "items_with_timing": len(elapsed),
            "inference_runs": runs,
            "inference_wall_seconds_total": round(sum(r["wall_seconds"] for r in runs), 1) if runs else None,
            "judge_seconds": round(judge_seconds, 1),
            "scoring_seconds": round(time.time() - t_score0, 1),
        },
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": summary, "records": graded}, f, ensure_ascii=False, indent=2)
    hms = lambda s: f"{int(s // 3600)}h {int(s % 3600 // 60)}m {int(s % 60)}s"
    t = meta["timing"]
    if t["generation_seconds_sum_of_items"] is not None:
        print("generation (sum of items):", hms(t["generation_seconds_sum_of_items"]))
    if t["inference_wall_seconds_total"] is not None:
        print("inference wall total      :", hms(t["inference_wall_seconds_total"]))
    print("judge                     :", hms(t["judge_seconds"]), f"({len(need_judge)} items)")
    print("scoring total             :", hms(t["scoring_seconds"]))
    print(json.dumps({k: summary[k] for k in ("total", "correct", "accuracy", "method_counts")},
                     ensure_ascii=False, indent=2))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
