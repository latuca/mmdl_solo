"""샘플 응답 채점 (평가 venv). 평가와 같은 규칙을 쓴다.

1) parsers.grade_response: "Final answer:" 파서 → 공식 fallback. 출력 길이 초과는 오답.
2) 1)에서 판정이 안 된 응답(pending_judge)은 평가와 같은 로컬 Qwen3-8B 저지 프롬프트로 판정한다.
   평가와 달리 여러 응답을 묶어서 돌리고, 재시도는 최대 3회. 끝까지 판정이 안 되면 label=None(쌍에서 제외).
출력: 응답마다 {"id","tag","k","label": true/false/null,"method",...}
"""
import argparse, json, os, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+", required=True)
    ap.add_argument("--source", required=True, help="vic_era_2000/data.json (question, options, answer)")
    ap.add_argument("--eval-src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-attempts", type=int, default=3)
    a = ap.parse_args()

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    sys.path.insert(0, a.eval_src)
    import parsers
    from evaluate import judge_prompt

    c = json.load(open(Path(a.eval_src) / "config.json"))
    src = {x["id"]: x for x in json.load(open(a.source, encoding="utf-8"))}


    def row_for(qid, raw, finish):
        s = src[qid]
        opts = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", s["options"])) if s["question_type"] == "multiple-choice" else {}
        return {"id": qid, "question_type": s["question_type"], "question": s["question"], "options": opts,
                "reference_answer": s["answer"], "model_response": {"raw": raw, "finish_reason": finish}}


    recs, pend = [], []
    for path in a.samples:
        for l in open(path, encoding="utf-8"):
            q = json.loads(l)
            for k, smp in enumerate(q["samples"]):
                row = row_for(q["id"], smp["raw"], smp["finish_reason"])
                g = parsers.grade_response(row)
                rec = {"id": q["id"], "tag": q["tag"], "k": k, "vic_type": src[q["id"]]["vic_type"],
                       "question_type": row["question_type"], "finish_reason": smp["finish_reason"],
                       "output_tokens": smp["output_tokens"], "method": g["method"],
                       "label": g["correct"] if g["status"] == "graded" else None}
                recs.append(rec)
                if g["status"] != "graded":
                    pend.append((rec, row))
    print(f"응답 {len(recs)}개, 규칙 채점 {len(recs) - len(pend)}개, 저지 필요 {len(pend)}개", flush=True)

    if pend:
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        spec = c["judge"]
        path = snapshot_download(spec["repo"], revision=spec["revision"], local_files_only=True)
        tok = AutoTokenizer.from_pretrained(path)
        llm = LLM(model=path, dtype="bfloat16", seed=spec["sampling"]["seed"], max_model_len=16384,
                  gpu_memory_utilization=0.90, max_num_seqs=64, max_num_batched_tokens=16384, enforce_eager=False)
        sp = SamplingParams(**spec["sampling"], max_tokens=spec["max_new_tokens"])
        todo = pend
        for attempt in range(1, a.max_attempts + 1):
            ids, keep = [], []
            for rec, row in todo:
                msgs = [{"role": "user", "content": judge_prompt(row, c)}]
                t = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                            enable_thinking=spec["enable_thinking"])
                if len(t) + spec["max_new_tokens"] > 16384:
                    rec["method"] = "judge_input_too_long"; continue
                ids.append({"prompt_token_ids": t}); keep.append((rec, row))
            outs = llm.generate(ids, sp, use_tqdm=False)
            todo = []
            for (rec, row), o in zip(keep, outs):
                g = parsers.grade_judge(row, o.outputs[0].text)
                if g["judge_parser_decision"]["decision_parsed"]:
                    rec.update(label=g["correct"], method="local_qwen3_8b_judge", judge_attempts=attempt)
                else:
                    todo.append((rec, row))
            print(f"저지 {attempt}회차: 미판정 {len(todo)}개", flush=True)
            if not todo:
                break
        for rec, _ in todo:
            rec.update(label=None, method="judge_undecided")

    with open(a.out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print("라벨:", Counter((r["tag"], r["label"]) for r in recs), flush=True)


if __name__ == "__main__":
    main()
