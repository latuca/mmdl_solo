"""vic_era_2000 -> SFT jsonl for the "general explanation only" condition.

input_prompt : built with mmdl_test/src/evaluate.py:prompt_for (identical to MMMU eval prompt + suffix)
target       : explanation with <think> removed, trailing \\boxed/Final-answer lines removed,
               then "\\nFinal answer: <answer>" appended (train_dr.py split_target format).
Optional --max-tokens: drop rows whose full (prompt+image+target) length exceeds the limit, using the
pinned processor and eval vision settings (needs the training venv).
"""
import argparse, json, re, sys, random, collections
from pathlib import Path

TAIL_RE = [
    re.compile(r"\n*\s*(?:\*\*)?\s*final\s*answer\s*(?:is)?\s*[:：]?.*$", re.I | re.S),
    re.compile(r"\n*\s*(?:the\s+(?:final\s+)?answer\s+is\s*[:：]?\s*)?\$*\\boxed\{.*$", re.I | re.S),
]


TAIL_MARK = re.compile(r"\\boxed\{|final\s*answer|the\s+(?:correct\s+)?(?:answer|option|choice)\s+is", re.I)
MIN_BODY = 60


def strip_answer_tail(e):
    """Remove the trailing sentence(s)/line(s) that state the final answer (\\boxed, 'final answer', ...)."""
    for _ in range(3):
        ms = [m for m in TAIL_MARK.finditer(e) if m.start() >= len(e) * 0.5]
        if not ms:
            break
        s = ms[0].start()
        cut = max(e.rfind("\n", 0, s), e.rfind(". ", 0, s) + 1 if e.rfind(". ", 0, s) >= 0 else -1)
        e = e[:cut + 1 if cut >= 0 else 0].rstrip()
    return e


def clean_target(expl, answer):
    """Returns (target, source) where source in {answer_block, think_fallback, plain}."""
    e = expl
    th = re.search(r"<think>(.*?)</think>", e, re.S)
    think = th.group(1).strip() if th else ""
    src = "plain"
    if "</think>" in e:
        e = e.split("</think>")[-1]
        src = "answer_block"
    m = re.search(r"<answer>(.*?)(?:</answer>|$)", e, re.S)
    if m:
        e = m.group(1)
    e = strip_answer_tail(e.replace("<think>", "").strip())
    if len(e) < MIN_BODY and think:
        # <answer> held only the final answer: the step-by-step solution lives in <think>.
        e, src = strip_answer_tail(think), "think_fallback"
    e = re.sub(r"\n{3,}", "\n\n", e).strip()
    if len(e) < MIN_BODY:
        return None, "no_reasoning"
    return e + "\nFinal answer: " + str(answer).strip(), src


def strip_inline_options(q, opts):
    if any(len(str(o).strip()) < 3 for o in opts):
        return q, False
    pos, start = [], 0
    for o in opts:
        i = q.find(str(o).strip(), start)
        if i < 0:
            return q, False
        pos.append(i)
        start = i + len(str(o).strip())
    tail = q[start:]
    if len(re.sub(r"[\s'\"\]\)\.]", "", tail)) > 0:
        return q, False
    pre = q[:pos[0]]
    pre = re.sub(r"[\[\s'\"]*\(?A\)?\s*[\.:\)、．]?\s*$", "", pre)
    pre = re.sub(r"(?:Answer\s+)?(?:Choices|Options)\s*[:：]?\s*$", "", pre.rstrip(), flags=re.I).rstrip()
    if len(pre) < 5:
        return q, False
    return pre, True


def clean_question(q):
    q = q.strip()
    if q.startswith("<image>"):
        q = q[len("<image>"):].lstrip()
    q = q.replace("<image>", "<image 1>")
    q = re.sub(r"^Question\s*[:：]\s*", "", q)
    return q.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--eval-src", required=True, help="mmdl_test/src (for prompt_for/config)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int)
    ap.add_argument("--seed", type=int, default=3407)
    a = ap.parse_args()
    sys.path.insert(0, a.eval_src)
    from evaluate import prompt_for
    cfg = json.load(open(Path(a.eval_src) / "config.json"))
    rows = json.load(open(Path(a.data_dir) / "data.json"))
    out, stats, drops = [], collections.Counter(), []
    for r in rows:
        q = clean_question(r["question"])
        if r["question_type"] == "multiple-choice":
            q, ok = strip_inline_options(q, r["options"])
            stats["mc_options_stripped" if ok else "mc_options_kept_inline"] += 1
            if not re.fullmatch(r"[A-Z]", r["answer"].strip()) or ord(r["answer"].strip()) - 65 >= len(r["options"]):
                drops.append((r["id"], "bad_mc_answer")); continue
        tgt, src = clean_target(r["explanation"], r["answer"])
        if tgt is None:
            drops.append((r["id"], src)); continue
        stats["target_" + src] += 1
        prompt = prompt_for({"question": q, "options": r["options"], "question_type": r["question_type"]}, cfg)
        out.append({"id": r["id"], "vic_type": r["vic_type"], "question_type": r["question_type"],
                    "image": r["image_1"], "input_prompt": prompt, "target": tgt})
    if a.max_tokens:
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(cfg["model"]["repo"], revision=cfg["model"]["revision"])
        vis = cfg["vision"]
        kept = []
        for i, r in enumerate(out):
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": Image.open(Path(a.data_dir) / r["image"]).convert("RGB"),
                 "min_pixels": vis["min_pixels"], "max_pixels": vis["max_pixels"]},
                {"type": "text", "text": r["input_prompt"]}]}]
            imgs, _, _ = process_vision_info(msgs, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            n_p = proc(text=[text], images=imgs, do_resize=False, return_tensors="pt")["input_ids"].shape[1]
            n_t = len(proc.tokenizer(r["target"] + "<|im_end|>", add_special_tokens=False)["input_ids"])
            r["n_prompt_tokens"], r["n_target_tokens"] = n_p, n_t
            if n_p + n_t > a.max_tokens:
                drops.append((r["id"], f"overlength_{n_p + n_t}"))
            else:
                kept.append(r)
            if i % 200 == 0:
                print(f"len check {i}/{len(out)}", flush=True)
        out = kept
    random.Random(a.seed).shuffle(out)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rep = {"n_in": len(rows), "n_out": len(out), "stats": dict(stats),
           "kept_by_type": dict(collections.Counter(r["vic_type"] for r in out)),
           "drops": collections.Counter(d[1].split("_")[0] for d in drops), "dropped_ids": drops}
    Path(a.out).with_suffix(".report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "dropped_ids"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
