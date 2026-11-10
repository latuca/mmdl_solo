"""5단계: held-out 20문항을 HF generate 또는 vLLM으로 생성한다 (base / G / E 모델).

실행 (mmdl_train 환경, 엔진마다 별도 프로세스):
  python train/gen_heldout.py --engine hf   --model E
  python train/gen_heldout.py --engine vllm --model E
샘플링은 평가와 동일: temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5, seed 3407.
HF generate에는 presence_penalty가 없어 vLLM과 같은 규칙(생성 토큰만 대상, logit에서 차감)의 processor를 쓴다.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import era_data  # noqa: E402
from era_data import MODEL_REPO, MODEL_REV, ROOT, SEED  # noqa: E402

MAX_NEW_TOKENS = 4096
MAX_MODEL_LEN = 64000   # 평가(scripts/config.py)와 동일
TEMP, TOP_P, TOP_K, PRESENCE = 0.7, 0.8, 20, 1.5


def model_path(name, exp):
    if name == "base":
        return MODEL_REPO, {"revision": MODEL_REV}
    return os.path.join(ROOT, "outputs", exp, name, "merged"), {}


def build_inputs(proc, rec, image_path):
    image, _, _ = era_data.load_image(image_path)   # 학습/평가와 같은 smart_resize
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": rec["prompt"]}]}]
    prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return prompt, image


def run_hf(args, recs, found):
    from transformers import AutoModelForImageTextToText, LogitsProcessor, LogitsProcessorList

    class PresencePenalty(LogitsProcessor):
        def __init__(self, penalty, prompt_len):
            self.penalty, self.prompt_len = penalty, prompt_len

        def __call__(self, input_ids, scores):
            for b in range(input_ids.shape[0]):
                seen = torch.unique(input_ids[b, self.prompt_len:])
                scores[b, seen] -= self.penalty
            return scores

    path, kw = model_path(args.model, args.exp)
    proc = era_data.load_processor()
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, **kw).cuda().eval()
    rows = []
    for rec in recs:
        prompt, image = build_inputs(proc, rec, found[rec["sample_id"]])
        enc = proc(text=[prompt], images=[image], return_tensors="pt", do_resize=False).to("cuda")
        n_in = enc["input_ids"].shape[1]
        torch.manual_seed(SEED)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**enc, do_sample=True, temperature=TEMP, top_p=TOP_P, top_k=TOP_K,
                                 max_new_tokens=MAX_NEW_TOKENS,
                                 logits_processor=LogitsProcessorList([PresencePenalty(PRESENCE, n_in)]))
        gen = out[0, n_in:]
        eos = {proc.tokenizer.convert_tokens_to_ids("<|im_end|>"), proc.tokenizer.convert_tokens_to_ids("<|endoftext|>")}
        finished = bool(gen.numel()) and int(gen[-1]) in eos
        rows.append({"sample_id": rec["sample_id"], "raw": proc.tokenizer.decode(gen, skip_special_tokens=True),
                     "n_tokens": int(gen.numel()), "finish_reason": "stop" if finished else "length",
                     "seconds": round(time.time() - t0, 1)})
        print(f"{rec['sample_id']} tok={rows[-1]['n_tokens']} {rows[-1]['finish_reason']}", flush=True)
    return rows


def run_vllm(args, recs, found):
    from vllm import LLM, SamplingParams
    path, kw = model_path(args.model, args.exp)
    proc = era_data.load_processor()
    llm = LLM(model=path, dtype="bfloat16", max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.9, max_num_seqs=1,
              max_num_batched_tokens=4096, enforce_eager=True, limit_mm_per_prompt={"image": 5}, seed=SEED,
              mm_processor_kwargs={"do_resize": False}, **kw)
    sp = SamplingParams(temperature=TEMP, top_p=TOP_P, top_k=TOP_K, repetition_penalty=1.0, presence_penalty=PRESENCE,
                        max_tokens=MAX_NEW_TOKENS, seed=SEED)
    rows = []
    for rec in recs:
        prompt, image = build_inputs(proc, rec, found[rec["sample_id"]])
        t0 = time.time()
        o = llm.generate([{"prompt": prompt, "multi_modal_data": {"image": [image]}}], sp)[0].outputs[0]
        rows.append({"sample_id": rec["sample_id"], "raw": o.text, "n_tokens": len(o.token_ids),
                     "finish_reason": o.finish_reason, "seconds": round(time.time() - t0, 1)})
        print(f"{rec['sample_id']} tok={rows[-1]['n_tokens']} {rows[-1]['finish_reason']}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["hf", "vllm"], required=True)
    ap.add_argument("--model", choices=["base", "G", "E"], required=True)
    ap.add_argument("--exp", default="era_lora_pilot")
    args = ap.parse_args()
    out_dir = os.path.join(ROOT, "outputs", args.exp, "stage5")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{args.model}_{args.engine}.jsonl")
    if os.path.exists(out):
        raise SystemExit(f"already exists: {out}")
    _, held, found, _ = era_data.prepare("E")   # held-out 분할은 조건과 무관하게 동일
    t0 = time.time()
    rows = (run_hf if args.engine == "hf" else run_vllm)(args, held, found)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"done {args.model}/{args.engine}: {len(rows)} samples, {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
