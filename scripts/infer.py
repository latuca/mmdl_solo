"""Qwen3-VL-4B-Instruct 추론. 결과는 한 줄씩 JSONL로 저장하고, 이미 있는 id는 건너뛴다(중단 후 재개 가능).

실행:
  python infer.py --limit 3                       # 스모크 테스트
  python infer.py --out results/predictions.jsonl  # 전체 900문항
"""
import argparse
import ast
import json
import os
import time

from datasets import load_dataset
from PIL import Image
from qwen_vl_utils.vision_process import smart_resize
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from config import (CONFIGS, DS_REPO, DS_REV, MODEL_REPO, MODEL_REV, SEED, MIN_PIXELS,
                    MAX_PIXELS, MAX_IMAGES, MAX_NEW_TOKENS, MAX_MODEL_LEN)

LETTERS = "ABCDEFGHIJKL"
PROMPT_MC = ("Question: {q}\nOptions:\n{opts}Please select the correct answer from the options above.\n"
             "On the last line, write \"Final answer: LETTER\", replacing LETTER with one uppercase option letter.")
PROMPT_OPEN = ("Question: {q}\n"
               "On the last line, write \"Final answer: ANSWER\", replacing ANSWER with a concise final answer.")


def parse_options(raw):
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        return raw
    return ast.literal_eval(raw)


def get_images(row):
    imgs = []
    for k in range(1, 8):
        im = row.get(f"image_{k}")
        if im is not None:
            imgs.append(im.convert("RGB"))
    return imgs[:MAX_IMAGES]


def resize(im):
    # 공식 qwen_vl_utils와 같은 smart_resize 규칙 (factor 32, 픽셀 범위 고정)
    w, h = im.size
    nh, nw = smart_resize(h, w, factor=32, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    return im.resize((nw, nh), Image.BICUBIC)


def build_text(row, opts):
    if row["question_type"] == "multiple-choice":
        lettered = "".join(f"{LETTERS[i]}. {o}\n" for i, o in enumerate(opts))
        return PROMPT_MC.format(q=row["question"], opts=lettered)
    return PROMPT_OPEN.format(q=row["question"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/predictions.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0이면 전체")
    args = ap.parse_args()
    t_start = time.time()
    t_start_iso = time.strftime("%Y-%m-%d %H:%M:%S")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    print(f"already done: {len(done)}", flush=True)

    processor = AutoProcessor.from_pretrained(MODEL_REPO, revision=MODEL_REV)
    llm = LLM(
        model=MODEL_REPO,
        revision=MODEL_REV,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.9,
        max_num_seqs=1,
        max_num_batched_tokens=4096,
        enforce_eager=True,
        limit_mm_per_prompt={"image": MAX_IMAGES},
        seed=SEED,
        mm_processor_kwargs={"do_resize": False},  # 이미 smart_resize 했으므로 vLLM 쪽 중복 리사이즈 끔
    )
    load_seconds = time.time() - t_start   # 모델 로딩까지 걸린 시간
    sp = SamplingParams(
        temperature=0.7, top_p=0.8, top_k=20,
        repetition_penalty=1.0, presence_penalty=1.5,
        max_tokens=MAX_NEW_TOKENS, seed=SEED,
    )

    n = 0
    with open(args.out, "a", encoding="utf-8") as f:
        for cfg in CONFIGS:
            ds = load_dataset(DS_REPO, cfg, split="validation", revision=DS_REV)
            for i in range(len(ds)):
                if args.limit and n >= args.limit:
                    break
                rid = f"validation_{cfg}_{i + 1}"
                if rid in done:
                    continue
                t0 = time.time()
                row = ds[i]
                opts = parse_options(row["options"])
                imgs = [resize(im) for im in get_images(row)]
                text = build_text(row, opts)

                content = [{"type": "image"} for _ in imgs] + [{"type": "text", "text": text}]
                prompt = processor.apply_chat_template(
                    [{"role": "user", "content": content}],
                    tokenize=False, add_generation_prompt=True,
                )
                req = {"prompt": prompt}
                if imgs:
                    req["multi_modal_data"] = {"image": imgs}
                out = llm.generate([req], sp)[0].outputs[0]
                elapsed = time.time() - t0   # 이미지 전처리 + 생성 시간

                rec = {
                    "id": rid,
                    "subject": cfg,
                    "dataset_index": i,
                    "question_type": row["question_type"],
                    "question": row["question"],
                    "options": dict(zip(LETTERS, opts)),
                    "reference_answer": row["answer"],
                    "num_images": len(imgs),
                    "raw": out.text,
                    "finish_reason": out.finish_reason,
                    "output_tokens": len(out.token_ids),
                    "elapsed_seconds": round(elapsed, 3),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                n += 1
                print(f"[{n}] {rid} finish={out.finish_reason} tok={len(out.token_ids)}", flush=True)
            if args.limit and n >= args.limit:
                break
    run = {"started": t_start_iso, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
           "model_load_seconds": round(load_seconds, 1),
           "wall_seconds": round(time.time() - t_start, 1), "new_records": n}
    runs_path = args.out.replace(".jsonl", ".runs.jsonl")   # 실행(재개 포함)마다 한 줄씩 누적
    with open(runs_path, "a", encoding="utf-8") as rf:
        rf.write(json.dumps(run) + "\n")
    print(f"done. wrote {n} new records -> {args.out}")
    print(f"run time: {run['wall_seconds']}s (model load {run['model_load_seconds']}s)")


if __name__ == "__main__":
    main()
