"""MMMU-Pro (standard, 10 options) 추론. infer.py와 같은 모델·샘플링·전처리를 재사용하고,
출력 레코드 형식도 같아서 score.py / export_csv.py를 그대로 쓸 수 있다.

실행:
  python infer_pro.py --limit 3 --out ../results/pro_smoke.jsonl
  python infer_pro.py --out ../results/pro_run1/predictions.jsonl
"""
import argparse
import json
import os
import time

from datasets import load_dataset
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from config import (MODEL_REPO, MODEL_REV, SEED, MAX_IMAGES, MAX_NEW_TOKENS, MAX_MODEL_LEN)
from infer import LETTERS, PROMPT_MC, parse_options, get_images, resize

PRO_REPO = "MMMU/MMMU_Pro"
PRO_CONFIG = "standard (10 options)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results/pro_run1/predictions.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0이면 전체(1730문항)")
    ap.add_argument("--revision", default=None, help="데이터셋 revision(커밋 해시). 재현용으로 고정 권장")
    args = ap.parse_args()
    t_start = time.time()
    t_start_iso = time.strftime("%Y-%m-%d %H:%M:%S")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    print(f"already done: {len(done)}", flush=True)

    ds = load_dataset(PRO_REPO, PRO_CONFIG, split="test", revision=args.revision)
    try:
        from huggingface_hub import HfApi
        print("dataset sha:", HfApi().dataset_info(PRO_REPO, revision=args.revision).sha, flush=True)
    except Exception as e:  # 네트워크 문제여도 평가 자체는 계속
        print("dataset sha 조회 실패:", e)
    print(f"{PRO_CONFIG}: {len(ds)} rows", flush=True)

    processor = AutoProcessor.from_pretrained(MODEL_REPO, revision=MODEL_REV)
    llm = LLM(
        model=MODEL_REPO, revision=MODEL_REV, dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.9,
        max_num_seqs=1, max_num_batched_tokens=4096, enforce_eager=True,
        limit_mm_per_prompt={"image": MAX_IMAGES}, seed=SEED,
        mm_processor_kwargs={"do_resize": False},
    )
    load_seconds = time.time() - t_start
    sp = SamplingParams(
        temperature=0.7, top_p=0.8, top_k=20,
        repetition_penalty=1.0, presence_penalty=1.5,
        max_tokens=MAX_NEW_TOKENS, seed=SEED,
    )

    n = 0
    with open(args.out, "a", encoding="utf-8") as f:
        for i in range(len(ds)):
            if args.limit and n >= args.limit:
                break
            row = ds[i]
            rid = row["id"]
            if rid in done:
                continue
            t0 = time.time()
            opts = parse_options(row["options"])
            imgs = [resize(im) for im in get_images(row)]
            lettered = "".join(f"{LETTERS[k]}. {o}\n" for k, o in enumerate(opts))
            text = PROMPT_MC.format(q=row["question"], opts=lettered)

            content = [{"type": "image"} for _ in imgs] + [{"type": "text", "text": text}]
            prompt = processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
            req = {"prompt": prompt}
            if imgs:
                req["multi_modal_data"] = {"image": imgs}
            out = llm.generate([req], sp)[0].outputs[0]
            elapsed = time.time() - t0

            rec = {
                "id": rid,
                "subject": row["subject"],
                "dataset_index": i,
                "question_type": "multiple-choice",
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

    run = {"started": t_start_iso, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
           "model_load_seconds": round(load_seconds, 1),
           "wall_seconds": round(time.time() - t_start, 1), "new_records": n,
           "benchmark": f"{PRO_REPO} / {PRO_CONFIG} / test", "dataset_revision": args.revision}
    with open(args.out.replace(".jsonl", ".runs.jsonl"), "a", encoding="utf-8") as rf:
        rf.write(json.dumps(run) + "\n")
    print(f"done. wrote {n} new records -> {args.out}")


if __name__ == "__main__":
    main()
