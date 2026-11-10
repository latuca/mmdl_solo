"""MMMU val 부분집합(ids 파일)을 병합 모델로 평가 생성한다. scripts/infer.py를 수정하지 않고 그 함수/설정을 가져다 쓴다.
출력 형식은 infer.py의 predictions.jsonl과 동일해서 scripts/score.py를 그대로 쓸 수 있다.
  python train/eval_subset.py --exp era_lora_ep5 --model E --ids outputs/era_lora_ep5/trunc111/ids_111.json --out-dir outputs/era_lora_ep5/trunc111
"""
import argparse, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from datasets import load_dataset
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from config import DS_REPO, DS_REV, MAX_IMAGES, MAX_MODEL_LEN
from infer import parse_options, get_images, resize, build_text, LETTERS

ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True); ap.add_argument("--model", required=True, choices=["G", "E"])
ap.add_argument("--ids", required=True); ap.add_argument("--out-dir", required=True)
ap.add_argument("--seeds", type=int, nargs="+", default=[3407, 3408, 3409])
ap.add_argument("--max-tokens", type=int, default=8192)
args = ap.parse_args()

path = os.path.join(ROOT, "outputs", args.exp, args.model, "merged")
ids = json.load(open(args.ids))
processor = AutoProcessor.from_pretrained(path)
reqs, metas, dsc = [], [], {}
for rid in ids:
    cfg, n = rid[len("validation_"):].rsplit("_", 1); i = int(n) - 1
    if cfg not in dsc:
        dsc[cfg] = load_dataset(DS_REPO, cfg, split="validation", revision=DS_REV)
    row = dsc[cfg][i]; opts = parse_options(row["options"])
    imgs = [resize(im) for im in get_images(row)]
    content = [{"type": "image"} for _ in imgs] + [{"type": "text", "text": build_text(row, opts)}]
    prompt = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    req = {"prompt": prompt}
    if imgs: req["multi_modal_data"] = {"image": imgs}
    reqs.append(req)
    metas.append({"id": rid, "subject": cfg, "dataset_index": i, "question_type": row["question_type"], "question": row["question"],
                  "options": dict(zip(LETTERS, opts)), "reference_answer": row["answer"], "num_images": len(imgs)})
print(f"prepared {len(reqs)} requests", flush=True)

llm = LLM(model=path, dtype="bfloat16", max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.9, max_num_seqs=1,
          max_num_batched_tokens=4096, enforce_eager=True, limit_mm_per_prompt={"image": MAX_IMAGES}, seed=3407,
          mm_processor_kwargs={"do_resize": False})
os.makedirs(args.out_dir, exist_ok=True)
for seed in args.seeds:
    out = os.path.join(args.out_dir, f"{args.model}_seed{seed}.jsonl")
    done = {json.loads(l)["id"] for l in open(out)} if os.path.exists(out) else set()
    sp = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0, presence_penalty=1.5,
                        max_tokens=args.max_tokens, seed=seed)
    with open(out, "a", encoding="utf-8") as f:
        for req, m in zip(reqs, metas):
            if m["id"] in done: continue
            t0 = time.time(); o = llm.generate([req], sp)[0].outputs[0]
            f.write(json.dumps({**m, "raw": o.text, "finish_reason": o.finish_reason, "output_tokens": len(o.token_ids),
                                "elapsed_seconds": round(time.time() - t0, 3)}, ensure_ascii=False) + "\n"); f.flush()
    print(f"seed {seed} done -> {out}", flush=True)

sys.stdout.flush()
os._exit(0)   # vLLM 종료 단계에서 프로세스가 멈추는 문제 회피 (결과는 문항마다 이미 저장됨)
