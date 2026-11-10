"""ERA LoRA-SFT 학습 (조건 G/E 공통 설정, 목표 응답만 다름).

실행 (mmdl_train 환경):
  # 2단계 메모리 점검: 가장 긴 샘플부터 20 optimizer step, 가중치는 저장하지 않음
  python train/train_lora.py --cond E --exp era_lora_pilot --max-steps 20 --longest-first --no-save
  # 3단계 학습: 조건별 1 epoch
  python train/train_lora.py --cond G --exp era_lora_pilot
  python train/train_lora.py --cond E --exp era_lora_pilot

결과: outputs/<exp>/<cond>/ 에 adapter/, config.yaml, loss.jsonl, data_meta.json, verify.json
"""
import argparse
import json
import math
import os
import random
import re
import sys
import time

import numpy as np
import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import era_data  # noqa: E402
from era_data import MODEL_REPO, MODEL_REV, ROOT, SEED  # noqa: E402

TARGET_MODULES = r".*language_model.*\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg):
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_REPO, revision=MODEL_REV, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": cfg["use_reentrant"]})
    # 임베딩이 동결이면 reentrant checkpoint 입력에 grad가 없어 LoRA까지 역전파가 끊긴다 → 입력에 grad를 켠다.
    # (non-reentrant에서는 불필요하지만 무해하므로 항상 켠다)
    model.enable_input_require_grads()
    lora = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                      bias="none", target_modules=TARGET_MODULES, task_type="CAUSAL_LM")
    return get_peft_model(model, lora)


def verify(model):
    """학습 직전 자동 검증: visual 미포함, 학습 파라미터 비율, LoRA 부착 모듈."""
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    lora_mods = sorted({re.sub(r"\.lora_[AB]\..*$", "", n) for n in trainable})
    bad = [n for n in trainable if "visual" in n or "lora_" not in n or ".mlp." in n]
    out = {
        "no_visual_in_trainable": not any("visual" in n for n in trainable),
        "only_lora_trainable": not bad,
        "trainable_params": n_train, "total_params": n_total, "ratio": n_train / n_total,
        "lora_modules_count": len(lora_mods), "lora_modules_first5": lora_mods[:5],
        "bad_trainable": bad[:10],
    }
    print(f"(1) 'visual' in trainable names: {'NONE' if out['no_visual_in_trainable'] else 'FOUND'}")
    print(f"(2) trainable {n_train:,} / total {n_total:,} = {out['ratio']:.4%}")
    print(f"(3) LoRA modules: {len(lora_mods)}; first5: {lora_mods[:5]}")
    if bad:
        raise SystemExit(f"unexpected trainable params: {bad[:10]}")
    return out


def to_cuda(ex):
    return {k: ex[k].unsqueeze(0).cuda() if k in ("input_ids", "attention_mask", "labels") else ex[k].cuda()
            for k in ("input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", choices=["G", "E"], required=True)
    ap.add_argument("--exp", default="era_lora_pilot")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0, help="0이면 epochs 기준. 2단계 메모리 점검용")
    ap.add_argument("--longest-first", action="store_true", help="가장 긴 샘플을 첫 step에 넣는다 (메모리 점검)")
    ap.add_argument("--no-save", action="store_true", help="adapter를 저장하지 않는다 (메모리 점검)")
    ap.add_argument("--allow-overlong", action="store_true", help=f"{era_data.MAX_LEN} 초과 샘플을 포함해 학습")
    ap.add_argument("--image-dir", default=era_data.IMAGE_DIR)
    args = ap.parse_args()

    cfg = {
        "model": MODEL_REPO, "revision": MODEL_REV, "dtype": "bfloat16", "quantization": None,
        "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "lora_bias": "none",
        "target_modules": TARGET_MODULES,
        "optimizer": "AdamW", "lr": 1e-4, "weight_decay": 0.01, "betas": [0.9, 0.999],
        "scheduler": "cosine", "warmup_ratio": 0.03,
        "micro_batch": 1, "grad_accum": 16, "max_grad_norm": 1.0,
        "gradient_checkpointing": True, "use_reentrant": False, "packing": False,
        "seed": SEED, "epochs": args.epochs, "max_steps": args.max_steps, "longest_first": args.longest_first,
        "max_len": era_data.MAX_LEN, "min_pixels": era_data.MIN_PIXELS, "max_pixels": era_data.MAX_PIXELS,
        "loss": "mean over response+<|im_end|> tokens per sample, averaged over accumulated samples",
    }
    out_dir = os.path.join(ROOT, "outputs", args.exp, args.cond + ("_memcheck" if args.no_save else ""))
    if os.path.exists(os.path.join(out_dir, "adapter")):
        raise SystemExit(f"already exists: {out_dir}/adapter (덮어쓰지 않음)")
    os.makedirs(out_dir, exist_ok=True)
    set_seed(SEED)

    proc = era_data.load_processor()
    train, _, found, meta = era_data.prepare(args.cond, proc, args.image_dir)
    meta["image_dir"] = os.path.abspath(args.image_dir)
    t0 = time.time()
    examples = [era_data.build_example(proc, r, args.cond, found[r["sample_id"]]) for r in train]
    print(f"built {len(examples)} examples in {time.time() - t0:.0f}s")
    over = {e["sample_id"]: e["meta"]["n_tokens"] for e in examples if e["meta"]["n_tokens"] > era_data.MAX_LEN}
    meta["over_max_len"] = over
    if over and not args.allow_overlong:
        raise SystemExit(f"{len(over)} samples exceed {era_data.MAX_LEN}: {over}  (--allow-overlong 로 포함 가능)")
    assert all(e["meta"]["ends_with_end_token"] for e in examples)

    model = build_model(cfg)
    ver = verify(model)

    # 순서: seed 고정 셔플을 epoch마다 새로 (longest-first면 가장 긴 샘플을 맨 앞으로)
    rng = random.Random(SEED)
    order = []
    for _ in range(args.epochs if not args.max_steps else 10 ** 6):
        perm = list(range(len(examples)))
        rng.shuffle(perm)
        order += perm
        if args.max_steps and len(order) >= args.max_steps * cfg["grad_accum"]:
            break
    if args.longest_first:
        longest = max(range(len(examples)), key=lambda i: examples[i]["meta"]["n_tokens"])
        order.remove(longest)
        order.insert(0, longest)
    ga = cfg["grad_accum"]
    total_steps = args.max_steps or math.ceil(len(order) / ga)
    order = order[: total_steps * ga]
    warmup = math.ceil(cfg["warmup_ratio"] * total_steps)
    cfg.update(total_steps=total_steps, warmup_steps=warmup, n_train=len(examples))
    print(f"optimizer steps {total_steps}, warmup {warmup}, samples/epoch {len(examples)}")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"], betas=tuple(cfg["betas"]))
    sched = get_cosine_schedule_with_warmup(opt, warmup, total_steps)

    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    with open(os.path.join(out_dir, "data_meta.json"), "w") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    with open(os.path.join(out_dir, "verify.json"), "w") as f:
        json.dump(ver, f, indent=1)

    model.train()
    torch.cuda.reset_peak_memory_stats()
    log = open(os.path.join(out_dir, "loss.jsonl"), "w")
    t_start = time.time()
    for step in range(total_steps):
        chunk = order[step * ga:(step + 1) * ga]
        acc = 0.0
        for i in chunk:
            loss = model(**to_cuda(examples[i])).loss
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at step {step + 1}, sample {examples[i]['sample_id']}")
            (loss / len(chunk)).backward()
            acc += loss.item() / len(chunk)
        gnorm = torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"]).item()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        rec = {"step": step + 1, "loss": round(acc, 5), "grad_norm": round(gnorm, 4),
               "lr": sched.get_last_lr()[0], "n_samples": len(chunk),
               "max_tokens_in_step": max(examples[i]["meta"]["n_tokens"] for i in chunk),
               "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
               "elapsed_s": round(time.time() - t_start, 1)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)
    log.close()

    summary = {"steps": total_steps, "wall_s": round(time.time() - t_start, 1),
               "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
               "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2 ** 30, 2)}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("summary:", summary)
    if not args.no_save:
        model.save_pretrained(os.path.join(out_dir, "adapter"))
        print("adapter ->", os.path.join(out_dir, "adapter"))


if __name__ == "__main__":
    main()
