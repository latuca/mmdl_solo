# -*- coding: utf-8 -*-
"""LoRA DPO (Qwen3-VL-4B, 시작 모델 = SFT ep4 병합 모델).

- 입력 처리는 train_sft.py / 평가와 같다: process_vision_info(min/max pixels) → chat template → processor(do_resize=False).
- 정책(policy) = 시작 모델 + 새 LoRA. 기준(reference) = 같은 모델에서 LoRA를 끈 상태 (메모리 모델 1개분).
- loss = -log σ(β·[(logπ(c) - logπ_ref(c)) - (logπ(r) - logπ_ref(r))]), 응답 토큰(…<|im_end|>) 로그확률 합.
- 큰 logits 텐서를 들고 있지 않도록 lm_head + log_softmax를 512토큰씩 checkpoint로 계산한다.
- 선택: --rpo-alpha > 0 이면 chosen 토큰 평균 NLL을 더한다 (기본 0 = 순수 DPO).
- 끝나면 어댑터 저장 → 시작 모델에 병합 → 토크나이저/프로세서 파일 복사 (평가 바로 가능).

쌍 jsonl 한 줄: {"id","image","input_prompt","chosen","rejected", ...}
"""
import argparse, json, math, random, shutil, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_sft import LORA_TARGET, BASE_FILES, md5

CHUNK = 512


class PairData:
    def __init__(self, rows, images_root, proc, vision):
        self.rows, self.root, self.proc, self.vision = rows, Path(images_root), proc, vision

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        r = self.rows[i]
        content = [dict(type="image", image=Image.open(self.root / r["image"]).convert("RGB"),
                        min_pixels=self.vision["min_pixels"], max_pixels=self.vision["max_pixels"])]
        msgs = [{"role": "user", "content": content + [{"type": "text", "text": r["input_prompt"]}]}]
        imgs, _, _ = process_vision_info(msgs, image_patch_size=self.vision.get("patch_size", 16),
                                         return_video_kwargs=True, return_video_metadata=True)
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.proc(text=[text], images=imgs, do_resize=False, return_tensors="pt")
        tok = self.proc.tokenizer
        c = tok(r["chosen"] + "<|im_end|>", add_special_tokens=False, return_tensors="pt")["input_ids"]
        j = tok(r["rejected"] + "<|im_end|>", add_special_tokens=False, return_tensors="pt")["input_ids"]
        return {"id": r["id"], "prompt_ids": enc["input_ids"], "pixel_values": enc["pixel_values"],
                "image_grid_thw": enc["image_grid_thw"], "chosen_ids": c, "rejected_ids": j}


def seq_logp(qwen, b, resp):
    """응답 토큰 로그확률 합과 토큰 수. qwen = Qwen3VLForConditionalGeneration (LoRA 주입된 상태)."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    p = b["prompt_ids"]
    ids = torch.cat([p, resp], dim=1).cuda()
    h = qwen.model(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=b["pixel_values"].cuda(),
                   image_grid_thw=b["image_grid_thw"].cuda()).last_hidden_state
    h = h[:, p.shape[1] - 1:-1]
    tgt = resp.cuda()

    def piece(hc, tc):
        lp = F.log_softmax(qwen.lm_head(hc).float(), dim=-1)
        return lp.gather(-1, tc.unsqueeze(-1)).sum()

    total = 0.0
    for s in range(0, tgt.shape[1], CHUNK):
        hc, tc = h[:, s:s + CHUNK], tgt[:, s:s + CHUNK]
        total = total + (checkpoint(piece, hc, tc, use_reentrant=False) if torch.is_grad_enabled() else piece(hc, tc))
    return total, tgt.shape[1]


def merge_local(start_dir, adapter_dir, out_dir):
    import torch
    from peft import PeftModel
    from transformers import Qwen3VLForConditionalGeneration
    base = Qwen3VLForConditionalGeneration.from_pretrained(start_dir, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    merged.save_pretrained(out_dir, safe_serialization=True)
    return [n for n in BASE_FILES if (Path(start_dir) / n).exists() and shutil.copy2(Path(start_dir) / n, out_dir / n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--start-model", required=True, help="시작 모델 폴더 (SFT ep4 병합 모델)")
    ap.add_argument("--vision-config", required=True, help="training_settings_only.json (해상도)")
    ap.add_argument("--diag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--rpo-alpha", type=float, default=0.0)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-tokens", type=int, default=8192, help="프롬프트+응답이 이보다 길면 그 쌍은 건너뛴다")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-merge", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, a.diag)
    from common import load_baseline
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    settings, _ = load_baseline(a.vision_config)
    vision = settings["vision"]
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"출력 폴더가 비어 있지 않다: {out}")
    out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed); torch.manual_seed(a.seed)
    rows = [json.loads(l) for l in open(a.pairs, encoding="utf-8") if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    proc = AutoProcessor.from_pretrained(a.start_model)
    data = PairData(rows, a.images, proc, vision)
    total_steps = a.max_steps or math.ceil(math.ceil(len(data) / a.grad_accum) * a.epochs)
    info = {"args": vars(a), "vision": vision, "pairs_md5": md5(a.pairs), "n_pairs": len(rows),
            "total_optimizer_steps": total_steps, "code_md5": md5(__file__), "torch": torch.__version__,
            "reference": "start model with LoRA disabled"}
    (out / "train_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    model = Qwen3VLForConditionalGeneration.from_pretrained(a.start_model, dtype=torch.bfloat16,
                                                            attn_implementation="flash_attention_2").cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
                                             target_modules=LORA_TARGET, task_type="CAUSAL_LM"))
    qwen = model.base_model.model
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"학습 파라미터 {sum(p.numel() for p in params):,}, 쌍 {len(rows)}, 옵티마이저 스텝 {total_steps}", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay, betas=(0.9, 0.999))
    warm = max(1, int(total_steps * a.warmup_ratio))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total_steps)))))

    log = open(out / "train_log.jsonl", "a", encoding="utf-8")
    keys = ("loss", "reward_acc", "margin", "reward_chosen", "reward_rejected", "logp_chosen", "logp_rejected", "nll_chosen")
    acc = {k: 0.0 for k in keys}; acc["n"] = 0
    step, seen, skipped, t0, epoch = 0, 0, 0, time.time(), 0
    model.train(); torch.cuda.reset_peak_memory_stats()
    while step < total_steps:
        order = list(range(len(data)))
        random.Random(a.seed + epoch).shuffle(order)
        loader = torch.utils.data.DataLoader(torch.utils.data.Subset(data, order), batch_size=None,
                                             shuffle=False, num_workers=a.workers)
        for bi, b in enumerate(loader):
            p_len = b["prompt_ids"].shape[1]
            if p_len + max(b["chosen_ids"].shape[1], b["rejected_ids"].shape[1]) > a.max_tokens:
                skipped += 1
            else:
                with torch.no_grad(), model.disable_adapter():
                    rc, _ = seq_logp(qwen, b, b["chosen_ids"])
                    rr, _ = seq_logp(qwen, b, b["rejected_ids"])
                pc, nc = seq_logp(qwen, b, b["chosen_ids"])
                pr, _ = seq_logp(qwen, b, b["rejected_ids"])
                z = a.beta * ((pc - rc) - (pr - rr))
                loss = -F.logsigmoid(z)
                if a.rpo_alpha > 0:
                    loss = loss + a.rpo_alpha * (-pc / nc)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss: {b['id']}")
                group = min(a.grad_accum, len(order) - (bi // a.grad_accum) * a.grad_accum)
                (loss / group).backward()
                vals = dict(loss=loss.item(), reward_acc=float(z.item() > 0), margin=z.item(),
                            reward_chosen=a.beta * (pc - rc).item(), reward_rejected=a.beta * (pr - rr).item(),
                            logp_chosen=pc.item(), logp_rejected=pr.item(), nll_chosen=-pc.item() / nc)
                for k in keys:
                    acc[k] += vals[k]
                acc["n"] += 1
            seen += 1
            if (bi + 1) % a.grad_accum and bi + 1 != len(order):
                continue
            gn = torch.nn.utils.clip_grad_norm_(params, a.max_grad_norm, error_if_nonfinite=True)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            el = time.time() - t0
            n = max(1, acc["n"])
            rec = {"step": step, "epoch": epoch, **{k: round(acc[k] / n, 4) for k in keys},
                   "lr": sched.get_last_lr()[0], "sec_per_step": round(el / step, 2),
                   "eta_min": round(el / step * (total_steps - step) / 60, 1),
                   "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                   "grad_norm": round(gn.item(), 6), "pairs_in_step": acc["n"], "skipped_total": skipped,
                   "pairs_seen": seen}
            log.write(json.dumps(rec) + "\n"); log.flush()
            if step % 5 == 0 or step == 1:
                print(rec, flush=True)
            acc = {k: 0.0 for k in keys}; acc["n"] = 0
            if step >= total_steps:
                break
        epoch += 1
    adapter_dir = out / "adapter"
    model.save_pretrained(adapter_dir)
    print("어댑터 저장:", adapter_dir, "건너뛴 쌍:", skipped, flush=True)
    if not a.no_merge:
        del model, opt, qwen
        torch.cuda.empty_cache()
        print("merge:", merge_local(a.start_model, adapter_dir, out / "merged"), "→", out / "merged", flush=True)


if __name__ == "__main__":
    main()
