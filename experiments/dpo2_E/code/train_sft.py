# -*- coding: utf-8 -*-
"""LoRA SFT (Qwen3-VL-4B-Instruct): 베이스라인 입력 템플릿 + "풀이\\nFinal answer: X" 정답.

- 입력은 평가와 같은 방식으로 만든다: process_vision_info(min/max pixels) → chat template → processor(do_resize=False).
  (check_train_env.py에서 평가 토큰과 같음을 확인한 경로)
- 정답 = reasoning 구간(마지막 줄 앞까지 + 줄바꿈) + answer 구간("Final answer: X<|im_end|>").
  loss = w_reason * CE(reasoning 토큰 평균) + w_answer * CE(answer 토큰 평균). 두 값은 따로 기록한다.
- micro batch 1 + gradient accumulation. flash-attention 2, gradient checkpointing, bf16.
- 끝나면 어댑터 저장 → base에 merge → base revision의 tokenizer/프로세서 파일을 복사 → 평가 스크립트
  (run_generate.py --model-path)로 바로 평가할 수 있는 폴더를 만든다.

데이터 jsonl 한 줄: {"id", "image" (또는 "images": [...]), "input_prompt", "target"}

사용 (서버, 학습 환경)
  ~/mmdl/.venv-train/bin/python train_sft.py --data ~/mmdl/datasets/arxivqa_prepared/sft_train_20k_balanced.jsonl \
      --images ~/mmdl/datasets/arxivqa --limit 5000 --baseline ~/mmdl/mmdl_test/outputs/baseline_results.json \
      --diag ~/mmdl/mmdl_test/diag --out ~/mmdl/train_runs/arxivqa_sft_5k
  속도·메모리만 재기:  ... --max-steps 30 --no-merge
  merge만 다시:        ... --merge-only --out <같은 폴더>
"""
import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

LORA_TARGET = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
BASE_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.json",
              "preprocessor_config.json", "video_preprocessor_config.json"]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_target(target):
    """정답을 reasoning 구간과 마지막 줄(Final answer)로 나눈다."""
    # 서술형 정답은 여러 줄일 수 있으므로 마지막 "\nFinal answer:" 위치에서 나눈다.
    t = target.rstrip()
    k = t.rfind("\nFinal answer:")
    if k < 0:
        raise ValueError("정답에 'Final answer:' 줄이 없음")
    return t[:k].rstrip() + "\n", t[k + 1:].strip()


class SFTData:
    def __init__(self, rows, images_root, proc, vision):
        self.rows, self.root, self.proc, self.vision = rows, Path(images_root), proc, vision

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        r = self.rows[i]
        paths = r.get("images") or [r["image"]]
        content = [dict(type="image", image=Image.open(self.root / p).convert("RGB"),
                        min_pixels=self.vision["min_pixels"], max_pixels=self.vision["max_pixels"]) for p in paths]
        msgs = [{"role": "user", "content": content + [{"type": "text", "text": r["input_prompt"]}]}]
        imgs, _, _ = process_vision_info(msgs, image_patch_size=self.vision.get("patch_size", 16),
                                         return_video_kwargs=True, return_video_metadata=True)
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.proc(text=[text], images=imgs, do_resize=False, return_tensors="pt")
        reason, answer = split_target(r["target"])
        tok = self.proc.tokenizer
        r_ids = tok(reason, add_special_tokens=False, return_tensors="pt")["input_ids"]
        a_ids = tok(answer + "<|im_end|>", add_special_tokens=False, return_tensors="pt")["input_ids"]
        p_len = enc["input_ids"].shape[1]
        input_ids = torch.cat([enc["input_ids"], r_ids, a_ids], dim=1)
        return {"id": r["id"], "loss_weights": r.get("loss_weights"), "input_ids": input_ids, "pixel_values": enc["pixel_values"],
                "image_grid_thw": enc["image_grid_thw"], "p_len": p_len,
                "r_len": r_ids.shape[1], "a_len": a_ids.shape[1]}


def step_loss(model, b, w_r, w_a, loss_mode="token_mean"):
    """한 문항의 loss. 프롬프트 부분 logits는 만들지 않는다 (logits_to_keep).

    loss_mode=token_mean: 표준 SFT (target 전체 토큰 CE 평균). 반환 l_r/l_a는 기록용.
    loss_mode=split: 원래 train_dr.py 방식 (w_r*mean(reason) + w_a*mean(answer)).
    """
    import torch
    import torch.nn.functional as F
    if b.get("loss_weights") is not None:
        policy = b["loss_weights"]
        w_r, w_a = float(policy["reason"]), float(policy["answer"])
        import math
        if not (math.isfinite(w_r) and math.isfinite(w_a) and w_r >= 0 and w_a >= 0 and w_r + w_a > 0):
            raise ValueError("Invalid per-record loss_weights")
    ids = b["input_ids"].cuda()
    T = b["r_len"] + b["a_len"]
    kw = dict(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=b["pixel_values"].cuda(),
              image_grid_thw=b["image_grid_thw"].cuda())
    try:
        logits = model(**kw, logits_to_keep=T + 1).logits[:, :-1]
    except TypeError:
        logits = model(**kw).logits[:, b["p_len"] - 1:-1]
    tgt = ids[:, b["p_len"]:]
    ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="none")
    l_r, l_a = ce[:b["r_len"]].mean(), ce[b["r_len"]:].mean()
    if loss_mode == "token_mean":
        return ce.mean(), l_r.detach(), l_a.detach()
    return w_r * l_r + w_a * l_a, l_r.detach(), l_a.detach()


def merge_and_save(repo, rev, adapter_dir, out_dir):
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import Qwen3VLForConditionalGeneration
    base = Qwen3VLForConditionalGeneration.from_pretrained(repo, revision=rev, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    merged.save_pretrained(out_dir, safe_serialization=True)
    snap = Path(snapshot_download(repo, revision=rev, allow_patterns=BASE_FILES))
    copied = [n for n in BASE_FILES if (snap / n).exists() and shutil.copy2(snap / n, out_dir / n)]
    return copied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--baseline", required=True, help="모델 repo/revision과 vision 설정을 읽는다")
    ap.add_argument("--diag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, help="앞에서 N문항만 사용 (데이터 파일은 이미 섞여 있음)")
    ap.add_argument("--val", help="loss만 재는 검증 jsonl (선택)")
    ap.add_argument("--val-limit", type=int, default=200)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, help="옵티마이저 스텝 상한 (속도 측정용)")
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--w-reason", type=float, default=1.0)
    ap.add_argument("--w-answer", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=8192, help="이보다 긴 문항은 건너뛴다 (메모리)")
    ap.add_argument("--save-every", type=int, default=100, help="옵티마이저 스텝마다 어댑터 저장")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--save-epochs", default="", help="쉼표 구분 epoch 번호(1부터). 해당 epoch 끝에 adapter_ep{N} 저장")
    ap.add_argument("--attn", default="flash_attention_2", choices=["flash_attention_2", "sdpa"])
    ap.add_argument("--loss-mode", default="token_mean", choices=["token_mean", "split"])
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--merge-only", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, a.diag)
    from common import load_baseline
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    settings, _ = load_baseline(a.baseline)
    repo, rev, vision = settings["model"]["repo"], settings["model"]["revision"], settings["vision"]
    out = Path(a.out)
    adapter_dir, merged_dir = out / "adapter", out / "merged"
    if a.merge_only:
        print("merge:", merge_and_save(repo, rev, adapter_dir, merged_dir))
        return
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"출력 폴더가 비어 있지 않다: {out}")
    out.mkdir(parents=True, exist_ok=True)

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    proc = AutoProcessor.from_pretrained(repo, revision=rev)
    data = SFTData(rows, a.images, proc, vision)
    n_steps_epoch = math.ceil(len(data) / a.grad_accum)
    total_steps = a.max_steps or math.ceil(n_steps_epoch * a.epochs)
    info = {"args": vars(a), "base": {"repo": repo, "revision": rev}, "vision": vision,
            "data_md5": md5(a.data), "n_rows": len(rows), "total_optimizer_steps": total_steps,
            "code_md5": md5(__file__), "torch": torch.__version__,
            "epoch_tail_policy": "flush final partial batch using its actual size",
            "teacher_forced_validation_is_accuracy": False}
    (out / "train_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    model = Qwen3VLForConditionalGeneration.from_pretrained(repo, revision=rev, dtype=torch.bfloat16,
                                                            attn_implementation=a.attn).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    cfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
                     target_modules=LORA_TARGET, task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"학습 파라미터 {sum(p.numel() for p in params):,}, 문항 {len(rows)}, 옵티마이저 스텝 {total_steps}")
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay, betas=(0.9, 0.999))
    save_epochs = {int(x) for x in a.save_epochs.split(",") if x.strip()}
    warm = max(1, int(total_steps * a.warmup_ratio))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total_steps)))))

    log = open(out / "train_log.jsonl", "a", encoding="utf-8")
    step, micro, t0 = 0, 0, time.time()
    acc = {"loss": 0.0, "lr_": 0.0, "la": 0.0, "tok": 0, "n": 0, "skipped": 0}
    model.train()
    epoch = 0
    torch.cuda.reset_peak_memory_stats()
    while step < total_steps:
        order = list(range(len(data)))
        random.Random(a.seed + epoch).shuffle(order)
        loader = torch.utils.data.DataLoader(torch.utils.data.Subset(data, order), batch_size=None,
                                             shuffle=False, num_workers=a.workers)
        for batch_i, b in enumerate(loader):
            if b["input_ids"].shape[1] > a.max_tokens:
                raise RuntimeError(f"Unexpected overlength row: {b['id']}")
            loss, l_r, l_a = step_loss(model, b, a.w_reason, a.w_answer, a.loss_mode)
            if not torch.isfinite(torch.stack([loss.detach(), l_r, l_a])).all():
                raise RuntimeError(f"Nonfinite loss: {b['id']}")
            group_size = min(a.grad_accum, len(order) - (batch_i // a.grad_accum) * a.grad_accum)
            (loss / group_size).backward()
            micro += 1
            acc["loss"] += loss.item(); acc["lr_"] += l_r.item(); acc["la"] += l_a.item()
            acc["tok"] += b["input_ids"].shape[1]; acc["n"] += 1
            if (batch_i + 1) % a.grad_accum and batch_i + 1 != len(order):
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(params, a.max_grad_norm, error_if_nonfinite=True)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            el = time.time() - t0
            rec = {"step": step, "epoch": epoch, "loss": round(acc["loss"] / acc["n"], 4),
                   "loss_reason": round(acc["lr_"] / acc["n"], 4), "loss_answer": round(acc["la"] / acc["n"], 4),
                   "lr": sched.get_last_lr()[0], "tokens_per_s": round(acc["tok"] / max(1e-6, el), 1),
                   "sec_per_step": round(el / step, 2), "eta_min": round(el / step * (total_steps - step) / 60, 1),
                   "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "skipped": acc["skipped"],
                   "gradient_norm_before_clip": round(grad_norm.item(), 6),
                   "microbatch_count": acc["n"], "examples_seen": micro}
            log.write(json.dumps(rec) + "\n"); log.flush()
            if step % 5 == 0 or step == 1:
                print(rec, flush=True)
            acc.update(loss=0.0, lr_=0.0, la=0.0, n=0)
            if step % a.save_every == 0:
                model.save_pretrained(out / f"adapter_step{step}")
            if step >= total_steps:
                break
        epoch += 1
        if epoch in save_epochs:
            d = out / f"adapter_ep{epoch}"
            model.save_pretrained(d)
            (d / "ckpt_info.json").write_text(json.dumps({"epoch": epoch, "optimizer_step": step}) + "\n")
            print(f"epoch {epoch} 끝: {d} 저장 (step {step})", flush=True)
    model.save_pretrained(adapter_dir)
    print("어댑터 저장:", adapter_dir)

    if a.val:
        vrows = [json.loads(l) for l in open(a.val, encoding="utf-8") if l.strip()][:a.val_limit]
        vdata = SFTData(vrows, a.images, proc, vision)
        model.eval()
        s_r = s_a = s_total = 0.0
        from collections import defaultdict
        by_role = defaultdict(lambda: {"n": 0, "loss_sum": 0.0, "answer_loss_sum": 0.0})
        with torch.no_grad():
            for i in range(len(vdata)):
                v_loss, l_r, l_a = step_loss(model, vdata[i], 1.0, 1.0)
                if not torch.isfinite(torch.stack([v_loss.detach(), l_r, l_a])).all():
                    raise RuntimeError(f"Nonfinite validation loss: {vrows[i]['id']}")
                s_r += l_r.item(); s_a += l_a.item(); s_total += v_loss.item()
                role = vrows[i]["role"]; stats = by_role[role]; stats["n"] += 1
                stats["loss_sum"] += v_loss.item(); stats["answer_loss_sum"] += l_a.item()
        val = {"val_n": len(vdata), "val_loss_reason": round(s_r / len(vdata), 4),
               "val_loss_answer": round(s_a / len(vdata), 4),
               "val_weighted_loss": round(s_total / len(vdata), 6),
               "teacher_forced_loss_only": True,
               "by_role": {role: {"n": s["n"], "weighted_loss": round(s["loss_sum"] / s["n"], 6),
                                   "answer_loss": round(s["answer_loss_sum"] / s["n"], 6)} for role, s in by_role.items()}}
        print("검증:", val)
        (out / "val.json").write_text(json.dumps(val) + "\n", encoding="utf-8")

    if not a.no_merge:
        del model, opt
        torch.cuda.empty_cache()
        print("merge:", merge_and_save(repo, rev, adapter_dir, merged_dir), "→", merged_dir)


if __name__ == "__main__":
    main()
