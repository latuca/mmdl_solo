"""DPO 학습률·SFT 항 가중치(λ)를 1차 근사로 정한다 (실제 모델·쌍에서 기울기 측정).

정책 = 시작 모델 + LoRA(초기 상태). 쌍 B개씩 묶은 배치 K개에 대해
  g_k = 배치 평균 ∇(log π(c) − log π(r))  (LoRA 파라미터)
Adam 정상 상태 근사: 파라미터 i 한 스텝 이동 ≈ lr · m_i/√v_i,  m_i ≈ ḡ_i,  v_i ≈ mean_k g_{k,i}²
  → 한 스텝 마진 변화 Δ ≈ lr · Σ_i ḡ_i² / rms_i,   z = β·마진
코사인 스케줄 평균 계수 s̄ ≈ 0.5, 총 스텝 N → 학습 끝 z ≈ β · lr · Δ/lr · N · s̄
  → lr = z* / (β · (Δ/lr) · N · s̄)
λ: 초기 DPO 손실 기울기 크기(β·0.5·‖ḡ‖₂)와 chosen 평균 NLL 기울기 크기(‖∇NLL‖₂)가 같아지도록
  λ = β·0.5·‖ḡ‖₂ / ‖∇NLL‖₂
"""
import argparse, json, math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_dpo import PairData, seq_logp
from train_sft import LORA_TARGET

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", required=True)
ap.add_argument("--images", required=True)
ap.add_argument("--start-model", required=True)
ap.add_argument("--vision-config", required=True)
ap.add_argument("--diag", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--beta", type=float, default=0.1)
ap.add_argument("--grad-accum", type=int, default=16)
ap.add_argument("--batches", type=int, default=4)
ap.add_argument("--z-target", type=float, default=1.0)
ap.add_argument("--n-pairs-total", type=int, required=True, help="실제 학습 쌍 수 (스텝 수 계산용)")
ap.add_argument("--max-tokens", type=int, default=8400)
a = ap.parse_args()

sys.path.insert(0, a.diag)
from common import load_baseline
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

torch.manual_seed(3407)
vision = load_baseline(a.vision_config)[0]["vision"]
rows = [json.loads(l) for l in open(a.pairs, encoding="utf-8")][: a.batches * a.grad_accum * 2]
proc = AutoProcessor.from_pretrained(a.start_model)
data = PairData(rows, a.images, proc, vision)
model = Qwen3VLForConditionalGeneration.from_pretrained(a.start_model, dtype=torch.bfloat16,
                                                        attn_implementation="flash_attention_2").cuda()
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=LORA_TARGET,
                                         task_type="CAUSAL_LM"))
model.train()
qwen = model.base_model.model
params = [p for p in model.parameters() if p.requires_grad]


def flat():
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).float().flatten() for p in params])


batch_m, batch_nll, used, i = [], [], 0, 0
while len(batch_m) < a.batches and i < len(data):
    gm = torch.zeros(sum(p.numel() for p in params), device="cuda")
    gn = torch.zeros_like(gm)
    n = 0
    while n < a.grad_accum and i < len(data):
        b = data[i]; i += 1
        if b["prompt_ids"].shape[1] + max(b["chosen_ids"].shape[1], b["rejected_ids"].shape[1]) > a.max_tokens:
            continue
        model.zero_grad(set_to_none=True)
        pc, nc = seq_logp(qwen, b, b["chosen_ids"]); pc.backward(); gc = flat(); gc_tok = gc / nc
        model.zero_grad(set_to_none=True)
        pr, _ = seq_logp(qwen, b, b["rejected_ids"]); pr.backward(); gr = flat()
        gm += gc - gr; gn += -gc_tok; n += 1
    batch_m.append(gm / n); batch_nll.append(gn / n); used += n
    print(f"배치 {len(batch_m)}/{a.batches} 측정 ({n}쌍)", flush=True)

G = torch.stack(batch_m)                  # K × P
gbar = G.mean(0)
rms = G.pow(2).mean(0).sqrt().clamp_min(1e-12)
delta_per_lr = float((gbar.pow(2) / rms).sum())          # 한 스텝 마진 변화 / lr
l1 = float(gbar.abs().sum())
N = math.ceil(a.n_pairs_total / a.grad_accum)
lr = a.z_target / (a.beta * delta_per_lr * N * 0.5)
lr_sign = a.z_target / (a.beta * l1 * N * 0.5)
gnll = torch.stack(batch_nll).mean(0)
lam = a.beta * 0.5 * float(gbar.norm()) / float(gnll.norm())
res = {"pairs_measured": used, "batches": len(batch_m), "steps_N": N, "beta": a.beta, "z_target": a.z_target,
       "grad_l1_mean": l1, "adam_delta_per_lr": delta_per_lr, "snr_ratio": delta_per_lr / l1,
       "lr_adam_estimate": lr, "lr_signsgd_upper_estimate": lr_sign,
       "dpo_grad_norm_init": a.beta * 0.5 * float(gbar.norm()), "nll_grad_norm": float(gnll.norm()),
       "lambda_balanced": lam}
Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps(res, indent=1), flush=True)
