"""4단계 보조: 응답 토큰 loss를 base / adapter(merge 전) / merged로 비교해 merge가 adapter 효과를 보존하는지 본다.
held-out 20문항(학습에 쓰지 않음)과 학습 샘플 20개 각각에 대해 계산한다."""
import json, os, sys, argparse
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import era_data
from era_data import MODEL_REPO, MODEL_REV, ROOT

ap = argparse.ArgumentParser(); ap.add_argument("--cond", required=True, choices=["G", "E"]); args = ap.parse_args()
proc = era_data.load_processor()
train, held, found, _ = era_data.prepare(args.cond, proc)
sets = {"heldout": held, "train20": train[:20]}
exs = {k: [era_data.build_example(proc, r, args.cond, found[r["sample_id"]]) for r in v] for k, v in sets.items()}
dev = lambda ex: {k: (ex[k].unsqueeze(0) if k in ("input_ids", "attention_mask", "labels") else ex[k]).cuda()
                  for k in ("input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw")}

@torch.no_grad()
def losses(model):
    out = {}
    for name, lst in exs.items():
        out[name] = [model(**dev(e)).loss.item() for e in lst]
    return out

res = {}
base = AutoModelForImageTextToText.from_pretrained(MODEL_REPO, revision=MODEL_REV, dtype=torch.bfloat16).cuda().eval()
pm = PeftModel.from_pretrained(base, os.path.join(ROOT, "outputs", "era_lora_pilot", args.cond, "adapter")).eval()
with pm.disable_adapter():
    res["base"] = losses(pm)
res["adapter_unmerged"] = losses(pm)
del pm, base; torch.cuda.empty_cache()
mm = AutoModelForImageTextToText.from_pretrained(os.path.join(ROOT, "outputs", "era_lora_pilot", args.cond, "merged"), dtype=torch.bfloat16).cuda().eval()
res["merged"] = losses(mm)
mean = lambda x: sum(x) / len(x)
summary = {m: {s: round(mean(v), 4) for s, v in d.items()} for m, d in res.items()}
print(args.cond, json.dumps(summary, indent=1))
json.dump({"summary": summary, "per_sample": res}, open(os.path.join(ROOT, "outputs", "era_lora_pilot", "stage4", f"fidelity_{args.cond}.json"), "w"), indent=1)
