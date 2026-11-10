"""held-out 20문항에서 응답 첫 토큰 분포를 잰다: P(첫 토큰 = ERA 시작 토큰 'Evidence')와 top-1 첫 토큰 빈도.
학습된 모델이 ERA로 시작하는 쪽으로 얼마나 움직였는지(형식 학습 여부)를 생성 없이 직접 본다.
  python train/first_token_prob.py --exp era_lora_pilot --models base G E
"""
import argparse, collections, json, os, sys
import torch
from transformers import AutoModelForImageTextToText
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import era_data, gen_heldout
from era_data import ROOT

ap = argparse.ArgumentParser(); ap.add_argument("--exp", default="era_lora_pilot")
ap.add_argument("--models", nargs="+", default=["base", "G", "E"]); args = ap.parse_args()
proc = era_data.load_processor(); tok = proc.tokenizer
_, held, found, _ = era_data.prepare("E")
ev_id = tok.encode("Evidence", add_special_tokens=False)[0]
print("first token id of 'Evidence':", ev_id, repr(tok.decode([ev_id])), "| full:", tok.encode("Evidence:", add_special_tokens=False))
inputs = []
for r in held:
    prompt, image = gen_heldout.build_inputs(proc, r, found[r["sample_id"]])
    inputs.append(proc(text=[prompt], images=[image], return_tensors="pt", do_resize=False))
res = {}
for m in args.models:
    path, kw = gen_heldout.model_path(m, args.exp)
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, **kw).cuda().eval()
    p_ev, top1 = [], collections.Counter()
    with torch.no_grad():
        for enc in inputs:
            logits = model(**enc.to("cuda")).logits[0, -1].float()
            pr = torch.softmax(logits, -1)
            p_ev.append(pr[ev_id].item()); top1[tok.decode([int(pr.argmax())])] += 1
    res[m] = {"mean_P(Evidence)": round(sum(p_ev) / len(p_ev), 5), "max_P(Evidence)": round(max(p_ev), 5),
              "top1_first_tokens": dict(top1.most_common(6))}
    print(m, json.dumps(res[m], ensure_ascii=False)); del model; torch.cuda.empty_cache()
os.makedirs(os.path.join(ROOT, "outputs", args.exp, "stage5"), exist_ok=True)
json.dump(res, open(os.path.join(ROOT, "outputs", args.exp, "stage5", "first_token.json"), "w"), indent=1, ensure_ascii=False)
