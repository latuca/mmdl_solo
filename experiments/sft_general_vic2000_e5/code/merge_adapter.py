"""LoRA 어댑터를 베이스 모델에 병합해 평가용 BF16 전체 모델 폴더를 만든다."""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_sft import merge_and_save

ap = argparse.ArgumentParser()
ap.add_argument("--baseline", required=True)
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
s = json.load(open(a.baseline))["evaluation"]["settings"]["model"]
print("copied:", merge_and_save(s["repo"], s["revision"], a.adapter, a.out), "->", a.out, flush=True)
