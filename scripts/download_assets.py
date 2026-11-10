"""MMMU validation 30개 config + 두 모델을 고정 revision으로 미리 받아 둔다.
실행: python download_assets.py
"""
from datasets import load_dataset
from huggingface_hub import snapshot_download

from config import CONFIGS, DS_REPO, DS_REV, MODEL_REPO, MODEL_REV, JUDGE_REPO, JUDGE_REV

total = 0
for cfg in CONFIGS:
    ds = load_dataset(DS_REPO, cfg, split="validation", revision=DS_REV)
    total += len(ds)
    print(f"[data] {cfg}: {len(ds)}", flush=True)
print(f"[data] total = {total} (기대값 900)")

for repo, rev in [(MODEL_REPO, MODEL_REV), (JUDGE_REPO, JUDGE_REV)]:
    path = snapshot_download(repo_id=repo, revision=rev)
    print(f"[model] {repo} -> {path}", flush=True)
