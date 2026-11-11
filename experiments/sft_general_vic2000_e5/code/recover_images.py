"""Recover missing vic_era_2000 images from the original HF source datasets.

- MMFineReason (sft split): matched by mmfinereason_source.row_index, verified by id + question.
- OpenMedReason (train split): no ids, so matched on exact (question, answer, reasoning==explanation).
Images are written to <data_dir>/<image_1> as RGB JPEG. Existing files are never overwritten.
A report is written to <data_dir>/recover_report.json.
"""
import argparse, io, json, os, sys, glob
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image

MMFR = "OpenDataArena/MMFineReason-1.8M-Qwen3-VL-235B-Thinking"
OMR = "researcher2026/OpenMedReason"


def save_image(img_cell, dest):
    raw = img_cell["bytes"] if isinstance(img_cell, dict) else img_cell
    im = Image.open(io.BytesIO(raw))
    im.load()
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".tmp"
    im.convert("RGB").save(tmp, format="JPEG", quality=95)
    os.replace(tmp, dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cache", required=True, help="download dir for parquet shards")
    a = ap.parse_args()
    rows = json.load(open(os.path.join(a.data_dir, "data.json")))
    missing = [r for r in rows if not os.path.exists(os.path.join(a.data_dir, r["image_1"]))]
    print(f"missing {len(missing)}", flush=True)
    report = {"recovered": [], "mismatch": [], "not_found": []}

    # ---- MMFineReason: walk sft shards in order, tracking global row offset
    need = {r["mmfinereason_source"]["row_index"]: r for r in missing if r["mmfinereason_source"]}
    if need:
        files = sorted(f for f in list_repo_files(MMFR, repo_type="dataset") if f.startswith("data/sft-"))
        offset, max_need = 0, max(need)
        for f in files:
            if offset > max_need:
                break
            p = hf_hub_download(MMFR, f, repo_type="dataset", cache_dir=a.cache)
            pf = pq.ParquetFile(p)
            n = pf.metadata.num_rows
            hit = [i for i in need if offset <= i < offset + n]
            print(f"{f}: rows {offset}-{offset+n-1}, hits {len(hit)}", flush=True)
            if hit:
                t = pf.read(columns=["id", "question", "image"]).to_pydict()
                for gi in hit:
                    r, li = need[gi], gi - offset
                    src = r["mmfinereason_source"]
                    if str(t["id"][li]) != str(src["id"]) or t["question"][li].strip() != r["question"].strip():
                        report["mismatch"].append({"id": r["id"], "row_index": gi})
                        continue
                    img = t["image"][li]
                    img = img[0] if isinstance(img, list) else img
                    save_image(img, os.path.join(a.data_dir, r["image_1"]))
                    report["recovered"].append({"id": r["id"], "source": "MMFineReason", "row_index": gi})
            offset += n

    # ---- OpenMedReason: exact (question, answer, reasoning) match across train shards
    norm = lambda s: (s or "").strip()
    need2 = {}
    for r in missing:
        if not r["mmfinereason_source"]:
            need2.setdefault((norm(r["question"]), norm(r["answer"]), norm(r["explanation"])), []).append(r)
    if need2:
        files = sorted(f for f in list_repo_files(OMR, repo_type="dataset") if f.startswith("data/train-"))
        found = set()
        for f in files:
            p = hf_hub_download(OMR, f, repo_type="dataset", cache_dir=a.cache)
            pf = pq.ParquetFile(p)
            t = pf.read(columns=["question", "answer", "reasoning"]).to_pydict()
            idx = [i for i in range(len(t["question"]))
                   if (norm(t["question"][i]), norm(t["answer"][i]), norm(t["reasoning"][i])) in need2]
            idx = [i for i in idx if (norm(t["question"][i]), norm(t["answer"][i]), norm(t["reasoning"][i])) not in found]
            print(f"{f}: hits {len(idx)}", flush=True)
            if idx:
                imgs = pf.read(columns=["image"]).column("image").to_pylist()
                for i in idx:
                    k = (norm(t["question"][i]), norm(t["answer"][i]), norm(t["reasoning"][i]))
                    if k in found:
                        continue
                    found.add(k)
                    for r in need2[k]:
                        save_image(imgs[i], os.path.join(a.data_dir, r["image_1"]))
                        report["recovered"].append({"id": r["id"], "source": "OpenMedReason", "shard": f, "row": i})
            if len(found) == len(need2):
                break
        for k, rs in need2.items():
            if k not in found:
                report["not_found"] += [r["id"] for r in rs]

    still = [r["id"] for r in rows if not os.path.exists(os.path.join(a.data_dir, r["image_1"]))]
    report["still_missing"] = still
    json.dump(report, open(os.path.join(a.data_dir, "recover_report.json"), "w"), indent=1)
    print(f"recovered {len(report['recovered'])}, mismatch {len(report['mismatch'])}, "
          f"not_found {len(report['not_found'])}, still_missing {len(still)}", flush=True)


if __name__ == "__main__":
    main()
