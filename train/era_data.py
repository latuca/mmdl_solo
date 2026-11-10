"""ERA LoRA-SFT 데이터 준비: 필터링, held-out 분할, 이미지 매칭, 렌더링/토큰화, loss mask.

입력은 평가(scripts/infer.py)와 같은 chat template + 이미지 1장 + JSONL의 prompt 그대로이며,
"ERA로 풀어라" 같은 지시는 넣지 않는다. 조건 G/E는 목표 응답만 다르다.

1단계 입력 점검(대표 8문항, owner별 2개):
  python train/era_data.py --inspect --out outputs/era_lora_pilot/stage1
"""
import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict

from PIL import Image
from qwen_vl_utils.vision_process import smart_resize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from config import MODEL_REPO, MODEL_REV, MIN_PIXELS, MAX_PIXELS, SEED  # noqa: E402

DATA = os.path.join(ROOT, "data", "era_dataset_v1.1_partial_187.jsonl")
IMAGE_DIR = os.path.join(ROOT, "data", "images")
TARGET_KEY = {"G": "general_solution", "E": "era_solution"}
EXCLUDE_STATUS = {"hold_question_edit_undecided"}
FLAG_STATUS = {"awaiting_reviewer_D"}   # 학습에는 포함, 메타데이터에 표시
EXCLUDE_NO_IMAGE = {"C_10598", "C_10611"}   # 이미지 zip(C_50.zip)에 파일이 없어 파일럿에서 제외 (2026-10-28 사용자 결정)
N_HELDOUT = 20
MAX_LEN = 4096
END_TOKEN = "<|im_end|>"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_records(path=DATA):
    rs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    drop = lambda r: r["final_status"] in EXCLUDE_STATUS or r["sample_id"] in EXCLUDE_NO_IMAGE
    kept = [r for r in rs if not drop(r)]
    excluded = [r["sample_id"] for r in rs if drop(r)]
    return kept, excluded


def split_heldout(records, n=N_HELDOUT, seed=SEED):
    """owner별 층화로 n개를 held-out으로 뽑는다 (두 조건 공통, seed 고정)."""
    by_owner = defaultdict(list)
    for r in sorted(records, key=lambda r: r["sample_id"]):
        by_owner[r["owner"]].append(r)
    owners = sorted(by_owner)
    per = {o: n // len(owners) for o in owners}
    for o in owners[: n - sum(per.values())]:
        per[o] += 1
    rng = random.Random(seed)
    held = set()
    for o in owners:
        held.update(r["sample_id"] for r in rng.sample(by_owner[o], per[o]))
    train = [r for r in records if r["sample_id"] not in held]
    heldout = [r for r in records if r["sample_id"] in held]
    return train, heldout


def index_images(image_dir=IMAGE_DIR):
    """basename -> 경로 목록. 같은 basename이 여러 개면 매칭 실패로 취급한다."""
    idx = defaultdict(list)
    for dp, _, fns in os.walk(image_dir):
        for fn in fns:
            idx[fn].append(os.path.join(dp, fn))
    return idx


def match_images(records, idx):
    """basename으로 매칭하되, 같은 basename이 여러 owner에 있으면(예: A/10100.png, C/10100.png)
    상위 폴더까지 일치하는 것만 남긴다."""
    found, missing, ambiguous = {}, [], []
    for r in records:
        hits = idx.get(os.path.basename(r["image_path"]), [])
        if len(hits) > 1:
            tail = os.path.join(*r["image_path"].split("/")[-2:])
            hits = [h for h in hits if h.endswith(os.sep + tail)]
        if len(hits) == 1:
            found[r["sample_id"]] = hits[0]
        elif not hits:
            missing.append(r["sample_id"])
        else:
            ambiguous.append((r["sample_id"], hits))
    return found, missing, ambiguous


def load_image(path):
    # 평가와 같은 smart_resize 규칙 (factor 32, 픽셀 범위 고정)
    im = Image.open(path).convert("RGB")
    w, h = im.size
    nh, nw = smart_resize(h, w, factor=32, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    return im.resize((nw, nh), Image.BICUBIC), (w, h), (nw, nh)


def load_processor():
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(MODEL_REPO, revision=MODEL_REV)


def build_example(processor, rec, cond, image_path):
    """한 문항을 학습 입력으로 만든다. labels는 응답 토큰 + 종료 토큰(<|im_end|>)만 남기고 -100."""
    target = rec[TARGET_KEY[cond]]
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": rec["prompt"]}]}]
    prompt_text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    # 학습 텍스트는 템플릿이 만든 assistant 턴을 <|im_end|>에서 자른 것과 같아야 한다
    templ_full = processor.apply_chat_template(
        msgs + [{"role": "assistant", "content": [{"type": "text", "text": target}]}], tokenize=False)
    full_text = prompt_text + target + END_TOKEN
    assert templ_full.startswith(full_text), f"{rec['sample_id']}: chat template mismatch"

    image, orig_size, new_size = load_image(image_path)
    enc_full = processor(text=[full_text], images=[image], return_tensors="pt", do_resize=False)
    enc_prompt = processor(text=[prompt_text], images=[image], return_tensors="pt", do_resize=False)
    input_ids = enc_full["input_ids"][0]
    n_prompt = enc_prompt["input_ids"].shape[1]
    assert input_ids[:n_prompt].equal(enc_prompt["input_ids"][0]), f"{rec['sample_id']}: prompt prefix mismatch"

    labels = input_ids.clone()
    labels[:n_prompt] = -100
    end_id = processor.tokenizer.convert_tokens_to_ids(END_TOKEN)
    return {
        "sample_id": rec["sample_id"],
        "input_ids": input_ids,
        "attention_mask": enc_full["attention_mask"][0],
        "pixel_values": enc_full["pixel_values"],
        "image_grid_thw": enc_full["image_grid_thw"],
        "labels": labels,
        "meta": {
            "n_tokens": int(input_ids.shape[0]),
            "n_prompt": int(n_prompt),
            "n_image_tokens": int((input_ids == processor.image_token_id).sum()),
            "n_loss_tokens": int((labels != -100).sum()),
            "ends_with_end_token": int(input_ids[-1]) == end_id and int(labels[-1]) == end_id,
            "orig_size": orig_size,
            "resized": new_size,
        },
    }


def prepare(cond, processor=None, image_dir=IMAGE_DIR):
    """조건 cond(G/E)의 train/held-out 문항과 메타데이터를 돌려준다. 이미지 누락 시 중단."""
    records, excluded = load_records()
    train, heldout = split_heldout(records)
    found, missing, ambiguous = match_images(records, index_images(image_dir))
    if missing or ambiguous:
        raise SystemExit(f"image match failed: missing {len(missing)} {missing[:10]}, "
                         f"ambiguous {len(ambiguous)} {ambiguous[:3]}")
    meta = {
        "data_path": os.path.relpath(DATA, ROOT),
        "data_sha256": sha256_file(DATA),
        "condition": cond,
        "target_key": TARGET_KEY[cond],
        "excluded_hold": excluded,
        "heldout_ids": [r["sample_id"] for r in heldout],
        "flag_awaiting_reviewer_D": [r["sample_id"] for r in train if r["final_status"] in FLAG_STATUS],
        "n_train": len(train),
        "owner_counts_train": dict(Counter(r["owner"] for r in train)),
    }
    return train, heldout, found, meta


def inspect(out_dir):
    """1단계: owner별 2문항씩 8개를 사람이 읽을 수 있게 저장한다."""
    from score import explicit_answer, mc_letter
    os.makedirs(out_dir, exist_ok=True)
    proc = load_processor()
    train, _, found, meta = prepare("E", proc)
    rng = random.Random(SEED)
    by_owner = defaultdict(list)
    for r in train:
        by_owner[r["owner"]].append(r)
    picks = [r for o in sorted(by_owner) for r in rng.sample(by_owner[o], 2)]
    rows = []
    for r in picks:
        for cond in ("G", "E"):
            ex = build_example(proc, r, cond, found[r["sample_id"]])
            ids, lab = ex["input_ids"], ex["labels"]
            loss_ids = ids[lab != -100]
            ans = explicit_answer(r[TARGET_KEY[cond]])
            rows.append({
                "sample_id": r["sample_id"], "owner": r["owner"], "cond": cond,
                "final_status": r["final_status"], **ex["meta"],
                "loss_span": [ex["meta"]["n_prompt"], ex["meta"]["n_tokens"]],
                "rendered_input": proc.tokenizer.decode(ids[: ex["meta"]["n_prompt"]]).replace(
                    "<|image_pad|>" * ex["meta"]["n_image_tokens"], f"<|image_pad|>x{ex['meta']['n_image_tokens']}"),
                "loss_text": proc.tokenizer.decode(loss_ids),
                "parsed_answer": (mc_letter(ans) or ans) if ans else None,
                "original_answer": r["original_answer"],
            })
    path = os.path.join(out_dir, "inspect_8.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    for row in rows:
        print(f"{row['sample_id']} [{row['owner']}/{row['cond']}] tok={row['n_tokens']} img_tok={row['n_image_tokens']} "
              f"loss_tok={row['n_loss_tokens']} end={row['ends_with_end_token']} "
              f"parsed={row['parsed_answer']} gold={row['original_answer']} size={row['orig_size']}->{row['resized']}")
    print("saved ->", path)


def length_report(out_dir):
    """전체 학습 문항의 이미지 포함 길이를 재고 MAX_LEN 초과 목록을 저장한다."""
    os.makedirs(out_dir, exist_ok=True)
    proc = load_processor()
    report = {}
    for cond in ("G", "E"):
        train, heldout, found, _ = prepare(cond, proc)
        lens = {r["sample_id"]: build_example(proc, r, cond, found[r["sample_id"]])["meta"]["n_tokens"]
                for r in train + heldout}
        over = sorted((k for k, v in lens.items() if v > MAX_LEN), key=lambda k: -lens[k])
        report[cond] = {"max": max(lens.values()), "longest": max(lens, key=lens.get),
                        "over_max_len": {k: lens[k] for k in over}, "lengths": lens}
        print(f"[{cond}] max={report[cond]['max']} ({report[cond]['longest']}) over {MAX_LEN}: {len(over)} {over[:10]}")
    with open(os.path.join(out_dir, "lengths.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true", help="대표 8문항 입력 점검")
    ap.add_argument("--lengths", action="store_true", help="전 문항 길이/초과 목록")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "era_lora_pilot", "stage1"))
    args = ap.parse_args()
    if args.inspect:
        inspect(args.out)
    if args.lengths:
        length_report(args.out)
    if not (args.inspect or args.lengths):
        records, excluded = load_records()
        train, heldout = split_heldout(records)
        found, missing, ambiguous = match_images(records, index_images())
        print(f"records {len(records)} (excluded {excluded}), train {len(train)}, held-out {len(heldout)} "
              f"{dict(Counter(r['owner'] for r in heldout))}")
        print(f"images matched {len(found)}, missing {len(missing)}, ambiguous {len(ambiguous)}")
        if missing:
            print("missing:", missing)
