"""4단계: LoRA adapter를 base에 merge해 HF 형식으로 저장하고, 원본과 대조 검증한다.

실행 (mmdl_train 환경):
  python train/merge_lora.py --cond E --exp era_lora_pilot

저장: outputs/<exp>/<cond>/merged/ (모델 + processor + tokenizer + chat template), merge_report.json
검증: (1) merge 전후 logits 차이 (2) 가중치 키가 원본과 동일하고 lora 키가 없음
      (3) 비전 인코더 가중치가 원본과 비트 단위로 동일 (4) tokenizer/processor/chat template 파일 hash를 원본과 대조
"""
import argparse
import hashlib
import json
import os
import sys

import torch
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoModelForImageTextToText
from huggingface_hub import snapshot_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import era_data  # noqa: E402
from era_data import MODEL_REPO, MODEL_REV, ROOT, sha256_file  # noqa: E402

# 토크나이저/프로세서/채팅 템플릿에 해당하는 파일들 (원본 snapshot에 있는 것만 대조)
TEXT_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.json",
              "chat_template.jinja", "preprocessor_config.json", "video_preprocessor_config.json",
              "special_tokens_map.json", "added_tokens.json", "generation_config.json"]


def sha(path):
    return sha256_file(path) if os.path.exists(path) else None


COPY_FROM_BASE = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.json",
                  "preprocessor_config.json", "video_preprocessor_config.json", "generation_config.json"]
REMOVE_REWRITTEN = ["added_tokens.json", "special_tokens_map.json", "chat_template.jinja"]  # save_pretrained가 새로 만든 파일


def restore_original_text_files(base_dir, out):
    """save_pretrained는 tokenizer/chat template을 새 형식으로 다시 쓴다. 원본 snapshot의 파일로 바이트 단위 복원한다."""
    import shutil
    for fn in REMOVE_REWRITTEN:
        if os.path.exists(os.path.join(out, fn)) and not os.path.exists(os.path.join(base_dir, fn)):
            os.remove(os.path.join(out, fn))
    for fn in COPY_FROM_BASE:
        src = os.path.join(base_dir, fn)
        if os.path.exists(src):
            shutil.copyfile(os.path.realpath(src), os.path.join(out, fn))


def weight_keys(d):
    keys = {}
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".safetensors"):
            with safe_open(os.path.join(d, fn), "pt") as f:
                for k in f.keys():
                    keys[k] = fn
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", choices=["G", "E"], required=True)
    ap.add_argument("--exp", default="era_lora_pilot")
    ap.add_argument("--fix-files-only", action="store_true", help="이미 만든 merged 폴더의 tokenizer/template 파일만 원본으로 복원하고 hash 대조")
    args = ap.parse_args()

    cdir = os.path.join(ROOT, "outputs", args.exp, args.cond)
    adapter, out = os.path.join(cdir, "adapter"), os.path.join(cdir, "merged")
    if not os.path.isdir(adapter):
        raise SystemExit(f"no adapter: {adapter}")
    base_dir = snapshot_download(MODEL_REPO, revision=MODEL_REV, local_files_only=True)
    if args.fix_files_only:
        restore_original_text_files(base_dir, out)
        cmp = {fn: sha(os.path.join(base_dir, fn)) == sha(os.path.join(out, fn)) for fn in TEXT_FILES
               if os.path.exists(os.path.join(base_dir, fn)) or os.path.exists(os.path.join(out, fn))}
        extra = [fn for fn in TEXT_FILES if os.path.exists(os.path.join(out, fn)) and not os.path.exists(os.path.join(base_dir, fn))]
        print(json.dumps({"identical": cmp, "files_only_in_merged": extra}, indent=1))
        rp = os.path.join(cdir, "merge_report.json")
        r = json.load(open(rp))
        r["files_after_restore"] = cmp
        json.dump(r, open(rp, "w"), indent=1, ensure_ascii=False)
        return
    if os.path.exists(out):
        raise SystemExit(f"already exists: {out} (덮어쓰지 않음)")
    print("base snapshot:", base_dir)

    proc = era_data.load_processor()
    train, heldout, found, _ = era_data.prepare(args.cond, proc)
    rec = heldout[0]
    ex = era_data.build_example(proc, rec, args.cond, found[rec["sample_id"]])
    batch = {k: (ex[k].unsqueeze(0) if k in ("input_ids", "attention_mask") else ex[k]).cuda()
             for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}

    base = AutoModelForImageTextToText.from_pretrained(MODEL_REPO, revision=MODEL_REV, dtype=torch.bfloat16).cuda()
    model = PeftModel.from_pretrained(base, adapter).eval()
    with torch.no_grad():
        before = model(**batch).logits.float().cpu()

    merged = model.merge_and_unload().eval()
    with torch.no_grad():
        after = merged(**batch).logits.float().cpu()
    diff = (before - after).abs()
    agree = (before.argmax(-1) == after.argmax(-1)).float().mean().item()
    print(f"logits max|diff| {diff.max():.4f} mean {diff.mean():.5f}; argmax agree {agree:.4%}")

    merged.save_pretrained(out, safe_serialization=True)
    proc.save_pretrained(out)
    restore_original_text_files(base_dir, out)

    # (2) 키 대조
    bk, mk = weight_keys(base_dir), weight_keys(out)
    only_base, only_merged = sorted(set(bk) - set(mk)), sorted(set(mk) - set(bk))
    lora_keys = [k for k in mk if "lora" in k.lower()]
    # (3) 비전 인코더 비트 단위 동일성 + LLM에서 바뀐 텐서 수
    n_visual_same = n_visual = n_changed = n_llm = 0
    changed_names = set()
    for k in sorted(set(bk) & set(mk)):
        with safe_open(os.path.join(base_dir, bk[k]), "pt") as fb, safe_open(os.path.join(out, mk[k]), "pt") as fm:
            same = torch.equal(fb.get_tensor(k), fm.get_tensor(k))
        if "visual" in k:
            n_visual += 1
            n_visual_same += same
        elif "language_model" in k:
            n_llm += 1
            if not same:
                n_changed += 1
                changed_names.add(k.rsplit(".", 2)[-2])
    # (4) 파일 hash 대조
    file_cmp = {fn: {"base": sha(os.path.join(base_dir, fn)), "merged": sha(os.path.join(out, fn))} for fn in TEXT_FILES}
    file_cmp = {fn: v for fn, v in file_cmp.items() if v["base"] or v["merged"]}
    for v in file_cmp.values():
        v["identical"] = v["base"] == v["merged"]
    # hash가 달라도 의미가 같은지: 렌더링 결과와 토큰화 결과를 직접 대조
    from transformers import AutoProcessor
    p0 = AutoProcessor.from_pretrained(base_dir)
    p1 = AutoProcessor.from_pretrained(out)
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": rec["prompt"]}]}]
    same_render = p0.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) == \
        p1.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    probe = rec["prompt"] + rec[era_data.TARGET_KEY[args.cond]]
    same_tok = p0.tokenizer(probe)["input_ids"] == p1.tokenizer(probe)["input_ids"]

    report = {
        "cond": args.cond, "base_revision": MODEL_REV, "adapter": os.path.relpath(adapter, ROOT),
        "adapter_sha256": sha(os.path.join(adapter, "adapter_model.safetensors")),
        "logits_max_abs_diff": diff.max().item(), "logits_mean_abs_diff": diff.mean().item(), "argmax_agree": agree,
        "keys_only_in_base": only_base[:10], "keys_only_in_merged": only_merged[:10], "lora_keys_in_merged": lora_keys[:10],
        "visual_tensors": n_visual, "visual_bitwise_identical": n_visual_same,
        "llm_tensors": n_llm, "llm_tensors_changed": n_changed, "llm_changed_module_types": sorted(changed_names),
        "files": file_cmp, "chat_template_render_identical": same_render, "tokenization_identical": same_tok,
    }
    with open(os.path.join(cdir, "merge_report.json"), "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=1, ensure_ascii=False))
    for fn, v in file_cmp.items():
        print(f"  {fn:32s} {'IDENTICAL' if v['identical'] else 'DIFFERENT'}")


if __name__ == "__main__":
    main()
