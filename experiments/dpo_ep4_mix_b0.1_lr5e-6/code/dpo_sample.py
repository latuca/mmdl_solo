"""DPO용 응답 샘플링 (평가 venv: vLLM).

학습 문항(SFT와 같은 train.jsonl의 input_prompt + 이미지)에 대해 모델이 응답을 n개씩 생성한다.
- 프롬프트·이미지 처리·해상도·샘플링 설정(temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5)은 평가와 같다.
- 평가와 다른 점: 여러 문항을 묶어서 생성(속도), 응답 길이 상한 = max_total - 입력 토큰
  (DPO 학습 최대 길이를 넘는 응답은 어차피 학습에 못 쓰기 때문).
- 결과는 문항마다 한 줄씩 jsonl에 추가하며, 다시 실행하면 끝난 문항은 건너뛴다.
"""
import argparse, json, os, sys, time
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="vLLM으로 띄울 모델 폴더 또는 HF repo")
    ap.add_argument("--revision")
    ap.add_argument("--tag", required=True, help="응답 출처 이름 (ep4 / base)")
    ap.add_argument("--data", required=True, help="SFT train.jsonl (id, image, input_prompt)")
    ap.add_argument("--images", required=True)
    ap.add_argument("--eval-src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-total", type=int, default=8192)
    ap.add_argument("--chunk", type=int, default=32, help="한 번에 vLLM에 넣는 문항 수")
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    sys.path.insert(0, a.eval_src)
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    c = json.load(open(Path(a.eval_src) / "config.json"))
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    out = Path(a.out)
    done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    pending = [r for r in rows if r["id"] not in done]
    print(f"[{a.tag}] 전체 {len(rows)}문항, 남은 {len(pending)}문항, 문항당 {a.n}개", flush=True)
    if not pending:
        sys.exit(0)

    proc = AutoProcessor.from_pretrained(a.model, revision=a.revision)
    llm = LLM(model=a.model, revision=a.revision, dtype="bfloat16", max_model_len=a.max_total,
              gpu_memory_utilization=0.90, max_num_seqs=a.max_num_seqs, max_num_batched_tokens=16384,
              enforce_eager=False, limit_mm_per_prompt={"image": 5}, seed=c["sampling"]["seed"],
              mm_processor_kwargs={"size": {"shortest_edge": c["vision"]["min_pixels"],
                                            "longest_edge": c["vision"]["max_pixels"]}, "do_resize": False})


    def build(r):
        img = Image.open(Path(a.images) / r["image"]).convert("RGB")
        msgs = [{"role": "user", "content": [dict(type="image", image=img, **c["vision"]),
                                             {"type": "text", "text": r["input_prompt"]}]}]
        rendered = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        images, _, kw = process_vision_info(msgs, image_patch_size=proc.image_processor.patch_size,
                                            return_video_kwargs=True, return_video_metadata=True)
        kw["do_resize"] = False
        n_in = int(proc(text=[rendered], images=images, return_tensors="pt", **kw)["input_ids"].shape[-1])
        return {"prompt": rendered, "multi_modal_data": {"image": images}, "mm_processor_kwargs": kw}, n_in


    t0, n_done = time.time(), 0
    with open(out, "a", encoding="utf-8") as f:
        for s in range(0, len(pending), a.chunk):
            batch = pending[s:s + a.chunk]
            reqs, params, kept = [], [], []
            for r in batch:
                req, n_in = build(r)
                room = a.max_total - n_in
                if room < 256:
                    f.write(json.dumps({"id": r["id"], "tag": a.tag, "input_tokens": n_in, "skipped": "prompt_too_long",
                                        "samples": []}, ensure_ascii=False) + "\n")
                    continue
                reqs.append(req); kept.append((r, n_in))
                params.append(SamplingParams(**c["sampling"], n=a.n, max_tokens=room, stop_token_ids=[]))
            outs = llm.generate(reqs, params, use_tqdm=False) if reqs else []
            for (r, n_in), o in zip(kept, outs):
                if len(o.prompt_token_ids) != n_in:
                    raise RuntimeError(f"processor/vLLM 입력 토큰 수 불일치: {r['id']}")
                samples = [{"raw": x.text, "finish_reason": x.finish_reason, "output_tokens": len(x.token_ids)}
                           for x in o.outputs]
                f.write(json.dumps({"id": r["id"], "tag": a.tag, "input_tokens": n_in, "samples": samples},
                                   ensure_ascii=False) + "\n")
            f.flush()
            n_done += len(batch)
            el = time.time() - t0
            print(f"[{a.tag}] {len(done) + n_done}/{len(rows)}문항, {el / 60:.1f}분 경과, "
                  f"남은 예상 {el / n_done * (len(pending) - n_done) / 60:.0f}분", flush=True)
    print(f"[{a.tag}] 샘플링 완료", flush=True)


if __name__ == "__main__":
    main()
