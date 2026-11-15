"""실험 E용 학습 데이터 = 친구(팀원) 루프 쌍 + D의 ERA 정답>오답 쌍 + D의 보존 데이터.

1) 친구 루프 쌍 326개 (datasets/dpo_r4_loop/train.jsonl, 친구 모델 604/900의 학습 데이터)
   - rejected = 8,192에서 잘린 베이스 응답, rej_loop_start = 루프 시작 '토큰' 위치, seg = 512.
   - 우리 학습 코드(train_dpo_d.py)는 문자 위치(rej_span_char)를 받으므로, 같은 토크나이저로
     rejected를 토큰화해 rej_loop_start 토큰의 시작 문자 위치로 바꾼다. 다시 토큰화하면 같은 토큰이 나오는지 검사한다.
   - 친구는 4 epoch 학습했으므로 --loop-repeat(기본 4)번 복제해서, 1 epoch 동안 같은 횟수만큼 보게 한다.
   - 이미지 경로는 친구 서버 경로(/home/jinu/mmdl)를 이 서버의 압축 해제 위치로 바꾼다.
2) ERA 정답 > 베이스 오답 쌍과 3) 보존 데이터(replay)는 pairs_D.jsonl에서 그대로 가져온다.
"""
import argparse, json, random
from collections import Counter
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--friend", required=True, help="dpo_r4_loop/train.jsonl")
ap.add_argument("--friend-root", required=True, help="친구 묶음을 푼 폴더 (/home/jinu/mmdl 대신)")
ap.add_argument("--pairs-d", required=True)
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--loop-repeat", type=int, default=4)
ap.add_argument("--seed", type=int, default=3407)
ap.add_argument("--out", required=True)
a = ap.parse_args()

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.tokenizer)
rng = random.Random(a.seed)

loops, bad = [], 0
for r in map(json.loads, open(a.friend, encoding="utf-8")):
    t = tok(r["rejected"], add_special_tokens=False, return_offsets_mapping=True)
    ls = r["rej_loop_start"]
    if ls is None or ls >= len(t["input_ids"]):
        bad += 1; continue
    cs = t["offset_mapping"][ls][0]
    # train_dpo_d.py와 같은 방식으로 문자 위치 → 토큰 위치를 되찾아 같은지 확인
    lo = next(k for k, (s0, e0) in enumerate(t["offset_mapping"]) if e0 > cs)
    if lo != ls:
        bad += 1; continue
    img = r["images"][0].replace("/home/jinu/mmdl", a.friend_root)
    assert Path(img).exists(), img
    loops.append({"id": r["id"], "vic_type": "loop", "image": img, "input_prompt": r["input_prompt"],
                  "chosen": r["chosen"], "rejected": r["rejected"], "rejected_truncated": True,
                  "rej_span_char": cs, "rej_span_tokens": r["seg"],
                  "chosen_src": "friend_base", "rejected_src": "friend_loop",
                  "chosen_tokens": r["chosen_tokens"], "rejected_tokens": ls + r["seg"]})

d = [json.loads(l) for l in open(a.pairs_d, encoding="utf-8")]
ew = [r for r in d if r.get("kind") != "replay" and r["chosen_src"] == "era" and r["rejected_src"] == "wrong"]
rep = [r for r in d if r.get("kind") == "replay"]

rows = loops * a.loop_repeat + ew + rep
rng.shuffle(rows)
with open(a.out, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
report = {"rows": len(rows), "friend_loop_pairs": len(loops), "friend_loop_dropped": bad,
          "loop_repeat": a.loop_repeat, "era_wrong_pairs": len(ew), "replay": len(rep),
          "by_vic_type": dict(Counter(r["vic_type"] for r in rows))}
json.dump(report, open(a.out.replace(".jsonl", ".report.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(report, ensure_ascii=False))
