"""predictions.jsonl + results.json → 문항별 CSV (모델 답 / 정답 / 정오 / 응답 원문)."""
import argparse, csv, json, re

ap = argparse.ArgumentParser()
ap.add_argument("--pred", default="../results/predictions.jsonl")
ap.add_argument("--res", default="../results/results.json")
ap.add_argument("--out", default="../results/mmmu_answers.csv")
args = ap.parse_args()

FINAL_RE = re.compile(r"(?im)^\W*final answer\W*:\s*(.+?)\s*$")

preds = {}
for line in open(args.pred, encoding="utf-8"):
    if line.strip():
        r = json.loads(line)
        preds[r["id"]] = r
graded = {g["id"]: g for g in json.load(open(args.res, encoding="utf-8"))["records"]}

def model_answer(g, raw):
    if g.get("extracted"):
        return g["extracted"]
    hits = FINAL_RE.findall(raw.rsplit("</think>", 1)[-1])
    return re.sub(r"[*_`#>\s]+", " ", hits[-1]).strip(" .:;") if hits else ""

cols = ["id", "subject", "question_type", "reference_answer", "model_answer",
        "model_answer_text", "correct", "method", "finish_reason", "output_tokens", "response"]
with open(args.out, "w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig: 엑셀에서 바로 열림
    w = csv.writer(f)
    w.writerow(cols)
    for rid, r in preds.items():
        g = graded.get(rid, {})
        ans = model_answer(g, r["raw"])
        text = r["options"].get(ans, "") if r["question_type"] == "multiple-choice" else ""
        w.writerow([rid, r["subject"], r["question_type"], r["reference_answer"], ans, text,
                    g.get("correct"), g.get("method"), r["finish_reason"], r["output_tokens"], r["raw"]])
print(f"-> {args.out} ({len(preds)} rows)")
