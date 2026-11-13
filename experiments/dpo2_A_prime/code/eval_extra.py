"""평가 결과의 보조 지표: 답변 길이, 출력 한도 초과(생각하다 끊김), 정상 추출 문항 정답률.
베이스라인 값(mmdl_test/reports/mmmu_baseline.md): 한도 초과 81, explicit parser 539/759."""
import argparse, json, statistics as st
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--eval-dir", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
R = json.load(open(Path(a.eval_dir) / "results.json"))["records"]
ok = lambda r: bool(r["grading"]["final"]["correct"])
toks = [r["model_response"]["output_tokens"] for r in R]
ex = [r for r in R if r["grading"]["final"]["method"] == "explicit_final_parser"]
res = {"correct": sum(map(ok, R)), "n": len(R),
       "output_tokens_median": st.median(toks), "output_tokens_mean": round(st.mean(toks)),
       "output_limit_truncated": sum(r["model_response"]["finish_reason"] == "length" for r in R),
       "explicit_parser_n": len(ex), "explicit_parser_correct": sum(map(ok, ex)),
       "explicit_parser_acc": round(sum(map(ok, ex)) / max(1, len(ex)), 4),
       "baseline_ref": {"output_limit_truncated": 81, "explicit_parser_n": 759, "explicit_parser_correct": 539,
                        "explicit_parser_acc": 0.7101}}
Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps(res))
