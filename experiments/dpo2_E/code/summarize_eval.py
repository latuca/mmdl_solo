"""체크포인트 평가 결과(summary.json)를 베이스라인 64.00%(576/900)와 비교한 표를 만든다.
베이스라인 과목별 정답 수는 mmdl_test/reports/mmmu_baseline.md 표 기준."""
import argparse, json
from pathlib import Path

BASE = {"Accounting": 19, "Agriculture": 15, "Architecture_and_Engineering": 11, "Art": 18, "Art_Theory": 24,
        "Basic_Medical_Science": 23, "Biology": 17, "Chemistry": 16, "Clinical_Medicine": 23,
        "Computer_Science": 20, "Design": 24, "Diagnostics_and_Laboratory_Medicine": 10, "Economics": 26,
        "Electronics": 19, "Energy_and_Power": 14, "Finance": 23, "Geography": 18, "History": 23,
        "Literature": 25, "Manage": 18, "Marketing": 26, "Materials": 15, "Math": 18,
        "Mechanical_Engineering": 11, "Music": 6, "Pharmacy": 21, "Physics": 25, "Psychology": 24,
        "Public_Health": 26, "Sociology": 18}

ap = argparse.ArgumentParser()
ap.add_argument("--eval-dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--label", default="")
a = ap.parse_args()
s = json.load(open(Path(a.eval_dir) / "summary.json"))
c = s["correct"]
lines = [f"# {a.label} vs 베이스라인",
         "",
         f"**{a.label}: {c}/900 = {c/9:.2f}%** (베이스라인 576/900 = 64.00%, 차이 {c-576:+d}문항 / {(c-576)/9:+.2f}%p)",
         "",
         "| 과목 | 베이스라인 | 이번 | 차이 |", "|---|---:|---:|---:|"]
for subj, b in BASE.items():
    n = s["by_subject"][subj]["correct"]
    lines.append(f"| {subj} | {b} | {n} | {n-b:+d} |")
lines += ["", "## 기타 집계 (summary.json)", "", "```json",
          json.dumps({k: v for k, v in s.items() if k != "by_subject"}, ensure_ascii=False, indent=1), "```"]
Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
print(lines[2])
