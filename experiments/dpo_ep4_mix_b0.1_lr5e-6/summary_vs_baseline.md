# DPO(ep4 시작) vs 베이스라인

**DPO(ep4 시작): 503/900 = 55.89%** (베이스라인 576/900 = 64.00%, 차이 -73문항 / -8.11%p)

| 과목 | 베이스라인 | 이번 | 차이 |
|---|---:|---:|---:|
| Accounting | 19 | 19 | +0 |
| Agriculture | 15 | 17 | +2 |
| Architecture_and_Engineering | 11 | 9 | -2 |
| Art | 18 | 22 | +4 |
| Art_Theory | 24 | 24 | +0 |
| Basic_Medical_Science | 23 | 16 | -7 |
| Biology | 17 | 12 | -5 |
| Chemistry | 16 | 11 | -5 |
| Clinical_Medicine | 23 | 15 | -8 |
| Computer_Science | 20 | 16 | -4 |
| Design | 24 | 22 | -2 |
| Diagnostics_and_Laboratory_Medicine | 10 | 12 | +2 |
| Economics | 26 | 20 | -6 |
| Electronics | 19 | 16 | -3 |
| Energy_and_Power | 14 | 19 | +5 |
| Finance | 23 | 18 | -5 |
| Geography | 18 | 13 | -5 |
| History | 23 | 21 | -2 |
| Literature | 25 | 23 | -2 |
| Manage | 18 | 19 | +1 |
| Marketing | 26 | 21 | -5 |
| Materials | 15 | 10 | -5 |
| Math | 18 | 17 | -1 |
| Mechanical_Engineering | 11 | 11 | +0 |
| Music | 6 | 8 | +2 |
| Pharmacy | 21 | 22 | +1 |
| Physics | 25 | 15 | -10 |
| Psychology | 24 | 16 | -8 |
| Public_Health | 26 | 20 | -6 |
| Sociology | 18 | 19 | +1 |

## 기타 집계 (summary.json)

```json
{
 "total": 900,
 "graded": 900,
 "correct": 503,
 "incorrect": 397,
 "accuracy": 0.5588888888888889,
 "accuracy_percent": 55.888888888888886,
 "aggregation": "unweighted mean of 30 subject accuracies",
 "method_counts": {
  "explicit_final_parser": 846,
  "output_limit_incorrect": 4,
  "official_parser_fallback": 6,
  "local_qwen3_8b_judge_parser": 44
 },
 "length_limit_incorrect": 4,
 "judge_count": 44,
 "judge_states": {
  "valid_option": 19,
  "no_match": 25
 },
 "random_fallback": 0,
 "exact_official_report_reproduction": false
}
```
