# DPO2-E (재현) vs 베이스라인

**DPO2-E (재현): 602/900 = 66.89%** (베이스라인 576/900 = 64.00%, 차이 +26문항 / +2.89%p)

| 과목 | 베이스라인 | 이번 | 차이 |
|---|---:|---:|---:|
| Accounting | 19 | 25 | +6 |
| Agriculture | 15 | 17 | +2 |
| Architecture_and_Engineering | 11 | 18 | +7 |
| Art | 18 | 19 | +1 |
| Art_Theory | 24 | 25 | +1 |
| Basic_Medical_Science | 23 | 23 | +0 |
| Biology | 17 | 14 | -3 |
| Chemistry | 16 | 17 | +1 |
| Clinical_Medicine | 23 | 22 | -1 |
| Computer_Science | 20 | 18 | -2 |
| Design | 24 | 26 | +2 |
| Diagnostics_and_Laboratory_Medicine | 10 | 11 | +1 |
| Economics | 26 | 27 | +1 |
| Electronics | 19 | 17 | -2 |
| Energy_and_Power | 14 | 18 | +4 |
| Finance | 23 | 23 | +0 |
| Geography | 18 | 17 | -1 |
| History | 23 | 21 | -2 |
| Literature | 25 | 24 | -1 |
| Manage | 18 | 19 | +1 |
| Marketing | 26 | 26 | +0 |
| Materials | 15 | 17 | +2 |
| Math | 18 | 22 | +4 |
| Mechanical_Engineering | 11 | 15 | +4 |
| Music | 6 | 12 | +6 |
| Pharmacy | 21 | 23 | +2 |
| Physics | 25 | 22 | -3 |
| Psychology | 24 | 22 | -2 |
| Public_Health | 26 | 26 | +0 |
| Sociology | 18 | 16 | -2 |

## 기타 집계 (summary.json)

```json
{
 "total": 900,
 "graded": 900,
 "correct": 602,
 "incorrect": 298,
 "accuracy": 0.6688888888888889,
 "accuracy_percent": 66.88888888888889,
 "aggregation": "unweighted mean of 30 subject accuracies",
 "method_counts": {
  "explicit_final_parser": 823,
  "local_qwen3_8b_judge_parser": 65,
  "official_parser_fallback": 8,
  "output_limit_incorrect": 4
 },
 "length_limit_incorrect": 4,
 "judge_count": 65,
 "judge_states": {
  "valid_option": 40,
  "invalid_option": 1,
  "no_match": 23,
  "unparsed": 1
 },
 "random_fallback": 0,
 "exact_official_report_reproduction": false
}
```
