# DPO2-E vs 베이스라인

**DPO2-E: 608/900 = 67.56%** (베이스라인 576/900 = 64.00%, 차이 +32문항 / +3.56%p)

| 과목 | 베이스라인 | 이번 | 차이 |
|---|---:|---:|---:|
| Accounting | 19 | 22 | +3 |
| Agriculture | 15 | 18 | +3 |
| Architecture_and_Engineering | 11 | 20 | +9 |
| Art | 18 | 19 | +1 |
| Art_Theory | 24 | 25 | +1 |
| Basic_Medical_Science | 23 | 21 | -2 |
| Biology | 17 | 17 | +0 |
| Chemistry | 16 | 15 | -1 |
| Clinical_Medicine | 23 | 21 | -2 |
| Computer_Science | 20 | 20 | +0 |
| Design | 24 | 25 | +1 |
| Diagnostics_and_Laboratory_Medicine | 10 | 11 | +1 |
| Economics | 26 | 27 | +1 |
| Electronics | 19 | 19 | +0 |
| Energy_and_Power | 14 | 20 | +6 |
| Finance | 23 | 24 | +1 |
| Geography | 18 | 19 | +1 |
| History | 23 | 22 | -1 |
| Literature | 25 | 25 | +0 |
| Manage | 18 | 17 | -1 |
| Marketing | 26 | 24 | -2 |
| Materials | 15 | 18 | +3 |
| Math | 18 | 20 | +2 |
| Mechanical_Engineering | 11 | 17 | +6 |
| Music | 6 | 10 | +4 |
| Pharmacy | 21 | 22 | +1 |
| Physics | 25 | 26 | +1 |
| Psychology | 24 | 22 | -2 |
| Public_Health | 26 | 26 | +0 |
| Sociology | 18 | 16 | -2 |

## 기타 집계 (summary.json)

```json
{
 "total": 900,
 "graded": 900,
 "correct": 608,
 "incorrect": 292,
 "accuracy": 0.6755555555555556,
 "accuracy_percent": 67.55555555555556,
 "aggregation": "unweighted mean of 30 subject accuracies",
 "method_counts": {
  "explicit_final_parser": 819,
  "local_qwen3_8b_judge_parser": 70,
  "official_parser_fallback": 10,
  "output_limit_incorrect": 1
 },
 "length_limit_incorrect": 1,
 "judge_count": 70,
 "judge_states": {
  "valid_option": 51,
  "invalid_option": 1,
  "no_match": 18
 },
 "random_fallback": 0,
 "exact_official_report_reproduction": false
}
```
