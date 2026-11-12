# epoch 4 vs 베이스라인

**epoch 4: 510/900 = 56.67%** (베이스라인 576/900 = 64.00%, 차이 -66문항 / -7.33%p)

| 과목 | 베이스라인 | 이번 | 차이 |
|---|---:|---:|---:|
| Accounting | 19 | 20 | +1 |
| Agriculture | 15 | 15 | +0 |
| Architecture_and_Engineering | 11 | 12 | +1 |
| Art | 18 | 20 | +2 |
| Art_Theory | 24 | 22 | -2 |
| Basic_Medical_Science | 23 | 20 | -3 |
| Biology | 17 | 14 | -3 |
| Chemistry | 16 | 11 | -5 |
| Clinical_Medicine | 23 | 13 | -10 |
| Computer_Science | 20 | 13 | -7 |
| Design | 24 | 23 | -1 |
| Diagnostics_and_Laboratory_Medicine | 10 | 8 | -2 |
| Economics | 26 | 18 | -8 |
| Electronics | 19 | 14 | -5 |
| Energy_and_Power | 14 | 13 | -1 |
| Finance | 23 | 18 | -5 |
| Geography | 18 | 15 | -3 |
| History | 23 | 20 | -3 |
| Literature | 25 | 27 | +2 |
| Manage | 18 | 16 | -2 |
| Marketing | 26 | 21 | -5 |
| Materials | 15 | 18 | +3 |
| Math | 18 | 20 | +2 |
| Mechanical_Engineering | 11 | 17 | +6 |
| Music | 6 | 8 | +2 |
| Pharmacy | 21 | 19 | -2 |
| Physics | 25 | 16 | -9 |
| Psychology | 24 | 17 | -7 |
| Public_Health | 26 | 24 | -2 |
| Sociology | 18 | 18 | +0 |

## 기타 집계 (summary.json)

```json
{
 "total": 900,
 "graded": 900,
 "correct": 510,
 "incorrect": 390,
 "accuracy": 0.5666666666666667,
 "accuracy_percent": 56.666666666666664,
 "aggregation": "unweighted mean of 30 subject accuracies",
 "method_counts": {
  "explicit_final_parser": 848,
  "local_qwen3_8b_judge_parser": 43,
  "official_parser_fallback": 5,
  "output_limit_incorrect": 4
 },
 "length_limit_incorrect": 4,
 "judge_count": 43,
 "judge_states": {
  "no_match": 24,
  "valid_option": 19
 },
 "random_fallback": 0,
 "exact_official_report_reproduction": false
}
```
