# epoch 5 vs 베이스라인

**epoch 5: 505/900 = 56.11%** (베이스라인 576/900 = 64.00%, 차이 -71문항 / -7.89%p)

| 과목 | 베이스라인 | 이번 | 차이 |
|---|---:|---:|---:|
| Accounting | 19 | 19 | +0 |
| Agriculture | 15 | 15 | +0 |
| Architecture_and_Engineering | 11 | 13 | +2 |
| Art | 18 | 17 | -1 |
| Art_Theory | 24 | 23 | -1 |
| Basic_Medical_Science | 23 | 19 | -4 |
| Biology | 17 | 17 | +0 |
| Chemistry | 16 | 16 | +0 |
| Clinical_Medicine | 23 | 13 | -10 |
| Computer_Science | 20 | 14 | -6 |
| Design | 24 | 21 | -3 |
| Diagnostics_and_Laboratory_Medicine | 10 | 10 | +0 |
| Economics | 26 | 22 | -4 |
| Electronics | 19 | 16 | -3 |
| Energy_and_Power | 14 | 11 | -3 |
| Finance | 23 | 18 | -5 |
| Geography | 18 | 10 | -8 |
| History | 23 | 21 | -2 |
| Literature | 25 | 23 | -2 |
| Manage | 18 | 19 | +1 |
| Marketing | 26 | 23 | -3 |
| Materials | 15 | 12 | -3 |
| Math | 18 | 15 | -3 |
| Mechanical_Engineering | 11 | 16 | +5 |
| Music | 6 | 8 | +2 |
| Pharmacy | 21 | 17 | -4 |
| Physics | 25 | 20 | -5 |
| Psychology | 24 | 16 | -8 |
| Public_Health | 26 | 19 | -7 |
| Sociology | 18 | 22 | +4 |

## 기타 집계 (summary.json)

```json
{
 "total": 900,
 "graded": 900,
 "correct": 505,
 "incorrect": 395,
 "accuracy": 0.5611111111111111,
 "accuracy_percent": 56.111111111111114,
 "aggregation": "unweighted mean of 30 subject accuracies",
 "method_counts": {
  "explicit_final_parser": 852,
  "output_limit_incorrect": 3,
  "local_qwen3_8b_judge_parser": 41,
  "official_parser_fallback": 4
 },
 "length_limit_incorrect": 3,
 "judge_count": 41,
 "judge_states": {
  "no_match": 24,
  "valid_option": 17
 },
 "random_fallback": 0,
 "exact_official_report_reproduction": false
}
```
