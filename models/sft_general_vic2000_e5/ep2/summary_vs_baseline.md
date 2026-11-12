# epoch 2 vs 베이스라인

**epoch 2: 497/900 = 55.22%** (베이스라인 576/900 = 64.00%, 차이 -79문항 / -8.78%p)

| 과목 | 베이스라인 | 이번 | 차이 |
|---|---:|---:|---:|
| Accounting | 19 | 22 | +3 |
| Agriculture | 15 | 13 | -2 |
| Architecture_and_Engineering | 11 | 14 | +3 |
| Art | 18 | 18 | +0 |
| Art_Theory | 24 | 22 | -2 |
| Basic_Medical_Science | 23 | 18 | -5 |
| Biology | 17 | 14 | -3 |
| Chemistry | 16 | 16 | +0 |
| Clinical_Medicine | 23 | 12 | -11 |
| Computer_Science | 20 | 13 | -7 |
| Design | 24 | 22 | -2 |
| Diagnostics_and_Laboratory_Medicine | 10 | 9 | -1 |
| Economics | 26 | 21 | -5 |
| Electronics | 19 | 16 | -3 |
| Energy_and_Power | 14 | 13 | -1 |
| Finance | 23 | 20 | -3 |
| Geography | 18 | 14 | -4 |
| History | 23 | 22 | -1 |
| Literature | 25 | 23 | -2 |
| Manage | 18 | 15 | -3 |
| Marketing | 26 | 20 | -6 |
| Materials | 15 | 11 | -4 |
| Math | 18 | 15 | -3 |
| Mechanical_Engineering | 11 | 10 | -1 |
| Music | 6 | 11 | +5 |
| Pharmacy | 21 | 19 | -2 |
| Physics | 25 | 16 | -9 |
| Psychology | 24 | 22 | -2 |
| Public_Health | 26 | 20 | -6 |
| Sociology | 18 | 16 | -2 |

## 기타 집계 (summary.json)

```json
{
 "total": 900,
 "graded": 900,
 "correct": 497,
 "incorrect": 403,
 "accuracy": 0.5522222222222222,
 "accuracy_percent": 55.222222222222214,
 "aggregation": "unweighted mean of 30 subject accuracies",
 "method_counts": {
  "explicit_final_parser": 839,
  "local_qwen3_8b_judge_parser": 52,
  "output_limit_incorrect": 8,
  "official_parser_fallback": 1
 },
 "length_limit_incorrect": 8,
 "judge_count": 52,
 "judge_states": {
  "no_match": 35,
  "unparsed": 1,
  "valid_option": 15,
  "invalid_option": 1
 },
 "random_fallback": 0,
 "exact_official_report_reproduction": false
}
```
