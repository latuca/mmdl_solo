# DPO2-A_prime vs 베이스라인

**DPO2-A_prime: 573/900 = 63.67%** (베이스라인 576/900 = 64.00%, 차이 -3문항 / -0.33%p)

| 과목 | 베이스라인 | 이번 | 차이 |
|---|---:|---:|---:|
| Accounting | 19 | 20 | +1 |
| Agriculture | 15 | 14 | -1 |
| Architecture_and_Engineering | 11 | 16 | +5 |
| Art | 18 | 17 | -1 |
| Art_Theory | 24 | 24 | +0 |
| Basic_Medical_Science | 23 | 22 | -1 |
| Biology | 17 | 18 | +1 |
| Chemistry | 16 | 18 | +2 |
| Clinical_Medicine | 23 | 20 | -3 |
| Computer_Science | 20 | 20 | +0 |
| Design | 24 | 25 | +1 |
| Diagnostics_and_Laboratory_Medicine | 10 | 11 | +1 |
| Economics | 26 | 26 | +0 |
| Electronics | 19 | 14 | -5 |
| Energy_and_Power | 14 | 13 | -1 |
| Finance | 23 | 23 | +0 |
| Geography | 18 | 14 | -4 |
| History | 23 | 22 | -1 |
| Literature | 25 | 25 | +0 |
| Manage | 18 | 18 | +0 |
| Marketing | 26 | 28 | +2 |
| Materials | 15 | 14 | -1 |
| Math | 18 | 20 | +2 |
| Mechanical_Engineering | 11 | 12 | +1 |
| Music | 6 | 5 | -1 |
| Pharmacy | 21 | 23 | +2 |
| Physics | 25 | 24 | -1 |
| Psychology | 24 | 23 | -1 |
| Public_Health | 26 | 25 | -1 |
| Sociology | 18 | 19 | +1 |

## 기타 집계 (summary.json)

```json
{
 "total": 900,
 "graded": 900,
 "correct": 573,
 "incorrect": 327,
 "accuracy": 0.6366666666666667,
 "accuracy_percent": 63.66666666666667,
 "aggregation": "unweighted mean of 30 subject accuracies",
 "method_counts": {
  "explicit_final_parser": 772,
  "output_limit_incorrect": 63,
  "local_qwen3_8b_judge_parser": 57,
  "official_parser_fallback": 8
 },
 "length_limit_incorrect": 63,
 "judge_count": 57,
 "judge_states": {
  "valid_option": 37,
  "invalid_option": 1,
  "no_match": 18,
  "unparsed": 1
 },
 "random_fallback": 0,
 "exact_official_report_reproduction": false
}
```
