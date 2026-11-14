# DPO2-C vs 베이스라인

**DPO2-C: 600/900 = 66.67%** (베이스라인 576/900 = 64.00%, 차이 +24문항 / +2.67%p)

| 과목 | 베이스라인 | 이번 | 차이 |
|---|---:|---:|---:|
| Accounting | 19 | 23 | +4 |
| Agriculture | 15 | 15 | +0 |
| Architecture_and_Engineering | 11 | 18 | +7 |
| Art | 18 | 19 | +1 |
| Art_Theory | 24 | 23 | -1 |
| Basic_Medical_Science | 23 | 22 | -1 |
| Biology | 17 | 17 | +0 |
| Chemistry | 16 | 17 | +1 |
| Clinical_Medicine | 23 | 19 | -4 |
| Computer_Science | 20 | 18 | -2 |
| Design | 24 | 25 | +1 |
| Diagnostics_and_Laboratory_Medicine | 10 | 10 | +0 |
| Economics | 26 | 27 | +1 |
| Electronics | 19 | 20 | +1 |
| Energy_and_Power | 14 | 19 | +5 |
| Finance | 23 | 21 | -2 |
| Geography | 18 | 18 | +0 |
| History | 23 | 21 | -2 |
| Literature | 25 | 25 | +0 |
| Manage | 18 | 19 | +1 |
| Marketing | 26 | 25 | -1 |
| Materials | 15 | 19 | +4 |
| Math | 18 | 19 | +1 |
| Mechanical_Engineering | 11 | 22 | +11 |
| Music | 6 | 10 | +4 |
| Pharmacy | 21 | 23 | +2 |
| Physics | 25 | 19 | -6 |
| Psychology | 24 | 24 | +0 |
| Public_Health | 26 | 27 | +1 |
| Sociology | 18 | 16 | -2 |

## 기타 집계 (summary.json)

```json
{
 "total": 900,
 "graded": 900,
 "correct": 600,
 "incorrect": 300,
 "accuracy": 0.6666666666666666,
 "accuracy_percent": 66.66666666666666,
 "aggregation": "unweighted mean of 30 subject accuracies",
 "method_counts": {
  "explicit_final_parser": 830,
  "local_qwen3_8b_judge_parser": 62,
  "official_parser_fallback": 7,
  "output_limit_incorrect": 1
 },
 "length_limit_incorrect": 1,
 "judge_count": 62,
 "judge_states": {
  "valid_option": 34,
  "no_match": 28
 },
 "random_fallback": 0,
 "exact_official_report_reproduction": false
}
```
