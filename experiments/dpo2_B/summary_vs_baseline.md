# DPO2-B vs 베이스라인

**DPO2-B: 597/900 = 66.33%** (베이스라인 576/900 = 64.00%, 차이 +21문항 / +2.33%p)

| 과목 | 베이스라인 | 이번 | 차이 |
|---|---:|---:|---:|
| Accounting | 19 | 21 | +2 |
| Agriculture | 15 | 18 | +3 |
| Architecture_and_Engineering | 11 | 16 | +5 |
| Art | 18 | 19 | +1 |
| Art_Theory | 24 | 23 | -1 |
| Basic_Medical_Science | 23 | 20 | -3 |
| Biology | 17 | 15 | -2 |
| Chemistry | 16 | 16 | +0 |
| Clinical_Medicine | 23 | 21 | -2 |
| Computer_Science | 20 | 20 | +0 |
| Design | 24 | 24 | +0 |
| Diagnostics_and_Laboratory_Medicine | 10 | 9 | -1 |
| Economics | 26 | 27 | +1 |
| Electronics | 19 | 18 | -1 |
| Energy_and_Power | 14 | 19 | +5 |
| Finance | 23 | 24 | +1 |
| Geography | 18 | 13 | -5 |
| History | 23 | 23 | +0 |
| Literature | 25 | 25 | +0 |
| Manage | 18 | 22 | +4 |
| Marketing | 26 | 26 | +0 |
| Materials | 15 | 14 | -1 |
| Math | 18 | 21 | +3 |
| Mechanical_Engineering | 11 | 18 | +7 |
| Music | 6 | 8 | +2 |
| Pharmacy | 21 | 26 | +5 |
| Physics | 25 | 24 | -1 |
| Psychology | 24 | 25 | +1 |
| Public_Health | 26 | 26 | +0 |
| Sociology | 18 | 16 | -2 |

## 기타 집계 (summary.json)

```json
{
 "total": 900,
 "graded": 900,
 "correct": 597,
 "incorrect": 303,
 "accuracy": 0.6633333333333333,
 "accuracy_percent": 66.33333333333333,
 "aggregation": "unweighted mean of 30 subject accuracies",
 "method_counts": {
  "explicit_final_parser": 807,
  "official_parser_fallback": 27,
  "local_qwen3_8b_judge_parser": 64,
  "output_limit_incorrect": 2
 },
 "length_limit_incorrect": 2,
 "judge_count": 64,
 "judge_states": {
  "valid_option": 45,
  "unparsed": 3,
  "no_match": 16
 },
 "random_fallback": 0,
 "exact_official_report_reproduction": false
}
```
