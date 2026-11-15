# DPO2-D vs 베이스라인

**DPO2-D: 598/900 = 66.44%** (베이스라인 576/900 = 64.00%, 차이 +22문항 / +2.44%p)

| 과목 | 베이스라인 | 이번 | 차이 |
|---|---:|---:|---:|
| Accounting | 19 | 24 | +5 |
| Agriculture | 15 | 16 | +1 |
| Architecture_and_Engineering | 11 | 18 | +7 |
| Art | 18 | 19 | +1 |
| Art_Theory | 24 | 26 | +2 |
| Basic_Medical_Science | 23 | 20 | -3 |
| Biology | 17 | 15 | -2 |
| Chemistry | 16 | 20 | +4 |
| Clinical_Medicine | 23 | 15 | -8 |
| Computer_Science | 20 | 21 | +1 |
| Design | 24 | 21 | -3 |
| Diagnostics_and_Laboratory_Medicine | 10 | 9 | -1 |
| Economics | 26 | 26 | +0 |
| Electronics | 19 | 21 | +2 |
| Energy_and_Power | 14 | 20 | +6 |
| Finance | 23 | 24 | +1 |
| Geography | 18 | 13 | -5 |
| History | 23 | 22 | -1 |
| Literature | 25 | 25 | +0 |
| Manage | 18 | 19 | +1 |
| Marketing | 26 | 26 | +0 |
| Materials | 15 | 13 | -2 |
| Math | 18 | 22 | +4 |
| Mechanical_Engineering | 11 | 19 | +8 |
| Music | 6 | 10 | +4 |
| Pharmacy | 21 | 24 | +3 |
| Physics | 25 | 25 | +0 |
| Psychology | 24 | 24 | +0 |
| Public_Health | 26 | 22 | -4 |
| Sociology | 18 | 19 | +1 |

## 기타 집계 (summary.json)

```json
{
 "total": 900,
 "graded": 900,
 "correct": 598,
 "incorrect": 302,
 "accuracy": 0.6644444444444445,
 "accuracy_percent": 66.44444444444446,
 "aggregation": "unweighted mean of 30 subject accuracies",
 "method_counts": {
  "explicit_final_parser": 815,
  "local_qwen3_8b_judge_parser": 67,
  "output_limit_incorrect": 14,
  "official_parser_fallback": 4
 },
 "length_limit_incorrect": 14,
 "judge_count": 67,
 "judge_states": {
  "valid_option": 41,
  "unparsed": 2,
  "no_match": 24
 },
 "random_fallback": 0,
 "exact_official_report_reproduction": false
}
```
