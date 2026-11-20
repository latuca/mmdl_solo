# DPO2-E (재현, 학습 seed 42) vs 베이스라인

**DPO2-E (재현, 학습 seed 42): 612/900 = 68.00%** (베이스라인 576/900 = 64.00%, 차이 +36문항 / +4.00%p)

| 과목 | 베이스라인 | 이번 | 차이 |
|---|---:|---:|---:|
| Accounting | 19 | 23 | +4 |
| Agriculture | 15 | 18 | +3 |
| Architecture_and_Engineering | 11 | 18 | +7 |
| Art | 18 | 19 | +1 |
| Art_Theory | 24 | 24 | +0 |
| Basic_Medical_Science | 23 | 23 | +0 |
| Biology | 17 | 19 | +2 |
| Chemistry | 16 | 16 | +0 |
| Clinical_Medicine | 23 | 22 | -1 |
| Computer_Science | 20 | 18 | -2 |
| Design | 24 | 25 | +1 |
| Diagnostics_and_Laboratory_Medicine | 10 | 12 | +2 |
| Economics | 26 | 27 | +1 |
| Electronics | 19 | 20 | +1 |
| Energy_and_Power | 14 | 19 | +5 |
| Finance | 23 | 25 | +2 |
| Geography | 18 | 17 | -1 |
| History | 23 | 22 | -1 |
| Literature | 25 | 25 | +0 |
| Manage | 18 | 17 | -1 |
| Marketing | 26 | 26 | +0 |
| Materials | 15 | 17 | +2 |
| Math | 18 | 22 | +4 |
| Mechanical_Engineering | 11 | 14 | +3 |
| Music | 6 | 11 | +5 |
| Pharmacy | 21 | 24 | +3 |
| Physics | 25 | 25 | +0 |
| Psychology | 24 | 23 | -1 |
| Public_Health | 26 | 25 | -1 |
| Sociology | 18 | 16 | -2 |

## 기타 집계 (summary.json)

```json
{
 "total": 900,
 "graded": 900,
 "correct": 612,
 "incorrect": 288,
 "accuracy": 0.6799999999999999,
 "accuracy_percent": 68.0,
 "aggregation": "unweighted mean of 30 subject accuracies",
 "method_counts": {
  "explicit_final_parser": 807,
  "local_qwen3_8b_judge_parser": 78,
  "output_limit_incorrect": 3,
  "official_parser_fallback": 12
 },
 "length_limit_incorrect": 3,
 "judge_count": 78,
 "judge_states": {
  "valid_option": 53,
  "no_match": 24,
  "invalid_option": 1
 },
 "random_fallback": 0,
 "exact_official_report_reproduction": false
}
```
