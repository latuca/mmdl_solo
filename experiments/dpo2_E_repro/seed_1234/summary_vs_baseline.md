# DPO2-E (재현, 학습 seed 1234) vs 베이스라인

**DPO2-E (재현, 학습 seed 1234): 601/900 = 66.78%** (베이스라인 576/900 = 64.00%, 차이 +25문항 / +2.78%p)

| 과목 | 베이스라인 | 이번 | 차이 |
|---|---:|---:|---:|
| Accounting | 19 | 22 | +3 |
| Agriculture | 15 | 16 | +1 |
| Architecture_and_Engineering | 11 | 19 | +8 |
| Art | 18 | 19 | +1 |
| Art_Theory | 24 | 24 | +0 |
| Basic_Medical_Science | 23 | 21 | -2 |
| Biology | 17 | 15 | -2 |
| Chemistry | 16 | 17 | +1 |
| Clinical_Medicine | 23 | 19 | -4 |
| Computer_Science | 20 | 18 | -2 |
| Design | 24 | 26 | +2 |
| Diagnostics_and_Laboratory_Medicine | 10 | 12 | +2 |
| Economics | 26 | 27 | +1 |
| Electronics | 19 | 17 | -2 |
| Energy_and_Power | 14 | 17 | +3 |
| Finance | 23 | 25 | +2 |
| Geography | 18 | 17 | -1 |
| History | 23 | 22 | -1 |
| Literature | 25 | 25 | +0 |
| Manage | 18 | 21 | +3 |
| Marketing | 26 | 28 | +2 |
| Materials | 15 | 13 | -2 |
| Math | 18 | 18 | +0 |
| Mechanical_Engineering | 11 | 18 | +7 |
| Music | 6 | 11 | +5 |
| Pharmacy | 21 | 22 | +1 |
| Physics | 25 | 26 | +1 |
| Psychology | 24 | 23 | -1 |
| Public_Health | 26 | 27 | +1 |
| Sociology | 18 | 16 | -2 |

## 기타 집계 (summary.json)

```json
{
 "total": 900,
 "graded": 900,
 "correct": 601,
 "incorrect": 299,
 "accuracy": 0.6677777777777777,
 "accuracy_percent": 66.77777777777777,
 "aggregation": "unweighted mean of 30 subject accuracies",
 "method_counts": {
  "explicit_final_parser": 808,
  "local_qwen3_8b_judge_parser": 76,
  "official_parser_fallback": 14,
  "output_limit_incorrect": 2
 },
 "length_limit_incorrect": 2,
 "judge_count": 76,
 "judge_states": {
  "valid_option": 54,
  "no_match": 19,
  "unparsed": 3
 },
 "random_fallback": 0,
 "exact_official_report_reproduction": false
}
```
