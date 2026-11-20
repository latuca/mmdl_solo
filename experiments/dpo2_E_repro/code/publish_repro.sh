#!/usr/bin/env bash
# DPO2-E 재현(seed별) 코드·결과를 레포 experiments/dpo2_E_repro/ 에 모아 커밋한다. 푸시는 사용자가 직접 한다.
# 가중치·샘플 원문·쌍 데이터·팀원 데이터는 올리지 않는다 (원래 publish_dpo2.sh 와 같은 기준).
# 사용: bash repro/publish_repro.sh 3407 42 [1234 ...]   (seed 3407 결과는 runs/dpo2/E, 나머지는 runs/dpo2/E_s<seed>)
set -euo pipefail
ROOT=/shared_dir/MMDL_solo; RUN=$ROOT/runs/dpo2; DST=$ROOT/experiments/dpo2_E_repro
cd "$ROOT"
mkdir -p "$DST/code" "$DST/data_report"
cp repro/{env.sh,run_repro_e.sh,run_seed_e.sh,repro_gate.py,blocked_window.py,blocked_times.txt,publish_repro.sh} "$DST/code/"
cp $RUN/data/{pairs_B,pairs_B_final,pairs_D,pairs_E}.report.json "$DST/data_report/"
cp $RUN/gate.json "$DST/gate_fixed.json"; cp $RUN/hparams.json "$DST/"
grep -v "GPU 사용 중" $RUN/pipeline.log > "$DST/pipeline.log"
for S in "$@"; do
  X=E; [ "$S" = 3407 ] || X=E_s$S
  [ -f "$RUN/$X/extra_stats.json" ] || { echo "seed $S: 평가 결과 없음 → 건너뜀"; continue; }
  O=$DST/seed_$S; rm -rf "$O"; mkdir -p "$O/train"
  cp $RUN/$X/train.log "$O/train/"; cp $RUN/$X/train/{train_info.json,train_log.jsonl} $RUN/$X/train/adapter/gates.json "$O/train/"
  cp -r $RUN/$X/eval "$O/eval"; rm -f "$O/eval/.run.lock"; find "$O/eval" -type f -size +95M -delete
  cp $RUN/$X/summary_vs_baseline.md $RUN/$X/extra_stats.json "$O/"
done
python3 - "$DST" "$@" <<'EOF'
import json, re, sys, math
from pathlib import Path
dst = Path(sys.argv[1])
rows = []
for d in [dst / f"seed_{x}" for x in sys.argv[2:] if (dst / f"seed_{x}/extra_stats.json").exists()]:
    e = json.loads((d / "extra_stats.json").read_text())
    g = json.loads((d / "train/gates.json").read_text())["deepstack_gates"]
    rows.append((int(d.name[5:]), e["correct"], e["output_limit_truncated"], e["explicit_parser_acc"], g))
c = [r[1] for r in rows]; m = sum(c) / len(c); sd = math.sqrt(sum((x - m) ** 2 for x in c) / (len(c) - 1)) if len(c) > 1 else 0
tab = "\n".join(f"| 재현 seed {s} | {k} | {k/9:.2f}% | {k-576:+d} | {t} | {a*100:.1f}% | {', '.join(f'{x:.3f}' for x in gg)} |" for s, k, t, a, gg in rows)
readme = f"""# DPO2-E 재현 (실습실 RTX 4090, 학습 seed {len(rows)}개)

원래 [dpo2_E](../dpo2_E) 실험을 베이스 샘플링부터 다시 돌려 가중치를 다시 만들고, 학습 seed만 바꿔 반복했다.
원래 코드(../dpo2_E/code)는 고치지 않고, 경로만 바꾼 재현 스크립트(code/)로 돌렸다.

- 고정값: 베이스 Qwen3-VL-4B-Instruct @ ebb281ec, 저지 Qwen3-8B @ b968826d, MMMU @ 98e6ac0c, mmdl_test @ 7740786
- hparams.json(lr 5e-5·λ 1.0)·신호 시험 결과(keep combos, img_pref)는 원래 값 고정 → 측정·lr 시험·신호 시험 생략 (gate_fixed.json)
- E 학습: lr 1e-5, β 0.1, 1 epoch, LoRA r16/α32/0.05, grad accum 16, warmup 3%, wd 0.01, clip 1.0, max-tokens 8400,
  RPO α 1.0, vision LoRA, DeepStack 게이트(lr 1e-3), 이미지 조건 선호 손실(V·I), 보존 가중치 1.0
- seed 반복: 학습 쌍(pairs_E, 2,307행)과 평가 설정(평가 seed 3407)은 같고 **학습 seed만** 다르다 (데이터 순서·LoRA 초기값·dropout)
- 쌍 수: 재현 B 1,825 / B_final 1,170 / D 973 / E 2,307 (원래 1,810 / 1,134 / 931 / 2,287), data_report/
- 원래 환경과 다른 점: 학습 venv에 flash-attn 2.8.3 (공식 wheel) 설치, 샘플링은 다시 해서 쌍 내용이 원래와 조금 다르다

## 결과 (MMMU-val 900문항, 베이스라인 576/900 = 64.00%, 출력 한도 초과 81개)

| 실행 | 정답 | 정확도 | 베이스라인 대비 | 출력 한도 초과 | 정상 추출 정답률 | DeepStack 게이트 |
|---|---:|---:|---:|---:|---:|---|
| 원래 E (옛 서버, seed 3407) | 608 | 67.56% | +32 | 1 | 68.5% | 1.008, 0.986, 0.992 |
{tab}
| **재현 평균 ± 표준편차** | **{m:.1f} ± {sd:.1f}** | **{m/9:.2f} ± {sd/9:.2f}%p** | **{m-576:+.1f}** | | | |

seed별 과목 점수: seed_*/summary_vs_baseline.md, 문항별 출력: seed_*/eval/, 학습 로그: seed_*/train/
가중치(병합 모델·어댑터)는 레포에 올리지 않는다 (Google Drive로 따로 배포).
"""
(dst / "README.md").write_text(readme, encoding="utf-8")
print(readme.split("## 결과")[1])
EOF
git add -A "$DST"
if git diff --cached --quiet; then echo "변경 없음"; exit 0; fi
git -c user.name=latuca -c user.email=ychan0302@naver.com commit -q -m "dpo2_E_repro: DPO2-E 재현 결과 (학습 seed $*)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01MvRnwUXHgDwGPAEQoEKMzG"
echo "커밋 완료: $(git rev-parse --short HEAD) → 푸시: cd $ROOT && git push origin main"
