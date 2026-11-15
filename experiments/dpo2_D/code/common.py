"""공통 유틸: 베이스라인 로드, 최종 답 파서, 채점, 루프 감지.

베이스라인 JSON(baseline_results.json)의 설정·프롬프트를 그대로 읽어 쓰므로
평가 조건이 스크립트마다 달라지지 않는다.
"""
import ast
import json
import re
import sys
from pathlib import Path

try:  # Windows 콘솔(cp949)에서 한글·특수문자 출력이 깨지지 않게
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---------------------------------------------------------------- 로드

def load_baseline(path):
    """(settings, records_by_id) 반환."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    settings = d["evaluation"]["settings"]
    records = {r["id"]: r for r in d["records"]}
    return settings, records


def read_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def baseline_type(rec):
    """베이스라인 기준 유형 코드. 정답이면 'OK'."""
    if rec["grading"]["final"]["correct"]:
        return "OK"
    return (rec.get("error_analysis") or {}).get("primary_type_code", "?")

# ---------------------------------------------------------------- 루프 감지
# calibrate_loop.py로 베이스라인 900개에 보정한 기본값:
# 4,096토큰마다 마지막 4,096토큰 구간의 12-gram 반복률을 보고,
# 0.5 이상이 2구간 연속이면 루프로 판정한다.
# (베이스라인: 루프 81/81 검출, 정답 응답 중 오검출 0)

LOOP_N = 12
LOOP_WINDOW = 4096
LOOP_TH = 0.5
LOOP_CONSEC = 2


def ngram_repeat_rate(text, n=LOOP_N):
    w = text.split()
    if len(w) < n + 50:
        return 0.0
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)

# ---------------------------------------------------------------- 최종 답 파서

_FINAL_RE = re.compile(r"final\s*answer\s*(?:is)?\s*[:：]\s*(.*)", re.I)


def _clean(s):
    s = s.replace("**", "").replace("__", "").replace("`", "")
    s = re.sub(r"^\s*#+\s*", "", s, flags=re.M)
    for _ in range(2):
        s = re.sub(r"\\(?:boxed|text|mathrm|textbf)\{([^{}]*)\}", r"\1", s)
    return s.replace("$", "")


def _final_field_payloads(text):
    """줄 맨 앞에서 시작하는 'Final answer:' 필드만 모은다.

    문장 중간의 언급(예: '... write "Final answer: LETTER" ...')은 필드로 보지 않는다.
    """
    lines = text.splitlines()
    out = []
    for i, line in enumerate(lines):
        s = line.strip().lstrip(">-*• ").strip()
        m = _FINAL_RE.match(s)
        if not m:
            continue
        pay = m.group(1).strip()
        if not pay:  # 다음 줄에 답이 있는 경우
            pay = next((l.strip() for l in lines[i + 1:] if l.strip()), "")
        out.append(pay)
    return out


def _letters(payload, valid):
    found = re.findall(r"(?<![A-Za-z])\(?([A-Z])\)?(?![A-Za-z])", payload)
    return [f for f in found if f in valid]


def _option_text_match(payload, options):
    p = payload.strip().lower().rstrip(".")
    hits = [k for k, v in options.items() if str(v).strip().lower().rstrip(".") == p]
    return hits[0] if len(hits) == 1 else None


def parse_mc(raw, options):
    """객관식 응답에서 최종 보기 문자 추출.

    반환: (status, answer, reason)
      status: 'extracted' | 'needs_judge'
    """
    text = _clean(raw.split("</think>")[-1])
    valid = set(options.keys())
    payloads = _final_field_payloads(text)

    if payloads:
        # 마지막(terminal) 필드를 우선한다. 앞쪽 필드는 충돌 여부만 확인.
        def one(pay):
            ls = set(_letters(pay, valid))
            if len(ls) == 1:
                return ls.pop(), None
            if len(ls) > 1:
                return None, "multiple_letters_in_final_field"
            t = _option_text_match(pay, options)
            return (t, None) if t else (None, "final_field_unparsed")

        last, err = one(payloads[-1])
        if last is None:
            return "needs_judge", None, err
        for pay in payloads[:-1]:
            prev, _ = one(pay)
            if prev is not None and prev != last:
                return "needs_judge", None, "conflicting_final_fields"
        return "extracted", last, "final_field"

    # 최종 필드가 없을 때: 응답 전체가 보기 하나뿐이면 채택 (예: "B", "B. Shape")
    stripped = text.strip()
    m = re.fullmatch(r"\(?([A-Z])\)?[\.\)]?", stripped)
    if m and m.group(1) in valid:
        return "extracted", m.group(1), "fallback_single_letter"
    m = re.fullmatch(r"\(?([A-Z])\)?[\.\):]\s+[^\n]{1,200}", stripped)
    if m and m.group(1) in valid:
        return "extracted", m.group(1), "fallback_letter_with_text"
    cands = set(re.findall(r"\(([A-Z])\)", text)) & valid
    if len(cands) == 1:
        return "extracted", cands.pop(), "fallback_paren_letter"
    t = _option_text_match(stripped, options)
    if t:
        return "extracted", t, "fallback_option_text"
    return "needs_judge", None, "no_final_field"


def _gold_list(ref):
    if isinstance(ref, list):
        return [str(x) for x in ref]
    s = str(ref).strip()
    if s.startswith("["):
        try:
            v = ast.literal_eval(s)
            if isinstance(v, list):
                return [str(x) for x in v]
        except Exception:
            pass
    return [s]


def _nums(s):
    out = []
    for x in re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", s):
        try:
            out.append(round(float(x.replace(",", "")), 2))
        except ValueError:
            pass
    return out


def grade_open(raw, reference):
    """주관식: 공식 방식처럼 정답 문자열이 최종 답에 들어 있는지로 판정.

    - 정답이 대문자 한 글자면(특례) 최종 답의 보기 문자와 비교한다.
    - 정답 문자열이 최종 답에 포함되면 정답.
    - 그 외에는 needs_judge (공식 파이프라인에서도 저지로 넘어가는 경우).
    """
    text = _clean(raw.split("</think>")[-1])
    pays = _final_field_payloads(text)
    if not pays:
        return "needs_judge", None, "no_final_field"
    pay = pays[-1]
    golds = _gold_list(reference)
    if len(golds) == 1 and re.fullmatch(r"[A-Z]", golds[0].strip()):
        ls = set(re.findall(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])", pay))
        if len(ls) == 1:
            got = ls.pop()
            return ("correct" if got == golds[0].strip() else "incorrect"), got, "single_letter_gold"
        return "needs_judge", pay, "single_letter_gold_unparsed"
    p_low = pay.lower()
    for g in golds:
        g_low = g.lower().strip().rstrip(".")
        if g_low and g_low in p_low:
            return "correct", pay, "gold_in_answer"
    return "needs_judge", pay, "no_match_needs_judge"


def grade_prediction(pred, base_rec, judge=None):
    """한 응답을 채점.

    pred: run_generate.py 출력 한 줄 (model_response 포함)
    base_rec: 베이스라인 레코드 (options, reference_answer, question_type)
    judge: {'answer': 'B'} 형태의 저지 결과 (없으면 None)

    반환 dict: verdict ('correct'|'incorrect'|'needs_judge'), reason, answer
    """
    mr = pred["model_response"]
    status = mr.get("generation_status", "")
    if status == "loop_stopped":
        return {"verdict": "incorrect", "reason": "loop_stopped", "answer": None}
    if mr.get("output_limit_reached") or status == "output_token_limit":
        return {"verdict": "incorrect", "reason": "length", "answer": None}

    raw = mr["raw"]
    gold = base_rec["reference_answer"]
    if base_rec["question_type"] == "multiple-choice":
        st, ans, why = parse_mc(raw, base_rec["options"])
        if st == "extracted":
            return {"verdict": "correct" if ans == gold else "incorrect", "reason": why, "answer": ans}
    else:
        st, ans, why = grade_open(raw, gold)
        if st in ("correct", "incorrect"):
            return {"verdict": st, "reason": why, "answer": ans}

    if judge is not None:
        j = judge.get("answer")
        if base_rec["question_type"] == "multiple-choice":
            ok = j == gold
        else:  # 주관식 저지: A=정답, B=기타
            ok = j == "A"
        return {"verdict": "correct" if ok else "incorrect", "reason": "judge", "answer": j}
    return {"verdict": "needs_judge", "reason": why, "answer": ans}
