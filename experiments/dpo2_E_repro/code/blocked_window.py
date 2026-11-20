"""수업 차단 시간 검사. 사용: python3 blocked_window.py blocked_times.txt <필요한 분>
blocked_times.txt 한 줄 형식: "Mon 09:00-12:00" (요일: Mon..Sun, # 뒤는 주석).
지금부터 <필요한 분> 안에 차단 구간과 겹치면 그 구간이 끝날 때까지 남은 초를, 아니면 0을 출력한다."""
import sys, re
from datetime import datetime, timedelta

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
path, need = sys.argv[1], int(sys.argv[2])
now = datetime.now()
end_need = now + timedelta(minutes=need)
blocks = []
for line in open(path, encoding="utf-8"):
    line = line.split("#")[0].strip()
    if not line:
        continue
    d = re.fullmatch(r"(\d{4}-\d{2}-\d{2})\s+(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})", line)   # 하루만: "2026-11-18 12:00-19:45"
    if d:
        day = datetime.strptime(d[1], "%Y-%m-%d")
        s = day.replace(hour=int(d[2]), minute=int(d[3]))
        e = day.replace(hour=int(d[4]), minute=int(d[5]))
        blocks.append((s, e if e > s else e + timedelta(days=1)))
        continue
    m = re.fullmatch(r"(\w{3})\s+(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})", line)
    if not m:
        sys.exit(f"형식 오류: {line!r}")
    wd = DAYS.index(m[1])
    for week in (0, 1):   # 이번 주·다음 주 (자정 넘기기·주말 넘기기 대비)
        day = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0) \
              + timedelta(days=wd + 7 * week)
        s = day.replace(hour=int(m[2]), minute=int(m[3]))
        e = day.replace(hour=int(m[4]), minute=int(m[5]))
        if e <= s:
            e += timedelta(days=1)
        blocks.append((s, e))
hit = [e for s, e in blocks if s < end_need and e > now]
print(int((max(hit) - now).total_seconds()) + 60 if hit else 0)
