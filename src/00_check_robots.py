"""
00_check_robots.py
빅카인즈 엑셀에 나온 모든 언론사에 대해, 기사 원문 수집이 robots.txt로 허용되는지 확인 (1~2분)
판단 기준은 src/robots_util.py (RFC 9309) 참고

실행: python src/00_check_robots.py
"""
import warnings
from pathlib import Path

import pandas as pd

from robots_util import fix_url, robots_check

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

df = pd.concat([pd.read_excel(f, dtype=str) for f in RAW_DIR.glob("*.xlsx")], ignore_index=True)
counts = df["언론사"].value_counts()
sample = df.dropna(subset=["URL"]).drop_duplicates("언론사").set_index("언론사")["URL"]

rows = []
for press, n in counts.items():
    if n < 10 or press not in sample:      # 기사 10건 미만 언론사는 생략
        continue
    url = fix_url(sample[press])
    ok, reason = robots_check(url)
    rows.append({"언론사": press, "기사수": n, "허용": ok, "이유": reason, "예시URL": url[:70]})
    print(f"  {press:<8} 기사 {n:>4}건  {'허용' if ok else '제외'}  ({reason})", flush=True)

out = pd.DataFrame(rows)
out.to_csv(RAW_DIR / "robots_check.csv", index=False, encoding="utf-8-sig")
print("\n[허용] 기사 수 상위 20")
print(out[out["허용"]].head(20)[["언론사", "기사수", "이유"]].to_string(index=False))
print("\n[제외]")
print(out[~out["허용"]][["언론사", "기사수", "이유"]].to_string(index=False))
print(f"\n저장 → {RAW_DIR / 'robots_check.csv'}")
