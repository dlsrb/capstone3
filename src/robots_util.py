"""
robots_util.py
robots.txt 확인 공통 함수 (RFC 9309 기준)

- robots.txt를 읽었으면 → 그 규칙대로 판단 (Disallow면 수집하지 않음)
- robots.txt가 없음(4xx, 429 제외) → 제한 없음으로 간주 (RFC 9309 2.3.1.3)
- 서버 오류(5xx)·접속 불가 → 수집하지 않음 (RFC 9309 2.3.1.4, 보수적으로 처리)

파이썬 기본 RobotFileParser.read()는 기본 User-Agent로 요청해서, 보안 장비가 막으면(403)
'전체 차단'으로 잘못 판단한다. 그래서 브라우저 User-Agent로 직접 받아서 해석한다.
"""
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
_cache = {}


def fix_url(url):
    url = str(url).strip()
    if url.startswith("//"):
        url = "https:" + url
    elif not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def robots_check(url, ua=UA):
    """(허용 여부, 이유) 반환"""
    p = urlparse(fix_url(url))
    base = f"{p.scheme}://{p.netloc}"
    if base not in _cache:
        try:
            r = requests.get(base + "/robots.txt", headers={"User-Agent": ua}, timeout=10)
            if r.status_code == 200:
                rp = RobotFileParser()
                rp.parse(r.text.splitlines())
                _cache[base] = ("rules", rp)
            elif 400 <= r.status_code < 500 and r.status_code != 429:
                _cache[base] = (f"robots.txt 없음({r.status_code}) → 제한 없음", None)
            else:
                _cache[base] = (f"robots.txt 서버오류({r.status_code}) → 제외", None)
        except Exception as e:
            _cache[base] = (f"robots.txt 접속불가({e.__class__.__name__}) → 제외", None)
    kind, rp = _cache[base]
    if kind == "rules":
        ok = rp.can_fetch(ua, fix_url(url))
        return ok, "robots.txt 허용" if ok else "robots.txt Disallow"
    return ("제한 없음" in kind), kind
