"""
01b_crawl_naver.py
해킹·개인정보 유출 관련 '메이저 언론' 네이버 뉴스 기사의 댓글 크롤링 (API 키 필요 없음)

준비: pip install requests pandas  (SEARCH_MODE="selenium"으로 쓸 때만 selenium, webdriver-manager 필요)
      (Chrome 브라우저가 설치되어 있어야 함. 드라이버는 webdriver-manager가 자동으로 받음)

동작:
  1단계) Selenium으로 네이버 뉴스 검색 화면을 열어 사고별 키워드·기간으로 기사를 모음
         → 메이저 언론 + 네이버 뉴스 링크만 남김 → data/raw/naver_articles.csv
         (크롬 창이 자동으로 열리고 스크롤됨. 끝날 때까지 창을 건드리지 말 것)
  2단계) 기사마다 댓글 수집 → data/raw/naver_comments.csv
         (중간에 멈춰도 다시 실행하면 이어서 수집)
"""
import json
import random
import re
import time
from datetime import datetime, timedelta
from html import unescape
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
ARTICLES_CSV = RAW_DIR / "naver_articles.csv"
COMMENTS_CSV = RAW_DIR / "naver_comments.csv"
DONE_TXT = RAW_DIR / "naver_done_articles.txt"   # 댓글 수집 끝난 기사 기록 (이어하기용)

# ------------------------------------------------------------------
# 메이저 언론 (네이버 언론사 코드 oid)
# ------------------------------------------------------------------
MAJOR_PRESS = {
    "001": "연합뉴스", "422": "연합뉴스TV", "023": "조선일보", "025": "중앙일보", "020": "동아일보",
    "028": "한겨레", "032": "경향신문", "005": "국민일보", "469": "한국일보", "081": "서울신문",
    "015": "한국경제", "009": "매일경제", "008": "머니투데이",
    "056": "KBS", "214": "MBC", "055": "SBS", "437": "JTBC", "052": "YTN",
    "448": "TV조선", "449": "채널A", "057": "MBN",
}

# ------------------------------------------------------------------
# 검색어 (사고별) — 사고일은 01_crawl.py와 같게 맞출 것
# 기사 기간: 사고일 7일 전 ~ 사고일 180일 후
# ------------------------------------------------------------------
GROUPS = {
    "SKT":    {"incident": "2025-04-22", "queries": ["SKT 해킹", "SK텔레콤 유심 해킹", "SKT 유심 교체", "SKT 위약금 면제"]},
    "예스24": {"incident": "2025-06-09", "queries": ["예스24 해킹", "예스24 랜섬웨어", "예스24 접속 장애"]},
    "롯데카드": {"incident": "2025-09-01", "queries": ["롯데카드 해킹", "롯데카드 개인정보 유출", "롯데카드 재발급"]},
    "KT":     {"incident": "2025-09-08", "queries": ["KT 소액결제", "KT 해킹", "KT 무단 결제", "KT 펨토셀"]},
    "쿠팡":   {"incident": "2025-11-29", "queries": ["쿠팡 개인정보 유출", "쿠팡 해킹", "쿠팡 탈퇴", "쿠팡 보상"]},
    "일반":   {"incident": None, "period": ("2025-01-01", "2026-10-31"),
               "queries": ["해킹 피해", "개인정보 유출 사고", "랜섬웨어 공격", "보이스피싱 피해", "스미싱 피해", "사이버 보안 투자"]},
}

MAX_SCROLL = 30                  # 검색 결과 화면을 아래로 내리는 최대 횟수 (검색어당)
MIN_COMMENTS = 10                # 댓글이 이보다 적은 기사는 건너뜀
MAX_COMMENTS_PER_ARTICLE = 3000  # 한 기사에서 최대 수집 댓글 수 (한 기사 쏠림 방지)
SLEEP = (0.4, 0.9)               # 요청 사이 무작위 대기(초)

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
ARTICLE_RE = re.compile(r"n\.news\.naver\.com/(?:mnews/)?article/(\d{3})/(\d{10})")


def nap():
    time.sleep(random.uniform(*SLEEP))


def strip_tags(s):
    return unescape(re.sub(r"<[^>]+>", "", s or ""))


# ==================================================================
# 1단계: 기사 목록
# ==================================================================
# 크롬 드라이버 경로를 직접 지정하고 싶으면 여기에 입력 (예: r"C:\chromedriver\chromedriver.exe")
CHROMEDRIVER_PATH = ""

# 기사 검색 방식: "requests" = 브라우저 없이 검색 결과 페이지를 직접 요청 (기본)
#                 "selenium" = 크롬 창을 띄워 스크롤 (requests 방식으로 기사가 거의 안 모일 때 사용)
SEARCH_MODE = "requests"
MAX_PAGES = 40           # requests 방식: 검색어당 최대 페이지 수 (페이지당 10건)


def make_driver():
    """크롬 드라이버 준비: ① 직접 지정 경로 → ② webdriver-manager(파이썬으로 다운로드) → ③ Selenium 기본."""
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    opts = webdriver.ChromeOptions()
    opts.add_argument("--window-size=1200,1000")
    opts.add_argument(f"user-agent={UA}")

    if CHROMEDRIVER_PATH:
        chrome = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
        if chrome.exists():
            opts.binary_location = str(chrome)
        return webdriver.Chrome(service=Service(CHROMEDRIVER_PATH), options=opts)
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        print("  크롬 드라이버 준비 중 (webdriver-manager)...", flush=True)
        return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
    except ImportError:
        print("  webdriver-manager가 없습니다 → pip install webdriver-manager", flush=True)
    except Exception as e:
        print(f"  webdriver-manager 실패({e.__class__.__name__}: {str(e)[:150]})", flush=True)
    return webdriver.Chrome(options=opts)


def search_links(driver, query, lo, hi):
    """네이버 뉴스 검색 결과에서 (oid, aid) 목록을 모음 (기간 지정, 무한 스크롤)."""
    from urllib.parse import quote
    url = ("https://search.naver.com/search.naver?where=news&sort=0&pd=3"
           f"&query={quote(query)}&ds={lo:%Y.%m.%d}&de={hi:%Y.%m.%d}"
           f"&nso=so:r,p:from{lo:%Y%m%d}to{hi:%Y%m%d}")
    driver.get(url)
    time.sleep(2)
    ids, same = set(), 0
    for _ in range(MAX_SCROLL):
        before = len(ids)
        ids |= set(ARTICLE_RE.findall(driver.page_source))
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(random.uniform(1.2, 2.0))
        same = same + 1 if len(ids) == before else 0
        if same >= 3:          # 3번 연속 새 기사가 없으면 끝
            break
    ids |= set(ARTICLE_RE.findall(driver.page_source))
    return ids


def search_links_requests(query, lo, hi):
    """브라우저 없이 네이버 뉴스 검색 결과 페이지를 10건씩 넘기며 (oid, aid) 수집."""
    ids, empty = set(), 0
    headers = {"User-Agent": UA, "Referer": "https://search.naver.com/"}
    for page in range(MAX_PAGES):
        params = {"where": "news", "query": query, "sort": 0, "pd": 3,
                  "ds": f"{lo:%Y.%m.%d}", "de": f"{hi:%Y.%m.%d}",
                  "nso": f"so:r,p:from{lo:%Y%m%d}to{hi:%Y%m%d}", "start": page * 10 + 1}
        try:
            r = requests.get("https://search.naver.com/search.naver", params=params,
                             headers=headers, timeout=10)
        except Exception as e:
            print(f"    검색 요청 오류({e.__class__.__name__}) → 이 검색어 중단", flush=True)
            break
        found = set(ARTICLE_RE.findall(r.text))
        new = found - ids
        ids |= found
        empty = 0 if new else empty + 1
        if empty >= 2:          # 2페이지 연속 새 기사가 없으면 끝
            break
        nap()
    return ids


def article_meta(oid, aid):
    """기사 페이지에서 제목·작성일 가져오기."""
    try:
        r = requests.get(f"https://n.news.naver.com/mnews/article/{oid}/{aid}",
                         headers={"User-Agent": UA}, timeout=10)
        t = re.search(r'<meta property="og:title" content="([^"]*)"', r.text)
        d = re.search(r'data-date-time="([0-9\- :]+)"', r.text)
        return (unescape(t.group(1)) if t else ""), (d.group(1) if d else "")
    except Exception:
        return "", ""


def search_articles():
    rows = []
    driver = make_driver() if SEARCH_MODE == "selenium" else None
    print(f"  검색 방식: {SEARCH_MODE}", flush=True)
    try:
        for group, g in GROUPS.items():
            if g["incident"]:
                inc = datetime.strptime(g["incident"], "%Y-%m-%d")
                lo, hi = inc - timedelta(days=7), inc + timedelta(days=180)
            else:
                lo, hi = (datetime.strptime(d, "%Y-%m-%d") for d in g["period"])
            for q in g["queries"]:
                ids = (search_links(driver, q, lo, hi) if driver
                       else search_links_requests(q, lo, hi))
                major = [(o, a) for o, a in ids if o in MAJOR_PRESS]
                for oid, aid in major:
                    rows.append({"group": group, "query": q, "oid": oid, "aid": aid,
                                 "press": MAJOR_PRESS[oid],
                                 "url": f"https://n.news.naver.com/mnews/article/{oid}/{aid}"})
                print(f"  [{group}] '{q}' → 전체 {len(ids)}건 중 메이저 언론 {len(major)}건", flush=True)
    finally:
        if driver:
            driver.quit()

    df = pd.DataFrame(rows).drop_duplicates(subset=["oid", "aid"]).reset_index(drop=True)
    print(f"\n기사 {len(df):,}건 제목·날짜 가져오는 중...", flush=True)
    metas = []
    for i, r in enumerate(df.itertuples(), 1):
        metas.append(article_meta(r.oid, r.aid))
        if i % 50 == 0:
            print(f"  {i:,}/{len(df):,}", flush=True)
        nap()
    df["title"] = [m[0] for m in metas]
    df["pub_date"] = [m[1] for m in metas]
    df.to_csv(ARTICLES_CSV, index=False, encoding="utf-8-sig")
    print(f"\n기사 목록 {len(df):,}건 저장 → {ARTICLES_CSV.name}")
    print(df.groupby("group").size().to_string())
    return df


# ==================================================================
# 2단계: 댓글
# ==================================================================
def fetch_comment_page(oid, aid, page):
    url = "https://apis.naver.com/commentBox/cbox/web_neo_list_jsonp.json"
    params = {
        "ticket": "news", "templateId": "default_society", "pool": "cbox5",
        "_callback": "cb", "lang": "ko", "country": "KR",
        "objectId": f"news{oid},{aid}", "pageSize": 100, "indexSize": 10,
        "listType": "OBJECT", "pageType": "more", "page": page,
        "sort": "new", "includeAllStatus": "false",
    }
    headers = {"User-Agent": UA, "Referer": f"https://n.news.naver.com/mnews/article/{oid}/{aid}"}
    r = requests.get(url, params=params, headers=headers, timeout=10)
    r.raise_for_status()
    text = r.text
    body = text[text.find("(") + 1: text.rfind(")")]   # JSONP → JSON
    return json.loads(body)


def crawl_comments(articles):
    done = set(DONE_TXT.read_text(encoding="utf-8").split()) if DONE_TXT.exists() else set()
    total = len(pd.read_csv(COMMENTS_CSV, usecols=["comment_no"])) if COMMENTS_CSV.exists() else 0
    todo = articles[~(articles["oid"].astype(str).str.zfill(3) + "_" +
                      articles["aid"].astype(str).str.zfill(10)).isin(done)]
    print(f"\n댓글 수집: 전체 기사 {len(articles):,} / 남은 기사 {len(todo):,} / 지금까지 댓글 {total:,}건")
    t0 = time.time()

    for i, a in enumerate(todo.itertuples(), 1):
        oid, aid = str(a.oid).zfill(3), str(a.aid).zfill(10)
        key = f"{oid}_{aid}"
        rows, page, n_total = [], 1, None
        try:
            while True:
                data = fetch_comment_page(oid, aid, page)
                if not data.get("success", True):
                    break
                res = data.get("result", {})
                if n_total is None:
                    n_total = (res.get("count") or {}).get("comment", 0)
                    if n_total < MIN_COMMENTS:
                        break
                clist = res.get("commentList") or []
                for c in clist:
                    if c.get("deleted") or not c.get("contents"):
                        continue
                    rows.append({
                        "comment_no": c.get("commentNo"),
                        "group": a.group, "press": a.press, "oid": oid, "aid": aid,
                        "article_title": a.title, "article_date": a.pub_date,
                        "date": c.get("regTime"),
                        "content": c.get("contents"),
                        "likes": c.get("sympathyCount"), "dislikes": c.get("antipathyCount"),
                        "replies": c.get("replyCount"),
                    })
                total_pages = (res.get("pageModel") or {}).get("totalPages", page)
                if not clist or page >= total_pages or len(rows) >= MAX_COMMENTS_PER_ARTICLE:
                    break
                page += 1
                nap()
        except Exception as e:
            print(f"  ! {key} 오류({e.__class__.__name__}) → 건너뜀", flush=True)
            time.sleep(3)
            continue

        if rows:
            pd.DataFrame(rows).to_csv(COMMENTS_CSV, mode="a", index=False, encoding="utf-8-sig",
                                      header=not COMMENTS_CSV.exists())
            total += len(rows)
        with open(DONE_TXT, "a", encoding="utf-8") as f:
            f.write(key + "\n")

        print(f"  [{i:>4}/{len(todo)}] {a.group:<5} {a.press:<6} 댓글 {len(rows):>5,}건 | 누적 {total:>8,}건 | "
              f"{time.time() - t0:6.0f}초 | {str(a.title)[:30]}", flush=True)
        nap()

    print(f"\n완료: 댓글 총 {total:,}건 → {COMMENTS_CSV.name}")


def main():
    if ARTICLES_CSV.exists():
        print(f"[1단계] 기존 기사 목록 사용 ({ARTICLES_CSV.name}) — 새로 검색하려면 이 파일을 지우고 실행")
        articles = pd.read_csv(ARTICLES_CSV, dtype={"oid": str, "aid": str})
    else:
        print("[1단계] 기사 검색")
        articles = search_articles()
    print("\n[2단계] 댓글 수집")
    crawl_comments(articles)

    if COMMENTS_CSV.exists():
        df = pd.read_csv(COMMENTS_CSV, usecols=["comment_no", "group"]).drop_duplicates("comment_no")
        print("\n========== 그룹별 댓글 수 ==========")
        print(df.groupby("group").size().to_string())
        print(f"합계: {len(df):,}건")


if __name__ == "__main__":
    main()
