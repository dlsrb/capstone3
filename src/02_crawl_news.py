"""
02_crawl_news.py
저출생 정책 관련 뉴스 기사 수집 (네이버 뉴스 경유, 7개 언론사)

준비:  pip install requests beautifulsoup4 pandas
       (SEARCH_MODE="selenium"을 쓸 때만: pip install selenium webdriver-manager)
실행:  PyCharm에서 Run  (또는 python src/02_crawl_news.py)

동작:
  1단계) 네이버 뉴스 검색으로 기사 링크 수집 (검색어 × 월 단위 기간)
         → 7개 언론사 기사만 남김 → data/raw/news_links.csv
  2단계) 기사 페이지에서 제목·날짜·본문·본문 속 링크 수집 → data/raw/news_articles.csv
         (중간에 멈춰도 다시 실행하면 이어서 수집)

PILOT = True  : 파일럿 (기사 약 100~200건, 10~20분) → 03_pilot_check.py로 데이터 판별
PILOT = False : 본 수집 (2021.07 ~ 2026.10 전체, 수 시간 걸릴 수 있음)
"""
import random
import re
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ==================================================================
# 설정
# ==================================================================
PILOT = True

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
SUFFIX = "_pilot" if PILOT else ""
LINKS_CSV = RAW_DIR / f"news_links{SUFFIX}.csv"
ARTICLES_CSV = RAW_DIR / f"news_articles{SUFFIX}.csv"

# 네이버 언론사 코드(oid) → (언론사, 성향)
OUTLETS = {
    "028": ("한겨레", "진보"),
    "032": ("경향신문", "진보"),
    "023": ("조선일보", "보수"),
    "025": ("중앙일보", "보수"),
    "020": ("동아일보", "보수"),
    "015": ("한국경제", "경제"),
    "009": ("매일경제", "경제"),
}

QUERIES = [
    "저출생 정책", "저출산 대책", "저출생 대책 실효성",
    "부모급여", "첫만남이용권", "아동수당", "출산지원금",
    "육아휴직 급여", "부모육아휴직", "육아기 근로시간 단축",
    "신생아 특례대출", "늘봄학교", "어린이집 보육", "일가정 양립",
]

START, END = date(2021, 7, 1), date(2026, 10, 31)    # 본 수집 기간 (README H2 기준 시점 전후 포함)
MAX_PAGES = 10                                       # 검색어·월마다 최대 페이지 (페이지당 10건)

# 파일럿: H2 기준 시점 근처 몇 달 + 검색어 일부만
PILOT_MONTHS = [(2023, 1), (2024, 1), (2025, 1)]
PILOT_QUERIES = ["저출생 정책", "부모급여", "육아휴직 급여", "신생아 특례대출", "늘봄학교"]
PILOT_MAX_PAGES = 3

SEARCH_MODE = "requests"   # 기사가 거의 안 모이면 "selenium"으로 바꿔서 실행
SLEEP = (1.0, 2.0)         # 요청 사이 무작위 대기(초) — README에 적은 1~2초

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
HEADERS = {"User-Agent": UA, "Referer": "https://search.naver.com/"}
ARTICLE_RE = re.compile(r"n\.news\.naver\.com/(?:mnews/)?article/(\d{3})/(\d{10})")
OPINION_RE = re.compile(r"^\s*[\[［<〈]\s*(사설|칼럼|시론|기고|오피니언|논설|포럼|데스크|기자수첩|특별기고|만물상|분수대|횡설수설)")


def nap():
    time.sleep(random.uniform(*SLEEP))


def month_windows():
    if PILOT:
        months = PILOT_MONTHS
    else:
        months, y, m = [], START.year, START.month
        while (y, m) <= (END.year, END.month):
            months.append((y, m))
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    for y, m in months:
        lo = date(y, m, 1)
        hi = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
        yield lo, date.fromordinal(hi.toordinal() - 1)


# ==================================================================
# 1단계: 검색으로 기사 링크 수집
# ==================================================================
def search_requests(query, lo, hi, max_pages):
    ids, empty = set(), 0
    for page in range(max_pages):
        params = {"where": "news", "query": query, "sort": 0, "pd": 3,
                  "ds": f"{lo:%Y.%m.%d}", "de": f"{hi:%Y.%m.%d}",
                  "nso": f"so:r,p:from{lo:%Y%m%d}to{hi:%Y%m%d}", "start": page * 10 + 1}
        try:
            r = requests.get("https://search.naver.com/search.naver", params=params,
                             headers=HEADERS, timeout=10)
        except Exception as e:
            print(f"    검색 오류({e.__class__.__name__}) → 이 검색어 중단", flush=True)
            break
        new = set(ARTICLE_RE.findall(r.text)) - ids
        ids |= new
        empty = 0 if new else empty + 1
        if empty >= 2:
            break
        nap()
    return ids


def make_driver():
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    opts = webdriver.ChromeOptions()
    opts.add_argument("--window-size=1200,1000")
    opts.add_argument(f"user-agent={UA}")
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
    except Exception:
        return webdriver.Chrome(options=opts)


def search_selenium(driver, query, lo, hi, max_pages):
    from urllib.parse import quote
    driver.get("https://search.naver.com/search.naver?where=news&sort=0&pd=3"
               f"&query={quote(query)}&ds={lo:%Y.%m.%d}&de={hi:%Y.%m.%d}"
               f"&nso=so:r,p:from{lo:%Y%m%d}to{hi:%Y%m%d}")
    time.sleep(2)
    ids, same = set(), 0
    for _ in range(max_pages * 2):
        before = len(ids)
        ids |= set(ARTICLE_RE.findall(driver.page_source))
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(random.uniform(1.2, 2.0))
        same = same + 1 if len(ids) == before else 0
        if same >= 3:
            break
    return ids


def collect_links():
    queries = PILOT_QUERIES if PILOT else QUERIES
    max_pages = PILOT_MAX_PAGES if PILOT else MAX_PAGES
    driver = make_driver() if SEARCH_MODE == "selenium" else None
    rows = []
    try:
        for lo, hi in month_windows():
            for q in queries:
                ids = (search_selenium(driver, q, lo, hi, max_pages) if driver
                       else search_requests(q, lo, hi, max_pages))
                keep = [(o, a) for o, a in ids if o in OUTLETS]
                rows += [{"query": q, "month": f"{lo:%Y-%m}", "oid": o, "aid": a} for o, a in keep]
                print(f"  {lo:%Y-%m} '{q}' → 전체 {len(ids)}건 중 7개 언론사 {len(keep)}건", flush=True)
    finally:
        if driver:
            driver.quit()

    df = pd.DataFrame(rows, columns=["query", "month", "oid", "aid"])
    # 같은 기사가 여러 검색어에 걸리면 검색어를 합쳐서 1건으로
    df = (df.groupby(["oid", "aid"], as_index=False)
            .agg(query=("query", lambda s: ";".join(sorted(set(s)))), month=("month", "first")))
    df.to_csv(LINKS_CSV, index=False, encoding="utf-8-sig")
    print(f"\n기사 링크 {len(df):,}건 저장 → {LINKS_CSV.name}")
    return df


# ==================================================================
# 2단계: 기사 본문 수집
# ==================================================================
REMOVE_SELECTORS = [
    ".end_photo_org", ".img_desc", "em.img_desc", ".nbd_table", ".vod_player_wrap",
    ".media_end_summary",   # 요약문(본문과 중복)
    "script", "style",
]


def parse_article(html):
    soup = BeautifulSoup(html, "html.parser")
    title = soup.select_one("#title_area") or soup.select_one("h2.media_end_head_headline")
    meta_title = soup.find("meta", property="og:title")
    t = title.get_text(" ", strip=True) if title else (meta_title["content"] if meta_title else "")

    d = soup.select_one("span.media_end_head_info_datestamp_time")
    pub = d.get("data-date-time", "") if d else ""

    body = soup.select_one("#dic_area") or soup.select_one("#newsct_article")
    if body is None:
        return t, pub, "", ""
    links = [a.get("href", "") for a in body.find_all("a") if a.get("href")]
    links += [f.get("src", "") for f in body.find_all("iframe") if f.get("src")]
    for sel in REMOVE_SELECTORS:
        for tag in body.select(sel):
            tag.decompose()
    for br in body.find_all("br"):
        br.replace_with("\n")
    text = body.get_text("\n")
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text).strip()
    return t, pub, text, " ".join(links)


def collect_articles(links):
    done = set()
    if ARTICLES_CSV.exists():
        prev = pd.read_csv(ARTICLES_CSV, dtype={"oid": str, "aid": str})
        done = set(prev["oid"] + "_" + prev["aid"])
    todo = links[~(links["oid"] + "_" + links["aid"]).isin(done)]
    print(f"\n본문 수집: 전체 {len(links):,} / 남은 {len(todo):,}", flush=True)

    buf, t0 = [], time.time()
    for i, r in enumerate(todo.itertuples(), 1):
        url = f"https://n.news.naver.com/mnews/article/{r.oid}/{r.aid}"
        try:
            resp = requests.get(url, headers={"User-Agent": UA}, timeout=10)
            title, pub, body, out_links = parse_article(resp.text)
        except Exception as e:
            print(f"  ! {url} 오류({e.__class__.__name__}) → 건너뜀", flush=True)
            time.sleep(3)
            continue
        press, lean = OUTLETS[r.oid]
        buf.append({
            "oid": r.oid, "aid": r.aid, "url": url, "press": press, "outlet_lean": lean,
            "article_type": "사설·칼럼·기고" if OPINION_RE.search(title) else "스트레이트",
            "query": r.query, "pub_date": pub, "title": title,
            "body": body, "body_links": out_links, "body_len": len(body),
        })
        if len(buf) >= 20 or i == len(todo):
            pd.DataFrame(buf).to_csv(ARTICLES_CSV, mode="a", index=False, encoding="utf-8-sig",
                                     header=not ARTICLES_CSV.exists())
            buf = []
        if i % 20 == 0 or i == len(todo):
            print(f"  [{i:>5}/{len(todo)}] {press:<5} {time.time() - t0:6.0f}초 | {title[:30]}", flush=True)
        nap()
    if buf:
        pd.DataFrame(buf).to_csv(ARTICLES_CSV, mode="a", index=False, encoding="utf-8-sig",
                                 header=not ARTICLES_CSV.exists())


def main():
    print(f"모드: {'파일럿' if PILOT else '본 수집'} | 검색 방식: {SEARCH_MODE}")
    if LINKS_CSV.exists():
        print(f"[1단계] 기존 링크 목록 사용 ({LINKS_CSV.name}) — 새로 검색하려면 이 파일을 지우고 실행")
        links = pd.read_csv(LINKS_CSV, dtype={"oid": str, "aid": str})
    else:
        print("[1단계] 기사 검색")
        links = collect_links()
    if links.empty:
        print("\n기사가 0건입니다. SEARCH_MODE를 'selenium'으로 바꾸고, news_links 파일을 지운 뒤 다시 실행하세요.")
        return

    print("\n[2단계] 기사 본문 수집")
    collect_articles(links)

    df = pd.read_csv(ARTICLES_CSV, dtype={"oid": str, "aid": str}).drop_duplicates(["oid", "aid"])
    print("\n========== 수집 결과 ==========")
    print(f"기사 수: {len(df):,}  (본문 없음 {int((df['body_len'] == 0).sum())}건)")
    print(df.groupby(["outlet_lean", "press"]).size().to_string())
    print(f"\n사설·칼럼·기고: {int((df['article_type'] != '스트레이트').sum())}건")
    print(f"저장 → {ARTICLES_CSV}")
    if PILOT:
        print("\n다음 단계: src/03_pilot_check.py 실행")


if __name__ == "__main__":
    main()
