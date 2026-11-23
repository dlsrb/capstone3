"""
02b_fetch_bigkinds.py
빅카인즈에서 내려받은 엑셀(기사 목록) → 7개 언론사 기사만 골라 → 언론사 원문 페이지에서 본문 전체 수집

빅카인즈 엑셀의 '본문'은 앞 200자만 들어 있어서, 'URL' 열의 언론사 원문 페이지에서 본문을 다시 가져온다.

준비:  pip install pandas openpyxl requests beautifulsoup4 trafilatura
실행:  python src/02b_fetch_bigkinds.py

입력:  data/raw/ 안의 빅카인즈 엑셀(.xlsx) 전부 — 파일 이름이 검색어 기록(query)으로 저장됨
출력:  data/raw/news_articles_pilot.csv  (PILOT=False면 news_articles.csv)
       → 다음 단계: src/03_pilot_check.py
"""
import json
import random
import re
import time
from pathlib import Path
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

import pandas as pd
import requests
import trafilatura
from bs4 import BeautifulSoup

from robots_util import fix_url, robots_check

# ==================================================================
# 설정
# ==================================================================
PILOT = False

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_CSV = RAW_DIR / ("news_articles_pilot.csv" if PILOT else "news_articles.csv")

# 언론사 → (코드, 성향)   ※ 코드는 문장 ID를 만들 때만 사용
OUTLETS = {
    "한겨레": ("028", "진보"), "경향신문": ("032", "진보"),
    "조선일보": ("023", "보수"), "중앙일보": ("025", "보수"), "동아일보": ("020", "보수"),
    "한국경제": ("015", "경제"), "매일경제": ("009", "경제"),
}

RETRY_FAILED = True            # True면 지난 실행에서 본문짧음·오류·robots 판정·미리보기 불일치 기사를 다시 수집
EXCLUDE_INTERNATIONAL = True   # 통합 분류가 '국제>…'인 기사(일본·중국 저출산 등) 제외 — 국내 정책 분석이 목적
SLEEP = (1.0, 2.0)             # 요청 사이 대기(초)
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# trafilatura가 실패할 때 쓰는 언론사별 본문 위치 (예비용)
FALLBACK_SELECTORS = {
    "hani.co.kr": [".article-text", "div.text"],
    "khan.co.kr": ["#articleBody", ".art_body"],
    "chosun.com": ["section.article-body", ".article-body"],
    "joongang.co.kr": ["#article_body"],
    "donga.com": ["section.news_view", ".article_txt"],
    "hankyung.com": ["#articletxt", ".article-body"],
    "mk.co.kr": [".news_cnt_detail_wrap", "#article_body"],
}

# 본문을 받기 전에 빅카인즈 엑셀의 제목·키워드만으로 관련 없는 기사를 거른다 (본 수집 시간 단축)
# 파일럿 검증: 기준 5 → 254건 중 113건만 받아도, 03의 관련도 필터를 통과한 93건을 하나도 놓치지 않음
PREFILTER_MIN = 5
MAX_PER_MONTH = 80             # 본 수집: 월별 최대 기사 수 (기간 고르게 + 수집 시간 제한). None이면 제한 없음
POLICY_TOPIC = re.compile(r"저출생|저출산|출산율|출생아|합계출산|인구\s*(정책|전략|위기|감소)")
AREA_WORDS = re.compile(
    r"다자녀|부모\s*급여|아동\s*수당|첫만남\s*이용권|출산\s*(지원금|장려금|축하금)|양육\s*수당|현금\s*(지원|성)|"
    r"육아\s*휴직|출산\s*휴가|배우자\s*출산|근로\s*시간\s*단축|6\s*\+\s*6|유연\s*근무|"
    r"어린이집|보육|늘봄|돌봄|유치원|유보\s*통합|"
    r"특례\s*대출|신혼\s*부부|공공\s*임대|주거\s*지원|청약|전세\s*대출|"
    r"사교육|교육비|학원비|등록금|일\s*[·ㆍ・-]?\s*가정\s*양립|워라밸|경력\s*단절|경단녀"
)


def prefilter_score(title, keywords):
    text = f"{title} {str(keywords).replace(',', ' ')}"
    in_title = bool(POLICY_TOPIC.search(title) or AREA_WORDS.search(title))
    return in_title, len(POLICY_TOPIC.findall(text)) + len(AREA_WORDS.findall(text))


OPINION_RE = re.compile(r"^\s*[\[［<〈]\s*(사설|칼럼|시론|기고|오피니언|논설|포럼|데스크|기자수첩|특별기고|만물상|분수대|횡설수설|사내칼럼)")
YT_EMBED_RE = re.compile(r"<iframe[^>]+src=[\"'][^\"']*(?:youtube\.com/embed/|youtu\.be/)([\w-]{11})")


def nap():
    time.sleep(random.uniform(*SLEEP))


# ==================================================================
# 1단계: 빅카인즈 엑셀 읽기
# ==================================================================
def load_bigkinds():
    files = sorted(RAW_DIR.glob("*.xlsx"))
    if not files:
        raise SystemExit(f"{RAW_DIR} 에 빅카인즈 엑셀(.xlsx)이 없습니다.")
    frames = []
    for f in files:
        d = pd.read_excel(f, dtype=str)
        if "뉴스 식별자" not in d.columns or "URL" not in d.columns:
            print(f"  - {f.name}: 빅카인즈 형식이 아니라 건너뜀")
            continue
        d["query"] = f.stem.replace(" 자료 엑셀", "").strip()
        frames.append(d)
        print(f"  - {f.name}: {len(d):,}건")
    df = pd.concat(frames, ignore_index=True)
    n_all = len(df)

    df = df[df["언론사"].isin(OUTLETS)]
    n_outlet = len(df)
    df = df[df["분석제외 여부"].fillna("") != "중복"]
    n_intl = 0
    if EXCLUDE_INTERNATIONAL:
        intl = df["통합 분류1"].fillna("").str.startswith("국제")
        n_intl = int(intl.sum())
        df = df[~intl]
    # 같은 기사가 여러 검색어에 걸리면 검색어를 합쳐서 1건으로
    df = (df.groupby("뉴스 식별자", as_index=False)
            .agg({**{c: "first" for c in df.columns if c not in ("뉴스 식별자", "query")},
                  "query": lambda s: ";".join(sorted(set(s)))}))
    print(f"\n  전체 {n_all:,}건 → 7개 언론사 {n_outlet:,}건 → 국제 기사 제외 {n_intl}건 → "
          f"중복 제거 후 {len(df):,}건")

    # 제목·키워드 사전 필터
    sc = [prefilter_score(str(t), k) for t, k in zip(df["제목"].fillna(""), df["키워드"].fillna(""))]
    df["prefilter"] = [s for _, s in sc]
    keep = pd.Series([t or s >= PREFILTER_MIN for t, s in sc], index=df.index)
    print(f"  제목·키워드 사전 필터(기준 {PREFILTER_MIN}) → {int(keep.sum()):,}건 "
          f"({keep.mean():.0%}, 나머지는 저출생과 관련이 적어 본문을 받지 않음)")
    df = df[keep]

    # 월별 상한 (본 수집만): 기간을 고르게 + 수집 시간 제한
    if not PILOT and MAX_PER_MONTH:
        df["month"] = df["일자"].str[:6]
        before = len(df)
        # 사설·칼럼은 평가 문장이 많아(파일럿) 상한에서 먼저 남기고, 나머지는 무작위
        df["opinion"] = df["제목"].fillna("").str.contains(OPINION_RE)
        df = (df.sample(frac=1, random_state=42).sort_values("opinion", ascending=False, kind="stable")
                .groupby("month", group_keys=False).head(MAX_PER_MONTH).drop(columns=["month", "opinion"]))
        print(f"  월별 최대 {MAX_PER_MONTH}건 → {len(df):,}건 (제외 {before - len(df):,}건)")
    print("  " + " / ".join(f"{k} {v}" for k, v in df["언론사"].value_counts().items()))
    return df


# ==================================================================
# 2단계: 원문 본문 수집
# ==================================================================
def clean_url(url):
    p = urlparse(fix_url(url))
    q = [(k, v) for k, v in parse_qsl(p.query) if not k.startswith("utm_")]
    return urlunparse(p._replace(query=urlencode(q)))


def allowed(url):
    return robots_check(url, UA)


def fetch_html(url):
    r = requests.get(url, headers={"User-Agent": UA}, timeout=15)
    r.raise_for_status()
    if not r.encoding or r.encoding.lower() == "iso-8859-1":
        r.encoding = r.apparent_encoding
    return r.text


def extract_fusion(html):
    """조선일보처럼 본문을 자바스크립트 데이터(Fusion.globalContent)로 넣어두는 사이트용."""
    m = re.search(r"Fusion\.globalContent\s*=\s*(\{.*?\});\s*Fusion\.", html, re.S)
    if not m:
        return ""
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        return ""
    paras = []
    for el in data.get("content_elements", []):
        if el.get("type") == "text" and el.get("content"):
            paras.append(BeautifulSoup(el["content"], "html.parser").get_text().strip())
    return "\n".join(p for p in paras if p)


AI_SUMMARY_RE = re.compile(r"Key Points|기사 내용 부족|[\U0001F300-\U0001FAFF]")


def _norm(t):
    return re.sub(r"\s+", "", str(t))


def preview_keys(preview):
    """빅카인즈 200자 미리보기에서 15글자 조각 몇 개 (부제목 때문에 앞부분이 달라도 확인되도록)."""
    n = _norm(preview).rstrip(".")
    return [n[i:i + 15] for i in (0, 30, 60, 90, 120) if len(n) >= i + 15]


def preview_match(preview, body):
    """미리보기 조각 중 하나라도 본문에 있으면 본문을 제대로 가져온 것으로 본다."""
    keys, b = preview_keys(preview), _norm(body)
    return bool(keys) and any(k in b for k in keys)


def _node_text(node):
    for t in node.select("script, style, figure, figcaption, .ad, [class*=ad_], [class*=summary], [class*=ai_]"):
        t.decompose()
    for br in node.find_all("br"):
        br.replace_with("\n")
    return re.sub(r"\n\s*\n+", "\n", node.get_text("\n")).strip()


def candidates(html, url):
    """본문 후보를 우선순위대로 만든다: trafilatura → 자바스크립트 데이터 → articleBody 표시 → 언론사별 위치"""
    yield "trafilatura", trafilatura.extract(html, include_comments=False, include_tables=False,
                                             favor_precision=True) or ""
    yield "fusion_json", extract_fusion(html)
    host = urlparse(url).netloc.replace("www.", "")
    sels = ['[itemprop="articleBody"]']
    for key, extra in FALLBACK_SELECTORS.items():
        if key in host:
            sels += extra
    for sel in sels:
        node = BeautifulSoup(html, "html.parser").select_one(sel)
        if node:
            yield f"selector:{sel}", _node_text(node)


def extract_body(html, url, preview=""):
    """미리보기와 일치하는 첫 후보를 본문으로 고른다. 없으면 가장 긴 후보(AI 요약으로 보이는 후보 제외)."""
    cands = [(m, b) for m, b in candidates(html, url) if len(b) >= 200]
    for m, b in cands:
        if preview_match(preview, b) and not AI_SUMMARY_RE.search(b):
            return b, m
    clean = [(m, b) for m, b in cands if not AI_SUMMARY_RE.search(b)]
    pool = clean or cands
    if not pool:
        return "", "none"
    m, b = max(pool, key=lambda x: len(x[1]))
    return b, m + ("" if clean else "(AI요약의심)")


def main():
    print(f"모드: {'파일럿' if PILOT else '본 수집'}\n[1단계] 빅카인즈 엑셀 읽기")
    df = load_bigkinds()

    done = set()
    if OUT_CSV.exists():
        prev = pd.read_csv(OUT_CSV, dtype=str)
        if RETRY_FAILED:
            retry = (prev["fetch_status"].fillna("").str.match(r"short_body|error|robots")
                     | (prev["preview_match"].astype(str) != "True")
                     | prev["body"].fillna("").str.contains("Key Points"))
            if retry.any():
                print(f"\n  지난 실행에서 실패한 {int(retry.sum())}건을 다시 수집합니다.")
                prev = prev[~retry]
                prev.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        done = set(prev["aid"])
    todo = df[~df["뉴스 식별자"].isin(done)]
    print(f"\n[2단계] 원문 본문 수집: 대상 {len(df):,} / 남은 {len(todo):,}")

    buf, t0, stats = [], time.time(), {"ok": 0, "robots": 0, "error": 0, "short": 0}
    for i, row in enumerate(todo.to_dict("records"), 1):
        url = clean_url(row["URL"])
        press = row["언론사"]
        oid, lean = OUTLETS[press]
        body, method, links, status = "", "", "", "ok"
        ok_robots, robots_reason = allowed(url)
        if not ok_robots:
            status = f"robots:{robots_reason}"
            stats["robots"] += 1
        else:
            try:
                html = fetch_html(url)
                body, method = extract_body(html, url, row["본문"])
                if PILOT and not preview_match(row["본문"], body):   # 원인 확인용으로 원본 페이지 저장
                    dbg = RAW_DIR / "debug_html"
                    dbg.mkdir(exist_ok=True)
                    (dbg / f"{press}_{row['뉴스 식별자']}.html").write_text(html, encoding="utf-8")
                links = " ".join(f"https://youtu.be/{v}" for v in sorted(set(YT_EMBED_RE.findall(html))))
                if len(body) < 200:
                    status = "short_body"
                    stats["short"] += 1
                else:
                    stats["ok"] += 1
            except Exception as e:
                status = f"error:{e.__class__.__name__}"
                stats["error"] += 1
        title = row["제목"] or ""
        buf.append({
            "oid": oid, "aid": row["뉴스 식별자"], "url": url, "press": press, "outlet_lean": lean,
            "article_type": "사설·칼럼·기고" if OPINION_RE.search(title) else "스트레이트",
            "query": row["query"], "pub_date": pd.to_datetime(row["일자"], format="%Y%m%d").strftime("%Y-%m-%d"),
            "title": title, "category": row["통합 분류1"], "body": body, "body_links": links,
            "body_len": len(body), "extract_method": method, "fetch_status": status,
            "preview_match": preview_match(row["본문"], body),
        })
        if len(buf) >= 20 or i == len(todo):
            pd.DataFrame(buf).to_csv(OUT_CSV, mode="a", index=False, encoding="utf-8-sig",
                                     header=not OUT_CSV.exists())
            buf = []
        if i % 10 == 0 or i == len(todo):
            print(f"  [{i:>5}/{len(todo)}] 성공 {stats['ok']} / 본문짧음 {stats['short']} / "
                  f"오류 {stats['error']} / robots차단 {stats['robots']} | {time.time() - t0:5.0f}초", flush=True)
        nap()

    out = pd.read_csv(OUT_CSV, dtype=str).drop_duplicates("aid")
    out["body_len"] = out["body_len"].astype(int)
    out["preview_match"] = out["preview_match"] == "True"
    print("\n========== 수집 결과 ==========")
    print(f"기사 {len(out):,}건")
    print(pd.crosstab(out["press"], out["fetch_status"]).to_string())
    ok = out[out["fetch_status"] == "ok"]
    print("\n언론사별 본문 길이(중앙값)와 미리보기 일치율 (일치율이 낮은 언론사는 본문 추출이 잘못됐을 가능성)")
    print(ok.groupby("press").agg(기사수=("aid", "count"), 본문길이=("body_len", "median"),
                                  미리보기일치=("preview_match", "mean")).round(2).to_string())
    print(f"\n저장 → {OUT_CSV}\n다음 단계: src/03_pilot_check.py")


if __name__ == "__main__":
    main()
