"""
00_debug_search.py
네이버 뉴스 검색에서 기사가 0건 나오는 원인 확인용 (1~2분)

실행: python src/00_debug_search.py
결과: data/raw/debug_requests.html, data/raw/debug_selenium.html 저장 + 진단 내용 출력
"""
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw"
OUT.mkdir(parents=True, exist_ok=True)

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
QUERY = "부모급여"
PARAMS = {"where": "news", "query": QUERY, "sort": 0, "pd": 3,
          "ds": "2024.01.01", "de": "2024.01.31",
          "nso": "so:r,p:from20240101to20240131", "start": 1}

PATTERNS = {
    "n.news.naver.com/mnews/article/숫자/숫자": r"n\.news\.naver\.com/(?:mnews/)?article/\d{3}/\d{10}",
    "news.naver.com (전체)": r"news\.naver\.com",
    "oid= 형식": r"oid=\d{3}",
    "언론사 원문 링크(http)": r"https?://(?:www\.)?(?:hani|khan|chosun|joongang|donga|hankyung|mk)\.co",
}


def diagnose(name, html):
    print(f"\n[{name}] HTML 길이 {len(html):,}자")
    for label, p in PATTERNS.items():
        print(f"  {label:<40} {len(re.findall(p, html)):>4}개")
    for word in ["부모급여", "네이버뉴스", "로봇", "captcha", "비정상"]:
        print(f"  '{word}' 등장 {html.count(word)}회")


def main():
    print("1) requests 방식")
    try:
        r = requests.get("https://search.naver.com/search.naver", params=PARAMS,
                         headers={"User-Agent": UA, "Referer": "https://search.naver.com/"}, timeout=10)
        print(f"  상태 코드 {r.status_code}, 최종 주소 {r.url[:120]}")
        (OUT / "debug_requests.html").write_text(r.text, encoding="utf-8")
        diagnose("requests", r.text)
    except Exception as e:
        print(f"  요청 실패: {e.__class__.__name__}: {e}")

    print("\n2) selenium 방식 (크롬 창이 잠깐 열립니다)")
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        opts = webdriver.ChromeOptions()
        opts.add_argument(f"user-agent={UA}")
        try:
            from webdriver_manager.chrome import ChromeDriverManager
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
        except Exception:
            driver = webdriver.Chrome(options=opts)
        from urllib.parse import urlencode
        driver.get("https://search.naver.com/search.naver?" + urlencode(PARAMS))
        time.sleep(3)
        for _ in range(3):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1.5)
        html = driver.page_source
        driver.quit()
        (OUT / "debug_selenium.html").write_text(html, encoding="utf-8")
        diagnose("selenium", html)
    except Exception as e:
        print(f"  selenium 실패: {e.__class__.__name__}: {str(e)[:200]}")

    print("\n완료. 위 출력 전체를 복사해서 알려주세요.")


if __name__ == "__main__":
    main()
