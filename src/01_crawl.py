"""
01_crawl.py
해킹 사고 기업 앱의 구글 플레이 리뷰 크롤링

실행 전:  pip install google-play-scraper pandas
실행:     PyCharm에서 Run (또는 python src/01_crawl.py)

동작:
  1) 앱 패키지명이 맞는지 확인 (틀리면 검색 후보를 보여주고 그 앱은 건너뜀)
  2) 최신 리뷰부터 과거로 거슬러 올라가며 수집
  3) '사고일 전 6개월 ~ 사고일 후 6개월' 안의 리뷰만 저장
  4) 앱마다 data/raw/reviews_<앱>.csv 로 저장 (중간 저장 포함)

※ 사고일(incident)은 보도자료로 꼭 다시 확인하고 고칠 것
"""
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from google_play_scraper import Sort, app, reviews, search
from google_play_scraper.exceptions import NotFoundError

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 수집 대상 (패키지명이 틀리면 실행 시 검색 후보가 출력됨 → 여기 고쳐서 다시 실행)
# ------------------------------------------------------------------
APPS = {
    "SKT":    {"package": "com.sktelecom.minit",  "search": "T world SK텔레콤",  "incident": "2025-04-22"},
    "예스24": {"package": "com.yes24.commerce",   "search": "예스24",            "incident": "2025-06-09"},
    "롯데카드": {"package": "com.lcacApp",          "search": "롯데카드 디지로카", "incident": "2025-09-01"},
    "KT":     {"package": "com.ktshow.cs",        "search": "마이케이티 KT",     "incident": "2025-09-08"},
    "쿠팡":   {"package": "com.coupang.mobile",   "search": "쿠팡",              "incident": "2025-11-29"},
}

MONTHS_BEFORE = 6        # 사고 전 수집 기간(개월)
MONTHS_AFTER = 6         # 사고 후 수집 기간(개월)
BATCH = 200              # 한 번에 요청할 리뷰 수 (최대 200)
SLEEP = 0.5              # 요청 사이 대기(초) - 서버 부담 줄이기
SAVE_EVERY = 25          # 몇 번 요청마다 중간 저장할지
MAX_EMPTY_RETRY = 3      # 오류 시 재시도 횟수


def check_package(name, info):
    """패키지명이 맞는지 확인. 틀리면 검색 후보 출력 후 False."""
    try:
        meta = app(info["package"], lang="ko", country="kr")
        print(f"  ✔ {name}: {meta['title']} (리뷰 수 약 {meta.get('reviews') or 0:,})")
        return True
    except NotFoundError:
        print(f"  ✘ {name}: 패키지명 '{info['package']}'를 찾지 못했습니다. 검색 후보:")
        for r in search(info["search"], lang="ko", country="kr", n_hits=5):
            print(f"      - {r['title']}  →  {r['appId']}")
        print(f"    위 후보 중 맞는 appId를 APPS['{name}']['package']에 넣고 다시 실행하세요.")
        return False


def to_rows(batch, name, incident):
    rows = []
    for r in batch:
        rows.append({
            "review_id": r["reviewId"],
            "app": name,
            "date": r["at"],
            "score": r["score"],
            "content": r["content"],
            "thumbs_up": r["thumbsUpCount"],
            "app_version": r.get("reviewCreatedVersion"),
            "reply_content": r.get("replyContent"),     # 기업 답변 (대응 분석에 활용)
            "replied_at": r.get("repliedAt"),
            "incident_date": incident,
        })
    return rows


def crawl(name, info):
    incident = datetime.strptime(info["incident"], "%Y-%m-%d")
    start = incident - timedelta(days=30 * MONTHS_BEFORE)
    end = incident + timedelta(days=30 * MONTHS_AFTER)
    out_path = RAW_DIR / f"reviews_{name}.csv"
    print(f"\n[{name}] 수집 기간 {start.date()} ~ {end.date()} (사고일 {incident.date()})")

    kept, token, n_req, seen, t0 = [], None, 0, 0, time.time()
    while True:
        for attempt in range(MAX_EMPTY_RETRY):
            try:
                batch, token = reviews(
                    info["package"], lang="ko", country="kr",
                    sort=Sort.NEWEST, count=BATCH, continuation_token=token,
                )
                break
            except Exception as e:  # 네트워크 오류 등
                print(f"    오류({e.__class__.__name__}), {attempt + 1}회 재시도...", flush=True)
                time.sleep(3)
        else:
            print("    재시도 실패 → 지금까지 수집한 것만 저장합니다.")
            break

        n_req += 1
        if not batch:
            print("    더 이상 리뷰가 없습니다.")
            break

        seen += len(batch)
        kept += [r for r in to_rows(batch, name, info["incident"]) if start <= r["date"] <= end]
        oldest = min(r["at"] for r in batch)

        if n_req % 5 == 0:
            print(f"    요청 {n_req:>4}회 | 확인 {seen:>7,}건 | 저장 대상 {len(kept):>7,}건 | "
                  f"현재 {oldest.date()}까지 내려옴 | {time.time() - t0:5.0f}초", flush=True)
        if n_req % SAVE_EVERY == 0:
            pd.DataFrame(kept).to_csv(out_path, index=False, encoding="utf-8-sig")

        if oldest < start or token is None:
            break
        time.sleep(SLEEP)

    df = pd.DataFrame(kept).drop_duplicates(subset="review_id")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    if len(df):
        before = (df["date"] < incident).sum()
        print(f"  → {name}: {len(df):,}건 저장 (사고 전 {before:,} / 사고 후 {len(df) - before:,})  {out_path.name}")
    else:
        print(f"  → {name}: 기간 안 리뷰 0건 (사고일·패키지명 확인 필요)")
    return len(df)


def main():
    print("[1단계] 패키지명 확인")
    ok = {name: info for name, info in APPS.items() if check_package(name, info)}

    print("\n[2단계] 리뷰 수집")
    total = {name: crawl(name, info) for name, info in ok.items()}

    print("\n========== 수집 결과 ==========")
    for name, n in total.items():
        print(f"  {name:<6}: {n:>8,}건")
    print(f"  {'합계':<6}: {sum(total.values()):>8,}건  (목표 50,000건 이상)")
    skipped = set(APPS) - set(ok)
    if skipped:
        print(f"\n  건너뛴 앱: {', '.join(skipped)} → 패키지명 고친 뒤 그 앱만 남기고 다시 실행")


if __name__ == "__main__":
    main()
