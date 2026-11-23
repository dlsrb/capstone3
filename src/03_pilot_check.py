"""
03_pilot_check.py
파일럿 기사로 "쓸만한 데이터인지" 판별 + 파일럿 라벨링용 200문장 추출

준비:  pip install pandas kiwipiepy
실행:  python src/03_pilot_check.py

1차 실행 (라벨 전)
  - 정제(상투 문구 제거) → 출처 신뢰도 필터 → 중복 기사 제거 → 문장 분리 → 정책 영역 태그
  - 판별 기준(양 / 정책 영역 / 출처 균형 / 신뢰도) 결과 출력
  - data/labeling/pilot_200.csv 생성 → 엑셀에서 label 열을 직접 채우기
      긍정 / 부정 / 중립 / 관계없음  (README 4장 기준)

2차 실행 (label 열을 채운 뒤 다시 실행)
  - 라벨 분포 판별 (긍정·부정이 각각 10~15% 이상인지) 결과 출력
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PILOT = False                 # False: 본 수집 기사(news_articles.csv) → 문장 파일(sentences.csv)만 만들고 라벨 샘플은 04에서
ARTICLES_CSV = ROOT / "data" / "raw" / ("news_articles_pilot.csv" if PILOT else "news_articles.csv")
SENT_CSV = ROOT / "data" / "processed" / ("pilot_sentences.csv" if PILOT else "sentences.csv")
LABEL_CSV = ROOT / "data" / "labeling" / "pilot_200.csv"
SENT_CSV.parent.mkdir(parents=True, exist_ok=True)
LABEL_CSV.parent.mkdir(parents=True, exist_ok=True)

TARGET_SENTENCES = 50_000
PERIOD = ("2021-07-01", "2026-10-31")   # 분석 기간 (파일럿 엑셀에 섞인 2026-11 기사 제외)
MIN_SENT_LEN = 10                        # 10자 미만 문장 제거 ("그뿐인가.", "[홍길동 기자]" 등 단독으로 의미 없는 문장)
DUP_WINDOW_DAYS = 3                      # 중복 기사 비교 범위: 발행일 ±3일 (받아쓰기·재송고는 며칠 안에 나옴)
N_LABEL = 200
OPINION_SHARE = 0.3          # 라벨링 샘플 중 사설·칼럼 문장 비율 (긍정·부정 확보용)
SEED = 42

# ------------------------------------------------------------------
# 정제: 기사 상투 문구 (줄 단위로 제거)
# ------------------------------------------------------------------
BOILERPLATE = re.compile(
    r"[\w.+-]+@[\w-]+\.[\w.]+"                 # 이메일
    r"|무단\s*전재|재배포\s*금지|AI\s*학습|Copyright|ⓒ|©"
    r"|^\s*[▶☞■◆▲△●]"                          # 관련기사·안내 기호로 시작
    r"|관련\s*기사|구독\s*하기|기사\s*제보"
    r"|^\s*\[?\s*사진\s*[=:]|^\s*\(?\s*사진\s*=|=\s*연합뉴스\s*\]?\s*$"
    r"|^\s*[가-힣]{2,4}\s*(기자|특파원|논설위원|선임기자)\s*$"
    r"|주요\s*뉴스\s*$|현장을\s*소개합니다|생중계될\s*예정"      # 파일럿에서 발견한 언론사 안내 문구
    r"|^\s*\[[^\]]{0,30}(기자|특파원)[^\]]{0,30}\]\s*$"          # 본 수집에서 발견: [홍길동 기자 / 김철수 기자]
    r"|^\s*[가-힣]{2,4}\s*\|\s*(논설위원|기자|특파원)"            # 황보연 | 논설위원
    r"|핵심\s*요약\s*쏙|AI\s*요약은|기사\s*본문을\s*함께\s*확인"   # 매일경제 AI 요약 안내
    r"|외부\s*필진|편집\s*방향과\s*다를\s*수|로그인\s*(후|해)"
)

BYLINE_TAIL = re.compile(r"\s*\[[^\]]{0,30}(기자|특파원)[^\]]{0,30}\]\s*$")

# ------------------------------------------------------------------
# 출처 신뢰도 필터 (README 2.3)
# ------------------------------------------------------------------
REMOVE_FLAGS = {
    "기사형광고": re.compile(r"\[\s*AD\s*\]|\[\s*광고\s*\]|\[\s*홍보\s*\]|협찬"),
}
MARK_FLAGS = {
    "커뮤니티·SNS인용": re.compile(r"온라인\s*커뮤니티|누리꾼|SNS에\s*올라온|게시글이\s*화제|커뮤니티에\s*올라온|댓글"),
    "익명출처": re.compile(r"관계자에\s*따르면|라는\s*말이\s*나온다|는\s*후문이다|는\s*전언이다"),
    "정정기사": re.compile(r"바로잡습니다|정정\s*보도|반론\s*보도"),
}
VIDEO_MENTION = re.compile(r"(유튜브|틱톡|인스타그램|SNS)\s*(에서|에|을\s*통해)")
CHANNEL_NAMED = re.compile(r"(채널|계정)\s*['\"‘“]?[\w가-힣]+")
YOUTUBE_LINK = re.compile(r"youtube\.com|youtu\.be")

# ------------------------------------------------------------------
# 정책 영역 태그 (README 4.3)
# ------------------------------------------------------------------
AREAS = {
    "현금 지원": r"다자녀|부모\s*급여|아동\s*수당|첫만남\s*이용권|출산\s*(지원금|장려금|축하금)|양육\s*수당|현금\s*(지원|성)",
    "육아휴직·근로시간": r"육아\s*휴직|출산\s*휴가|배우자\s*출산|근로\s*시간\s*단축|6\s*\+\s*6|유연\s*근무",
    "보육·돌봄": r"어린이집|보육|늘봄|돌봄|유치원|유보\s*통합",
    "주거": r"특례\s*대출|신혼\s*부부\s*(주택|특공|특별\s*공급)|공공\s*임대|주거\s*지원|청약|전세\s*대출",
    "교육비": r"사교육|교육비|학원비|등록금",
    "일·가정 양립": r"일\s*[·ㆍ・-]?\s*가정\s*양립|워라밸|경력\s*단절|경단녀",
}
AREAS = {k: re.compile(v) for k, v in AREAS.items()}
POLICY_TOPIC = re.compile(r"저출생|저출산|출산율|출생아|합계출산|인구\s*(정책|전략|위기|감소)")
MIN_RELEVANCE = 10  # 기사 관련도 = 정책 영역 키워드 수 + 저출생 키워드 수. 제목에 저출생·정책 키워드가 없으면 이 값 이상이어야 유지
                    # (파일럿 라벨링 결과: 관련도 10 이하 기사 문장의 72%가 관계없음 → 3에서 10으로 상향)


def article_relevance(title, body):
    text = f"{title}\n{body}"
    area_hits = {k: len(p.findall(text)) for k, p in AREAS.items()}
    topic_hits = len(POLICY_TOPIC.findall(text))
    keep = bool(POLICY_TOPIC.search(title)) or any(p.search(title) for p in AREAS.values()) \
        or (sum(area_hits.values()) + topic_hits >= MIN_RELEVANCE)
    top = max(area_hits, key=area_hits.get)
    main_area = top if area_hits[top] >= 2 else ""      # 기사 전체에서 2번 이상 나온 대표 영역
    return keep, main_area


DEMONSTRATIVE = re.compile(r"^\s*(이는|이것은|이 제도|해당 제도|이 정책|해당 정책|이같은|이 같은|이러한|이런)")


def tag_areas(text):
    return [k for k, p in AREAS.items() if p.search(text)]


def clean_body(body):
    lines = [ln.strip() for ln in str(body).split("\n")]
    return "\n".join(ln for ln in lines if ln and not BOILERPLATE.search(ln))


def article_flags(title, body, links):
    text = f"{title}\n{body}"
    remove = [k for k, p in REMOVE_FLAGS.items() if p.search(text)]
    mark = [k for k, p in MARK_FLAGS.items() if p.search(text)]
    if VIDEO_MENTION.search(body) and not CHANNEL_NAMED.search(body) and not YOUTUBE_LINK.search(str(links)):
        remove.append("출처불명인용")
    if YOUTUBE_LINK.search(str(links)):
        mark.append("영상링크(확인필요)")      # 본 수집 때 oEmbed로 공식 채널·링크 확인
    return remove, mark


def near_duplicates(df, threshold=0.8, window_days=DUP_WINDOW_DAYS):
    """본문 5글자 조각(5-gram) 자카드 유사도로 중복 기사 찾기.
    모든 쌍을 비교하면 기사 수의 제곱만큼 걸리므로, 발행일이 window_days 이내인 기사끼리만 비교한다.
    (보도자료 받아쓰기·반복 송고는 보통 며칠 안에 나오기 때문)"""
    d = df.assign(_date=pd.to_datetime(df["pub_date"], errors="coerce")).sort_values("_date")
    idx = list(d.index)
    dates = list(d["_date"])
    grams = [set(b[i:i + 5] for i in range(max(len(b) - 4, 0))) for b in d["clean"]]
    win = pd.Timedelta(days=window_days)
    dup = set()
    for i in range(len(idx)):
        if idx[i] in dup or not grams[i]:
            continue
        for j in range(i + 1, len(idx)):
            if dates[j] - dates[i] > win:
                break
            if idx[j] in dup or not grams[j]:
                continue
            inter = len(grams[i] & grams[j])
            if inter and inter / len(grams[i] | grams[j]) >= threshold:
                dup.add(idx[j])          # 먼저 나온 기사를 대표로 남김
    return dup


def split_sentences(df):
    from kiwipiepy import Kiwi
    kiwi = Kiwi()
    rows = []
    for a in df.itertuples():
        title_areas = tag_areas(a.title)
        article_area = [a.main_area] if a.main_area else []
        prev_areas, n = [], 0
        for line in a.clean.split("\n"):
            for s in kiwi.split_into_sents(line):
                sent = BYLINE_TAIL.sub("", s.text).strip()     # 문장 끝에 붙은 "[홍길동 기자]" 제거
                if len(sent) < MIN_SENT_LEN or BOILERPLATE.search(sent):
                    continue
                areas = tag_areas(sent)
                source = "직접"
                if not areas and DEMONSTRATIVE.search(sent):
                    areas, source = (prev_areas, "앞문장") if prev_areas else (title_areas, "제목")
                if areas:
                    prev_areas = areas
                elif article_area:                       # 영역 언급이 없는 문장은 기사 대표 영역으로
                    areas, source = article_area, "기사"
                n += 1
                rows.append({
                    "sent_id": f"{a.oid}_{a.aid}_{n:03d}", "oid": a.oid, "aid": a.aid,
                    "press": a.press, "outlet_lean": a.outlet_lean, "article_type": a.article_type,
                    "pub_date": a.pub_date, "title": a.title, "sentence": sent,
                    "policy_area": ";".join(areas) if areas else "기타",
                    "area_source": source if areas else "",
                    "flag": a.flag,
                })
    return pd.DataFrame(rows)


def verdict(ok):
    return "통과" if ok else "주의"


def first_run():
    if not ARTICLES_CSV.exists():
        sys.exit(f"{ARTICLES_CSV} 가 없습니다. 먼저 02_crawl_news.py (PILOT=True)를 실행하세요.")
    df = pd.read_csv(ARTICLES_CSV, dtype={"oid": str, "aid": str}).drop_duplicates(["oid", "aid"])
    if "fetch_status" in df.columns:      # 02b(빅카인즈) 결과: 본문을 제대로 가져온 기사만
        df = df[df["fetch_status"] == "ok"]
    df = df[df["body_len"].astype(int) > 0].reset_index(drop=True)
    df["title"] = df["title"].fillna("")
    in_period = df["pub_date"].between(*PERIOD)
    if (~in_period).any():
        print(f"  분석 기간({PERIOD[0]} ~ {PERIOD[1]}) 밖 기사 {(~in_period).sum()}건 제외")
    df = df[in_period].reset_index(drop=True)
    n0 = len(df)

    # 정제 + 필터
    df["clean"] = df["body"].map(clean_body)
    fl = [article_flags(t, b, l) for t, b, l in zip(df["title"], df["body"], df["body_links"].astype("string").fillna(""))]
    df["remove_reason"] = [";".join(r) for r, _ in fl]
    # 언론사가 붙인 AI 요약문이 본문으로 들어간 기사 (기자가 쓴 문장이 아니므로 제외)
    ai = df["body"].fillna("").str.contains("Key Points")
    if "extract_method" in df.columns:
        ai |= df["extract_method"].fillna("").str.contains("AI요약")
    df.loc[ai, "remove_reason"] = (df.loc[ai, "remove_reason"] + ";AI요약본문").str.strip(";")
    # 저출생 정책과 거의 관련 없는 기사 (예: '저출산'이 한 번 스치듯 언급된 기사)
    rel = [article_relevance(t, b) for t, b in zip(df["title"], df["body"].fillna(""))]
    df["main_area"] = [m for _, m in rel]
    off = ~pd.Series([k for k, _ in rel], index=df.index)
    df.loc[off, "remove_reason"] = (df.loc[off, "remove_reason"] + ";주제무관").str.strip(";")
    df["flag"] = [";".join(m) for _, m in fl]
    removed = df[df["remove_reason"] != ""]
    df = df[df["remove_reason"] == ""]
    dup = near_duplicates(df)
    df = df.drop(index=list(dup))

    sents = split_sentences(df)
    n_sent_raw = len(sents)
    sents = sents.drop_duplicates("sentence", keep="first")    # 같은 문장이 여러 기사에 반복 (받아쓰기·연재 안내 문구)
    n_sent_dup = n_sent_raw - len(sents)
    sents.to_csv(SENT_CSV, index=False, encoding="utf-8-sig")

    # ---------------- 판별 결과 ----------------
    print("=" * 60)
    print("파일럿 데이터 판별 결과")
    print("=" * 60)
    print(f"\n[기사] 수집 {n0}건 → 필터 제거 {len(removed)}건 → 중복 제거 {len(dup)}건 → 최종 {len(df)}건")
    if len(removed):
        print("  제거 사유:", removed["remove_reason"].str.split(";").explode().value_counts().to_dict())
    marked = df["flag"].str.split(";").explode()
    marked = marked[marked != ""]
    print("  표시(flag):", marked.value_counts().to_dict() if len(marked) else "없음")

    spa = len(sents) / max(len(df), 1)
    need = int(TARGET_SENTENCES / spa) if spa else 0
    print(f"\n[1. 양] 문장 {len(sents):,}개 (중복 문장 {n_sent_dup:,}개 제거), 기사당 평균 {spa:.1f}문장")
    print(f"  → 5만 문장에 필요한 기사 수 ≈ {need:,}건  (본 수집 링크 수와 비교해서 판단)")

    print("\n[2. 문장 길이]")
    ln = sents["sentence"].str.len()
    print(f"  평균 {ln.mean():.0f}자 / 중앙값 {ln.median():.0f}자 / 최대 {ln.max()}자 / 200자 초과 {(ln > 200).mean():.1%}")

    print("\n[3. 정책 영역] (다중 태그는 각각 집계)")
    areas = sents["policy_area"].str.split(";").explode().value_counts()
    share = areas / len(sents)
    for k, v in areas.items():
        print(f"  {k:<10} {v:>6,}  ({share[k]:.1%})")
    tagged = [a for a in AREAS if a in areas.index]
    print(f"  → 6개 영역 중 등장 {len(tagged)}개  [{verdict(len(tagged) >= 5)}]")
    src = sents["area_source"].replace("", "태그없음").value_counts(normalize=True)
    print("  태그 근거: " + " / ".join(f"{k} {v:.0%}" for k, v in src.items()))

    print("\n[4. 출처 균형]")
    lean = df["outlet_lean"].value_counts(normalize=True)
    print("  " + " / ".join(f"{k} {v:.0%}" for k, v in lean.items()))
    print(f"  → 가장 큰 쪽 비율 {lean.max():.0%}  [{verdict(lean.max() <= 0.5)}]")
    print(f"  사설·칼럼·기고 기사: {(df['article_type'] != '스트레이트').sum()}건")

    print("\n[5. 기간]")
    months = pd.to_datetime(df["pub_date"], errors="coerce").dt.to_period("M").value_counts().sort_index()
    print("  " + ", ".join(f"{k}: {v}" for k, v in months.items()))

    # ---------------- 라벨링 샘플 ----------------
    if not PILOT:
        print(f"\n문장 파일 저장 → {SENT_CSV}\n다음 단계: src/04_make_label_set.py")
        return
    if LABEL_CSV.exists():
        print(f"\n{LABEL_CSV.name} 이 이미 있어 새로 만들지 않습니다. (다시 만들려면 파일을 지우고 실행)")
        return
    op = sents[sents["article_type"] != "스트레이트"]
    st = sents[sents["article_type"] == "스트레이트"]
    n_op = min(len(op), int(N_LABEL * OPINION_SHARE))
    sample = pd.concat([op.sample(n_op, random_state=SEED),
                        st.sample(min(len(st), N_LABEL - n_op), random_state=SEED)])
    sample = sample.sample(frac=1, random_state=SEED)
    sample["label"] = ""
    sample["note"] = ""
    cols = ["sent_id", "press", "outlet_lean", "article_type", "title", "sentence",
            "policy_area", "label", "note"]
    sample[cols].to_csv(LABEL_CSV, index=False, encoding="utf-8-sig")
    print(f"\n라벨링용 {len(sample)}문장 저장 → {LABEL_CSV}")
    print("  엑셀에서 label 열에 긍정/부정/중립/관계없음을 입력하고, policy_area가 틀리면 고친 뒤")
    print("  이 스크립트를 다시 실행하면 라벨 분포를 판별합니다.")


def second_run(lab):
    lab = lab[lab["label"].notna() & (lab["label"].astype(str).str.strip() != "")].copy()
    lab["label"] = lab["label"].astype(str).str.replace(r"\s+", "", regex=True)   # '관계 없음' → '관계없음'
    valid = {"긍정", "부정", "중립", "관계없음"}
    bad = lab[~lab["label"].isin(valid)]
    if len(bad):
        print(f"잘못된 라벨 {len(bad)}건 (긍정/부정/중립/관계없음만 가능):")
        print(bad[["sent_id", "label"]].to_string(index=False))
    lab = lab[lab["label"].isin(valid)]
    print("=" * 60)
    print(f"파일럿 라벨 분포 판별 ({len(lab)}문장)")
    print("=" * 60)
    dist = lab["label"].value_counts(normalize=True).reindex(list(valid), fill_value=0)
    for k in ["긍정", "부정", "중립", "관계없음"]:
        print(f"  {k:<5} {dist[k]:6.1%}")
    ok = dist["긍정"] >= 0.10 and dist["부정"] >= 0.10
    print(f"\n  → 긍정·부정 각각 10% 이상: [{verdict(ok)}]")
    if not ok:
        print("     사설·칼럼 비중을 높이거나(OPINION_SHARE), 평가 표현이 많은 검색어를 추가하세요.")
    print("\n[기사 유형별 라벨 분포]")
    print(pd.crosstab(lab["article_type"], lab["label"], normalize="index").round(2).to_string())
    print("\n[언론 성향별 라벨 분포]")
    print(pd.crosstab(lab["outlet_lean"], lab["label"], normalize="index").round(2).to_string())


def main():
    if PILOT and LABEL_CSV.exists():
        lab = pd.read_csv(LABEL_CSV, dtype=str)
        if lab["label"].notna().any() and (lab["label"].fillna("").str.strip() != "").any():
            second_run(lab)
            return
    first_run()


if __name__ == "__main__":
    main()
