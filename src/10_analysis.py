"""
10_analysis.py
EDA(원본·학습 데이터)와 결과 분석(H1~H3, 키워드 네트워크) 그래프·표를 한 번에 만든다.

입력
    data/raw/news_articles.csv          수집 원본 (EDA)
    data/processed/sentences.csv        정제 문장 78,935개 (언론사·날짜·정책 영역)
    data/labeling/label_set.csv         라벨링 데이터 (학습 데이터 EDA)
    koelectra/predictions_all.csv       KoELECTRA 전체 분류 결과
출력
    images/eda_*.png      원본 데이터 EDA
    images/train_*.png    학습 데이터 EDA
    images/result_*.png   결과 분석
    results/*.csv         그래프에 쓴 수치 (원문 문장 없음 → GitHub 공개 가능)

준비:  pip install kiwipiepy networkx matplotlib
실행:  python src/10_analysis.py      (전체 약 3~5분, 대부분 형태소 분석 시간)

방법 설명 (보고서용)
    - 비율: 각 집단에서 '관계없음'을 뺀 문장(긍정+부정+중립) 중 긍정·부정이 차지하는 비율
    - 키워드: kiwipiepy로 명사(NNG·NNP)만 추출, 불용어 제거 후 빈도 계산
    - 키워드 네트워크: 같은 문장에 함께 나온 명사 쌍을 연결(동시출현). 원 크기 = 빈도, 선 굵기 = 함께 나온 횟수
"""
import re
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "images"
RES = ROOT / "results"
IMG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

# ---------------------------------------------------------------- 공통 설정
for f in ["Malgun Gothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"]:
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

COLOR = {"긍정": "#2E8B57", "부정": "#C8443A", "중립": "#A9A9A9", "관계없음": "#E3E3E3"}
LEAN_COLOR = {"진보": "#3A6EA5", "보수": "#C8443A", "경제": "#D9A441"}
LABELS = ["긍정", "부정", "중립", "관계없음"]
AREAS = ["현금 지원", "육아휴직·근로시간", "보육·돌봄", "주거", "교육비", "일·가정 양립"]
LEANS = ["진보", "보수", "경제"]

# H2 기준 시점 (README 1.3 표. 보도자료로 확인 후 날짜를 고칠 것)
POLICIES = {
    "첫만남이용권": ("2022-01", r"첫만남"),
    "부모급여": ("2023-01", r"부모\s*급여"),
    "6+6 부모육아휴직제": ("2024-01", r"6\s*\+\s*6|부모\s*육아\s*휴직제"),
    "신생아 특례대출": ("2024-01", r"특례\s*대출|신생아\s*(?:특례\s*)?대출"),
    "늘봄학교 전국 확대": ("2024-09", r"늘봄"),
    "육아휴직 급여 인상": ("2025-01", r"육아\s*휴직\s*급여"),
}
STOP = set("""것 수 등 때 년 월 일 명 개 이 그 중 위 말 곳 기자 정도 경우 이번 지난해 올해 내년 국내 우리 이상 이하 때문
관련 대상 지원 정책 문제 정부 저출생 저출산 출산 사회 가운데 상황 최근 생각 이후 전 후 측 바 뿐 데 만 원 억 조 만원 억원
씨 분 점 앞 뒤 안 밖 간 내 외 해 당시 현재 지금 기준 수준 결과 이유 부분 방안 계획 사업 제도 대책 추진 확대 운영 마련
지적 분석 한국 평가 전문가 설명 강조 주장 발표 조사 응답 비율 가능 필요""".split())
# 정책 용어를 한 단어로 묶어 형태소 분석 (예: 육아휴직이 '육아'+'휴직'으로 쪼개지지 않게)
USER_WORDS = ["육아휴직", "부모급여", "아동수당", "첫만남이용권", "늘봄학교", "신생아특례대출", "특례대출", "출산율",
              "출생아", "저출생", "저출산", "어린이집", "유보통합", "공공임대", "출산장려금", "난임", "다자녀",
              "일가정양립", "경력단절", "사각지대", "돌봄교실", "가사관리사", "인구전략기획부", "저출산고령사회위원회"]


def mark_policies(ax, color="#888"):
    """정책 시행 시점 세로선. 같은 날짜의 정책은 이름을 한 줄로 합친다"""
    by_date = {}
    for name, (d, _) in POLICIES.items():
        by_date.setdefault(d, []).append(name)
    top = ax.get_ylim()[1]
    for d, names in by_date.items():
        x = pd.Period(d).to_timestamp()
        ax.axvline(x, color=color, ls="--", lw=0.8)
        ax.text(x, top, " · ".join(names), rotation=90, va="top", ha="right", fontsize=7, color=color)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG / name, dpi=200)
    plt.close(fig)
    print("  저장:", name)


def stacked_share(table, ax, title, order=None, cols=("긍정", "부정", "중립")):
    """table: index=집단, columns=라벨 비율(0~1). 가로 누적 막대"""
    t = table.reindex(order) if order else table
    left = pd.Series(0.0, index=t.index)
    for c in cols:
        ax.barh(t.index, t[c], left=left, color=COLOR[c], label=c)
        for i, (v, l) in enumerate(zip(t[c], left)):
            if v >= 0.04:
                ax.text(l + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if c != "중립" else "black")
        left += t[c]
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.invert_yaxis()
    ax.set_title(title)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.25), frameon=False)


def share(df, by):
    t = pd.crosstab(df[by], df["pred"], normalize="index")
    return t.reindex(columns=["긍정", "부정", "중립"], fill_value=0)


# ---------------------------------------------------------------- 데이터 읽기
print("데이터 읽는 중...")
art = pd.read_csv(ROOT / "data/raw/news_articles.csv", dtype=str, keep_default_na=False)
sent = pd.read_csv(ROOT / "data/processed/sentences.csv", dtype=str, keep_default_na=False)
lab = pd.read_csv(ROOT / "data/labeling/label_set.csv", dtype=str, keep_default_na=False)
pred = pd.read_csv(ROOT / "koelectra/predictions_all.csv", dtype={"sent_id": str})
df = sent.merge(pred, on="sent_id", how="inner")
df["month"] = pd.to_datetime(df["pub_date"], errors="coerce").dt.to_period("M")
df["year"] = df["pub_date"].str[:4]
rel = df[df["pred"] != "관계없음"].copy()                 # 저출생 관련 문장
area_long = (rel.assign(area=rel["policy_area"].str.split(";")).explode("area").reset_index(drop=True))
area_direct = area_long[area_long["area_source"] == "직접"]   # 문장에 정책 키워드가 직접 있는 경우만

# ================================================================ 1. 원본 데이터 EDA
print("\n[1] 원본 데이터 EDA")
ok = art[art["fetch_status"] == "ok"].copy()
ok["year"] = ok["pub_date"].str[:4]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
cnt = ok.groupby(["press", "outlet_lean"]).size().reset_index(name="n").sort_values("n")
axes[0].barh(cnt["press"], cnt["n"], color=[LEAN_COLOR[l] for l in cnt["outlet_lean"]])
axes[0].set_title("언론사별 수집 기사 수 (본문 수집 성공)")
for l in LEANS:
    axes[0].bar(0, 0, color=LEAN_COLOR[l], label=l)
axes[0].legend(frameon=False)
yr = pd.crosstab(ok["year"], ok["press"])
yr.plot(kind="bar", stacked=True, ax=axes[1], colormap="tab20", width=0.8)
axes[1].set_title("연도별·언론사별 기사 수"); axes[1].set_xlabel(""); axes[1].legend(fontsize=7, frameon=False)
axes[1].tick_params(axis="x", rotation=0)
save(fig, "eda_01_press_year.png")
yr.to_csv(RES / "eda_articles_year_press.csv", encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(12, 3.8))
m = df.groupby("month")["sent_id"].count()
art_m = pd.to_datetime(sent.drop_duplicates("aid")["pub_date"], errors="coerce").dt.to_period("M").value_counts().sort_index()
ax.plot(art_m.index.to_timestamp(), art_m.values, color="#3A6EA5", marker=".", label="기사 수")
mark_policies(ax, "#C8443A")
ax.set_title("월별 분석 대상 기사 수와 주요 정책 시행 시점"); ax.set_ylabel("기사 수")
save(fig, "eda_02_monthly_articles.png")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ln = sent["sentence"].str.len()
axes[0].hist(ln.clip(upper=250), bins=50, color="#3A6EA5")
axes[0].axvline(ln.mean(), color="#C8443A", ls="--")
axes[0].set_title(f"문장 길이 분포 (평균 {ln.mean():.0f}자, 중앙값 {ln.median():.0f}자)"); axes[0].set_xlabel("글자 수 (250자 이상은 250으로 표시)")
ar = sent.assign(area=sent["policy_area"].str.split(";")).explode("area")
ac = ar.loc[ar["area_source"] == "직접", "area"].value_counts().reindex(AREAS).sort_values()
axes[1].barh(ac.index, ac.values, color="#3A6EA5")
axes[1].set_title("정책 영역별 문장 수 (문장에 키워드가 직접 있는 경우)")
save(fig, "eda_03_length_area.png")

flow = pd.DataFrame({"단계": ["빅카인즈 검색 결과", "7개 언론사·국제 제외·중복 제거", "제목·키워드 사전 필터",
                              "월별 상한(80건) 적용", "본문 수집 성공", "주제무관·출처불명 등 필터, 중복 기사 제거", "문장 (정제 후)"],
                     "건수": [28455, 24018, 8138, 4498, 3852, 2968, len(sent)]})
flow.to_csv(RES / "eda_pipeline.csv", index=False, encoding="utf-8-sig")

# ================================================================ 2. 학습 데이터 EDA
print("\n[2] 학습 데이터 EDA")
lab["label"] = lab["label"].str.replace(r"\s+", "", regex=True)
lab = lab[lab["label"].isin(LABELS)]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
lc = lab["label"].value_counts().reindex(LABELS)
axes[0].bar(lc.index, lc.values, color=[COLOR[l] for l in lc.index], edgecolor="#666")
for i, v in enumerate(lc.values):
    axes[0].text(i, v, f"{v:,}\n({v / len(lab):.1%})", ha="center", va="bottom", fontsize=9)
axes[0].set_title(f"라벨 분포 (총 {len(lab):,}문장)"); axes[0].set_ylim(0, lc.max() * 1.2)
gl = pd.crosstab(lab["sample_group"], lab["label"], normalize="index").reindex(columns=LABELS)
gl = gl.reindex(["random", "cue", "opinion", "pilot", "positive_boost"]).dropna(how="all")
left = pd.Series(0.0, index=gl.index)
for c in LABELS:
    axes[1].barh(gl.index, gl[c], left=left, color=COLOR[c], label=c, edgecolor="white")
    left += gl[c]
axes[1].invert_yaxis(); axes[1].set_xlim(0, 1)
axes[1].set_title("샘플링 그룹별 라벨 비율 (random = 실제 분포)")
axes[1].legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.28), frameon=False)
save(fig, "train_01_label_dist.png")
pd.crosstab(lab["sample_group"], lab["label"]).to_csv(RES / "train_group_label.csv", encoding="utf-8-sig")

print("  형태소 분석 중...")
from kiwipiepy import Kiwi  # noqa: E402
kiwi = Kiwi()
for w in USER_WORDS:
    kiwi.add_user_word(w, "NNP")


def nouns(text):
    return [t.form for t in kiwi.tokenize(text) if t.tag in ("NNG", "NNP") and len(t.form) > 1 and t.form not in STOP]


lab["nouns"] = [nouns(s) for s in lab["sentence"]]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
top_rows = []
for ax, c in zip(axes, ["긍정", "부정", "중립"]):
    vc = pd.Series([w for ws in lab.loc[lab["label"] == c, "nouns"] for w in ws]).value_counts().head(15)
    ax.barh(vc.index[::-1], vc.values[::-1], color=COLOR[c])
    ax.set_title(f"'{c}' 문장 상위 명사")
    top_rows += [{"label": c, "rank": i + 1, "noun": w, "freq": n} for i, (w, n) in enumerate(vc.items())]
save(fig, "train_02_top_nouns.png")
pd.DataFrame(top_rows).to_csv(RES / "train_top_nouns.csv", index=False, encoding="utf-8-sig")

# ================================================================ 3. 결과 분석
print("\n[3] 결과 분석")
overall = df["pred"].value_counts().reindex(LABELS)
overall.to_csv(RES / "result_overall.csv", encoding="utf-8-sig")
print("  전체 분류:", overall.to_dict())

# H1 정책 영역별
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
h1_all = share(area_long[area_long["area"].isin(AREAS)], "area")
h1_dir = share(area_direct[area_direct["area"].isin(AREAS)], "area")
order = h1_dir.sort_values("부정", ascending=False).index.tolist()
stacked_share(h1_dir, axes[0], "정책 영역별 논조 (문장에 키워드 직접 등장)", order)
stacked_share(h1_all, axes[1], "정책 영역별 논조 (기사 대표 영역 상속 포함)", order)
save(fig, "result_H1_area_tone.png")
pd.concat({"직접": h1_dir, "전체": h1_all}, axis=1).round(4).to_csv(RES / "result_H1_area_tone.csv", encoding="utf-8-sig")
area_direct[area_direct["area"].isin(AREAS)]["area"].value_counts().to_csv(RES / "result_H1_area_n.csv", encoding="utf-8-sig")

# H1 키워드 네트워크: 부정 문장이 많은 상위 3개 영역
print("  키워드 네트워크 계산 중...")
net_areas = h1_dir.index[h1_dir.index.isin(AREAS)].tolist()
net_areas = area_direct[area_direct["pred"] == "부정"]["area"].value_counts().reindex(AREAS).dropna().sort_values(ascending=False).index[:3]
fig, axes = plt.subplots(1, 3, figsize=(20, 7.5))
net_rows = []
for ax, a in zip(axes, net_areas):
    texts = area_direct.loc[(area_direct["area"] == a) & (area_direct["pred"] == "부정"), "sentence"]
    docs = [list(dict.fromkeys(nouns(t))) for t in texts]
    freq = pd.Series([w for d in docs for w in d]).value_counts()
    keep = set(freq.head(25).index)
    pairs = pd.Series([tuple(sorted(p)) for d in docs for p in combinations([w for w in d if w in keep], 2)]).value_counts()
    pairs = pairs[pairs >= max(3, pairs.quantile(0.7))]
    G = nx.Graph()
    for (u, v), w in pairs.items():
        G.add_edge(u, v, weight=w)
    if G.number_of_nodes() == 0:
        ax.set_axis_off(); continue
    pos = nx.spring_layout(G, k=0.9, seed=42, weight="weight")
    nx.draw_networkx_edges(G, pos, ax=ax, width=[0.3 + 3 * G[u][v]["weight"] / pairs.max() for u, v in G.edges()],
                           edge_color="#BBBBBB")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=[150 + 2500 * freq[n] / freq.max() for n in G.nodes()],
                           node_color=COLOR["부정"], alpha=0.75)
    nx.draw_networkx_labels(G, pos, ax=ax, font_family=plt.rcParams["font.family"][0], font_size=12)
    ax.set_title(f"{a} — 부정 문장 키워드 네트워크 (n={len(texts):,})", fontsize=14)
    ax.set_axis_off()
    net_rows += [{"area": a, "noun": w, "freq": n} for w, n in freq.head(25).items()]
save(fig, "result_H1_keyword_network.png")
pd.DataFrame(net_rows).to_csv(RES / "result_H1_negative_keywords.csv", index=False, encoding="utf-8-sig")

# H2 시기별
mt = pd.crosstab(rel["month"], rel["pred"], normalize="index").reindex(columns=["긍정", "부정"]).rolling(3, min_periods=1).mean()
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(mt.index.to_timestamp(), mt["긍정"], color=COLOR["긍정"], label="긍정 비율")
ax.plot(mt.index.to_timestamp(), mt["부정"], color=COLOR["부정"], label="부정 비율")
mark_policies(ax)
ax.set_title("월별 긍정·부정 비율 추이 (저출생 관련 문장, 3개월 이동평균)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}")); ax.legend(frameon=False)
save(fig, "result_H2_monthly_tone.png")
mt.to_csv(RES / "result_H2_monthly_tone.csv", encoding="utf-8-sig")

rows = []
for name, (d, pat) in POLICIES.items():
    t0 = pd.Period(d)
    sub = rel[rel["sentence"].str.contains(pat, regex=True)].copy()
    diff = (sub["month"] - t0).apply(lambda x: x.n if pd.notna(x) else None)
    for period, lo, hi in [("시행 전 6개월", -6, -1), ("직후 0~3개월", 0, 3), ("6~12개월 후", 6, 12)]:
        s = sub[(diff >= lo) & (diff <= hi)]
        rows.append({"정책": name, "시점": d, "구간": period, "문장 수": len(s),
                     "긍정": (s["pred"] == "긍정").mean() if len(s) else None,
                     "부정": (s["pred"] == "부정").mean() if len(s) else None})
h2 = pd.DataFrame(rows)
h2.to_csv(RES / "result_H2_policy_periods.csv", index=False, encoding="utf-8-sig")
fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
for ax, (name, g) in zip(axes.ravel(), h2.groupby("정책", sort=False)):
    x = range(len(g))
    ax.bar([i - 0.2 for i in x], g["긍정"].fillna(0), 0.4, color=COLOR["긍정"], label="긍정")
    ax.bar([i + 0.2 for i in x], g["부정"].fillna(0), 0.4, color=COLOR["부정"], label="부정")
    ax.set_xticks(list(x)); ax.set_xticklabels([f"{p}\n(n={n})" for p, n in zip(g["구간"], g["문장 수"])], fontsize=8)
    ax.set_title(f"{name} ({g['시점'].iloc[0]})")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
axes[0, 0].legend(frameon=False)
fig.suptitle("정책 시행 전후 해당 정책 언급 문장의 긍정·부정 비율 (n<30 구간은 해석 주의)")
save(fig, "result_H2_policy_periods.png")

# H3 언론 성향별
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
h3_all = share(rel, "outlet_lean")
h3_23 = share(rel[rel["year"] >= "2023"], "outlet_lean")          # 경제지 표본이 2023년 이후에 몰려 있어 기간을 맞춘 비교
stacked_share(h3_all, axes[0], "언론 성향별 논조 (전체 기간)", LEANS)
stacked_share(h3_23, axes[1], "언론 성향별 논조 (2023년 이후, 기간 통일)", LEANS)
save(fig, "result_H3_lean_tone.png")
pd.concat({"전체": h3_all, "2023이후": h3_23}, axis=1).round(4).to_csv(RES / "result_H3_lean_tone.csv", encoding="utf-8-sig")

hm = (area_direct[area_direct["area"].isin(AREAS) & (area_direct["year"] >= "2023")]
      .pipe(lambda x: pd.crosstab(x["area"], x["outlet_lean"], values=(x["pred"] == "부정"), aggfunc="mean"))
      .reindex(index=order, columns=LEANS))
fig, ax = plt.subplots(figsize=(6.5, 4.2))
im = ax.imshow(hm.values, cmap="Reds", vmin=0)
ax.set_xticks(range(len(LEANS))); ax.set_xticklabels(LEANS)
ax.set_yticks(range(len(hm.index))); ax.set_yticklabels(hm.index)
for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
        v = hm.values[i, j]
        ax.text(j, i, "" if pd.isna(v) else f"{v:.0%}", ha="center", va="center", fontsize=9)
ax.set_title("정책 영역 × 언론 성향 부정 비율 (2023년 이후)")
plt.colorbar(im, ax=ax, format=plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
save(fig, "result_H3_area_lean_heatmap.png")
hm.round(4).to_csv(RES / "result_H3_area_lean_negative.csv", encoding="utf-8-sig")

neg = rel[(rel["pred"] == "부정") & (rel["year"] >= "2023")]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
h3k = []
for ax, l in zip(axes, LEANS):
    vc = pd.Series([w for t in neg.loc[neg["outlet_lean"] == l, "sentence"] for w in set(nouns(t))]).value_counts().head(15)
    ax.barh(vc.index[::-1], vc.values[::-1], color=LEAN_COLOR[l])
    ax.set_title(f"{l} 언론 부정 문장 상위 명사")
    h3k += [{"lean": l, "rank": i + 1, "noun": w, "freq": n} for i, (w, n) in enumerate(vc.items())]
save(fig, "result_H3_lean_keywords.png")
pd.DataFrame(h3k).to_csv(RES / "result_H3_lean_keywords.csv", index=False, encoding="utf-8-sig")

# 기사 유형
stype = share(rel, "article_type")
stype.round(4).to_csv(RES / "result_article_type_tone.csv", encoding="utf-8-sig")
print("\n완료. 그래프 → images/, 수치 → results/")
