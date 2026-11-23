"""
06_extra_label.py
검수가 끝난 label_set.csv를 보완하는 스크립트. MODE를 바꿔 가며 실행한다.

  MODE = "blind"     블라인드 재라벨링 파일 만들기
                     label_set.csv(파일럿 제외)에서 무작위 200문장을 뽑아, 라벨을 숨긴 채
                     data/labeling/blind_200.csv 로 저장 → my_label 열을 직접 채운다.
                     (AI 라벨·최종 라벨을 보지 않고 문장만 보고 판단할 것)

  MODE = "check"     blind_200.csv의 my_label vs label_set.csv의 최종 label 비교
                     → 일치율, Cohen's Kappa, 혼동행렬 (보고서 5.3절 '라벨링 일관성')

  MODE = "positive"  긍정 보충 후보 파일 만들기
                     sentences.csv 전체에서 아직 라벨링하지 않은 문장 중
                     ① 저출생·정책 영역 키워드가 있고 ② 긍정 평가 표현이 있고 ③ 부정 표현이 없는 문장을
                     골라 data/labeling/positive_candidates.csv 로 저장 → label 열을 채운다.
                     (후보일 뿐 실제 라벨은 판단 규칙대로. 부정·중립·관계없음도 나올 수 있다)

  MODE = "add"       라벨을 채운 positive_candidates.csv를 label_set.csv에 합친다.
                     sample_group = "positive_boost" 로 표시 (테스트셋에서는 제외할 것)

실행:  python src/06_extra_label.py
"""
import re
import shutil
from pathlib import Path

import pandas as pd

MODE = "blind"          # "blind" → (직접 라벨) → "check" / "positive" → (라벨) → "add"
N_BLIND = 200
N_POSITIVE = 400
MAX_PER_ARTICLE = 3
SEED = 7                 # 04(SEED=42)와 다른 값

ROOT = Path(__file__).resolve().parents[1]
LABEL_DIR = ROOT / "data" / "labeling"
LABEL_CSV = LABEL_DIR / "label_set.csv"
SENT_CSV = ROOT / "data" / "processed" / "sentences.csv"
BLIND_CSV = LABEL_DIR / "blind_200.csv"
POS_CSV = LABEL_DIR / "positive_candidates.csv"
LABELS = ["긍정", "부정", "중립", "관계없음"]

# 저출생 대책과 관련된 문장인지 (03의 POLICY_TOPIC + 정책 영역 키워드를 간단히 합친 것)
TOPIC = re.compile(
    r"저출생|저출산|출산율|출생아|인구\s*(정책|전략)|부모\s*급여|아동\s*수당|첫만남|출산\s*(지원금|장려금|휴가)|"
    r"육아\s*휴직|어린이집|보육|늘봄|돌봄|유보\s*통합|신혼\s*부부|특례\s*대출|난임|다자녀|일\s*[·ㆍ]?\s*가정\s*양립|"
    r"양육|가사\s*(관리사|도우미)|만남\s*(행사|프로그램)"
)
# 긍정 평가 표현
POS_CUE = re.compile(
    r"효과를\s*(거두|보|냈|내고)|효과가\s*(있|크|컸|나타)|호응|도움이\s*(되|됐|된)|성과를\s*(거두|냈|내고)|"
    r"환영|기여|덕분|긍정적|호평|만족|반가|인기|몰려|실효성(이|을)?\s*(있|높|입증)|힘을\s*보|고무적|바람직"
)
# 부정 표현 (같이 있으면 긍정일 가능성이 낮아 후보에서 제외)
NEG_CUE = re.compile(r"미미|없었|없다|못했|못한|부족|한계|그쳤|그친|불과|우려|비판|역부족|실패|문제|논란")


def cohen_kappa(a, b):
    """(실제 일치율 - 우연 일치율) / (1 - 우연 일치율)"""
    a, b = pd.Series(list(a)), pd.Series(list(b))
    po = (a == b).mean()
    pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True)
    pe = sum(pa.get(k, 0) * pb.get(k, 0) for k in LABELS)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def load_label_set():
    df = pd.read_csv(LABEL_CSV, dtype=str, keep_default_na=False)
    df["label"] = df["label"].str.replace(r"\s+", "", regex=True)
    return df


# ------------------------------------------------------------------
def blind():
    if BLIND_CSV.exists():
        raise SystemExit(f"{BLIND_CSV.name} 가 이미 있습니다. 직접 채운 내용을 덮어쓰지 않도록 중단합니다.")
    df = load_label_set()
    pool = df[df["sample_group"] != "pilot"]          # 파일럿은 이미 직접 라벨링했으므로 제외
    out = pool.sample(N_BLIND, random_state=SEED)[["sent_id", "sentence"]].copy()
    out["my_label"] = ""
    out["my_note"] = ""
    out.to_csv(BLIND_CSV, index=False, encoding="utf-8-sig")
    print(f"블라인드 재라벨링 {len(out)}문장 저장 → {BLIND_CSV}")
    print("엑셀에서 my_label 열만 채우세요. label_set.csv는 열어보지 말고, 문장만 보고 판단합니다.")


def check():
    b = pd.read_csv(BLIND_CSV, dtype=str, keep_default_na=False)
    b["my_label"] = b["my_label"].str.replace(r"\s+", "", regex=True)
    empty = b[~b["my_label"].isin(LABELS)]
    if len(empty):
        print(f"! my_label이 비었거나 잘못된 문장 {len(empty)}건은 제외하고 계산합니다.")
    df = load_label_set()[["sent_id", "label", "ai_label"]]
    m = b[b["my_label"].isin(LABELS)].merge(df, on="sent_id", how="left")

    for col, name in [("label", "최종 라벨(검수 후)"), ("ai_label", "AI 1차 라벨")]:
        acc = (m["my_label"] == m[col]).mean()
        print(f"\n[블라인드 재라벨링 vs {name}] {len(m)}문장  일치율 {acc:.1%}  "
              f"Cohen's Kappa {cohen_kappa(m['my_label'], m[col]):.3f}")
    cm = pd.crosstab(m["label"], m["my_label"], rownames=["최종 라벨"], colnames=["블라인드"])
    print(cm.reindex(index=LABELS, columns=LABELS, fill_value=0).to_string())
    diff = m[m["my_label"] != m["label"]]
    if len(diff):
        diff[["sent_id", "sentence", "label", "my_label", "my_note"]].to_csv(
            LABEL_DIR / "blind_disagree.csv", index=False, encoding="utf-8-sig")
        print(f"\n불일치 {len(diff)}건 → {LABEL_DIR / 'blind_disagree.csv'} (보고서 '주요 불일치 사례'에 활용)")


def positive():
    if POS_CSV.exists():
        raise SystemExit(f"{POS_CSV.name} 가 이미 있습니다. 라벨링한 내용을 덮어쓰지 않도록 중단합니다.")
    s = pd.read_csv(SENT_CSV, dtype=str, keep_default_na=False)
    used = set(load_label_set()["sent_id"])
    s = s[~s["sent_id"].isin(used)]
    cand = s[s["sentence"].str.contains(TOPIC) & s["sentence"].str.contains(POS_CUE)
             & ~s["sentence"].str.contains(NEG_CUE)]
    print(f"아직 라벨링하지 않은 문장 {len(s):,}개 중 긍정 후보 {len(cand):,}개")
    cand = cand.sample(frac=1, random_state=SEED).groupby("aid", group_keys=False).head(MAX_PER_ARTICLE)
    out = cand.head(N_POSITIVE).copy()
    out["sample_group"] = "positive_boost"
    out["label"] = ""
    out["note"] = ""
    cols = ["sent_id", "sample_group", "press", "outlet_lean", "article_type", "pub_date",
            "title", "sentence", "policy_area", "label", "note"]
    out[cols].to_csv(POS_CSV, index=False, encoding="utf-8-sig")
    print(f"긍정 보충 후보 {len(out)}문장 저장 → {POS_CSV}")
    print(out["press"].value_counts().to_string())
    print("label 열을 판단 규칙대로 채우세요. (긍정이 아니면 다른 라벨을 그대로 적으면 됩니다)")


def add():
    p = pd.read_csv(POS_CSV, dtype=str, keep_default_na=False)
    p["label"] = p["label"].str.replace(r"\s+", "", regex=True)
    bad = p[~p["label"].isin(LABELS)]
    if len(bad):
        raise SystemExit(f"label이 비었거나 잘못된 문장이 {len(bad)}건 있습니다. 모두 채운 뒤 다시 실행하세요.")
    df = load_label_set()
    if (df["sample_group"] == "positive_boost").any():
        raise SystemExit("label_set.csv에 이미 positive_boost가 들어 있습니다. 중복 추가를 막기 위해 중단합니다.")
    shutil.copy(LABEL_CSV, LABEL_CSV.with_suffix(".bak.csv"))
    out = pd.concat([df, p.reindex(columns=df.columns, fill_value="")], ignore_index=True)
    out.to_csv(LABEL_CSV, index=False, encoding="utf-8-sig")
    print(f"label_set.csv에 {len(p)}문장 추가 → 총 {len(out):,}문장")
    print("\n[추가한 문장의 라벨]")
    print(p["label"].value_counts().reindex(LABELS, fill_value=0).to_string())
    print("\n[전체 라벨 분포]")
    dist = out["label"].value_counts().reindex(LABELS, fill_value=0)
    for k, v in dist.items():
        print(f"  {k:<5} {v:>5,}  ({v / len(out):.1%})")


if __name__ == "__main__":
    {"blind": blind, "check": check, "positive": positive, "add": add}[MODE]()
