"""
04_make_label_set.py
본 수집 문장(sentences.csv)에서 직접 라벨링할 문장 2,000건+를 뽑는다.

왜 무작위로 뽑지 않나?
  파일럿 200문장을 무작위로 라벨링했더니 긍정 4.5% / 부정 4.5% / 중립 30.7% / 관계없음 60.3%였다.
  이대로면 2,000건 중 긍정·부정이 각각 90건 정도라 학습이 어렵다.
  파일럿에서 평가 표현(효과·실효성·한계·그쳤다·환영·우려 등)이 들어간 문장은 32%가 긍정·부정이었고,
  없는 문장은 5%였다. 그래서 그룹을 나눠 뽑는다.

  그룹         비율   설명
  cue          50%   평가 표현이 들어간 문장
  opinion      20%   사설·칼럼·기고 문장
  random       30%   무작위 (실제 기사 분포 그대로 → 테스트셋은 이 그룹에서만 뽑아 현실 성능을 잰다)
  pilot        +α    이미 라벨링한 파일럿 200문장 (라벨 그대로 포함)

실행:  python src/04_make_label_set.py
출력:  data/labeling/label_set.csv  → label, note 열을 채운다 (README 4장 기준)
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SENT_CSV = ROOT / "data" / "processed" / "sentences.csv"
PILOT_CSV = ROOT / "data" / "labeling" / "pilot_200.csv"
OUT_CSV = ROOT / "data" / "labeling" / "label_set.csv"

N_TOTAL = 2200              # 2,000건 + 라벨링하다 제외할 문장(깨진 문장 등) 여유분
SHARES = {"cue": 0.5, "opinion": 0.2, "random": 0.3}
MAX_PER_ARTICLE = 5         # 한 기사에서 너무 많이 뽑히지 않도록 (그룹별)
SEED = 42

CUE = re.compile(
    r"효과|실효성|한계|비판|지적|그쳤|그친|불과|성과|환영|우려|실패|미흡|부족|문제|도움|기여|힘을 보|덕분|"
    r"바람직|긍정|부정적|반발|논란|외면|사각지대|역부족|눈먼|퍼주기|낭비|호응|인기|몰려|반가"
)


def take(pool, n, group, used):
    pool = pool[~pool["sent_id"].isin(used)]
    pool = pool.sample(frac=1, random_state=SEED).groupby("aid", group_keys=False).head(MAX_PER_ARTICLE)
    out = pool.head(n).copy()
    out["sample_group"] = group
    used.update(out["sent_id"])
    if len(out) < n:
        print(f"  ! {group}: 목표 {n}건 중 {len(out)}건만 확보 (후보 부족)")
    return out


def main():
    if not SENT_CSV.exists():
        raise SystemExit(f"{SENT_CSV} 가 없습니다. 03_pilot_check.py를 PILOT=False로 먼저 실행하세요.")
    if OUT_CSV.exists():
        raise SystemExit(f"{OUT_CSV} 가 이미 있습니다. 라벨링한 내용을 덮어쓰지 않도록 중단합니다.")
    s = pd.read_csv(SENT_CSV, dtype=str)
    s["aid"] = s["sent_id"].str.rsplit("_", n=1).str[0]

    used = set()
    pilot = pd.DataFrame()
    if PILOT_CSV.exists():
        pilot = pd.read_csv(PILOT_CSV, dtype=str)
        pilot["label"] = pilot["label"].fillna("").str.replace(r"\s+", "", regex=True)
        pilot["sample_group"] = "pilot"
        used.update(pilot["sent_id"])

    n = {g: int(N_TOTAL * r) for g, r in SHARES.items()}
    parts = [
        take(s[s["sentence"].str.contains(CUE)], n["cue"], "cue", used),
        take(s[s["article_type"] != "스트레이트"], n["opinion"], "opinion", used),
        take(s, n["random"], "random", used),
    ]
    out = pd.concat(parts, ignore_index=True)
    out["label"] = ""
    out["note"] = ""
    cols = ["sent_id", "sample_group", "press", "outlet_lean", "article_type", "pub_date",
            "title", "sentence", "policy_area", "label", "note"]
    out = out.sample(frac=1, random_state=SEED)      # 그룹이 섞이게 (라벨링할 때 그룹을 의식하지 않도록)
    if len(pilot):
        out = pd.concat([pilot.reindex(columns=cols), out[cols]], ignore_index=True)
    out[cols].to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"라벨링 대상 {len(out):,}건 저장 → {OUT_CSV}")
    print(out["sample_group"].value_counts().to_string())
    # 파일럿 결과로 추정한 긍정·부정 비율: cue 32%, 사설·칼럼 15%, 무작위 9%
    est = (len(parts[0]) * 0.32 + len(parts[1]) * 0.15 + len(parts[2]) * 0.09) / max(len(out) - len(pilot), 1)
    print(f"\n예상 긍정+부정 비율 ≈ {est:.0%} (파일럿 기준 추정, 무작위로 뽑으면 약 9%)")
    print("label, note 열을 채운 뒤 05 단계(학습 데이터셋 구축)로 넘어갑니다.")


if __name__ == "__main__":
    main()
