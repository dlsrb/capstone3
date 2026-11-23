"""
05_ai_label.py
AI(Gems / GPTs) 1차 라벨링을 돕는 스크립트. 위에서 MODE를 바꿔 가며 세 번 실행한다.

  MODE = "split"   label_set.csv → data/labeling/batches/batch_001.txt ... (100문장씩)
                   각 txt 내용을 docs/labeling_prompt.md 지침을 넣은 Gem/GPT에 붙여넣고,
                   AI가 출력한 CSV를 data/labeling/ai_out/batch_001.csv 처럼 같은 번호로 저장한다.
                   (파일럿 200문장도 섞여 있다. AI에게는 내 라벨을 보여주지 않고 AI 정확도를 재기 위함)

  MODE = "merge"   ai_out/*.csv → label_set.csv에 ai_label, ai_note 열 추가
                   - 빠진 sent_id, 잘못된 라벨이 있으면 다시 돌릴 배치 번호를 알려준다
                   - 파일럿 200문장으로 AI 정확도(일치율, Cohen's Kappa)를 계산한다
                   - 아직 비어 있는 label 칸은 AI 라벨로 미리 채운다 → 엑셀에서 전수 검수하며 틀린 것만 고친다
                     (검수 후에도 ai_label 열은 그대로 두어야 report에서 수정 내역을 계산할 수 있다)

  MODE = "report"  검수가 끝난 뒤: AI 1차 라벨 vs 최종 라벨 비교 (보고서 5.3절 표)

실행:  python src/05_ai_label.py
"""
import re
import shutil
from pathlib import Path

import pandas as pd

MODE = "merge"          # "split" → "merge" → (엑셀 검수) → "report"
BATCH_SIZE = 100

ROOT = Path(__file__).resolve().parents[1]
LABEL_DIR = ROOT / "data" / "labeling"
LABEL_CSV = LABEL_DIR / "label_set.csv"
BATCH_DIR = LABEL_DIR / "batches"
AI_DIR = LABEL_DIR / "ai_out"
LABELS = ["긍정", "부정", "중립", "관계없음"]


def load_label_set():
    df = pd.read_csv(LABEL_CSV, dtype=str, keep_default_na=False)
    df["label"] = df["label"].str.replace(r"\s+", "", regex=True)
    return df


def save_label_set(df):
    shutil.copy(LABEL_CSV, LABEL_CSV.with_suffix(".bak.csv"))      # 덮어쓰기 전 백업
    df.to_csv(LABEL_CSV, index=False, encoding="utf-8-sig")


# ------------------------------------------------------------------
# 일치도 계산
# ------------------------------------------------------------------
def cohen_kappa(a, b):
    """Cohen's Kappa = (실제 일치율 - 우연 일치율) / (1 - 우연 일치율)
    우연 일치율: 두 사람이 각자의 라벨 비율대로 아무렇게나 골랐을 때 같은 라벨이 나올 확률"""
    a, b = pd.Series(list(a)), pd.Series(list(b))
    po = (a == b).mean()
    pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True)
    pe = sum(pa.get(k, 0) * pb.get(k, 0) for k in LABELS)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def agreement(df, ai_col, human_col, title):
    d = df[(df[ai_col] != "") & (df[human_col] != "")]
    if d.empty:
        print(f"\n[{title}] 비교할 문장이 없습니다.")
        return
    acc = (d[ai_col] == d[human_col]).mean()
    print(f"\n[{title}] {len(d)}문장  일치율 {acc:.1%}  Cohen's Kappa {cohen_kappa(d[ai_col], d[human_col]):.3f}")
    cm = pd.crosstab(d[human_col], d[ai_col], rownames=["사람(최종)"], colnames=["AI"])
    cm = cm.reindex(index=LABELS, columns=LABELS, fill_value=0)
    print(cm.to_string())
    diff = d[d[ai_col] != d[human_col]]
    if len(diff):
        pairs = (diff[ai_col] + " → " + diff[human_col]).value_counts().head(5)
        print("  가장 많이 고친 라벨 (AI → 사람):")
        for k, v in pairs.items():
            print(f"    {k}: {v}건")
    return cm


# ------------------------------------------------------------------
def split():
    df = load_label_set()
    BATCH_DIR.mkdir(exist_ok=True)
    AI_DIR.mkdir(exist_ok=True)
    n = 0
    for i in range(0, len(df), BATCH_SIZE):
        n += 1
        part = df.iloc[i:i + BATCH_SIZE]
        lines = [f"{sid}\t{re.sub(r'\s+', ' ', s)}" for sid, s in zip(part["sent_id"], part["sentence"])]
        (BATCH_DIR / f"batch_{n:03d}.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"{len(df):,}문장 → {n}개 배치 저장 → {BATCH_DIR}")
    print(f"AI 출력은 {AI_DIR}\\batch_001.csv 처럼 같은 번호로 저장한 뒤 MODE='merge'로 실행하세요.")


def parse_ai_file(path):
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip().strip("`")
        if not line or line.lower().startswith("sent_id") or "," not in line:
            continue
        parts = [p.strip().strip('"') for p in line.split(",", 2)]   # note에 쉼표가 있어도 안전하게
        parts += [""] * (3 - len(parts))
        rows.append({"sent_id": parts[0], "ai_label": re.sub(r"\s+", "", parts[1]),
                     "ai_note": parts[2], "batch": path.stem})
    return rows


def merge():
    df = load_label_set()
    files = sorted(AI_DIR.glob("batch_*.csv"))
    if not files:
        raise SystemExit(f"{AI_DIR} 에 AI 출력 파일이 없습니다.")
    ai = pd.DataFrame([r for f in files for r in parse_ai_file(f)])
    ai = ai.drop_duplicates("sent_id", keep="last")

    bad = ai[~ai["ai_label"].isin(LABELS)]
    unknown = ai[~ai["sent_id"].isin(df["sent_id"])]
    ai = ai[ai["ai_label"].isin(LABELS) & ai["sent_id"].isin(df["sent_id"])]

    # 배치별로 빠진 문장 확인
    batch_of = {sid: f"batch_{i // BATCH_SIZE + 1:03d}" for i, sid in enumerate(df["sent_id"])}
    missing = df.loc[~df["sent_id"].isin(ai["sent_id"]), "sent_id"]
    print(f"AI 출력 {len(files)}개 파일, 라벨 {len(ai):,}건 반영 / 전체 {len(df):,}건")
    if len(bad):
        print(f"  ! 잘못된 라벨 {len(bad)}건 (예: {bad['ai_label'].head(3).tolist()}) → 해당 배치 다시 실행")
    if len(unknown):
        print(f"  ! label_set에 없는 sent_id {len(unknown)}건 (AI가 id를 바꿔 씀) → 해당 배치 다시 실행")
    if len(missing):
        redo = sorted({batch_of[s] for s in missing})
        print(f"  ! 라벨이 없는 문장 {len(missing)}건 → 다시 돌릴 배치: {', '.join(redo)}")

    df = df.drop(columns=[c for c in ("ai_label", "ai_note") if c in df.columns])
    df = df.merge(ai[["sent_id", "ai_label", "ai_note"]], on="sent_id", how="left").fillna("")

    # 파일럿: AI에게 내 라벨을 보여주지 않았으므로 AI 정확도를 재는 기준이 된다
    agreement(df[df["sample_group"] == "pilot"], "ai_label", "label", "파일럿 200문장: AI vs 직접 라벨")

    # 비어 있는 label을 AI 라벨로 미리 채움 (이미 채운 라벨은 건드리지 않음)
    empty = (df["label"] == "") & (df["ai_label"] != "")
    df.loc[empty, "label"] = df.loc[empty, "ai_label"]
    df.loc[empty & (df["note"] == ""), "note"] = df.loc[empty & (df["note"] == ""), "ai_note"]
    save_label_set(df)
    print(f"\nlabel 칸 {empty.sum():,}건을 AI 라벨로 채워 저장 → {LABEL_CSV}")
    print("엑셀에서 전수 검수하며 틀린 label만 고치세요. (ai_label 열은 지우지 말 것)")


def report():
    df = load_label_set()
    if "ai_label" not in df.columns:
        raise SystemExit("ai_label 열이 없습니다. MODE='merge'를 먼저 실행하세요.")
    bad = df[~df["label"].isin(LABELS)]
    if len(bad):
        print(f"! label이 비었거나 잘못된 문장 {len(bad)}건: {bad['label'].value_counts().to_dict()}")
    new = df[df["sample_group"] != "pilot"]
    cm = agreement(new, "ai_label", "label", "검수 결과: AI 1차 라벨 vs 최종 라벨 (파일럿 제외)")
    if cm is not None:
        cm.to_csv(LABEL_DIR / "ai_vs_final_confusion.csv", encoding="utf-8-sig")

    ok = df[df["label"].isin(LABELS)]
    print("\n[최종 라벨 분포]")
    dist = ok["label"].value_counts().reindex(LABELS, fill_value=0)
    for k, v in dist.items():
        print(f"  {k:<5} {v:>5,}  ({v / len(ok):.1%})")
    print("\n[샘플 그룹별 라벨 비율]")
    print((pd.crosstab(ok["sample_group"], ok["label"], normalize="index")
           .reindex(columns=LABELS, fill_value=0) * 100).round(1).to_string())


if __name__ == "__main__":
    {"split": split, "merge": merge, "report": report}[MODE]()
