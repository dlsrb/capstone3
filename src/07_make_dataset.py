"""
07_make_dataset.py
라벨링이 끝난 label_set.csv → 수업 KoELECTRA 코드(KOELECTRA-Finetune.py)가 읽는 형식으로 변환

수업 코드의 입력 형식 (ratings_train2.txt, 네이버 영화 리뷰)
    id <TAB> document <TAB> label        ← label은 정수

만드는 파일 (koelectra 폴더)
    train_data.txt  학습용. 수업 코드가 이 안에서 다시 80:20으로 train/valid를 나눈다.
    test_data.txt   최종 평가용. 학습에 전혀 쓰지 않는다.
                    실제 기사 분포를 반영하는 'random' 그룹에서만 층화 추출한다.
                    (cue·사설·긍정보충 그룹은 일부러 긍정·부정을 많이 뽑았으므로 테스트에 넣지 않는다)
    label_map.txt   숫자 라벨 ↔ 한글 라벨 대응표

실행:  python src/07_make_dataset.py
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
LABEL_CSV = ROOT / "data" / "labeling" / "label_set.csv"
OUT_DIR = ROOT / "koelectra"

LABEL_MAP = {"관계없음": 0, "중립": 1, "부정": 2, "긍정": 3}
TEST_SHARE = 0.4          # random 그룹(660문장) 중 테스트로 쓸 비율 → 약 260문장
SEED = 2026               # 수업 코드의 random_state와 같은 값


def main():
    df = pd.read_csv(LABEL_CSV, dtype=str, keep_default_na=False)
    df["label"] = df["label"].str.replace(r"\s+", "", regex=True)
    bad = df[~df["label"].isin(LABEL_MAP)]
    if len(bad):
        print(f"! 라벨이 비었거나 잘못된 {len(bad)}문장은 제외합니다: {bad['label'].value_counts().to_dict()}")
    df = df[df["label"].isin(LABEL_MAP)].copy()
    df["document"] = df["sentence"].str.replace(r"[\t\r\n]+", " ", regex=True).str.strip()
    df["y"] = df["label"].map(LABEL_MAP)
    df["id"] = df["sent_id"]

    rand = df[df["sample_group"] == "random"]
    _, test = train_test_split(rand, test_size=TEST_SHARE, stratify=rand["y"], random_state=SEED)
    train = df[~df["id"].isin(test["id"])]

    cols = ["id", "document", "y"]
    for name, part in [("train_data.txt", train), ("test_data.txt", test)]:
        part[cols].rename(columns={"y": "label"}).to_csv(OUT_DIR / name, sep="\t", index=False)
    (OUT_DIR / "label_map.txt").write_text(
        "\n".join(f"{v}\t{k}" for k, v in LABEL_MAP.items()), encoding="utf-8")

    inv = {v: k for k, v in LABEL_MAP.items()}
    print(f"학습 데이터 {len(train):,}문장 → {OUT_DIR / 'train_data.txt'}  (수업 코드가 다시 80:20으로 train/valid 분할)")
    print(f"테스트 데이터 {len(test):,}문장 → {OUT_DIR / 'test_data.txt'}  (random 그룹에서 층화 추출)")
    summary = pd.DataFrame({
        "train": train["y"].value_counts(),
        "test": test["y"].value_counts(),
    }).fillna(0).astype(int).sort_index()
    summary.index = [f"{i} {inv[i]}" for i in summary.index]
    print(summary.to_string())


if __name__ == "__main__":
    main()
