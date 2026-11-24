"""
09_predict_all.py  (koelectra 폴더에서 실행)
학습한 모델로 전체 문장(data/processed/sentences.csv, 78,935문장)을 4분류한다.

- CPU에서는 1시간 이상 걸릴 수 있다. 5,000문장마다 결과를 저장하므로
  중간에 멈춰도 다시 실행하면 이어서 진행한다.
- 결과: koelectra/predictions_all.csv
    sent_id, pred(라벨), p_관계없음, p_중립, p_부정, p_긍정(각 라벨일 확률)
  기사 정보(언론사·날짜·정책 영역 등)는 분석 단계에서 sentences.csv와 sent_id로 합친다.
"""
import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import ElectraForSequenceClassification, ElectraTokenizer, logging

os.chdir(os.path.dirname(os.path.abspath(__file__)))
logging.set_verbosity_error()

MODEL_DIR = "model_koelectra_capstone"
TOKENIZER_DIR = "koelectra-base-v3-discriminator"
SENT_CSV = "../data/processed/sentences.csv"
OUT_CSV = "predictions_all.csv"
LABEL_NAMES = ["관계없음", "중립", "부정", "긍정"]
BATCH = 64
CHUNK = 5000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용하는 장치:", device)
tokenizer = ElectraTokenizer.from_pretrained(TOKENIZER_DIR)
model = ElectraForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

sents = pd.read_csv(SENT_CSV, usecols=["sent_id", "sentence"], dtype=str, keep_default_na=False)
done = set()
if os.path.exists(OUT_CSV):
    done = set(pd.read_csv(OUT_CSV, usecols=["sent_id"], dtype=str)["sent_id"])
    print(f"이미 분류한 {len(done):,}문장은 건너뜁니다.")
todo = sents[~sents["sent_id"].isin(done)].reset_index(drop=True)
print(f"분류할 문장: {len(todo):,} / 전체 {len(sents):,}")

for start in range(0, len(todo), CHUNK):
    part = todo.iloc[start:start + CHUNK]
    texts = list(part["sentence"].values)
    probs = []
    for i in tqdm(range(0, len(texts), BATCH), desc=f"{start + len(done):,}~", leave=False):
        batch = tokenizer(texts[i:i + BATCH], truncation=True, max_length=128, padding=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)).logits
        probs.append(torch.softmax(logits, dim=1).cpu())
    p = torch.cat(probs).numpy()
    out = pd.DataFrame({"sent_id": part["sent_id"].values,
                        "pred": [LABEL_NAMES[k] for k in p.argmax(axis=1)]})
    for k, name in enumerate(LABEL_NAMES):
        out[f"p_{name}"] = p[:, k].round(4)
    out.to_csv(OUT_CSV, mode="a", header=not os.path.exists(OUT_CSV), index=False, encoding="utf-8")  # 이어쓰기라 BOM 없는 utf-8
    print(f"  저장: {start + len(part) + len(done):,} / {len(sents):,}")

res = pd.read_csv(OUT_CSV)
print("\n=== 전체 분류 결과 ===")
print(res["pred"].value_counts().reindex(LABEL_NAMES, fill_value=0).to_string())
print(f"→ {OUT_CSV}")
