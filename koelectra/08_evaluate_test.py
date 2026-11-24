"""
08_evaluate_test.py  (koelectra 폴더에서 실행)
학습이 끝난 모델(model_koelectra_capstone)을 테스트 데이터(test_data.txt)로 최종 평가한다.
테스트 데이터는 학습·검증에 한 번도 쓰지 않았고, 실제 기사 분포(random 그룹)에서 뽑았다.

출력
    - 정확도, 라벨별 정밀도(precision)·재현율(recall)·F1, macro F1
    - 혼동행렬 그림 → images/koelectra_confusion_matrix.png
    - 틀린 문장 목록 → koelectra/test_errors.csv  (보고서 '오분류 사례')

용어
    정밀도: 모델이 '부정'이라고 한 것 중 실제로 부정인 비율
    재현율: 실제 부정 문장 중 모델이 부정으로 맞힌 비율
    F1    : 정밀도와 재현율의 조화평균. macro F1은 4개 라벨 F1의 단순 평균(적은 라벨도 똑같이 반영)
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import ElectraForSequenceClassification, ElectraTokenizer, logging

os.chdir(os.path.dirname(os.path.abspath(__file__)))
logging.set_verbosity_error()
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

MODEL_DIR = "model_koelectra_capstone"
TOKENIZER_DIR = "koelectra-base-v3-discriminator"
LABEL_NAMES = ["관계없음", "중립", "부정", "긍정"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = ElectraTokenizer.from_pretrained(TOKENIZER_DIR)
model = ElectraForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

test = pd.read_csv("test_data.txt", sep="\t").dropna(axis=0)
texts = list(test["document"].values)
y_true = test["label"].values

y_pred = []
for i in range(0, len(texts), 32):
    batch = tokenizer(texts[i:i + 32], truncation=True, max_length=128, padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)).logits
    y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())
y_pred = np.array(y_pred)

print(f"테스트 문장 수: {len(y_true)}")
print(f"정확도(Accuracy): {accuracy_score(y_true, y_pred):.4f}")
print(f"Macro F1: {f1_score(y_true, y_pred, average='macro'):.4f}\n")
print(classification_report(y_true, y_pred, labels=[0, 1, 2, 3], target_names=LABEL_NAMES, digits=3, zero_division=0))

cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
fig, ax = plt.subplots(figsize=(5.5, 4.5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(4)); ax.set_xticklabels(LABEL_NAMES)
ax.set_yticks(range(4)); ax.set_yticklabels(LABEL_NAMES)
ax.set_xlabel("모델 예측"); ax.set_ylabel("실제 라벨"); ax.set_title("테스트 데이터 혼동행렬")
for r in range(4):
    for c in range(4):
        ax.text(c, r, cm[r, c], ha="center", va="center", color="white" if cm[r, c] > cm.max() / 2 else "black")
plt.colorbar(im)
plt.tight_layout()
os.makedirs("../images", exist_ok=True)
plt.savefig("../images/koelectra_confusion_matrix.png", dpi=200)

err = test.assign(pred=y_pred)
err = err[err["label"] != err["pred"]]
err["label"] = err["label"].map(dict(enumerate(LABEL_NAMES)))
err["pred"] = err["pred"].map(dict(enumerate(LABEL_NAMES)))
err.to_csv("test_errors.csv", index=False, encoding="utf-8-sig")
print("혼동행렬 → images/koelectra_confusion_matrix.png")
print(f"틀린 문장 {len(err)}개 → koelectra/test_errors.csv")
