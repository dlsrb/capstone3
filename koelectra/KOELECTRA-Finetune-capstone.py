# =====================================================================
# 수업 코드(KOELECTRA-Finetune.py)를 캡스톤 데이터에 적용한 버전
# 바뀐 곳은 모두 "# [변경]" 으로 표시했다. 나머지는 수업 코드 그대로다.
#   - 데이터: 네이버 영화 리뷰 2분류 → 저출생 기사 문장 4분류 (src/07_make_dataset.py로 생성)
#   - num_labels 2 → 4
#   - 검증 데이터 loss 계산 추가, epoch별 결과 CSV·그래프 저장 (과제 요구사항)
# 실행: PyCharm에서 이 파일 오른쪽 클릭 → Run (작업 폴더 = koelectra 폴더)
# =====================================================================
import os                                                     # [변경] 작업 폴더 고정용
import matplotlib.pyplot as plt                               # [변경] 그래프
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from transformers import get_linear_schedule_with_warmup, logging
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

os.chdir(os.path.dirname(os.path.abspath(__file__)))         # [변경] 어디서 실행해도 koelectra 폴더 기준으로 파일을 찾음
plt.rcParams["font.family"] = "Malgun Gothic"                 # [변경] 그래프 한글 (윈도우 기본 글꼴)
plt.rcParams["axes.unicode_minus"] = False

# 0. 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용하는 장치: ", device)

# 1. 경고 메시지 제거
logging.set_verbosity_error()

# 2. 데이터 로드
train_data_path = "train_data.txt"                            # [변경] 07_make_dataset.py가 만든 파일
dataset = pd.read_csv(train_data_path, sep='\t').dropna(axis=0)
text = list(dataset['document'].values)
label = dataset['label'].values

num_to_print = 3
print("=== 데이터 확인 ===")
for j in range(num_to_print):
    print(f"기사 문장: {text[j][:20]}, \t라벨 : {label[j]}" )    # [변경]
print(f"\t * 학습 데이터의 수: {len(text)}")
label_names = {0: "관계없음", 1: "중립", 2: "부정", 3: "긍정"}  # [변경] 4분류 (koelectra/label_map.txt)
for k, v in label_names.items():
    print(f"\t * {v} 문장 수: {list(label).count(k)}")

# 3. 텍스트 토큰화
tokenizer = ElectraTokenizer.from_pretrained("koelectra-base-v3-discriminator")
inputs = tokenizer(text, truncation=True, max_length=128, add_special_tokens=True,   # [변경] 256→128: 문장 평균 58자라 충분, 학습 시간 절반
                   padding="max_length")
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
print("\n=== 토큰과 결과 ===")
for j in range(num_to_print):
    print(f"{j+1}번째 데이터")
    print("== 토큰 ==")
    print(input_ids[j])
    print("== 어텐션 마스크 ==")
    print(attention_mask[j])

# 4. 데이터 분리
train, valid, train_y, valid_y = train_test_split(input_ids, label, test_size=0.2, random_state=2026)
train_masks, valid_masks, _, _ = train_test_split(attention_mask, label, test_size=0.2, random_state=2026)

# 5. pytorch의 데이터로더 설정
batch_size = 32
train_inputs = torch.tensor(train)
train_labels = torch.tensor(train_y)
train_masks = torch.tensor(train_masks)
train_data = TensorDataset(train_inputs, train_labels, train_masks)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

valid_inputs = torch.tensor(valid)
valid_labels = torch.tensor(valid_y)
valid_masks = torch.tensor(valid_masks)
valid_data = TensorDataset(valid_inputs, valid_labels, valid_masks)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_size)

# 6. 모델 설정
model = ElectraForSequenceClassification.from_pretrained('koelectra-base-v3-discriminator',
                                                         num_labels=4)   # [변경] 2→4
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-04, eps=1e-06, betas=(0.9, 0.999))

epoch = 4
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                            num_training_steps=len(train_dataloader) * epoch)

epoch_results = []

# 7. 학습 진행 (중간에 학습 데이터와 검증 데이터의 정확도를 계산)
for e in range(0, epoch):
    # === 학습 시작
    model.train()
    total_train_loss = 0

    progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {e+1}, leave=False")

    for batch in progress_bar:
        batch_ids, batch_label, batch_mask = tuple(t.to(device) for t in batch)
        model.zero_grad()

        outputs = model(batch_ids, attention_mask=batch_mask, labels=batch_label)
        loss = outputs.loss
        total_train_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({'loss': loss.item()})

    avg_train_loss = total_train_loss / len(train_dataloader)
    # === 학습 종료

    # === 학습데이터에 대한 정확도 측정
    model.eval()
    progress_bar = tqdm(train_dataloader, desc=f"Evaluating Train Epoch {e+1}, leave=False")
    train_preds = []; train_true = []
    for batch in progress_bar:
        batch_ids, batch_label, batch_mask = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            outputs = model(batch_ids, attention_mask=batch_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        train_preds.extend(preds.cpu().numpy())
        train_true.extend(batch_label.cpu().numpy())
    train_accuracy = np.sum( np.array(train_preds) == np.array(train_true)) / len(train_preds)

    # === 학습데이터에 대한 정확도 측정 완료

    # === 검증데이터에 대한 정확도 측정
    progress_bar = tqdm(valid_dataloader, desc=f"Evaluating Train Epoch {e + 1}, leave=False")
    valid_preds = []; valid_true = []
    total_valid_loss = 0                                                       # [변경] 검증 loss
    for batch in progress_bar:
        batch_ids, batch_label, batch_mask = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            outputs = model(batch_ids, attention_mask=batch_mask, labels=batch_label)  # [변경] labels를 넣어 loss도 계산
        total_valid_loss += outputs.loss.item()                                 # [변경]
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        valid_preds.extend(preds.cpu().numpy())
        valid_true.extend(batch_label.cpu().numpy())
    valid_accuracy = np.sum(np.array(valid_preds) == np.array(valid_true)) / len(valid_preds)

    # === 검증데이터에 대한 정확도 측정 완료
    avg_valid_loss = total_valid_loss / len(valid_dataloader)                  # [변경]
    epoch_results.append((avg_train_loss, train_accuracy, valid_accuracy, avg_valid_loss))  # [변경] valid loss 추가

print("\n === 학습 결과 요약 ===")
for j, (loss, t_acc, v_acc, v_loss) in enumerate(epoch_results, start=1):   # [변경]
    print(f"Epoch: {j}, Train loss: {loss: .4f}, Valid loss: {v_loss: .4f}, Train ACC: {t_acc: .4f}, Valid ACC: {v_acc: .4f}")

# [변경] epoch별 결과 저장 + 그래프 (과제: epoch별 loss 및 정확도 그래프)
result_df = pd.DataFrame(epoch_results, columns=["train_loss", "train_acc", "valid_acc", "valid_loss"])
result_df.index = range(1, len(result_df) + 1)
result_df.index.name = "epoch"
result_df.to_csv("epoch_results.csv", encoding="utf-8-sig")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(result_df.index, result_df["train_loss"], marker="o", label="Train loss")
axes[0].plot(result_df.index, result_df["valid_loss"], marker="o", label="Valid loss")
axes[0].set_title("Epoch별 Loss"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend()
axes[1].plot(result_df.index, result_df["train_acc"], marker="o", label="Train accuracy")
axes[1].plot(result_df.index, result_df["valid_acc"], marker="o", label="Valid accuracy")
axes[1].set_title("Epoch별 정확도"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].legend()
for ax in axes:
    ax.set_xticks(result_df.index); ax.grid(alpha=0.3)
plt.tight_layout()
os.makedirs("../images", exist_ok=True)
plt.savefig("../images/koelectra_epoch_loss_acc.png", dpi=200)
print("그래프 저장 → images/koelectra_epoch_loss_acc.png, 수치 → koelectra/epoch_results.csv")

print("\n === 모델 저장 ===")
save_path = "model_koelectra_capstone"                       # [변경] 저장 폴더 이름
model.cpu()
for param in model.parameters():
    if not param.is_contiguous:
        param.data = param.data.contiguous()
model.save_pretrained(save_path, '.pt')
print("\n === 종료 ===")







