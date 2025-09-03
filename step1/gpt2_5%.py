import pandas as pd
import re
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, pipeline, EarlyStoppingCallback
)
import transformers, torch
from datasets import Dataset
data_path = "NOTEEVENTS.csv"
df = pd.read_csv(data_path)


# 觀察欄位
#print(df.columns)

# 看看 CATEGORY 欄位裡面有哪些值
#print(df["CATEGORY"].value_counts())

#nursing_df = df[df["CATEGORY"] == "Nursing"]
# 只取 TEXT 欄位
#texts = nursing_df["TEXT"].tolist()
# 觀察資料
#print(nursing_df.head())

# 印出第0筆 nursing note
#print(nursing_df["TEXT"].iloc[0])
# 篩選 Nursing 類別
nursing_df = df[df["CATEGORY"] == "Nursing"].dropna(subset=["TEXT"])

# 移除標點符號
def remove_punct(text):
    return re.sub(r"[^\w\s]", "", text)

nursing_df["TEXT"] = nursing_df["TEXT"].apply(remove_punct)
# 抽樣
sample_size = 0.05
nursing_df = nursing_df.sample(frac=sample_size, random_state=42)
# 轉成 HF Dataset
dataset = Dataset.from_pandas(nursing_df[["TEXT"]].rename(columns={"TEXT": "text"}))
dataset = dataset.train_test_split(test_size=0.1, seed=42)

from transformers import AutoTokenizer

model_name = "gpt2"  # 英文筆記用 gpt2，中文可換 ckiplab/gpt2-base-chinese
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    encodings = tokenizer(batch["text"], truncation=True, padding="max_length")
    encodings["labels"] = encodings["input_ids"].copy()  # 加上 labels
    return encodings


tokenized_train = dataset["train"].map(tokenize, batched=True, remove_columns=["text"])
tokenized_eval  = dataset["test"].map(tokenize, batched=True, remove_columns=["text"])


model = AutoModelForCausalLM.from_pretrained(model_name)

from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

# 只使用舊版本支持的參數
from transformers import Trainer, TrainingArguments
import torch

# 設定 TrainingArguments（舊版本也可用）
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=10,            # 設大一點，讓 early stopping 發揮作用
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    logging_steps=50,
    logging_dir="./logs",
    save_total_limit=1,

    eval_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
)

# --- 手動 early stopping ---
best_loss = float('inf')
patience = 3
patience_counter = 0

for epoch in range(int(training_args.num_train_epochs)):
    print(f"\n===== Epoch {epoch+1} =====")
    trainer.train()
    
    # 評估
    metrics = trainer.evaluate()
    val_loss = metrics["eval_loss"]
    print(f"Validation loss: {val_loss:.4f}")
    
    # 判斷是否是最佳模型
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
        print("New best model! Saving...")
        trainer.save_model("./best_model")
    else:
        patience_counter += 1
        print(f"No improvement. Patience: {patience_counter}/{patience}")
    
    # 判斷是否 early stop
    if patience_counter >= patience:
        print("Early stopping triggered.")
        break

# 訓練完成後，載入最佳模型
model = trainer.model.from_pretrained("./best_model")

from transformers import pipeline

text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "Patient is"
outputs = text_generator(prompt, max_length=80, num_return_sequences=3, do_sample=True, top_k=50)
  
for i, out in enumerate(outputs, 1):
    print(f"建議 {i}: {out['generated_text']}\n")
