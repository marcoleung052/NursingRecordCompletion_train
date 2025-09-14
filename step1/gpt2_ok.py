#資料
import pandas as pd

df = pd.read_csv("NOTEEVENTS.csv")
'''
print("\n1.\n")
print(len(df))
print("\n2.\n")
print(df.head())

# 看看 CATEGORY 欄位裡面有哪些值
print("\n3.\n")
print(df["CATEGORY"].value_counts())

# 觀察欄位
print("\n4.\n")
print(df.columns)

# 觀察資料
print("\n5.\n")
nursing_df = df[df["CATEGORY"] == "Nursing"]
print(nursing_df.head())
'''
nursing_df = df[df["CATEGORY"] == "Nursing"]

# 印出第0筆 nursing note
print("\n6.\n")
print(nursing_df["TEXT"].iloc[0])

# 清理文字：去除多餘空白、特殊符號
def clean_text(text):
    text = str(text).strip().replace("\n", " ").replace("\r", " ")
    return text

nursing_texts = nursing_df["TEXT"].dropna().apply(clean_text).tolist()
print("\n7.\n")
print(nursing_texts[0])  # 印出清理後的第0筆

with open("nursing_corpus.txt", "w", encoding="utf-8") as f:
    for line in nursing_texts:
        f.write(line + "\n")

#GPT2
from datasets import Dataset
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

dataset = Dataset.from_dict({"text": nursing_texts})

def tokenize_function(examples):
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    # 加上 labels，讓模型能計算 loss
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize_function, batched=True)
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments

model = GPT2LMHeadModel.from_pretrained("gpt2")

training_args = TrainingArguments(
    output_dir="./gpt2-nursing",
    per_device_train_batch_size=4,
    num_train_epochs=2,
    logging_steps=100,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

from transformers import pipeline

generator = pipeline("text-generation", model="./gpt2-nursing", tokenizer=tokenizer)

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 載入 checkpoint 模型
print("正在載入模型...")
model = GPT2LMHeadModel.from_pretrained("./gpt2-nursing")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # 或你原本用的 tokenizer
print("載入完成")

# 儲存成 Hugging Face 格式
print("正在儲存模型...")
model.save_pretrained("./gpt2-nursing-final")
tokenizer.save_pretrained("./gpt2-nursing-final")
print("儲存完成")

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 載入模型與 tokenizer
model = GPT2LMHeadModel.from_pretrained("./gpt2-nursing-final", local_files_only=True)
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-nursing-final", local_files_only=True)
model.eval()

while True:
    prompt = input("請輸入開頭句子：").strip()
    if prompt == "stop":
        break

    while True:
        # 編碼輸入
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=inputs["input_ids"].shape[1] + 20,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=1.0,
                num_return_sequences=5
            )

        # 解碼候選句子
        candidates = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        # 顯示選項
        print("\n請選擇下一個句子：")
        for i, sentence in enumerate(candidates):
            print(f"{chr(97+i)}. {sentence.strip()}")

        # 使用者輸入
        choice = input("請輸入選項或指令(re=重新生成, stop=結束)：").strip().lower()

        if choice == "stop":
            break
        elif choice == "re":
            continue  # 重新生成候選句子
        elif choice in [chr(97+i) for i in range(len(candidates))]:
            prompt = candidates[ord(choice) - 97]
            print(f"\n目前句子：{prompt}")
        else:
            print("無效選項或指令，請重新輸入。")

