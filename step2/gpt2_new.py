import os
import torch
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# ===== 讀取資料 =====
print("讀取 Nursing 資料...")

dataset = load_dataset(
    "text",
    data_files={
        "train": "data/train.txt",
        "test": "data/test.txt"  # 這裡直接用 test 作為 eval
    }
)

# ===== Tokenizer =====
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# ===== 模型 =====
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

# ===== 訓練設定 =====
training_args = TrainingArguments(
    output_dir="./checkpoints",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    save_steps=5000,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=100,
    do_train=True,
    do_eval=True,
    eval_strategy="steps",  # ⚠️ 改成 eval_strategy
)

# ===== Trainer =====
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
)

print("開始訓練 GPT-2...")
trainer.train()

trainer.save_model("./checkpoints/gpt2-nursing")
tokenizer.save_pretrained("./checkpoints/gpt2-nursing")
print("✅ 訓練完成，模型已存到 ./checkpoints/gpt2-nursing")
