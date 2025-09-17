import pandas as pd
import re
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, pipeline
import torch

# 資料前處理

data_path = "NOTEEVENTS.csv"
df = pd.read_csv(data_path)

# 篩選 Nursing 類別
nursing_df = df[df["CATEGORY"] == "Nursing"].dropna(subset=["TEXT"])

def remove_punct_keep(text):
    #保留 . , ; :
    return re.sub(r"[^\w\s\.,;:]", "", text)

nursing_df["TEXT"] = nursing_df["TEXT"].apply(remove_punct_keep)
nursing_df = nursing_df.sample(frac=0.1, random_state=42)


# 轉成 HF Dataset
dataset = Dataset.from_pandas(
    nursing_df[["TEXT"]].rename(columns={"TEXT": "text"})
)
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# Tokenizer

model_name = "facebook/opt-350M"
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token="my token")#放自己的token

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token  

def tokenize(batch):
    encodings = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )
    labels = [[-100 if token==tokenizer.pad_token_id else token for token in seq] for seq in encodings["input_ids"]]
    encodings["labels"] = labels
    return encodings



tokenized_train = dataset["train"].map(tokenize, batched=True, remove_columns=["text"])
tokenized_eval  = dataset["test"].map(tokenize, batched=True, remove_columns=["text"])

# 設定 torch tensor 格式
tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_eval.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
print(tokenized_train[0]["input_ids"])
print(tokenized_train[0]["labels"])
print(tokenized_train[0]["labels"])


device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)


# 訓練設定

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    logging_steps=50,
    save_total_limit=1,

    eval_steps=50,
    learning_rate=3e-5,
    fp16=False,  # 測試用 float32
)


from transformers import EarlyStoppingCallback

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,

)



# EarlyStopping + 保存最佳模型

best_loss = float('inf')
patience = 3
patience_counter = 0

for epoch in range(int(training_args.num_train_epochs)):
    print(f"\n===== Epoch {epoch+1} =====")
    trainer.train()
    # 手動梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    metrics = trainer.evaluate()
    val_loss = metrics["eval_loss"]
    print(f"Validation loss: {val_loss:.4f}")
    
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
        print("New best model! Saving...")
        trainer.save_model("./OPT_model")
    else:
        patience_counter += 1
        print(f"No improvement. Patience: {patience_counter}/{patience}")
    
    if patience_counter >= patience:
        print("Early stopping triggered.")
        break
