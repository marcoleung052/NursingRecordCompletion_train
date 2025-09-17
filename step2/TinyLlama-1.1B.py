# 資料前處理
import pandas as pd
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

df = pd.read_csv("NOTEEVENTS.csv", low_memory=False)
nursing_df = df[df["CATEGORY"] == "Nursing"]

def clean_text(text):
    text = str(text).strip().replace("\n", " ").replace("\r", " ")
    return text

nursing_texts = nursing_df["TEXT"].dropna().apply(clean_text).tolist()

with open("nursing_corpus.txt", "w", encoding="utf-8") as f:
    for line in nursing_texts:
        f.write(line + "\n")

# 載入模型與 LoRA
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

import torch

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ✅ 強制關閉 use_cache，避免梯度丟失
model.enable_input_require_grads()  # ✅ 確保輸入能回傳梯度
model.gradient_checkpointing_enable()  # ✅ 啟用 checkpointing
model.config.use_cache = False  # ✅ 避免衝突


# 建立資料集
from datasets import Dataset

dataset = Dataset.from_dict({"text": nursing_texts})

def tokenize_function(examples):
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])

# ✅ 手動測試 loss 是否存在
sample = tokenized_dataset[0]
input_ids = torch.tensor([sample["input_ids"]]).to(model.device)
attention_mask = torch.tensor([sample["attention_mask"]]).to(model.device)
labels = torch.tensor([sample["labels"]]).to(model.device)

model.train()
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels
)

print("✅ 測試 loss:", outputs.loss)
assert outputs.loss is not None, "❌ 模型未回傳 loss，請檢查 labels 是否正確"

# 設定 Trainer 微調
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

training_args = TrainingArguments(
    output_dir="./tinyllama-nursing",
    per_device_train_batch_size=4,
    num_train_epochs=2,
    logging_steps=50,
    save_steps=200,
    save_total_limit=2,
    fp16=True,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    learning_rate=2e-4,
    warmup_ratio=0.03,
    report_to=[],
    remove_unused_columns=False
)


# ✅ 使用官方推薦的 data_collator
def data_collator(features):
    batch = {
        "input_ids": torch.tensor([f["input_ids"] for f in features], dtype=torch.long),
        "attention_mask": torch.tensor([f["attention_mask"] for f in features], dtype=torch.long),
        "labels": torch.tensor([f["labels"] for f in features], dtype=torch.long)
    }
    return batch


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.shape, param.dtype)
        break

trainer.train()

# 儲存微調後模型
model.save_pretrained("./tinyllama-nursing-final")
tokenizer.save_pretrained("./tinyllama-nursing-final")
