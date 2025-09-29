import pandas as pd
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

# 確保 TOKENIZERS_PARALLELISM 環境變數設定
os.environ["TOKENIZERS_PARALLELISM"] = "false"
data_path = "data/train.txt"

# 1. 資料讀取與前處理
print(f"1. 讀取資料：{data_path}")

try:
    # 假設 data/test.txt 已經存在，每行是一筆資料
    with open(data_path, "r", encoding="utf-8") as f:
        raw_texts = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print(f"錯誤：找不到 {data_path} 檔案。請確保檔案存在。")
    exit()

def clean_text(text):
    """清理文字：去除換行符並移除前後空白"""
    text = str(text).strip().replace("\n", " ").replace("\r", " ")
    return text

# 對讀入的資料進行清理
nursing_texts = [clean_text(t) for t in raw_texts if t]
print(f"成功清理 {len(nursing_texts)} 筆文本記錄。")


# 2. 載入模型與 LoRA
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print(f"2. 載入模型：{model_name}...")

# 設置 BitsAndBytes 4-bit 量化配置
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

# 設置 LoRA 配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
print("LoRA 設置完成。可訓練參數：")
model.print_trainable_parameters()

# 3. 記憶體優化設置
model.enable_input_require_grads()  # 確保輸入能回傳梯度
model.gradient_checkpointing_enable()  # 啟用 checkpointing
model.config.use_cache = False  # 避免梯度衝突


# 4. 建立與標記化資料集
print("4. 建立與標記化資料集...")

# 建立單一 Dataset 物件
full_dataset = Dataset.from_dict({"text": nursing_texts})

def tokenize_function(examples):
    """將文本標記化並複製 input_ids 作為 labels"""
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length", # 靜態 padding
        max_length=512
    )
    # Causal LM 訓練的 labels 就是 input_ids
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = full_dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])

# 切分資料集為訓練集 (90%) 和驗證集 (10%)
split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)

train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print(f"   - 訓練集大小: {len(train_dataset)}")
print(f"   - 驗證集大小: {len(eval_dataset)}")

# 5. 手動測試 loss (保持驗證)
sample = train_dataset[0]
input_ids = torch.tensor([sample["input_ids"]]).to(model.device)
attention_mask = torch.tensor([sample["attention_mask"]]).to(model.device)
labels = torch.tensor([sample["labels"]]).to(model.device)

model.train()
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels
)

print(f"測試 loss: {outputs.loss:.4f}")
assert outputs.loss is not None, "模型未回傳 loss，請檢查 labels 是否正確"


# 6. 設定 Trainer 微調
print("6. 設定 Trainer 並開始訓練...")

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
    remove_unused_columns=False,
    evaluation_strategy="steps", # 啟用每隔 steps 進行驗證
    eval_steps=200,              # 每 200 步驗證一次
)

# 使用官方的 DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # 使用切分後的訓練集
    eval_dataset=eval_dataset,    # 使用切分後的驗證集
    tokenizer=tokenizer,
    data_collator=data_collator   # 使用官方 Collator
)

# 確保至少一個參數需要梯度
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"   - 正在訓練的參數範例: {name} (Shape: {param.shape})")
        break

# 開始訓練
trainer.train()

# 儲存微調後模型
model.save_pretrained("./tinyllama-nursing-final2")
tokenizer.save_pretrained("./tinyllama-nursing-final2")
print("\n訓練完成！模型已儲存到 ./tinyllama-nursing-final2")
