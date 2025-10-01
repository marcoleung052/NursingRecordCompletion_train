import os
import torch
# 導入 PyTorch 相關模組
from torch.utils.data import DataLoader
from tqdm import tqdm
# 導入 Hugging Face 模組
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForLanguageModeling, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from datasets import Dataset

# 確保 TOKENIZERS_PARALLELISM 環境變數設定
os.environ["TOKENIZERS_PARALLELISM"] = "false"
data_path = "data/train.txt"

# ***** 記憶體極限安全設定：大幅縮小資料集 *****
# 確定 VRAM 極限後，使用 64 萬筆資料進行訓練
MAX_SAMPLES_TO_USE = 640000
# **********************************************

# 1. 資料讀取與前處理
print(f"1. 讀取資料：{data_path}")

try:
    with open(data_path, "r", encoding="utf-8") as f:
        raw_texts = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print(f"錯誤：找不到 {data_path} 檔案。請確保檔案存在。")
    exit()

def clean_text(text):
    """清理文字：去除換行符並移除前後空白"""
    text = str(text).strip().replace("\n", " ").replace("\r", " ")
    return text

# 對讀入的資料進行清理並應用記憶體優化限制
cleaned_texts = [clean_text(t) for t in raw_texts if t]
nursing_texts = cleaned_texts[:MAX_SAMPLES_TO_USE]

print(f"成功清理 {len(cleaned_texts)} 筆原始記錄。")
print(f"記憶體安全限制：本次訓練僅使用 {len(nursing_texts)} 筆記錄。")


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
print("4. 建立與標記化資料集 (採用硬碟寫入優化)...")

# 建立單一 Dataset 物件
full_dataset = Dataset.from_dict({"text": nursing_texts})

def tokenize_function(examples):
    """將文本標記化並複製 input_ids 作為 labels"""
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    # Causal LM 訓練的 labels 就是 input_ids
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

# *** 保持 writer_batch_size 設置，利用硬碟緩存標記化結果 ***
tokenized_dataset = full_dataset.map(
    tokenize_function,
    batched=True,
    writer_batch_size=1000
)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])

# 切分資料集為訓練集 (90%) 和驗證集 (10%)
split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)

train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print(f"   - 訓練集大小: {len(train_dataset)}")
print(f"   - 驗證集大小: {len(eval_dataset)}")

# 使用官方的 DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# 5. 手動測試 loss 
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


# 6. 設定 PyTorch 訓練循環
print("6. 設定 PyTorch 訓練循環並開始訓練...")

# --- 訓練參數 ---
BATCH_SIZE = 32 # 最佳安全 Batch Size
NUM_EPOCHS = 1  # 最高效率訓練 Epoch 數
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03
LOGGING_STEPS = 50
SAVE_STEPS = 200

# 獲取模型所在設備 (GPU)
device = model.device

# 1. 設定 DataLoader
train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE, 
    collate_fn=data_collator,
    shuffle=True,
    drop_last=True # 確保每個批次大小一致
)

# 2. 設定優化器
trainable_params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.AdamW(
    trainable_params,
    lr=LEARNING_RATE, 
    eps=1e-8
)
print(f"   - 正在訓練的參數範例: {trainable_params[0].name if hasattr(trainable_params[0], 'name') else 'LoRA adapter'} (Shape: {trainable_params[0].shape})")

# 3. 設定學習率排程器
num_training_steps = len(train_dataloader) * NUM_EPOCHS
num_warmup_steps = int(num_training_steps * WARMUP_RATIO)

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)
print(f"   - 總訓練步數 (Total Steps): {num_training_steps}")

# 4. 訓練循環
model.train()
global_step = 0
total_loss = 0

for epoch in range(NUM_EPOCHS):
    print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
    
    # 使用 tqdm 顯示進度條
    for step, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")):
        
        # 1. 將批次資料移動到 GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 2. 前向傳播
        with torch.autocast(device_type="cuda", dtype=torch.float16):
             outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        # 3. 反向傳播
        loss.backward()

        # 4. 優化器步進
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        global_step += 1
        
        # 5. 紀錄 Loss
        if global_step % LOGGING_STEPS == 0:
            current_lr = lr_scheduler.get_last_lr()[0]
            avg_loss = total_loss / LOGGING_STEPS
            print(f" | Loss: {avg_loss:.4f} | LR: {current_lr:.8f}")
            total_loss = 0 # 重設累積 loss

        # 6. 儲存檢查點
        if global_step % SAVE_STEPS == 0:
            output_dir = f"./tinyllama-nursing/checkpoint-{global_step}"
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"\n模型檢查點已儲存至 {output_dir}")


# 儲存微調後模型
model.save_pretrained("./tinyllama-nursing-final2")
tokenizer.save_pretrained("./tinyllama-nursing-final2")
print("\n訓練完成！模型已儲存到 ./tinyllama-nursing-final2")
