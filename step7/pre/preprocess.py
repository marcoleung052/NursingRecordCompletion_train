import pandas as pd
import re
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import os
import sys
from pprint import pprint

# --- 導入字典 ---
try:
    from abbreviations import NORMALIZATION_MAP
except ImportError:
    print("錯誤：找不到 abbreviations.py 檔案。")
    NORMALIZATION_MAP = {}

# --- 參數設定 ---
FILE_LIST = [
    "1-2.xlsx",     
    "3-12.xlsx",
    "13-40.xlsx",
    "41-70.xlsx"
]
SHEET_NAME = "工作表1"
PATIENT_DELIMITER = "長庚醫療財團法人"
MODEL_NAME = "MediaTek-Research/Breeze-7B-Base-v0.1"

# !! 這是您要儲存處理後資料的位置
OUTPUT_DATA_DIR = "./processed_data"

# (這裡的 normalize_text_for_lm 和 load_and_merge_excel 函數
#  與我們前一版完全相同，此處為求簡潔先折疊)
# --- 正規化函數 ---
def normalize_text_for_lm(text):
    for key, value in NORMALIZATION_MAP.items():
        text = re.sub(r'\b' + re.escape(key) + r'\b', value, text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"記錄者：.*$", "", text)
    return text.strip()

# --- Excel 讀取與合併函數 ---
def load_and_merge_excel(file_path, sheet_name, ts_col, content_col, delimiter):
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 '{file_path}'。")
        return []
    except Exception as e:
        print(f"讀取 Excel 檔案 '{file_path}' 時發生錯誤：{e}")
        return []
    df = df.fillna('').astype(str)
    log_entries = [] 
    current_timestamp = None
    current_content = [] 
    timestamp_pattern = re.compile(r"^\d{2}:\d{2}(:\d{2})?$") # 接受 HH:MM 或 HH:MM:SS
    for index, row in df.iterrows():
        cell_A_content = row[ts_col].strip()
        cell_B_content = row[content_col].strip()
        if delimiter in cell_A_content or delimiter in cell_B_content:
            if current_timestamp: 
                full_text = " ".join(current_content)
                log_entries.append({"text": normalize_text_for_lm(full_text)})
            current_timestamp = None
            current_content = []
            continue 
        if timestamp_pattern.match(cell_A_content):
            if current_timestamp: 
                full_text = " ".join(current_content)
                log_entries.append({"text": normalize_text_for_lm(full_text)})
            current_timestamp = cell_A_content
            current_content = [cell_B_content]
        else:
            if current_timestamp: 
                if cell_A_content: current_content.append(cell_A_content)
                if cell_B_content: current_content.append(cell_B_content)
    if current_timestamp and current_content:
        full_text = " ".join(current_content)
        log_entries.append({"text": normalize_text_for_lm(full_text)})
    print(f"檔案 '{file_path}' 處理完成，讀取了 {len(log_entries)} 筆紀錄。")
    return log_entries

# --- Tokenization 函數 ---
def create_tokenizer(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception as e:
        print(f"下載 Tokenizer '{model_name}' 失敗：{e}")
        sys.exit(1)

def tokenize_function(batch, tokenizer, max_length=512):
    encodings = tokenizer(
        batch["text"], truncation=True, padding="max_length", max_length=max_length
    )
    labels_batch = []
    for seq in encodings["input_ids"]:
        labels_batch.append(
            [-100 if token == tokenizer.pad_token_id else token for token in seq]
        )
    encodings["labels"] = labels_batch
    return encodings

# --- 主要執行區塊 ---
def main():
    """
    執行完整的前處理流程並儲存到磁碟。
    """
    all_records = []
    print("--- 開始執行前處理 ---")
    for file_path in FILE_LIST:
        print(f"\n正在處理檔案: {file_path}")
        records_from_this_file = load_and_merge_excel(
            file_path, SHEET_NAME, 0, 1, PATIENT_DELIMITER
        )
        all_records.extend(records_from_this_file)

    if not all_records:
        print("未讀取到任何資料，程式終止。")
        return

    print(f"\n--- 所有檔案處理完畢 ---")
    print(f"總共從 {len(FILE_LIST)} 個檔案中，合併了 {len(all_records)} 筆獨立紀錄。")

    text_corpus = [r['text'] for r in all_records if r['text']] 
    print(f"過濾空紀錄後，剩下 {len(text_corpus)} 筆資料。")
    
    if not text_corpus:
        print("沒有可用的文本資料，程式終止。")
        return
        
    hf_dataset = Dataset.from_dict({"text": text_corpus})
    dataset_split = hf_dataset.train_test_split(test_size=0.1, seed=42)
    print(f"\n--- 資料集切分 (Hugging Face 格式) ---")
    print(dataset_split)

    print("\n--- 正在載入 Tokenizer ---")
    tokenizer = create_tokenizer(MODEL_NAME)
    
    print("\n--- 正在對總資料集進行 Tokenization (這可能需要幾分鐘) ---")
    tokenized_datasets = dataset_split.map(
        lambda batch: tokenize_function(batch, tokenizer),
        batched=True,
        remove_columns=["text"], 
        num_proc=os.cpu_count()
    )
    
    print("\n--- Tokenization 完成 ---")
    
    # --- !! 關鍵：儲存到磁碟 !! ---
    print(f"\n--- 正在將 Tokenized 資料集保存到 {OUTPUT_DATA_DIR} ---")
    tokenized_datasets.save_to_disk(OUTPUT_DATA_DIR)
    
    # 我們也需要儲存 Tokenizer，訓練時才能載入
    tokenizer.save_pretrained(OUTPUT_DATA_DIR)
    
    print(f"\n✅ 前處理完成！")
    print(f"處理好的資料已儲存在 '{OUTPUT_DATA_DIR}' 資料夾中。")
    print("您現在可以執行 `train.py` 來載入這些資料並開始訓練。")

if __name__ == "__main__":
    main()
