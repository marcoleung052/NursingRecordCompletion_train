import pandas as pd
import os
from sklearn.model_selection import train_test_split

# 設定環境變數
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 1. 資料載入與前處理
print("1. 載入 NOTEEVENTS.csv...")
# 假設 NOTEEVENTS.csv 存在於當前目錄
try:
    df = pd.read_csv("NOTEEVENTS.csv", low_memory=False)
except FileNotFoundError:
    print("錯誤：找不到 NOTEEVENTS.csv 檔案。請確保檔案在當前目錄中。")
    exit()

# 篩選 "Nursing" 類別的筆記
nursing_df = df[df["CATEGORY"] == "Nursing"]

def clean_text(text):
    """清理文字：去除換行符並移除前後空白"""
    text = str(text).strip().replace("\n", " ").replace("\r", " ")
    return text

# 取得清理後的護理筆記列表
nursing_texts = nursing_df["TEXT"].dropna().apply(clean_text).tolist()

total_records = len(nursing_texts)
print(f"成功清理 {total_records} 筆護理筆記。")

# 2. 資料切分 (2:8 比例)
print("2. 進行資料切分 (2:8 比例)...")

# random_state 確保每次切分的結果一致
part_a_texts, part_b_texts = train_test_split(
    nursing_texts,
    test_size=0.8,  # train set 佔 80%
    random_state=42 # 保持可重現性
)

part_a_count = len(part_a_texts)
part_b_count = len(part_b_texts)

print(f"   - test set (20%): {part_a_count} 筆")
print(f"   - train set (80%): {part_b_count} 筆")
print(f"   - 總和檢查: {part_a_count + part_b_count} / {total_records} 筆")


# 3. 儲存為 .txt 檔案
def save_texts_to_file(filepath, text_list):
    """將文字列表寫入檔案，每行一條記錄"""
    with open(filepath, "w", encoding="utf-8") as f:
        for line in text_list:
            f.write(line + "\n")
    print(f"   - 檔案儲存成功: {filepath} ({len(text_list)} 筆記錄)")

print("3. 儲存切分後的檔案...")

# 儲存 test_set (20%)
save_texts_to_file("nursing_test_set.txt", part_a_texts)

# 儲存 train_set (80%)
save_texts_to_file("nursing_train_set.txt", part_b_texts)

print("\n資料切分與儲存完成。")
