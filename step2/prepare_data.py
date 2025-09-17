import pandas as pd
import os
from sklearn.model_selection import train_test_split

data_path = "data/NOTEEVENTS.csv"
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

print("開始篩選 Nursing 類別...")

chunksize = 100000
nursing_texts = []

for chunk in pd.read_csv(data_path, chunksize=chunksize, low_memory=False):
    nursing_df = chunk[chunk["CATEGORY"] == "Nursing"]
    nursing_texts.extend(nursing_df["TEXT"].astype(str).tolist())

print(f"總共收集到 {len(nursing_texts)} 筆 Nursing 記錄")

train_texts, test_texts = train_test_split(nursing_texts, test_size=0.1, random_state=42)

with open(os.path.join(output_dir, "train.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(train_texts))

with open(os.path.join(output_dir, "test.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(test_texts))

print("✅ 資料已存到 data/train.txt 和 data/test.txt")
