# GPT2 護理記錄補全

本專案使用 Hugging Face 的 GPT-2 模型，訓練於 MIMIC-III 資料集中「Nursing」類別的醫療筆記，並提供互動式補全介面，讓使用者逐字或逐句選擇生成內容，探索語言模型在醫療語境下的表現。

---

## 資料來源：MIMIC-III

- 使用 `NOTEEVENTS.csv` 中 `CATEGORY == "Nursing"` 的筆記，共 223,556 筆  
- 其他類別包含 Radiology、ECG、Physician、Discharge summary 等  
- 資料下載方式：  
  ```bash
  pip install gdown  
  gdown 1QFIWLsFP6_MzCNe8euuupAK6hFxaueXl
  ```

---

## 補全介面功能

### 1. 逐詞補全

- 顯示 top-5 詞彙選項與機率  
- 使用者逐字選擇，控制語意走向  
- 支援指令：`re`（重新生成）、`stop`（結束）

### 2. 句子級補全

- 使用 `model.generate()` 產生 5 個候選句子  
- 使用者選擇其中一個作為新 prompt  
- 支援指令：`re`（重新生成）、`stop`（結束）

---

## 🔍 不同版本比較摘要

| 檔案名稱       | 資料量     | 補全方式     | 互動性     | 模型訓練策略       | 輸出結果           | 
|----------------|------------|--------------|------------|--------------------|--------------------|
| [gpt2_all.py](https://github.com/marcoleung052/NursingRecordCompletion_train/blob/bdd58bcc5b37a6d8cee679fca3fa03ec33e8b781/step1/gpt2_all.py "游標顯示")      | 全部資料   | pipeline 自動 | 無         | 標準訓練           | - | 
| [gpt2_5000.py](https://github.com/marcoleung052/NursingRecordCompletion_train/blob/bdd58bcc5b37a6d8cee679fca3fa03ec33e8b781/step1/gpt2_5000.py "游標顯示")    | 前 5000 筆 | 逐詞補全     | 高度互動   | LSTM 預訓練 + GPT2 | 在專有名詞斷句預測範圍較多 |  
| [gpt2_5%.py](https://github.com/marcoleung052/NursingRecordCompletion_train/blob/bdd58bcc5b37a6d8cee679fca3fa03ec33e8b781/step1/gpt2_5%25.py "游標顯示")      | 5% 抽樣    | 逐詞補全     | 高度互動   | 防過擬合訓練       | 只要不是連接詞等等能預測幾個字與樣本一樣 | 
| [gpt2_ok.py](https://github.com/marcoleung052/NursingRecordCompletion_train/blob/7aa012855cdff76302c3ea8a8313a7385919003a/step1/gpt2_ok.py "游標顯示")      | 全部資料   | 逐詞補全     | 高度互動   | 標準訓練           | 只要不是連接詞等等能預測幾個字與樣本一樣 |
| [gpt2_ok_all.py](https://github.com/marcoleung052/NursingRecordCompletion_train/blob/bdd58bcc5b37a6d8cee679fca3fa03ec33e8b781/step1/gpt2_ok_all.py "游標顯示")  | 全部資料   | 句子補全     | 中度互動   | 標準訓練           | 第一個字跟樣本一樣其他不同 |
