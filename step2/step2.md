# 護理記錄補全

這個專案使用 Hugging Face 的不同模型比較，訓練於 MIMIC-III 的 Nursing Notes，並評估模型品質，探索語言模型在醫療語境下表現的比較。

## 模型

### 無法運行：
- MPT-7b (GPU記憶體不足)
- GPT-neo-1.3B (GPU記憶體不足)
- OpenELM-270M(apple) (未允權)
- Qwen3-1.3B (GPU記憶體不足)
- DCLM-Baseline-7B(apple) (GPU記憶體不足)
- bitnet-b1.58-2B-4T(microsoft) (GPU記憶體不足)

### 可運行：
- [GPT2](-)
- [TinyLlama-1.1B](https://github.com/marcoleung052/NursingRecordCompletion_train/blob/0a83c97a7ad4afd9d8b347bfbc98fcb5fcea7aae/step2/TinyLlama-1.1B.py)
- [Qwen3-0.6B](https://github.com/marcoleung052/NursingRecordCompletion_train/blob/0a83c97a7ad4afd9d8b347bfbc98fcb5fcea7aae/step2/Qwen3-0.6.py)
- [Opt-350m(facebook)](https://github.com/marcoleung052/NursingRecordCompletion_train/blob/0a83c97a7ad4afd9d8b347bfbc98fcb5fcea7aae/step2/opt-350M.py)

## 主要功能對比

| 項目       | GPT2                         | TinyLlama-1.1B                     | Qwen3-0.6B | Opt-350m                                |
|------------|------------------------------|------------------------------------|------------|------------------------------------------|
| 資料規模   | 全部資料                     | 全部資料                           | -          | 抽樣 10%                                 |
| 訓練方式   | 全參數微調 (fine-tuning)     | LoRA 微調 + 量化 QLoRA            | -          | 自回歸式訓練 + 梯度累積 + 梯度裁剪       |
| 輸出方式   | 訓練完成：儲存模型與 tokenizer | 訓練 log 輸出 loss，模型輸出目錄 | -          | 輸出 loss，保存最佳模型與中間輸出       |

## 資料處理對比

| 項目       | GPT2                                                                 | TinyLlama-1.1B                         | Qwen3-0.6B                                                                 | Opt-350m                          |
|------------|----------------------------------------------------------------------|----------------------------------------|---------------------------------------------------------------------------|-----------------------------------|
| 補 0       | 右側                                                                | 右側                                   | 右側                                                                      | 右側                              |
| 篩選條件   | 資料過濾時：刪除過長的序列（block_size 以上）、空白樣本。         | Nursing 類別                            | 長度 50–500 字元，句子至少 20 字 5 詞                                     | 移除 NAN                          |
| 預處理     | - 使用 GPT2Tokenizer 分詞<br> - 將文字轉換成 token id<br> - 補零/截斷到固定長度 | 去除多餘空白、換行                      | regular expression,<br>chunksize 分塊讀檔減少記憶體,<br>句子切分：前半輸入、後半輸出 | 移除部分標點符號，256            |
| 資料分割   | 9：1                                                                 | 無分割                                  | 8：2                                                                      | 9：1                              |
| 儲存格式   | 文字檔                                                              | 文字檔                                  | JSON 檔                                                                   | 文字檔                            |

## 模型訓練對比

| 項目         | GPT2                                                                 | TinyLlama-1.1B                          | Qwen3-0.6B                        | Opt-350m                          |
|--------------|----------------------------------------------------------------------|-----------------------------------------|----------------------------------|-----------------------------------|
| 基礎模型     | 預設使用 gpt2 (117M)                                                 | TinyLlama/TinyLlama-1.1B-Chat-v1.0      | Qwen 0.6B                         | Facebook/opt-350M                 |
| 訓練策略     | - AdamW optimizer（transformers 預設）<br> - 線性學習率衰減 (`lr_scheduler_type="linear"`)<br> - gradient clipping 預設啟用 | LoRA + VRAM                             | Trainer + gradient accumulation  | 標準訓練                          |
| Epochs       | 1                                                                    | 2                                       | 1                                | 5                                 |
| Batch Size   | 2                                                                    | 4                                       | 1                                | 4                                 |
| 評估機制     | 驗證集 loss 監控                                                     | 驗證集 loss 監控                        | 驗證集 loss 監控                 | 驗證集 loss 監控                 |
| 模型保存     | 新增資料夾儲存                                                       | 新增資料夾儲存                          | 新增資料夾儲存                   | 新增資料夾儲存                   |

## 評估

### ROUGE 分數

| 項目        | GPT2 | TinyLlama-1.1B | Qwen3-0.6B | Opt-350m | 字面重疊說明         |
|-------------|------|----------------|------------|----------|----------------------|
| ROUGE-1     | -    | 0.2208         | 0.1485     | 0.1749   | 單字重疊率           |
| ROUGE-2     | -    | 0.1914         | 0.1300     | 0.1532   | 雙字重疊率           |
| ROUGE-L     | -    | 0.2180         | 0.1489     | 0.1751   | 最長公共子序列       |
| ROUGE-Lsum  | -    | 0.2190         | 0.1479     | 0.1753   | 多句摘要             |

### BERTScore：

| 項目       | GPT2 | TinyLlama-1.1B | Qwen3-0.6B | Opt-350m | 說明                                           |
|------------|------|----------------|------------|----------|------------------------------------------------|
| Precision  | -    | 0.7672         | 0.7867     | 0.7821   | 生成的詞向量與參考相似度（語意正確？）        |
| Recall     | -    | 0.9253         | 0.9087     | 0.9194   | 參考的詞向量與生成相似度（參考被涵蓋？）      |
| F1         | -    | 0.8386         | 0.8431     | 0.8450   | Precision、Recall 調和平均（整體語意）        |
