import pandas as pd
import os
import re
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import warnings
import gc
warnings.filterwarnings("ignore")

class MIMICNursingTextCompletion:
    def __init__(self, model_name = "Qwen/Qwen3-0.6B", output_dir="./nursing_model"):
        """
        初始化MIMIC-III護理記錄文字補齊模型
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.test_dataset = None
        
        # 創建輸出目錄
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs("data", exist_ok=True)
    
    def preprocess_nursing_data(self, data_path="data/NOTEEVENTS.csv", max_samples=5000):
        """
        極度優化的資料預處理 - 大幅減少樣本數
        """
        print(f"🔄 開始篩選 Nursing 類別... (極限優化，最多 {max_samples} 筆)")
        
        chunksize = 5000  # 進一步減少chunk size
        nursing_texts = []
        total_rows = 0
        
        try:
            for chunk in pd.read_csv(data_path, chunksize=chunksize, low_memory=False):
                total_rows += len(chunk)
                nursing_df = chunk[chunk["CATEGORY"] == "Nursing"]
                
                # 清理和標準化文本
                clean_texts = self._clean_nursing_texts(nursing_df["TEXT"].astype(str).tolist())
                nursing_texts.extend(clean_texts)
                
                print(f"已處理 {total_rows} 行，收集到 {len(nursing_texts)} 筆護理記錄")
                
                # 提早停止
                if len(nursing_texts) >= max_samples:
                    nursing_texts = nursing_texts[:max_samples]
                    print(f"🛑 已達到最大樣本數 {max_samples}，停止讀取")
                    break
                
                # 強制垃圾回收
                del chunk, nursing_df
                gc.collect()
                
        except FileNotFoundError:
            print(f"❌ 找不到檔案: {data_path}")
            return False
        
        if len(nursing_texts) == 0:
            print("❌ 沒有找到護理記錄")
            return False
            
        print(f"✅ 收集到 {len(nursing_texts)} 筆 Nursing 記錄")
        
        # 分割訓練和測試資料 (8:2)
        train_texts, test_texts = train_test_split(
            nursing_texts, 
            test_size=0.2,
            random_state=42
        )
        
        print(f"📊 訓練資料: {len(train_texts)} 筆")
        print(f"📊 測試資料: {len(test_texts)} 筆")
        
        # 創建極少量的訓練樣本
        self._create_minimal_samples(train_texts, test_texts)
        
        # 清理記憶體
        del nursing_texts, train_texts, test_texts
        gc.collect()
        
        return True
    
    def _clean_nursing_texts(self, texts):
        """清理護理記錄文本"""
        cleaned_texts = []
        
        for text in texts:
            if pd.isna(text) or len(str(text).strip()) < 30:
                continue
                
            text = str(text).strip()
            text = re.sub(r'\s+', ' ', text)  # 標準化空白
            text = re.sub(r'\n+', ' ', text)  # 換行改為空白
            
            # 只取中等長度的文本
            if 50 < len(text) < 500:
                cleaned_texts.append(text)
                
        return cleaned_texts
    
    def _create_minimal_samples(self, train_texts, test_texts):
        """創建極少量的訓練樣本"""
        print("🔄 創建極少量訓練樣本...")
        
        def create_samples(texts, split_name, max_total_samples=1000):
            samples = []
            sample_count = 0
            
            for text in tqdm(texts[:min(len(texts), 500)], desc=f"處理{split_name}資料"):  # 最多處理500筆原文
                if sample_count >= max_total_samples:
                    break
                    
                # 簡單分句
                sentences = re.split(r'[.!?]\s+', text)
                
                for sentence in sentences[:2]:  # 每篇文章最多取2句
                    if sample_count >= max_total_samples:
                        break
                        
                    sentence = sentence.strip()
                    if len(sentence) < 20:
                        continue
                        
                    words = sentence.split()
                    if len(words) < 5:
                        continue
                    
                    # 只創建一個樣本：取前一半作為input，後一半作為output
                    mid_point = len(words) // 2
                    if mid_point >= 2:
                        prefix = ' '.join(words[:mid_point])
                        completion = ' '.join(words[mid_point:])
                        
                        if len(completion.strip()) > 3:
                            samples.append({
                                'input': f"請補齊以下護理記錄：{prefix}",
                                'output': completion
                            })
                            sample_count += 1
            
            return samples
        
        # 訓練集最多1000個樣本，測試集最多100個樣本
        train_samples = create_samples(train_texts, "訓練", max_total_samples=40000)
        test_samples = create_samples(test_texts, "測試", max_total_samples=10000)
        
        # 儲存樣本
        with open("data/train_samples.json", "w", encoding="utf-8") as f:
            json.dump(train_samples, f, ensure_ascii=False, indent=2)
            
        with open("data/test_samples.json", "w", encoding="utf-8") as f:
            json.dump(test_samples, f, ensure_ascii=False, indent=2)
        
        print(f"訓練樣本: {len(train_samples)} 個")
        print(f"測試樣本: {len(test_samples)} 個")
        
        # 清理記憶體
        del train_samples, test_samples
        gc.collect()
    
    def load_model_and_tokenizer(self):
        """載入模型和tokenizer"""
        print("🔄 載入Qwen模型和tokenizer...")
        
        try:
            # 先載入tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 載入模型 (更多記憶體優化)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                use_cache=False  # 減少記憶體使用
            )
            
            print("✅ 模型載入成功")
            return True
            
        except Exception as e:
            print(f"❌ 模型載入失敗: {e}")
            return False
    
    def prepare_datasets_streaming(self):
        """串流方式準備資料集，避免一次性載入所有資料"""
        print("🔄 以串流方式準備資料集...")
        
        # 載入樣本
        with open("data/train_samples.json", "r", encoding="utf-8") as f:
            train_samples = json.load(f)
            
        with open("data/test_samples.json", "r", encoding="utf-8") as f:
            test_samples = json.load(f)
        
        print(f"📊 載入樣本 - 訓練: {len(train_samples)}, 測試: {len(test_samples)}")
        
        # 逐一tokenize，而不是批量處理
        def tokenize_samples(samples, desc):
            tokenized_samples = []
            
            for sample in tqdm(samples, desc=desc):
                full_text = f"{sample['input']} {sample['output']}{self.tokenizer.eos_token}"
                
                # 單個樣本tokenize
                tokenized = self.tokenizer(
                    full_text,
                    truncation=True,
                    padding=False,  # 不padding，訓練時再處理
                    max_length=128,  # 大幅減少max_length
                    return_tensors=None  # 返回列表而不是張量
                )
                
                tokenized_samples.append({
                    'input_ids': tokenized['input_ids'],
                    'attention_mask': tokenized['attention_mask'],
                    'labels': tokenized['input_ids'].copy()  # labels和input_ids相同
                })
                
                # 定期垃圾回收
                if len(tokenized_samples) % 100 == 0:
                    gc.collect()
            
            return tokenized_samples
        
        # 串流tokenize
        train_tokenized = tokenize_samples(train_samples, "Tokenizing訓練資料")
        test_tokenized = tokenize_samples(test_samples, "Tokenizing測試資料")
        
        # 轉換為Dataset
        self.train_dataset = Dataset.from_list(train_tokenized)
        self.test_dataset = Dataset.from_list(test_tokenized)
        
        print(f"✅ 資料集準備完成 - 訓練: {len(self.train_dataset)}, 測試: {len(self.test_dataset)}")
        
        # 清理記憶體
        del train_samples, test_samples, train_tokenized, test_tokenized
        gc.collect()
    
    def train_model(self, num_epochs=1, batch_size=1, learning_rate=5e-6):
        """極簡訓練配置"""
        print("🔄 開始極簡訓練...")
        
        # 檢查transformers版本並調整參數
        import transformers
        transformers_version = transformers.__version__
        print(f"📦 Transformers版本: {transformers_version}")
        
        # 最小化訓練參數 - 移除了不相容的參數
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=16,  # 大幅增加梯度累積
            warmup_steps=10,
            learning_rate=learning_rate,
            logging_steps=20,
            save_steps=200,
            save_total_limit=1,
            load_best_model_at_end=False,  # 不載入最佳模型以節省記憶體
            report_to=None,
            dataloader_pin_memory=False,
            fp16=False,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            # 移除了 predict_with_generate 參數
        )
        
        # 自定義資料collator，處理padding
        def custom_data_collator(features):
            # 找到最長序列
            max_length = max(len(f['input_ids']) for f in features)
            max_length = min(max_length, 128)  # 限制最大長度
            
            batch = {}
            batch['input_ids'] = []
            batch['attention_mask'] = []
            batch['labels'] = []
            
            for feature in features:
                input_ids = feature['input_ids'][:max_length]
                attention_mask = feature['attention_mask'][:max_length]
                labels = feature['labels'][:max_length]
                
                # Padding
                pad_length = max_length - len(input_ids)
                if pad_length > 0:
                    input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_length
                    attention_mask = attention_mask + [0] * pad_length
                    labels = labels + [-100] * pad_length  # -100會被忽略
                
                batch['input_ids'].append(input_ids)
                batch['attention_mask'].append(attention_mask)
                batch['labels'].append(labels)
            
            # 轉換為張量
            return {
                'input_ids': torch.tensor(batch['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(batch['attention_mask'], dtype=torch.long),
                'labels': torch.tensor(batch['labels'], dtype=torch.long)
            }
        
        # 初始化Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            data_collator=custom_data_collator,
            tokenizer=self.tokenizer,
        )
        
        # 訓練
        print("🚀 開始訓練...")
        try:
            trainer.train()
            print("✅ 訓練成功完成")
        except Exception as e:
            print(f"❌ 訓練過程中出現錯誤: {e}")
            print("嘗試降低batch size或調整其他參數")
            raise e
        
        # 儲存模型
        print("💾 儲存模型...")
        try:
            trainer.save_model()
            self.tokenizer.save_pretrained(self.output_dir)
            print("✅ 模型儲存成功")
        except Exception as e:
            print(f"❌ 模型儲存失敗: {e}")
        
        # 清理
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def evaluate_model(self):
        """簡單評估"""
        print("🔄 簡單評估...")
        
        # 載入少量測試樣本
        with open("data/test_samples.json", "r", encoding="utf-8") as f:
            test_samples = json.load(f)
        
        # 只測試5個樣本
        results = []
        for i, sample in enumerate(test_samples[:5]):
            input_text = sample['input']
            expected_output = sample['output']
            predicted_output = self.generate_completion(input_text)
            
            results.append({
                'input': input_text,
                'expected': expected_output,
                'predicted': predicted_output
            })
            
            print(f"測試 {i+1}:")
            print(f"輸入: {input_text}")
            print(f"預期: {expected_output}")
            print(f"預測: {predicted_output}")
            print("-" * 40)
        
        return results
    
    def generate_completion(self, input_text, max_length=30):
        """生成文字補齊"""
        if self.model is None:
            return "模型尚未載入"
        
        try:
            inputs = self.tokenizer(input_text, return_tensors="pt")
            
            # 將輸入移到正確的設備
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=inputs['input_ids'].shape[1] + max_length,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=max_length  # 更明確的參數
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            completion = generated_text[len(input_text):].strip()
            
            return completion
        
        except Exception as e:
            print(f"生成時出現錯誤: {e}")
            return f"生成失敗: {str(e)}"

def main():
    """主要執行函數 - 極簡版本"""
    print("🏥 MIMIC-III 護理記錄文字補齊系統 (極簡版 v2)")
    print("=" * 50)
    
    # 檢查系統資訊
    print(f"🔧 PyTorch版本: {torch.__version__}")
    print(f"🔧 CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🔧 CUDA設備: {torch.cuda.get_device_name()}")
        print(f"🔧 CUDA記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 設定極小的資源使用
    system = MIMICNursingTextCompletion(
        model_name = "Qwen/Qwen3-0.6B",
        output_dir="./nursing_minimal_model"
    )
    
    try:
        # 步驟1: 極簡資料預處理
        print("\n🔄 步驟1: 資料預處理")
        if not system.preprocess_nursing_data(max_samples=50000):  # 進一步減少到1000筆
            print("❌ 資料預處理失敗")
            return
        
        gc.collect()
        
        # 步驟2: 載入模型
        print("\n🔄 步驟2: 載入模型")
        if not system.load_model_and_tokenizer():
            print("❌ 模型載入失敗")
            return
        
        # 步驟3: 串流準備資料集
        print("\n🔄 步驟3: 準備資料集")
        system.prepare_datasets_streaming()
        
        # 步驟4: 極簡訓練
        print("\n🔄 步驟4: 開始訓練")
        system.train_model(num_epochs=1, batch_size=1, learning_rate=5e-6)
        
        # 步驟5: 簡單評估
        print("\n🔄 步驟5: 評估結果")
        system.evaluate_model()
        
        print("\n🎉 極簡版訓練完成！")
        print(f"📁 模型儲存在: {system.output_dir}")
        
    except Exception as e:
        print(f"\n❌ 執行過程中出現錯誤: {e}")
        print("請檢查系統資源和數據檔案")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

'''
=== 📊 評估結果 ===
ROUGE 分數：
rouge1: 0.1485
rouge2: 0.1300
rougeL: 0.1489
rougeLsum: 0.1479

BERTScore (平均)：
Precision: 0.7867
Recall:    0.9087
F1:        0.8431
'''
