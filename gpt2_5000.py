#!/home/jovyan/mimic_test/.venv/bin/python
import pandas as pd
import numpy as np
import re
import gdown
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.tokenize import sent_tokenize
import pickle
import os
import tensorflow as tf
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ================== 數據處理 ==================
def download_and_load_data():
    """下載並載入 MIMIC-III NOTEEVENTS 數據"""
    print("正在下載 MIMIC-III NOTEEVENTS 資料集...")
    url = "https://drive.google.com/uc?id=1QFIWLsFP6_MzCNe8euuupAK6hFxaueXl"
    output = "NOTEEVENTS.csv"

    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

    print("正在載入數據...")
    df = pd.read_csv(output)

    print("數據欄位:", df.columns.tolist())
    print("\nCATEGORY 分布:")
    print(df["CATEGORY"].value_counts())

    nursing_df = df[df["CATEGORY"] == "Nursing"].copy()
    print(f"\n護理紀錄總數: {len(nursing_df)}")

    return nursing_df

def preprocess_nursing_notes(nursing_df):
    """預處理護理紀錄文本"""
    print("正在預處理護理紀錄...")

    nursing_df = nursing_df.dropna(subset=['TEXT'])

    def clean_text(text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^\w\s\.\,\!\?\;\:]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    nursing_df['CLEANED_TEXT'] = nursing_df['TEXT'].apply(clean_text)
    nursing_df = nursing_df[nursing_df['CLEANED_TEXT'] != '']

    print(f"清理後的護理紀錄數: {len(nursing_df)}")
    return nursing_df

def create_sentence_pairs(texts, max_pairs=10000):
    """從護理紀錄創建句子對"""
    print("正在創建句子對...")

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    sentence_pairs = []
    for text in texts[:1000]:
        if len(text) < 50:
            continue
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            continue
        for i in range(len(sentences) - 1):
            first, second = sentences[i].strip(), sentences[i+1].strip()
            if (10 <= len(first) <= 200 and 10 <= len(second) <= 200):
                sentence_pairs.append((first, second))
                if len(sentence_pairs) >= max_pairs:
                    break
        if len(sentence_pairs) >= max_pairs:
            break

    print(f"創建了 {len(sentence_pairs)} 個句子對")
    return sentence_pairs

# ================== 模型 ==================
def build_lstm_model(vocab_size, embedding_dim=100, lstm_units=128, max_length=50):
    """建立 LSTM 模型"""
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        LSTM(lstm_units, return_sequences=True),
        Dropout(0.3),
        LSTM(lstm_units),
        Dropout(0.3),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

class NursingTextCompletion:
    """護理文字補齊模型類"""

    def __init__(self, max_features=10000, max_length=50):
        self.max_features = max_features
        self.max_length = max_length
        self.tokenizer = None
        self.model = None

    def prepare_data(self, sentence_pairs):
        print("正在準備訓練數據...")
        all_texts = []
        for first, second in sentence_pairs:
            all_texts.append(first)
            all_texts.append(second)

        self.tokenizer = Tokenizer(num_words=self.max_features, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(all_texts)

        X_sequences, y_sequences = [], []
        for first_sentence, second_sentence in sentence_pairs:
            first_seq = self.tokenizer.texts_to_sequences([first_sentence])[0]
            second_seq = self.tokenizer.texts_to_sequences([second_sentence])[0]
            if len(first_seq) > 0 and len(second_seq) > 0:
                X_sequences.append(first_seq)
                y_sequences.append(second_seq[0])

        X = pad_sequences(X_sequences, maxlen=self.max_length, padding="post")
        y = np.array(y_sequences)

        print(f"訓練數據形狀: X={X.shape}, y={y.shape}")
        return X, y

    def train(self, sentence_pairs, epochs=10, batch_size=32):
        X, y = self.prepare_data(sentence_pairs)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        vocab_size = min(len(self.tokenizer.word_index) + 1, self.max_features)
        self.model = build_lstm_model(vocab_size, max_length=self.max_length)

        print("開始訓練模型...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        return history

    def predict_next_sentence_start(self, input_sentence, top_k=5):
        if self.model is None or self.tokenizer is None:
            return "模型尚未訓練"
        input_seq = self.tokenizer.texts_to_sequences([input_sentence.lower()])[0]
        input_padded = pad_sequences([input_seq], maxlen=self.max_length, padding="post")
        predictions = self.model.predict(input_padded)[0]
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        reverse_word_map = {v: k for k, v in self.tokenizer.word_index.items()}
        results = [(reverse_word_map[idx], predictions[idx]) for idx in top_indices if idx in reverse_word_map]
        return results

    def save_model(self, filepath):
        """保存模型與 tokenizer"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if self.model is not None:
            self.model.save(f"{filepath}.h5")
        if self.tokenizer is not None:
            with open(f"{filepath}_tokenizer.pkl", "wb") as f:
                pickle.dump(self.tokenizer, f)
        print(f"✅ 模型已保存到 {filepath}.h5 和 {filepath}_tokenizer.pkl")

    def load_model(self, filepath):
        """載入模型與 tokenizer"""
        try:
            self.model = tf.keras.models.load_model(f"{filepath}.h5")
            with open(f"{filepath}_tokenizer.pkl", "rb") as f:
                self.tokenizer = pickle.load(f)
            print(f"✅ 模型已從 {filepath} 載入成功")
        except Exception as e:
            print(f"❌ 載入模型時發生錯誤: {e}")

# ================== 主程序 ==================
def main():
    print("=== 護理紀錄文字補齊模型訓練系統 ===\n")

    nursing_df = download_and_load_data()
    nursing_df = preprocess_nursing_notes(nursing_df)
    sentence_pairs = create_sentence_pairs(nursing_df["CLEANED_TEXT"].tolist())

    print("\n=== 開始訓練文字補齊模型 ===")
    completion_model = NursingTextCompletion()
    training_pairs = sentence_pairs[:1000]
    completion_model.train(training_pairs, epochs=5, batch_size=32)

    # 保存模型
    completion_model.save_model("saved_models/nursing_completion_model")

    # ================== 補齊互動界面 ==================
    # 載入 LSTM 訓練好的模型轉成 Hugging Face GPT2 格式（假設已轉換）
    # 若尚未轉換，需先使用 Hugging Face 的 from_pretrained 保存
    model = GPT2LMHeadModel.from_pretrained("saved_models/nursing_completion_model", local_files_only=True)
    tokenizer = GPT2Tokenizer.from_pretrained("saved_models/nursing_completion_model", local_files_only=True)
    model.eval()

    print("\n=== 護理文字補齊互動模式 ===")
    while True:
        prompt = input("請輸入開頭句子（輸入 stop 結束）：").strip()
        if prompt.lower() == "stop":
            break

        while True:
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1).squeeze()

            temperature = 1.0
            probs = torch.pow(probs, 1.0 / temperature)
            probs = probs / probs.sum()

            top_k = 5
            top_probs, top_indices = torch.topk(probs, top_k)
            top_tokens = [tokenizer.decode([idx]) for idx in top_indices]
            options = list(zip(top_tokens, top_probs))

            print("\n請選擇下一個詞：")
            for i, (tok, prob) in enumerate(options):
                print(f"{chr(97+i)}. {tok.strip()} ({prob.item()*100:.1f}%)")

            choice = input
