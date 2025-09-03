#!/home/jovyan/mimic_test/.venv/bin/python
import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

# ------------------ 數據處理 ------------------
def download_and_load_data():
    df = pd.read_csv("NOTEEVENTS.csv", low_memory=False)
    nursing_df = df[df["CATEGORY"] == "Nursing"].copy()
    nursing_df = nursing_df.dropna(subset=['TEXT'])
    nursing_df['CLEANED_TEXT'] = nursing_df['TEXT'].apply(lambda x: re.sub(r'[^\w\s\.\,\!\?\;\:]', ' ', str(x).lower()))
    nursing_df = nursing_df[nursing_df['CLEANED_TEXT'] != '']
    return nursing_df

def create_sentence_pairs(texts, max_pairs=5000):
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    sentence_pairs = []
    for text in texts[:1000]:
        if len(text) < 50:
            continue
        sentences = sent_tokenize(text)
        for i in range(len(sentences)-1):
            first, second = sentences[i].strip(), sentences[i+1].strip()
            if 10 <= len(first) <= 200 and 10 <= len(second) <= 200:
                sentence_pairs.append(f"{first} {second}")
                if len(sentence_pairs) >= max_pairs:
                    break
        if len(sentence_pairs) >= max_pairs:
            break
    return sentence_pairs

# ------------------ 主程序 ------------------
def main():
    nursing_df = download_and_load_data()
    sentence_pairs = create_sentence_pairs(nursing_df["CLEANED_TEXT"].tolist())
    print(f"句子對數量: {len(sentence_pairs)}")

    # ------------------ GPT2 互動式補全 ------------------
    model_path = "./saved_models/nursing_completion_model"
    if not os.path.exists(model_path):
        print("模型路徑不存在，請先訓練或下載 GPT2 模型")
        return

    model = GPT2LMHeadModel.from_pretrained(model_path, local_files_only=True)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path, local_files_only=True)
    model.eval()

    while True:
        prompt = input("請輸入開頭句子：").strip()
        if prompt == "stop":
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

            print("\n請選擇下一個詞：")
            for i, tok in enumerate(top_tokens):
                print(f"{chr(97+i)}. {tok.strip()} ({top_probs[i].item()*100:.1f}%)")

            choice = input("請輸入選項或指令(re=重新生成, stop=結束)：").strip().lower()
            if choice == "stop":
                break
            elif choice == "re":
                continue
            elif choice in [chr(97+i) for i in range(top_k)]:
                selected_token = top_tokens[ord(choice)-97]
                prompt += selected_token
                print(f"\n目前句子：{prompt}")
            else:
                print("無效選項或指令，請重新輸入。")

if __name__ == "__main__":
    main()
