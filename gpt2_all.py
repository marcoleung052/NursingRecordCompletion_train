import os
import argparse
import pandas as pd
from transformers import (
    GPT2Tokenizer, 
    GPT2LMHeadModel, 
    Trainer, 
    TrainingArguments, 
    TextDataset, 
    DataCollatorForLanguageModeling, 
    pipeline
)

# ========== Step 1. 資料準備 ==========
def prepare_data(data_path="data/NOTEEVENTS.csv", output_path="data/nursing.txt", chunksize=5000):
    print("開始篩選 Nursing 類別...")
    os.makedirs("data", exist_ok=True)
    texts = []
    for chunk in pd.read_csv(data_path, chunksize=chunksize, low_memory=False):
        nursing_df = chunk[chunk["CATEGORY"] == "Nursing"]
        texts.extend(nursing_df["TEXT"].dropna().tolist())
    print(f"總共收集到 {len(texts)} 筆 Nursing 記錄")
    with open(output_path, "w", encoding="utf-8") as f:
        for line in texts:
            f.write(line.replace("\n", " ") + "\n")
    print(f"Nursing 資料已經存到 {output_path}")


# ========== Step 2. 模型訓練 ==========
def train_gpt2(data_file="data/nursing.txt", model_dir="gpt2-nursing", num_train_epochs=1):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=data_file,
        block_size=128
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir=model_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=2,
        save_steps=5000,
        save_total_limit=2,
        logging_steps=100,
        prediction_loss_only=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )

    print("開始訓練 GPT-2...")
    trainer.train()
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"模型已經存到 {model_dir}")


# ========== Step 3. 文字生成 ==========
def generate_text(model_dir="gpt2-nursing", prompt="Patient is resting comfortably."):
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    output = generator(prompt, max_length=100, num_return_sequences=1)
    print("\n=== 生成結果 ===")
    print(output[0]["generated_text"])


# ========== 主程式 ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, required=True, choices=["prepare", "train", "generate"], help="要執行的步驟")
    parser.add_argument("--prompt", type=str, default="Patient is resting comfortably.", help="生成文字的開頭")
    parser.add_argument("--epochs", type=int, default=1, help="訓練的 epoch 數")
    args = parser.parse_args()

    if args.step == "prepare":
        prepare_data()
    elif args.step == "train":
        train_gpt2(num_train_epochs=args.epochs)
    elif args.step == "generate":
        generate_text(prompt=args.prompt)
