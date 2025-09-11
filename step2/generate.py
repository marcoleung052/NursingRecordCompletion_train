import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
import os

model_dir = "checkpoints/gpt2-nursing"
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

test_path = "data/test.txt"
output_path = "outputs/generated.jsonl"
os.makedirs("outputs", exist_ok=True)

with open(test_path, "r", encoding="utf-8") as f:
    test_lines = [line.strip() for line in f.readlines() if line.strip()]

N = 100
test_lines = test_lines[:N]

results = []
for i, prompt in enumerate(test_lines, 1):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=2
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    results.append({"id": i, "prompt": prompt, "generated": generated_text})
    print(f"[{i}/{len(test_lines)}] 完成")

with open(output_path, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"✅ 已完成生成，結果存到 {output_path}")
