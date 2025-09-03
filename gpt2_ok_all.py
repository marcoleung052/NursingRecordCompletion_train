import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 載入模型與 tokenizer
model = GPT2LMHeadModel.from_pretrained("./gpt2-nursing-final", local_files_only=True)
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-nursing-final", local_files_only=True)
model.eval()

while True:
    prompt = input("請輸入開頭句子：").strip()
    if prompt == "stop":
        break

    while True:
        # 編碼輸入
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=inputs["input_ids"].shape[1] + 20,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=1.0,
                num_return_sequences=5
            )

        # 解碼候選句子
        candidates = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        # 顯示選項
        print("\n請選擇下一個句子：")
        for i, sentence in enumerate(candidates):
            print(f"{chr(97+i)}. {sentence.strip()}")

        # 使用者輸入
        choice = input("請輸入選項或指令(re=重新生成, stop=結束)：").strip().lower()

        if choice == "stop":
            break
        elif choice == "re":
            continue  # 重新生成候選句子
        elif choice in [chr(97+i) for i in range(len(candidates))]:
            prompt = candidates[ord(choice) - 97]
            print(f"\n目前句子：{prompt}")
        else:
            print("無效選項或指令，請重新輸入。")
