import json
from datasets import load_metric
import evaluate

generated_path = "outputs/generated.jsonl"
generated = []
with open(generated_path, "r", encoding="utf-8") as f:
    for line in f:
        generated.append(json.loads(line))

references = [g["prompt"] for g in generated]
predictions = [g["generated"] for g in generated]

rouge = load_metric("rouge")
rouge_scores = rouge.compute(predictions=predictions, references=references)

bertscore = evaluate.load("bertscore")
bert_scores = bertscore.compute(predictions=predictions, references=references, lang="en")

print("\n=== 📊 評估結果 ===")
print("ROUGE 分數：")
for k, v in rouge_scores.items():
    print(f"{k}: {v.mid.fmeasure:.4f}")

print("\nBERTScore (平均)：")
print(f"Precision: {sum(bert_scores['precision'])/len(bert_scores['precision']):.4f}")
print(f"Recall:    {sum(bert_scores['recall'])/len(bert_scores['recall']):.4f}")
print(f"F1:        {sum(bert_scores['f1'])/len(bert_scores['f1']):.4f}")
