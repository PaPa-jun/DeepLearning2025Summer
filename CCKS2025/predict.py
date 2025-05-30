import torch, pandas as pd
from torch.utils.data import DataLoader
from modules import TestDataset
from transformers import BertForSequenceClassification, AutoTokenizer, DistilBertForSequenceClassification, DistilBertTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

model = DistilBertForSequenceClassification.from_pretrained("./best_model_distil")
tokenizer = DistilBertTokenizer.from_pretrained("./best_model_distil")
model.to(device)
text = pd.read_json("test.jsonl", lines=True)["text"].tolist()

test_set = TestDataset(text, tokenizer)
test_loader = DataLoader(test_set)

model.eval()
predictions = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # 前向传播
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # 获取预测类别（0 或 1）
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(preds)

# 4. 写入结果文件
with open("submit.txt", "w") as f:
    for pred in predictions:
        f.write(f"{pred}\n")