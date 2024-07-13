import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, BertTokenizer, Trainer, DataCollatorWithPadding,AutoTokenizer

from datasets import Dataset
# 1. โหลดโมเดลและ Tokenizer
checkpoint_path = r"D:\machine_learning_AI_Builders\บท4\NLP\Text_classification\test-trainer\checkpoint-1000"  # ระบุเส้นทางของ checkpoint
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

sentence1 = "It is a pleasant day outside."
sentence2 = "She dislikes reading novels."

# เตรียมข้อมูล
inputs = tokenizer(sentence1, sentence2, return_tensors="pt", truncation=True)

import torch
import numpy as np

with torch.no_grad():
    logits = model(**inputs).logits

# แปลง logits เป็น predictions
pred = np.argmax(logits, axis=-1)

print(f"Predicted label: {pred[0]}")
