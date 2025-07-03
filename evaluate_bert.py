# evaluate_bert.py

import pandas as pd
import torch
from sklearn.metrics import classification_report
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
from datasets import Dataset

# Load test data
df = pd.read_csv('data/test_data.csv')
df = df[['text', 'label']].dropna()
df['label'] = df['label'].astype(int)
# Limit to 100 rows for faster evaluation
df = df.sample(1500, random_state=42)
# Load tokenizer and model
model_path = "models/bert_model"
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Tokenize
def tokenize_function(example):
    return tokenizer(example['text'], truncation=True, padding='max_length', max_length=512)

# Prepare dataset
test_dataset = Dataset.from_pandas(df)
test_dataset = test_dataset.map(tokenize_function, batched=False)

# Format for PyTorch
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# DataLoader
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
dataloader = DataLoader(test_dataset, batch_size=4, collate_fn=data_collator)

# Device setup
device = torch.device("cpu")
model.to(device)
model.eval()

# Inference
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Properly handle label extraction
        labels = batch['label'].to(device) if 'label' in batch else batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)

        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Evaluation
print(classification_report(all_labels, all_preds))