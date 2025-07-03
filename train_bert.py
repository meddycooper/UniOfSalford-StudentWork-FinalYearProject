# train_bert.py

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset

# Load data
real = pd.read_csv('data/real_news_data.csv')
fake = pd.read_csv('data/fake_news_data.csv')

real['label'] = 1
fake['label'] = 0

df = pd.concat([real, fake], ignore_index=True)
df = df[['text', 'label']].dropna()

# Sample smaller dataset for faster training on CPU
df = df.sample(1500, random_state=42)

# Train/test split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_dict({'text': train_texts.tolist(), 'label': train_labels.tolist()})
test_dataset = Dataset.from_dict({'text': test_texts.tolist(), 'label': test_labels.tolist()})

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Define model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Training setup
training_args = TrainingArguments(
    output_dir="models/bert_output",
    logging_dir="models/bert_logs",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

from sklearn.metrics import accuracy_score

def compute_metrics(p):
    preds = p.predictions.argmax(axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained("models/bert_model")
tokenizer.save_pretrained("models/bert_model")

print("DistilBERT model trained and saved.")

# Evaluate the model
eval_results = trainer.evaluate()
accuracy = eval_results.get('eval_accuracy', None)

if accuracy is not None:
    print(f"Accuracy: {accuracy:.4f}")
else:
    print("Accuracy not found in evaluation results.")