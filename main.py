# main.py

import train_model
import evaluate
import predict

# Train model on dataset
train_model.train_model('data/fake_news_data.csv')

# Evaluate model
evaluate.evaluate_model('data/test_data.csv')

# Make predictions
print(predict.predict_news("Breaking news: Something fake happened"))
