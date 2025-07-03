# evaluate.py

import pickle
import pandas as pd
from sklearn.metrics import classification_report
from preprocess import preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer
with open('models/fake_news_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    
with open('models/vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def evaluate_model(dataset_path):
    # Load the test dataset
    df = pd.read_csv(dataset_path)
    
    # Preprocess the text
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    
    # Transform text using the saved vectorizer
    X_test_tfidf = vectorizer.transform(df['cleaned_text'])
    
    # Predict using the model
    y_pred = model.predict(X_test_tfidf)
    
    # Print classification report
    print(classification_report(df['label'], y_pred))

# Evaluate model performance
evaluate_model('data/test_data.csv')
