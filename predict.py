# predict.py

import pickle
from preprocess import preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer
with open('models/fake_news_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('models/vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def predict_news(news_text):
    # Preprocess the text
    cleaned_text = preprocess_text(news_text)
    
    # Transform text using the saved vectorizer
    news_tfidf = vectorizer.transform([cleaned_text])
    
    # Predict using the trained model
    prediction = model.predict(news_tfidf)
    
    if prediction == 0:
        return "Real News"
    else:
        return "Fake News"

# Example usage
print(predict_news("This is a fake news example"))
