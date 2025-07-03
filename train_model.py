import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import pickle

# Import preprocess function
from preprocess import preprocess_text

def train_model(true_news_path, fake_news_path):
    # Load datasets
    true_news_df = pd.read_csv(true_news_path)
    fake_news_df = pd.read_csv(fake_news_path)
    
    # Add labels
    true_news_df['label'] = 1
    fake_news_df['label'] = 0  # Label fake news as 0
    
    # Combine both datasets
    df = pd.concat([true_news_df, fake_news_df], ignore_index=True)
    
    # Preprocess the text data
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    
    # Split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['label'], test_size=0.2)
    
    # Vectorize text using TF-IDF
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train a Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_tfidf)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    
    # Save model and vectorizer for future use
    with open('models/fake_news_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    with open('models/vectorizer.pkl', 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

# Train model with both datasets
train_model('data/real_news_data.csv', 'data/fake_news_data.csv')
