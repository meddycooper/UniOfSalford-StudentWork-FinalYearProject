# preprocess.py

import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords list
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove punctuation and stopwords, and lemmatize words
    cleaned_tokens = [
        lemmatizer.lemmatize(word.lower()) 
        for word in tokens if word not in stop_words and word not in string.punctuation
    ]
    
    # Join tokens back into a single string
    return " ".join(cleaned_tokens)
