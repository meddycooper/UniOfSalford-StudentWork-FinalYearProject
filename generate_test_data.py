# generate_test_data.py

import pandas as pd
from sklearn.model_selection import train_test_split

# Load datasets
real_news = pd.read_csv('data/real_news_data.csv')
fake_news = pd.read_csv('data/fake_news_data.csv')

# Add labels
real_news['label'] = 1
fake_news['label'] = 0

# Combine datasets
df = pd.concat([real_news, fake_news], ignore_index=True)

# Shuffle and split into train/test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save the test set
test_df.to_csv('data/test_data.csv', index=False)

print("Test data saved to data/test_data.csv")