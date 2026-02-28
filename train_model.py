import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import re

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Basic cleaning to remove noise that might confuse the model
    text = re.sub(r'https?://[A-Za-z0-9./]+', '', text) # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove punctuation and numbers
    text = text.lower() # Convert to lowercase
    return text

def train():
    print("Loading dataset...")
    # Load dataset from parent directory
    data_path = os.path.join("..", "fake_or_real_news.csv")
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return

    data = pd.read_csv(data_path)
    
    # 0 for REAL, 1 for FAKE (consistent with untitled.py)
    print("Preprocessing data...")
    # Clean the text using the new function
    data['clean_text'] = data['text'].apply(clean_text)
    data['fake'] = data['label'].apply(lambda x: 0 if x == 'REAL' else 1)
    
    # The feature is 'clean_text', target is 'fake'
    X = data['clean_text']
    y = data['fake']

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Vectorizing text... (Now using Bi-grams and cleaning)")
    # Upgraded vectorizer: using n-grams (1,2) to catch phrases, capping at max_features to prevent memory issues
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7, ngram_range=(1, 2), max_features=50000)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    print("Training PassiveAggressiveClassifier model...")
    # Using PassiveAggressiveClassifier which is notoriously effective for text classification and fake news
    clf = PassiveAggressiveClassifier(max_iter=50, random_state=42)
    clf.fit(X_train_vectorized, y_train)

    print("Evaluating model...")
    y_pred = clf.predict(X_test_vectorized)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['REAL (0)', 'FAKE (1)']))

    print("Saving model and vectorizer...")
    # Save the model and vectorizer to the current directory
    joblib.dump(clf, 'fake_news_model.joblib')
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
    print("Saved as 'fake_news_model.joblib' and 'tfidf_vectorizer.joblib' successfully!")

if __name__ == "__main__":
    train()
