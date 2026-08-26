import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from transformers import pipeline
import re
import json
from tqdm import tqdm

tqdm.pandas()

def clean_text(text):
    text = re.sub(r'@[A-Za-z0-9_]+', '', text) # Remove mentions
    text = re.sub(r'http\S+', '', text) # Remove links
    return text.strip()

def main():
    print("Loading data...")
    df = pd.read_csv('Tweets.csv')
    
    # Keep relevant columns
    df = df[['tweet_id', 'airline_sentiment', 'airline', 'text', 'tweet_created']]
    
    # Clean text
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Map sentiment labels to int
    # For a fair comparison with distilbert-sst2, we will focus on Positive vs Negative
    df_binary = df[df['airline_sentiment'].isin(['positive', 'negative'])].copy()
    
    X = df_binary['clean_text']
    y = df_binary['airline_sentiment'].map({'negative': 0, 'positive': 1})
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. TF-IDF + Logistic Regression
    print("Training TF-IDF + Logistic Regression...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train_tfidf, y_train)
    
    lr_preds = logreg.predict(X_test_tfidf)
    lr_acc = accuracy_score(y_test, lr_preds)
    lr_f1 = f1_score(y_test, lr_preds)
    
    print(f"LogReg - Accuracy: {lr_acc:.4f}, F1: {lr_f1:.4f}")
    
    # 2. Pretrained Transformer (DistilBERT SST-2)
    print("Running Pretrained Transformer (DistilBERT)...")
    sentiment_pipeline = pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english", 
        device=-1  # CPU for compatibility, change to 0 if GPU available
    )
    
    # Predict on test set
    transformer_preds = []
    # Using small batching / progress bar
    for text in tqdm(X_test, desc="Transformer Predict"):
        # Truncate to 512 tokens if needed, but tweets are short
        try:
            res = sentiment_pipeline(text)[0]
            pred = 1 if res['label'] == 'POSITIVE' else 0
            transformer_preds.append(pred)
        except Exception as e:
            transformer_preds.append(0) # fallback
            
    tf_acc = accuracy_score(y_test, transformer_preds)
    tf_f1 = f1_score(y_test, transformer_preds)
    
    print(f"DistilBERT - Accuracy: {tf_acc:.4f}, F1: {tf_f1:.4f}")
    
    # Save metrics
    metrics = {
        "LogReg": {"accuracy": lr_acc, "f1": lr_f1},
        "DistilBERT": {"accuracy": tf_acc, "f1": tf_f1}
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)
        
    # Now generate predictions on the FULL dataset (including neutral, but distilbert only does pos/neg)
    # Actually, for the dashboard, let's just use the LogReg for sentiment or the original true labels,
    # or the true labels vs predicted. Let's just predict on the whole dataset using LogReg for the dashboard if needed,
    # but since the dataset HAS true labels, we can just use the true labels for the business analytics part!
    # "join sentiment + topic back to time"
    # We will just save the cleaned text back to be used by the topic modeler.
    
    print("Saving processed data...")
    df.to_csv("processed_tweets.csv", index=False)
    print("Data pipeline complete.")

if __name__ == "__main__":
    main()
