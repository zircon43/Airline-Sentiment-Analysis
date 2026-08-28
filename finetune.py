import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
import evaluate
import torch

def main():
    print("Loading data...")
    # Make sure to run this on a machine with a GPU (e.g. Kaggle Notebook)
    df = pd.read_csv('Tweets.csv')
    
    # We will stick to Positive (1) vs Negative (0)
    df_binary = df[df['airline_sentiment'].isin(['positive', 'negative'])].copy()
    df_binary['label'] = df_binary['airline_sentiment'].map({'negative': 0, 'positive': 1})
    
    # Keep only necessary columns
    df_binary = df_binary[['text', 'label']]
    
    # Train/Test Split (80/20)
    train_df, test_df = train_test_split(df_binary, test_size=0.2, random_state=42)
    
    # Convert pandas dataframes to Hugging Face Dataset format
    hf_dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df, preserve_index=False),
        "test": Dataset.from_pandas(test_df, preserve_index=False)
    })
    
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    
    print("Tokenizing datasets...")
    tokenized_datasets = hf_dataset.map(tokenize_function, batched=True)
    
    # Load raw pre-trained DistilBERT (no existing sentiment tuning)
    print("Loading raw model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Define Evaluation Metrics
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_metric.compute(predictions=predictions, references=labels)
        f1 = f1_metric.compute(predictions=predictions, references=labels)
        return {"accuracy": acc["accuracy"], "f1": f1["f1"]}

    # Training Arguments (Optimized for a free Kaggle T4 GPU)
    training_args = TrainingArguments(
        output_dir="./airline_sentiment_model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    print("Starting Fine-tuning (This requires a GPU!)...")
    trainer.train()
    
    print("Evaluating on Test Set...")
    eval_results = trainer.evaluate()
    print("Evaluation Results:", eval_results)
    
    # Save the finalized model
    trainer.save_model("./airline_sentiment_model_final")
    print("Model saved to ./airline_sentiment_model_final")

if __name__ == "__main__":
    main()
