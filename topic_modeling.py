import pandas as pd
from bertopic import BERTopic
import json

def main():
    print("Loading processed data...")
    df = pd.read_csv('processed_tweets.csv')
    
    # We only care about why people are complaining
    neg_df = df[df['airline_sentiment'] == 'negative'].copy()
    
    # Drop NAs
    neg_df = neg_df.dropna(subset=['clean_text'])
    
    # BERTopic runs faster and better with more data, but if there's too much, we could sample.
    # The dataset has ~9k negative tweets, which is perfectly fine for BERTopic.
    docs = neg_df['clean_text'].tolist()
    
    print("Running BERTopic on", len(docs), "documents...")
    # Initialize BERTopic
    topic_model = BERTopic(language="english", calculate_probabilities=False, verbose=True)
    topics, probs = topic_model.fit_transform(docs)
    
    neg_df['topic'] = topics
    
    # Extract topic representations
    topic_info = topic_model.get_topic_info()
    print("Topics generated:")
    print(topic_info.head(10))
    
    # Map topics to human-readable labels programmatically based on keywords
    topic_mapping = {}
    for i, row in topic_info.iterrows():
        topic_id = row['Topic']
        if topic_id == -1:
            topic_mapping[topic_id] = "Outliers / Miscellaneous"
        else:
            top_words = [word for word, _ in topic_model.get_topic(topic_id)]
            words_str = " ".join(top_words[:10])
            
            if any(w in words_str for w in ['delay', 'late', 'wait', 'hour', 'time']):
                label = "Delays & Waiting"
            elif any(w in words_str for w in ['bag', 'luggage', 'lost', 'claim']):
                label = "Luggage Issues"
            elif any(w in words_str for w in ['cancel', 'rebook', 'flight', 'connection']):
                label = "Cancellations / Flights"
            elif any(w in words_str for w in ['customer', 'service', 'hold', 'phone', 'call', 'agent', 'help']):
                label = "Customer Service"
            elif any(w in words_str for w in ['seat', 'class', 'upgrade', 'sit']):
                label = "Seating / Comfort"
            elif any(w in words_str for w in ['weather', 'storm']):
                label = "Weather"
            else:
                label = f"Topic: {words_str[:30]}..."
                
            topic_mapping[topic_id] = label
            
    neg_df['topic_label'] = neg_df['topic'].map(topic_mapping)
    
    print("Saving topic modeled data...")
    neg_df.to_csv('negative_topics.csv', index=False)
    
    # Also save a sample for the dashboard
    samples = []
    for t_id in topic_mapping.keys():
        if t_id == -1:
            continue
        t_df = neg_df[neg_df['topic'] == t_id].head(3)
        for _, row in t_df.iterrows():
            samples.append({
                "Topic": topic_mapping[t_id],
                "Tweet": row['text'],
                "Airline": row['airline']
            })
    pd.DataFrame(samples).to_csv('topic_samples.csv', index=False)
    
    with open('topic_mapping.json', 'w') as f:
        json.dump(topic_mapping, f)
        
    print("Topic modeling complete.")

if __name__ == "__main__":
    main()
