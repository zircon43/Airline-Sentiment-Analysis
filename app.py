import streamlit as st
import pandas as pd
import json
import plotly.express as px

st.set_page_config(page_title="Airline Sentiment Analytics", layout="wide")
st.title("🛫 Airline Sentiment & Topic Analytics Dashboard")

@st.cache_data
def load_data():
    with open('metrics.json', 'r') as f:
        metrics = json.load(f)
    
    df = pd.read_csv('processed_tweets.csv')
    df['tweet_created'] = pd.to_datetime(df['tweet_created'])
    
    neg_topics = pd.read_csv('negative_topics.csv')
    neg_topics['tweet_created'] = pd.to_datetime(neg_topics['tweet_created'])
    
    samples = pd.read_csv('topic_samples.csv')
    
    return metrics, df, neg_topics, samples

try:
    metrics, df, neg_topics, samples = load_data()
except Exception as e:
    st.error(f"Please run the data pipeline and topic modeling scripts first. Error: {e}")
    st.stop()
    
# 1. Model Comparison
st.header("1. Sentiment Model Comparison")
col1, col2 = st.columns(2)
with col1:
    st.metric("LogReg Accuracy", f"{metrics['LogReg']['accuracy']:.2%}")
    st.metric("LogReg F1 Score", f"{metrics['LogReg']['f1']:.2%}")
with col2:
    st.metric("DistilBERT Accuracy", f"{metrics['DistilBERT']['accuracy']:.2%}")
    st.metric("DistilBERT F1 Score", f"{metrics['DistilBERT']['f1']:.2%}")

st.markdown("""
**Tradeoff Analysis:** The TF-IDF + Logistic Regression baseline is extremely fast and interpretable, 
while the Pretrained Transformer (DistilBERT) provides higher accuracy at the cost of computational speed and opacity.
""")

st.divider()

# 2. Topic Modeling Insights
st.header("2. What are people complaining about?")
topic_counts = neg_topics[neg_topics['topic'] != -1]['topic_label'].value_counts().reset_index()
topic_counts.columns = ['Topic', 'Complaint Volume']
fig_topics = px.bar(topic_counts.head(10), x='Topic', y='Complaint Volume', title="Top Complaint Topics")
st.plotly_chart(fig_topics, use_container_width=True)

with st.expander("View Example Reviews by Topic"):
    for topic in samples['Topic'].unique():
        st.subheader(topic)
        for _, row in samples[samples['Topic'] == topic].iterrows():
            st.markdown(f"- **{row['Airline']}**: {row['Tweet']}")
            
st.divider()

# 3. Business Analytics (Time Series)
st.header("3. Business Insights over Time")

# Daily volume of negative tweets by topic
# Some tweets might have timezone, let's normalize to date
neg_topics['date'] = pd.to_datetime(neg_topics['tweet_created'], utc=True).dt.date
daily_topics = neg_topics[neg_topics['topic'] != -1].groupby(['date', 'topic_label']).size().reset_index(name='count')

# Let's filter to top 3 topics to avoid clutter
if len(topic_counts) > 0:
    top_3_topics = topic_counts['Topic'].head(3).tolist()
    daily_topics_top = daily_topics[daily_topics['topic_label'].isin(top_3_topics)]

    fig_time = px.line(daily_topics_top, x='date', y='count', color='topic_label', 
                       title="Daily Complaint Volume by Top Topics", markers=True)
    st.plotly_chart(fig_time, use_container_width=True)

    # Airline breakdown for top complaint
    col_a, col_b = st.columns(2)
    with col_a:
        top_complaint = top_3_topics[0]
        top_complaint_df = neg_topics[neg_topics['topic_label'] == top_complaint]
        airline_counts = top_complaint_df['airline'].value_counts().reset_index()
        airline_counts.columns = ['Airline', 'Count']
        fig_airline = px.pie(airline_counts, names='Airline', values='Count', title=f"Most affected by '{top_complaint}'")
        st.plotly_chart(fig_airline, use_container_width=True)

    with col_b:
        # Find the day with the absolute highest spike for the top topic
        max_spike_row = daily_topics[daily_topics['topic_label'] == top_complaint].sort_values(by='count', ascending=False).iloc[0]
        max_date = max_spike_row['date']
        max_count = max_spike_row['count']
        
        st.info("💡 **Key Business Insight**")
        st.success(f"**{top_complaint}** is the leading cause of negative sentiment.")
        st.warning(f"We observed a significant spike in **{top_complaint}** complaints on **{max_date}**, reaching **{max_count}** negative mentions in a single day.")
        
        # Which airline was most responsible for this spike?
        spike_df = neg_topics[(neg_topics['topic_label'] == top_complaint) & (neg_topics['date'] == max_date)]
        worst_airline = spike_df['airline'].value_counts().index[0]
        st.error(f"**{worst_airline}** was the primary driver of this spike. Operations teams should investigate logs for {worst_airline} on {max_date} to understand the root cause.")
