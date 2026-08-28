# Twitter US Airline Sentiment Analytics & Topic Modeling

This project explores unstructured text data from the well-known [Twitter US Airline Sentiment dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment). By combining traditional NLP techniques with modern transformer-based models and dynamic topic modeling, this repository demonstrates an end-to-end data pipeline focused on extracting actionable business insights from raw social media complaints.

## 🎯 Business Objective
Airlines frequently struggle to categorize and prioritize customer feedback in real-time. The goal of this project is to:
1. **Classify Sentiment**: Automatically determine if a tweet is positive or negative.
2. **Discover Root Causes**: Automatically cluster and label negative tweets into specific, actionable operational failure categories (e.g., "Luggage Issues", "Flight Delays").
3. **Trend Analysis**: Correlate these specific complaint topics over time to identify operational breakdowns (e.g., identifying when a specific airline experienced a systemic baggage handling failure).

## 🧠 Methodology

### 1. Sentiment Classification Comparison
We implemented and compared two models for sentiment analysis to understand the tradeoff between speed/interpretability and raw accuracy:
- **Baseline Model**: A traditional `TF-IDF` vectorizer paired with a `Logistic Regression` classifier. Fast to train, highly interpretable, and lightweight.
- **Transformer Model**: A zero-shot evaluation using the HuggingFace `distilbert-base-uncased-finetuned-sst-2-english` pretrained pipeline. Slower, but captures semantic nuances significantly better.

**Results:**
While the Logistic Regression achieved a slightly higher raw accuracy by favoring the majority class, the **DistilBERT model achieved a much higher F1 Score (75.8% vs 70.4%)**, making it the superior choice for handling the inherent class imbalances found in real-world complaint data.

### 2. Topic Modeling with BERTopic
To answer *why* customers were complaining, we applied **BERTopic** to the subset of negative reviews. This allowed us to dynamically cluster thousands of tweets into specific issues without requiring thousands of manual labels.

We programmatically mapped the resulting dense clusters to human-readable labels based on keyword extraction:
- 🧳 **Luggage Issues** (lost bags, claim issues)
- ⏰ **Delays & Waiting** (late flights, hours on tarmac)
- ❌ **Cancellations / Flights** (rebooking, missed connections)
- 📞 **Customer Service** (hold times, rude agents)
- 💺 **Seating / Comfort** (legroom, upgrades)

## 📊 Interactive Dashboard
A Streamlit dashboard was built to bring these models to life. It visualizes the sentiment metrics, provides example tweets for each topic, and plots time-series graphs to track complaint volume over time.

### How to Run Locally

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   ```

2. **Download the Dataset**
   Download the `Tweets.csv` file from [Kaggle](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) and place it in the root directory.

3. **Install Dependencies**
   It's recommended to use a virtual environment or Conda:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: The pipeline uses the CPU version of PyTorch for compatibility).*

4. **Run the Data Pipeline (Optional)**
   The metrics and topic models are pre-computed in the repo, but you can re-run them:
   ```bash
   python data_pipeline.py
   python topic_modeling.py
   ```

5. **Launch the Dashboard**
   ```bash
   streamlit run app.py
   ```

## 🛠 Tech Stack
- **Data Manipulation**: `pandas`, `numpy`
- **Machine Learning**: `scikit-learn`, `BERTopic`, `PyTorch` (CPU)
- **Transformers**: `HuggingFace pipelines`
- **Visualization & App**: `Streamlit`, `Plotly`

---
*Created as a portfolio demonstration of end-to-end NLP analytics.*
