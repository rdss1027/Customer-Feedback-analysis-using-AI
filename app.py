import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
import gensim
from gensim.utils import simple_preprocess
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from transformers import T5ForConditionalGeneration, T5Tokenizer
from io import StringIO
import os

# -------------------------------------------------------------
# Load Sentiment Model + Word2Vec
# -------------------------------------------------------------
@st.cache_resource
def load_sentiment_model():
    model = load_model("Models/sentiment_lstm.h5")  # ✅ LSTM model (.h5)
    w2v = gensim.models.Word2Vec.load("models/word2vec_feedback.model")
    return model, w2v

# Load Summarization model (T5)
@st.cache_resource
def load_t5_model():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    return tokenizer, model

# -------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------
def text_to_sequence(texts, w2v, max_len=100):
    """Convert text to sequence of word indices based on Word2Vec vocabulary."""
    vocab = set(w2v.wv.index_to_key)
    seqs = []
    for text in texts:
        tokens = simple_preprocess(str(text))
        vecs = [w2v.wv.key_to_index[w] + 1 for w in tokens if w in vocab]
        seqs.append(vecs)
    return pad_sequences(seqs, maxlen=max_len, padding="post")


def predict_sentiment(texts, model, w2v):
    X = text_to_sequence(texts, w2v)
    preds = model.predict(X)
    return ["Positive" if p > 0.5 else "Negative" for p in preds.flatten()]

def generate_summary(text, tokenizer, model, summary_type="short"):
    if summary_type == "short":
        max_len = 40
    else:
        max_len = 100
    input_text = "summarize: " + text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True)
    summary_ids = model.generate(input_ids, max_length=max_len, min_length=10, length_penalty=2.0)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# -------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------
st.set_page_config(page_title="AI Customer Feedback Analysis", layout="wide")

st.title("🧠 Intelligent Customer Feedback Analysis System")

menu = ["Upload & Analyze", "Summarization", "Insights"]
choice = st.sidebar.selectbox("Navigate", menu)

# Load models once
model, w2v = load_sentiment_model()
tokenizer, t5_model = load_t5_model()

# -------------------------------------------------------------
# Upload & Analyze Section
# -------------------------------------------------------------
if choice == "Upload & Analyze":
    st.subheader("📤 Upload Feedback Dataset or Enter Text")

    # Option 1: File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    # Option 2: Direct text input
    st.write("OR")
    user_feedback = st.text_area("✍️ Enter feedback text manually:", "")

    if uploaded_file:
        # --- Existing CSV logic ---
        df = pd.read_csv(uploaded_file)
        if 'review' in df.columns:
            df.rename(columns={'review': 'feedback'}, inplace=True)
        st.write("✅ Dataset loaded successfully!")
        st.dataframe(df.head())

        # Predict sentiments
        st.subheader("📊 Sentiment Prediction")
        df["Predicted_Sentiment"] = predict_sentiment(df["feedback"], model, w2v)

        fig = px.pie(df, names="Predicted_Sentiment", title="Sentiment Distribution")
        st.plotly_chart(fig, use_container_width=True)

        # Word Cloud
        st.subheader("☁️ Word Cloud of Feedback")
        text = " ".join(df["feedback"].astype(str).tolist())
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        st.image(wc.to_array())

        st.download_button("📥 Download Results (CSV)", df.to_csv(index=False).encode('utf-8'), "analyzed_feedback.csv")

    elif user_feedback.strip():
        # --- New single text prediction logic ---
        st.subheader("🧾 Sentiment Prediction for Entered Text")

        # Add a button to submit the manual feedback
        if st.button("Submit Feedback"):
            pred = predict_sentiment([user_feedback], model, w2v)[0]

            # Display the predicted sentiment
            if pred == "Positive":
                st.success(f"🌟 **Predicted Sentiment:** {pred}")
            else:
                st.error(f"💢 **Predicted Sentiment:** {pred}")



# -------------------------------------------------------------
# Summarization Section
# -------------------------------------------------------------
elif choice == "Summarization":
    st.subheader("📝 Feedback Summarization")

    user_text = st.text_area("Enter a long feedback or paragraph:")
    summary_type = st.radio("Select Summary Type:", ["short", "detailed"])

    if st.button("Generate Summary"):
        if user_text.strip():
            summary = generate_summary(user_text, tokenizer, t5_model, summary_type)
            st.success("**Generated Summary:**")
            st.write(summary)
        else:
            st.warning("Please enter feedback text first!")

# -------------------------------------------------------------
# Insights Section
# -------------------------------------------------------------
elif choice == "Insights":
    st.subheader("📈 Insights & Trends")
    st.write("Upload the cleaned dataset (used in Part 4):")

    file = st.file_uploader("Upload cleaned feedback CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        if 'sentiment' not in df.columns:
            st.error("No 'sentiment' column found. Please upload the cleaned dataset with 'sentiment' column.")
        else:
            # Handle missing date column gracefully
            df['date'] = pd.to_datetime(df.get('date', pd.date_range('2024-01-01', periods=len(df))))
            df['sentiment'] = df['sentiment'].astype(int)

            # Sentiment trend
            trend = df.groupby(df['date'].dt.to_period('M'))['sentiment'].mean().reset_index()
            trend['date'] = trend['date'].astype(str)
            fig = px.line(trend, x='date', y='sentiment', title='Average Monthly Sentiment Trend')
            st.plotly_chart(fig, use_container_width=True)

            # Negative feedback word cloud
            st.subheader("🔍 Most Common Words in Negative Feedback")
            neg_text = " ".join(df[df['sentiment'] == 0]['feedback'].astype(str))
            wc_neg = WordCloud(width=800, height=400, background_color="white").generate(neg_text)
            st.image(wc_neg.to_array())


