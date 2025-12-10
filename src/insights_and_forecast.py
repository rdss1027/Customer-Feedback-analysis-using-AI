"""
Part 4 - Predictive Insight Generation
- Identify recurring issues (TF-IDF + NMF topic modeling)
- Forecast customer satisfaction score trends (Prophet preferred; ARIMA fallback)
- Produce AI_insights_report.pdf with visualizations
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import pairwise_distances_argmin_min

from datetime import datetime, timedelta

# Optional: try to import prophet, otherwise use statsmodels ARIMA
try:
    from prophet import Prophet
    HAVE_PROPHET = True
except Exception:
    HAVE_PROPHET = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    HAVE_ARIMA = True
except Exception:
    HAVE_ARIMA = False

# -------------------------
# Config
# -------------------------
CLEANED_PATH = "data/cleaned_imdb_reviews.csv"   # input
TEXT_COL = "cleaned_feedback"                   # or 'feedback' or 'processed_feedback'
LABEL_COL = "sentiment"                         # binary 0/1
DATE_COL = "date"                               # optional; script will simulate if missing
OUTPUT_PDF = "AI_insights_report.pdf"

N_TOPICS = 8   # number of topics to extract
TOP_N_WORDS = 8
FORECAST_DAYS = 30
AGG_FREQ = 'D'  # 'D' daily, 'W' weekly, 'M' monthly

# -------------------------
# Helper functions
# -------------------------
def load_and_prepare(path=CLEANED_PATH):
    df = pd.read_csv(path)
    # pick text column
    if TEXT_COL not in df.columns:
        # fallback options
        for alt in ['processed_feedback', 'cleaned_feedback', 'feedback', 'review']:
            if alt in df.columns:
                df[TEXT_COL] = df[alt]
                break
    # ensure label column exists
    if LABEL_COL not in df.columns:
        # if possible derive from sentiment strings
        if 'sentiment' in df.columns:
            df[LABEL_COL] = df['sentiment']
        else:
            raise ValueError("No sentiment/label column found.")
    # normalize label: if strings -> map
    if df[LABEL_COL].dtype == object:
        df[LABEL_COL] = df[LABEL_COL].astype(str).str.lower().map({'positive':1,'negative':0,'pos':1,'neg':0})
    # drop missing text or labels
    df = df.dropna(subset=[TEXT_COL, LABEL_COL]).reset_index(drop=True)
    # ensure label numeric
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    # date handling: if missing, create synthetic date spread over last 365 days
    if DATE_COL not in df.columns or df[DATE_COL].isnull().all():
        end = datetime.today()
        start = end - timedelta(days=364)
        rng = pd.to_datetime(np.random.randint(int(start.timestamp()), int(end.timestamp()), size=len(df)), unit='s')
        df[DATE_COL] = rng
    else:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
        # fill missing dates similarly
        missing = df[DATE_COL].isna()
        if missing.any():
            end = datetime.today()
            start = end - timedelta(days=364)
            rng = pd.to_datetime(np.random.randint(int(start.timestamp()), int(end.timestamp()), size=missing.sum()), unit='s')
            df.loc[missing, DATE_COL] = rng

    # If no continuous rating, create a proxy rating for forecasting: positive->4, negative->2
    if 'rating' not in df.columns:
        df['rating'] = df[LABEL_COL].apply(lambda x: 4.0 if x==1 else 2.0)

    return df

# -------------------------
# Topic modeling (TF-IDF + NMF)
# -------------------------
def topic_modeling(df, text_col=TEXT_COL, n_topics=N_TOPICS, top_n=TOP_N_WORDS):
    docs = df[text_col].astype(str).tolist()
    tfidf = TfidfVectorizer(max_df=0.95, min_df=10, stop_words='english', max_features=20000)
    X = tfidf.fit_transform(docs)
    nmf = NMF(n_components=n_topics, random_state=42, init='nndsvda', max_iter=400)
    W = nmf.fit_transform(X)  # document-topic
    H = nmf.components_       # topic-term

    feature_names = tfidf.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(H):
        top_features_idx = topic.argsort()[:-top_n - 1:-1]
        top_features = [feature_names[i] for i in top_features_idx]
        topics.append((topic_idx, top_features))

    # assign dominant topic to each doc
    dominant_topic = W.argmax(axis=1)
    df_topics = df.copy()
    df_topics['dominant_topic'] = dominant_topic
    return df_topics, topics, nmf, tfidf

# -------------------------
# Get sample texts per topic
# -------------------------
def sample_texts_per_topic(df_topics, text_col=TEXT_COL, n_samples=5):
    samples = {}
    for t in sorted(df_topics['dominant_topic'].unique()):
        subset = df_topics[df_topics['dominant_topic']==t]
        # choose top-n by closeness to topic center
        # if available, use the matrix distance; otherwise sample
        sample_texts = subset[text_col].sample(n=min(n_samples, len(subset)), random_state=42).tolist()
        samples[t] = sample_texts
    return samples

# -------------------------
# Forecasting using Prophet or ARIMA fallback (stable version)
# -------------------------
def forecast_timeseries(df, date_col=DATE_COL, value_col='rating', freq=AGG_FREQ, periods=FORECAST_DAYS):
    ts = df.set_index(date_col).resample(freq)[value_col].mean().reset_index().dropna()
    ts = ts.rename(columns={date_col:'ds', value_col:'y'})
    ts = ts.sort_values('ds')

    if len(ts) < 10:
        raise ValueError("Not enough data points for forecasting.")

    try:
        # Try Prophet first
        from prophet import Prophet
        m = Prophet()
        m.fit(ts)
        future = m.make_future_dataframe(periods=periods, freq='D')
        forecast = m.predict(future)
        engine = 'prophet'
        print("✅ Prophet forecast generated successfully.")
        return ts, forecast, engine

    except Exception as e:
        print(f"⚠️ Prophet failed due to: {e}")
        print("👉 Falling back to ARIMA forecasting...")

        # ARIMA fallback
        from statsmodels.tsa.arima.model import ARIMA
        ts_indexed = ts.set_index('ds')
        ts_indexed = ts_indexed.asfreq('D').fillna(method='ffill')

        model = ARIMA(ts_indexed['y'], order=(2, 1, 2))
        model_fit = model.fit()
        forecast_res = model_fit.get_forecast(steps=periods)
        forecast = forecast_res.summary_frame().reset_index().rename(columns={'index':'ds', 'mean':'yhat'})
        forecast['yhat_lower'] = forecast['mean_ci_lower']
        forecast['yhat_upper'] = forecast['mean_ci_upper']
        engine = 'arima'

        print("✅ ARIMA forecast generated successfully.")
        return ts.reset_index(drop=True), forecast, engine

# -------------------------
# Plotting helpers
# -------------------------
def plot_top_topics(topics, pdf, top_n=TOP_N_WORDS):
    # topics: list of (topic_idx, [words])
    for topic_idx, words in topics:
        fig, ax = plt.subplots(figsize=(8,3))
        ax.barh(range(len(words[::-1])), [1]*len(words), align='center')
        ax.set_yticks(range(len(words[::-1])))
        ax.set_yticklabels(words[::-1])
        ax.set_xlabel("Relative weight (NMF top terms)")
        ax.set_title(f"Topic {topic_idx} top {len(words)} keywords")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

def plot_topic_distribution(df_topics, pdf):
    counts = df_topics['dominant_topic'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_xlabel("Topic")
    ax.set_ylabel("Number of feedbacks")
    ax.set_title("Feedback count by dominant topic")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

def plot_forecast(ts, forecast, pdf, engine='prophet'):
    # ts: historical, forecast: prophet forecast or arima summary
    if engine == 'prophet':
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(111)
        ax.plot(ts['ds'], ts['y'], label='historical')
        ax.plot(forecast['ds'], forecast['yhat'], label='forecast')
        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2, label='uncertainty')
        ax.set_xlabel('Date'); ax.set_ylabel('Average rating')
        ax.set_title('Satisfaction trend and forecast (Prophet)')
        ax.legend()
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    else:
        # simple arima plot
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(111)
        ax.plot(ts['ds'], ts['y'], label='historical')
        ax.plot(forecast['ds'], forecast['yhat'], label='forecast')
        ax.set_xlabel('Date'); ax.set_ylabel('Average rating')
        ax.set_title('Satisfaction trend and forecast (ARIMA)')
        ax.legend()
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

# -------------------------
# Main analysis function
# -------------------------
def run_insights_and_forecast(input_csv=CLEANED_PATH, output_pdf=OUTPUT_PDF):
    df = load_and_prepare(input_csv)

    # Topic modeling
    df_topics, topics, nmf, tfidf = topic_modeling(df)
    samples = sample_texts_per_topic(df_topics, n_samples=5)

    # Forecasting
    ts, forecast, engine = forecast_timeseries(df)

    # Build PDF report
    with PdfPages(output_pdf) as pdf:
        # cover page
        fig = plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.text(0.5, 0.7, "AI Insights Report", fontsize=24, ha='center')
        plt.text(0.5, 0.6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ha='center')
        plt.text(0.1, 0.45, f"Dataset: {os.path.basename(input_csv)}", fontsize=10)
        plt.text(0.1, 0.4, f"Rows analyzed: {len(df)}", fontsize=10)
        plt.text(0.1, 0.35, f"Topics extracted: {len(topics)}", fontsize=10)
        pdf.savefig(fig); plt.close(fig)

        # topic keyword pages
        plot_topic_distribution(df_topics, pdf)
        plot_top_topics(topics, pdf)

        # add sample texts per topic
        for t, texts in samples.items():
            fig = plt.figure(figsize=(8.5, 4))
            plt.axis('off')
            plt.title(f"Sample feedback for Topic {t}")
            sample_text = "\n\n".join([f"- {txt[:400]}..." for txt in texts])
            plt.text(0.01, 0.99, sample_text, va='top', wrap=True)
            pdf.savefig(fig); plt.close(fig)

        # forecasting plots
        plot_forecast(ts, forecast, pdf, engine=engine)

        # summary page with actionable insights
        fig = plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.title("Actionable Insights & Recommendations")
        insights = [
            f"Top recurring topics (keywords):",
        ]
        for tid, kw in topics:
            insights.append(f"Topic {tid}: {', '.join(kw[:6])}")
        insights.append("")
        insights.append("Recommendations:")
        insights.append("- Review top topic(s) with increasing frequency and dig into sample feedback.")
        insights.append("- Prioritize fixes for issues appearing in the most frequent topics.")
        insights.append("- Monitor forecasted decline/increase in satisfaction and plan interventions.")
        plt.text(0.01, 0.99, "\n".join(insights), va='top', wrap=True)
        pdf.savefig(fig); plt.close(fig)

    print(f"\n✅ Report saved to {output_pdf}")
    return df_topics, topics, ts, forecast

if __name__ == "__main__":
    df_topics, topics, ts, forecast = run_insights_and_forecast()
