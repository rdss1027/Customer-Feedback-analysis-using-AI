## Intelligent Customer Feedback Analysis System using AI (with LSTM + Word2Vec + T5 + Streamlit)
# 🧠 Intelligent Customer Feedback Analysis System using AI

An end-to-end **AI-based system** that analyzes customer feedback, detects sentiment (Positive/Negative), summarizes feedback using NLP, and provides interactive visual insights — built using **LSTM**, **Word2Vec**, and **T5 Transformer** models, deployed via **Streamlit**.

---

## 🚀 Project Overview

This project aims to automatically analyze customer reviews or feedback data to extract insights and trends.  
It involves **five major parts**, from data preprocessing to model deployment.

---

## 📂 Project Structure


---

## 🧩 **PART 1 – Data Preprocessing**

**Goal:** Clean and prepare raw customer feedback text for modeling.

## 📊 Dataset Description

The project uses the **IMDb Movie Review Dataset**, which contains text reviews of movies along with their associated sentiment labels.
For Dataset Use Any Review Datasets like imdb,  ecommerce customer reviews etc

### **Attributes Used**
| Attribute Name | Description |
|----------------|--------------|
| `review` | The full text of the customer’s movie review. |
| `sentiment` | Binary label — `0` for Negative and `1` for Positive sentiment. |
| `cleaned_feedback` | Preprocessed version of the review after text cleaning (used for model training). |

> ✅ **Note:** The dataset is ideal for text classification tasks such as sentiment analysis because it contains balanced classes and diverse real-world language expressions.



**Steps:**
- Load IMDB dataset (review + sentiment columns)
- Remove stopwords, punctuation, and special characters
- Lemmatize tokens for normalization
- Save cleaned dataset → `data/cleaned_imdb_reviews.csv`

---

## 🤖 **PART 2 – Sentiment Classification (LSTM + Word2Vec)**

**Goal:** Build a deep learning model to classify feedback as Positive or Negative.

**Model Workflow:**
1. **Tokenization:** Split text into words using `nltk.word_tokenize`.
2. **Word2Vec Embedding:** Train word vectors (`vector_size=100`).
3. **LSTM Model:**  
   - Embedding Layer (initialized with Word2Vec matrix)  
   - LSTM Layer (128 units)  
   - Dense Layers for sentiment output  
4. **Training:** Binary cross-entropy loss, Adam optimizer  
5. **Metrics:** Accuracy, Precision, Recall, F1-score

**Outputs:**
- `models/sentiment_lstm.h5` – trained model  
- `models/word2vec_feedback.model` – word embedding  
- `models/tokenizer.pkl` – tokenizer for inference  

📘 **Explanation 1: Why LSTM + Word2Vec?**
> LSTMs are powerful for sequential data like text, as they can capture context and order of words.  
> Word2Vec provides semantically meaningful word embeddings that help the model understand similar words (e.g., *good*, *great*, *excellent*).  
> Together, they create a strong and interpretable sentiment classifier.

---

## ✍️ **PART 3 – Feedback Summarization (T5 Transformer)**

**Goal:** Generate short or detailed summaries of customer feedback.

**Model:**  
- Uses Hugging Face’s **T5-small** pretrained model.  
- Input format: `"summarize: <feedback text>"`  
- Output: Concise summary text.  

**Example:**
> Input: “The delivery was fast, but the packaging was poor and the product was scratched.”  
> Output: “Quick delivery, but poor packaging quality.”

---

## 📊 **PART 4 – Visualization and Insights**

**Goal:** Explore feedback sentiment trends and visualize results.

**Includes:**
- Sentiment distribution (Pie chart)
- Monthly sentiment trend (Line chart)
- WordCloud for Positive/Negative feedback
- Interactive analytics with Plotly

**Output Files:**
- Visualizations integrated into the Streamlit app.

---

## 🌐 **PART 5 – Deployment with Streamlit**

**Goal:** Create an interactive web dashboard for prediction, summarization, and insights.

### Features:
- Upload CSV file or enter feedback text manually.
- Predict sentiment instantly using trained LSTM model.
- Generate feedback summaries using T5.
- Visualize trends and word clouds.

**Run the App:**
```bash
cd app
streamlit run streamlit_app.py


| Component          | Technology                    |
| ------------------ | ----------------------------- |
| Language           | Python 3.10+                  |
| Deep Learning      | TensorFlow / Keras            |
| Word Embeddings    | Gensim Word2Vec               |
| Transformer Model  | Hugging Face T5               |
| Data Visualization | Plotly, WordCloud, Matplotlib |
| Frontend           | Streamlit                     |
| Evaluation         | scikit-learn metrics          |

---

## ⚙️ Environment Setup (Step-by-Step)

This section explains how to create and configure a clean environment for the project.

### 🧰 1. Prerequisites

Make sure you have installed:
- **Python 3.10 or above**
- **pip** or **conda**
- (Optional but recommended) **virtualenv** or **Anaconda**

---

### 🧱 2. Create a Virtual Environment

#### 🐍 Option A: Using Conda
```bash
conda create -n feedback python=3.10 -y
conda activate feedback

##Other Option

python -m venv feedback_env
feedback_env\Scripts\activate     # on Windows
# or
source feedback_env/bin/activate  # on macOS/Linux

```bash
pip install -r requirements.txt

🧠 5. Prepare Data and Models

Dataset:
Place your dataset (e.g., IMDB Dataset.csv) in the data/ folder.
Example: data/raw_feedback.csv

Run Part 1 – Data Preprocessing:
Cleans text and saves cleaned_imdb_reviews.csv.

Run Part 2 – Sentiment Classification:
Trains LSTM model and saves:
models/sentiment_lstm.h5
models/word2vec_feedback.model

Run Part 3 – Summarization (T5):
Generates summaries using pretrained T5-small model.

Run Part 4 File

Run the Streamlit App (Part 5 – Deployment)

```bash
cd app
streamlit run streamlit_app.py

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
Open the URL in your browser to launch the interactive dashboard.

🧩 Future Enhancements

Add Neutral class for 3-way sentiment detection.

Fine-tune summarization using custom datasets.

Integrate topic modeling (LDA or BERTopic) for theme analysis.

Deploy app on cloud (Streamlit Cloud / AWS / Hugging Face Spaces).

👨‍💻 Author

Developed by: [RDSSASI]
Project: Intelligent Customer Feedback Analysis System using AI

🎓 For Academic / Research Submission
