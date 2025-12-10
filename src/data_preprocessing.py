# ==============================================================
# Intelligent Customer Feedback Analysis System using AI
# Part 1 – Data Handling and Preprocessing
# ==============================================================

import os
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
tqdm.pandas()

# --------------------------------------------------------------
# 0. NLTK Setup
# --------------------------------------------------------------
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

for pkg in ["stopwords", "punkt", "punkt_tab", "wordnet", "omw-1.4"]:
    nltk.download(pkg, download_dir=nltk_data_dir, quiet=True)

# --------------------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------------------
input_path = r"C:/Users/RDSSASIDHAR/Desktop/IntelligentCustomerFeedbackAnalysisSystemusing_AI/main/data/imdb_Dataset.csv" # <-- change path if needed
df = pd.read_csv(input_path)
print("Original shape:", df.shape)
print(df.head())

# --------------------------------------------------------------
# 2. Convert sentiment to binary
# --------------------------------------------------------------
df["sentiment"] = df["sentiment"].str.lower().map({"positive": 1, "negative": 0})
df.dropna(subset=["sentiment"], inplace=True)
df["sentiment"] = df["sentiment"].astype(int)

# Rename for consistency
df.rename(columns={"review": "feedback"}, inplace=True)

# --------------------------------------------------------------
# 3. Remove missing / duplicate records
# --------------------------------------------------------------
df.dropna(subset=["feedback"], inplace=True)
df.drop_duplicates(subset=["feedback"], inplace=True)
df.reset_index(drop=True, inplace=True)
print("After removing duplicates & missing:", df.shape)

# --------------------------------------------------------------
# 4. Clean text (remove HTML, URLs, special chars)
# --------------------------------------------------------------
def clean_text(text):
    text = BeautifulSoup(str(text), "html.parser").get_text()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^A-Za-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

df["cleaned_feedback"] = df["feedback"].progress_apply(clean_text)

# --------------------------------------------------------------
# 5. Tokenization, Stopword Removal, Lemmatization
# --------------------------------------------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

df["processed_feedback"] = df["cleaned_feedback"].progress_apply(preprocess)

# --------------------------------------------------------------
# 6. Save cleaned dataset
# --------------------------------------------------------------
os.makedirs("data", exist_ok=True)
output_path = "data/cleaned_imdb_reviews.csv"
df.to_csv(output_path, index=False)

print("\n✅ Cleaned dataset saved to:", output_path)
print(df.head())
