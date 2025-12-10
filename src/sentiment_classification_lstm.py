# ==============================================================
# Intelligent Customer Feedback Analysis System using AI
# Part 2 – Sentiment Classification (LSTM + Word2Vec)
# ==============================================================

import os
import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# --------------------------------------------------------------
# 1. Load Cleaned Dataset
# --------------------------------------------------------------
df = pd.read_csv("data/cleaned_imdb_reviews.csv")
print("Dataset shape:", df.shape)

# Use processed feedback as input and sentiment as label
text_column = "processed_feedback"
label_column = "sentiment"

# Drop missing labels if any
df.dropna(subset=[text_column, label_column], inplace=True)
df[label_column] = df[label_column].astype(int)

# --------------------------------------------------------------
# 2. Tokenize for Word2Vec
# --------------------------------------------------------------
tokenized_sentences = [nltk.word_tokenize(text) for text in df[text_column].astype(str)]
print("✅ Tokenization complete. Sample:", tokenized_sentences[0][:10])

# --------------------------------------------------------------
# 3. Train Word2Vec
# --------------------------------------------------------------
w2v_model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=2, workers=4)
os.makedirs("models", exist_ok=True)
w2v_model.save("models/word2vec_feedback.model")
print("✅ Word2Vec trained. Vocabulary size:", len(w2v_model.wv))

# --------------------------------------------------------------
# 4. Tokenize & Pad Sequences for LSTM
# --------------------------------------------------------------
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df[text_column])
sequences = tokenizer.texts_to_sequences(df[text_column])

max_len = 100
X = pad_sequences(sequences, maxlen=max_len)
y = np.array(df[label_column])
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100

# --------------------------------------------------------------
# 5. Create Embedding Matrix from Word2Vec
# --------------------------------------------------------------
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]

# --------------------------------------------------------------
# 6. Build LSTM Model
# --------------------------------------------------------------
model = Sequential([
    Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_len, trainable=False),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# --------------------------------------------------------------
# 7. Train-Test Split
# --------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------------------------------------------
# 8. Train Model
# --------------------------------------------------------------
history = model.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.1, verbose=1)

# --------------------------------------------------------------
# 9. Evaluate Model
# --------------------------------------------------------------
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("\nEvaluation Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# --------------------------------------------------------------
# 10. Save Model and Tokenizer
# --------------------------------------------------------------
model.save("models/sentiment_lstm.h5")
joblib.dump(tokenizer, "models/tokenizer.pkl")

print("\n✅ Model training complete and saved to 'models/' folder.")
