# ==============================================================
# Part 3 — Text Summarization using T5
# File: 03_summarization_t5.py (or notebook)
# ==============================================================

import os
import pandas as pd
import nltk
from tqdm.auto import tqdm

# Transformers
from transformers import T5ForConditionalGeneration, T5TokenizerFast, pipeline

# --------------------------------------------------------------
# 0. Setup
# --------------------------------------------------------------
tqdm.pandas()
nltk.download('punkt')

# Use t5-small for speed; swap to 't5-base' if you want higher quality
MODEL_NAME = "t5-small"
DEVICE = 0  # -1 for CPU, 0 for first GPU (change as needed)

# Load model + tokenizer
tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# Optionally use the pipeline wrapper for convenience (we'll implement chunking ourselves)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=DEVICE)

# --------------------------------------------------------------
# 1. Utility: chunk long documents into manageable pieces
# --------------------------------------------------------------
def chunk_text_by_token_length(text, max_tokens=450):
    """
    Splits text into chunks whose tokenized length is <= max_tokens.
    Uses sentence tokenization and accumulates sentences until near the max_tokens limit.
    """
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_len = 0
    for sent in sentences:
        tokenized = tokenizer.encode(sent, truncation=False, add_special_tokens=False)
        tok_len = len(tokenized)
        # If single sentence longer than max_tokens, truncate it
        if tok_len >= max_tokens:
            # flush current chunk
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_len = 0
            # split the long sentence naively by words (fallback)
            words = sent.split()
            part = []
            part_len = 0
            for w in words:
                wlen = len(tokenizer.encode(w, add_special_tokens=False))
                if part_len + wlen > max_tokens:
                    chunks.append(" ".join(part))
                    part = [w]
                    part_len = wlen
                else:
                    part.append(w)
                    part_len += wlen
            if part:
                chunks.append(" ".join(part))
            continue

        # If adding sentence would exceed max_tokens, flush current chunk
        if current_len + tok_len > max_tokens:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sent]
            current_len = tok_len
        else:
            current_chunk.append(sent)
            current_len += tok_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# --------------------------------------------------------------
# 2. Summarization wrapper (short / detailed)
# --------------------------------------------------------------
def summarize_text_t5(text, summary_type="short", max_input_tokens=450, short_min=8, short_max=40, long_min=60, long_max=180):
    """
    Summarize `text` using T5 with chunking. Returns a single summary by concatenating chunk summaries.
      - summary_type: "short" or "detailed"
    """
    if not isinstance(text, str) or text.strip() == "":
        return ""

    # Choose params by summary type
    if summary_type == "short":
        min_len, max_len = short_min, short_max
    else:
        min_len, max_len = long_min, long_max

    # chunk text
    chunks = chunk_text_by_token_length(text, max_tokens=max_input_tokens)
    chunk_summaries = []

    for chunk in chunks:
        # T5 expects a "summarize: " prefix by some conventions (not strictly required)
        input_text = "summarize: " + chunk.strip()
        # Use the pipeline (handles tokenization & generation). You can tune num_beams, length_penalty, etc.
        try:
            outs = summarizer(
                input_text,
                max_length=max_len,
                min_length=min_len,
                do_sample=False,
                truncation=True
            )
            if isinstance(outs, list) and len(outs) > 0 and 'summary_text' in outs[0]:
                chunk_summaries.append(outs[0]['summary_text'].strip())
            elif isinstance(outs, list) and len(outs) > 0:
                # sometimes pipeline returns plain text
                chunk_summaries.append(outs[0].get('summary_text', str(outs[0])))
            else:
                # Fallback: empty
                chunk_summaries.append("")
        except Exception as e:
            # Best-effort: on failure append truncated chunk
            chunk_summaries.append(" ".join(chunk.split()[:max_len]))

    # combine chunk summaries into one text
    combined = " ".join([s for s in chunk_summaries if s])
    # Optionally, do a final summarization pass of the combined text to tighten it
    if len(combined.split()) > max_len:
        final_in = "summarize: " + combined
        final_out = summarizer(final_in, max_length=max_len, min_length=min_len, do_sample=False, truncation=True)
        if isinstance(final_out, list) and len(final_out) > 0:
            return final_out[0].get('summary_text', final_out[0])
        return combined
    else:
        return combined

# --------------------------------------------------------------
# 3. Batch processing helper for a DataFrame
# --------------------------------------------------------------
def summarize_dataframe(df, text_col="feedback", out_short_col="summary_short", out_long_col="summary_detailed", n_samples=None):
    """
    Add two columns to dataframe: short and detailed summaries for text_col.
    If n_samples provided, process only first n_samples rows.
    """
    df = df.copy()
    if n_samples is not None:
        df = df.head(n_samples)
    tqdm.pandas(desc="Summarizing (short)")
    df[out_short_col] = df[text_col].progress_apply(lambda t: summarize_text_t5(t, summary_type="short"))
    tqdm.pandas(desc="Summarizing (detailed)")
    df[out_long_col] = df[text_col].progress_apply(lambda t: summarize_text_t5(t, summary_type="detailed"))
    return df

# --------------------------------------------------------------
# 4. Example usage on cleaned dataset
# --------------------------------------------------------------
if __name__ == "__main__":
    # Load cleaned dataset (from Part 1)
    cleaned_path = "data/cleaned_imdb_reviews.csv"
    if not os.path.exists(cleaned_path):
        raise FileNotFoundError(f"Cleaned file not found: {cleaned_path}. Run Part 1 first.")

    df = pd.read_csv(cleaned_path)
    # prefer the cleaned column; fallback to original review
    text_col = "cleaned_feedback" if "cleaned_feedback" in df.columns else ("feedback" if "feedback" in df.columns else df.columns[0])

    # Example: summarize first 20 reviews (demo). Set n_samples=None to process all (slow).
    demo_df = summarize_dataframe(df, text_col=text_col, out_short_col="summary_short", out_long_col="summary_detailed", n_samples=20)

    # show examples
    for i, row in demo_df.head(8).iterrows():
        print("="*80)
        print("Original (truncated):", row[text_col][:400], "...")
        print("\nShort summary:\n", row["summary_short"])
        print("\nDetailed summary:\n", row["summary_detailed"])
        print("="*80, "\n")

    # Save demo results
    os.makedirs("results", exist_ok=True)
    demo_df.to_csv("results/summaries_demo.csv", index=False)
    print("\n✅ Demo summaries saved to results/summaries_demo.csv")
