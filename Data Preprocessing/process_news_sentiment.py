import pandas as pd
import numpy as np
import os
import glob
from transformers import pipeline
import torch
import sys

# Suppress a common but harmless warning from transformers
from transformers import logging
logging.set_verbosity_error()

def initialize_finbert():
    """
    Initializes and returns the FinBERT sentiment analysis pipeline.
    """
    device_name = 'cpu'
    if torch.cuda.is_available():
        device_name = 'cuda'
    
    print(f"Initializing FinBERT pipeline... (Using device: {device_name})")
    
    try:
        sentiment_pipeline = pipeline(
            "sentiment-analysis", 
            model="ProsusAI/finbert", 
            device=-1 if device_name == 'cpu' else 0
        )
        print("FinBERT pipeline loaded successfully.")
        return sentiment_pipeline
    except Exception as e:
        print(f"[Fatal Error] Could not load FinBERT model: {e}")
        print("Make sure you are connected to the internet and have 'transformers' and 'torch' installed.")
        sys.exit(1) # Exit the script if the model can't be loaded

def process_news_file(news_file, sentiment_pipeline):
    """
    Reads a raw news file, analyzes every article, and returns a DataFrame 
    of daily average sentiment scores.
    """
    
    # --- 1. Read and Clean News Data ---
    try:
        news_df = pd.read_csv(news_file)
        news_df['date'] = pd.to_datetime(news_df['date'])
    except Exception as e:
        print(f"  [Error] Could not read or parse news file {news_file}: {e}")
        return None

    # Clean text and create a 'full_text' column for analysis
    news_df['description'] = news_df['description'].fillna('').astype(str)
    news_df['title'] = news_df['title'].astype(str)
    news_df['full_text'] = news_df['title'] + ' ' + news_df['description']
    
    # Filter out empty text
    valid_texts_df = news_df[news_df['full_text'].str.strip() != ''].copy()
    
    if valid_texts_df.empty:
        print("  > No valid news text found. Returning empty DataFrame.")
        return pd.DataFrame(columns=['date', 'avg_polarity', 'avg_confidence']).set_index('date')

    # --- 2. Run FinBERT Analysis ---
    print(f"  > Analyzing {len(valid_texts_df)} individual news articles...")
    texts_to_analyze = valid_texts_df['full_text'].tolist()
    try:
        results = sentiment_pipeline(texts_to_analyze, truncation=True, batch_size=32)
    except Exception as e:
        print(f"  [Error] FinBERT analysis failed: {e}")
        return None
    
    # Map results back to the DataFrame
    results_df = pd.DataFrame(results)
    valid_texts_df['label'] = results_df['label'].values
    valid_texts_df['confidence'] = results_df['score'].values
    
    # --- 3. Calculate Polarity Score ---
    # Convert label to a single polarity score:
    # positive -> +confidence, negative -> -confidence, neutral -> 0
    def calculate_polarity(row):
        if row['label'] == 'positive':
            return row['confidence']
        elif row['label'] == 'negative':
            return -row['confidence']
        else:
            return 0.0
    
    valid_texts_df['polarity'] = valid_texts_df.apply(calculate_polarity, axis=1)
    
    # --- 4. Aggregate to get DAILY AVERAGE ---
    print("  > Aggregating sentiment scores by day...")
    valid_texts_df.set_index('date', inplace=True)
    daily_avg_sentiment = valid_texts_df.resample('D').agg(
        avg_polarity=('polarity', 'mean'),
        avg_confidence=('confidence', 'mean')
    )
    
    # Fill any gaps (days with news but no valid text?) with 0s
    daily_avg_sentiment.fillna(0, inplace=True)
    
    return daily_avg_sentiment

def main():
    # --- Configuration ---
    STOCK_NEWS_DIR = "Stock News"
    OUTPUT_DIR = "daily_sentiment_data" # New folder for this module's output

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load the model ONCE for the entire run
    sentiment_pipeline = initialize_finbert()
    
    search_path = os.path.join(STOCK_NEWS_DIR, "*_news.csv")
    news_files = glob.glob(search_path)
    
    if not news_files:
        print(f"No *_news.csv files found in '{STOCK_NEWS_DIR}' folder.")
        return

    print(f"\nFound {len(news_files)} stocks. Starting sentiment processing...")
    
    success_count = 0
    fail_count = 0

    for news_file in news_files:
        basename = os.path.basename(news_file)
        ticker = basename.split('_news.csv')[0]
        
        print(f"\n--- Processing: {ticker} ---")
            
        daily_sentiment_data = process_news_file(news_file, sentiment_pipeline)
        
        if daily_sentiment_data is not None:
            # Reset index to make 'date' a column
            daily_sentiment_data.reset_index(inplace=True)
            
            output_filename = os.path.join(OUTPUT_DIR, f"{ticker}_sentiment.csv")
            daily_sentiment_data.to_csv(output_filename, index=False)
            
            print(f"  [Success] Saved to: {output_filename}")
            print(f"  > Total days with sentiment: {len(daily_sentiment_data)}")
            success_count += 1
        else:
            print(f"  [Failed] Could not process {ticker}.")
            fail_count += 1

    print(f"\n--- Sentiment Processing Complete ---")
    print(f"Successfully processed: {success_count} stocks")
    print(f"Skipped/Failed: {fail_count} stocks")

if __name__ == "__main__":
    main()