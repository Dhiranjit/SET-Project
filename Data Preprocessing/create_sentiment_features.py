import pandas as pd
import numpy as np
import os
import glob
import sys

def process_features(sentiment_file, news_file):
    """
    Combines daily sentiment with daily news counts and engineers new
    features like change, momentum, and count.
    """
    
    # --- 1. Load Daily Sentiment Data ---
    try:
        sentiment_df = pd.read_csv(sentiment_file, parse_dates=['date'])
        sentiment_df.set_index('date', inplace=True)
    except Exception as e:
        print(f"  [Error] Could not read sentiment file {sentiment_file}: {e}")
        return None

    # --- 2. Load and Aggregate News Count ---
    try:
        news_df = pd.read_csv(news_file, parse_dates=['date'])
        news_df.set_index('date', inplace=True)
        # Resample by day ('D') and get the size (count) of each group
        count_df = news_df.resample('D').size().to_frame('news_article_count')
    except Exception as e:
        print(f"  [Error] Could not read or process news file {news_file}: {e}")
        return None

    # --- 3. Merge Sentiment and Counts ---
    # We join to the sentiment_df, which has the full daily index
    print("  > Merging sentiment scores and article counts...")
    merged_df = sentiment_df.join(count_df, how='left')
    
    # Fill days with no news with 0 articles
    merged_df['news_article_count'].fillna(0, inplace=True)
    merged_df['news_article_count'] = merged_df['news_article_count'].astype(int)
    
    # Ensure data is sorted by date for correct momentum/change calculation
    merged_df.sort_index(inplace=True)

    # --- 4. Engineer New Features ---
    print("  > Engineering features: change, momentum, count...")
    
    # 1. sentiment_change: Change in sentiment score from previous day
    # .diff() calculates the difference from the previous row
    merged_df['sentiment_change'] = merged_df['avg_polarity'].diff().fillna(0)
    
    # 2. sentiment_momentum: Moving average of sentiment (3-day and 5-day)
    # .rolling() creates a sliding window for calculations
    merged_df['momentum_3d'] = merged_df['avg_polarity'].rolling(window=3).mean().fillna(0)
    merged_df['momentum_5d'] = merged_df['avg_polarity'].rolling(window=5).mean().fillna(0)
    
    # 3. news_article_count: (Already created in step 3)
    
    return merged_df

def main():
    # --- Configuration ---
    INPUT_SENTIMENT_DIR = "daily_sentiment_data"
    INPUT_NEWS_DIR = "Stock News"
    OUTPUT_DIR = "final_sentiment_features" # New folder for this module's output

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find all the sentiment files we just created
    search_path = os.path.join(INPUT_SENTIMENT_DIR, "*_sentiment.csv")
    sentiment_files = glob.glob(search_path)
    
    if not sentiment_files:
        print(f"No *_sentiment.csv files found in '{INPUT_SENTIMENT_DIR}' folder.")
        print("Please run 'process_news_sentiment.py' first.")
        return

    print(f"\nFound {len(sentiment_files)} sentiment files. Starting feature engineering...")
    
    success_count = 0
    fail_count = 0

    for sentiment_file in sentiment_files:
        basename = os.path.basename(sentiment_file)
        ticker = basename.split('_sentiment.csv')[0]
        
        print(f"\n--- Processing: {ticker} ---")
        
        # Find the matching original news file
        news_file = os.path.join(INPUT_NEWS_DIR, f"{ticker}_news.csv")
        
        if not os.path.exists(news_file):
            print(f"  [Skipped] No matching raw news file found at: {news_file}")
            fail_count += 1
            continue
            
        feature_data = process_features(sentiment_file, news_file)
        
        if feature_data is not None:
            # Reset index to make 'date' a column again
            feature_data.reset_index(inplace=True)
            
            output_filename = os.path.join(OUTPUT_DIR, f"{ticker}_features.csv")
            feature_data.to_csv(output_filename, index=False)
            
            print(f"  [Success] Saved to: {output_filename}")
            success_count += 1
        else:
            print(f"  [Failed] Could not process {ticker}.")
            fail_count += 1

    print(f"\n--- Feature Engineering Complete ---")
    print(f"Successfully processed: {success_count} stocks")
    print(f"Skipped/Failed: {fail_count} stocks")

if __name__ == "__main__":
    main()