import pandas as pd
import numpy as np
import os
import glob

def process_stock(candle_file, news_file):
    """
    Aggregates and merges a single pair of candle and news files.
    """
    try:
        candles_df = pd.read_csv(candle_file)
        news_df = pd.read_csv(news_file)
    except Exception as e:
        print(f"  [Error] Could not read files: {e}")
        return None

    # --- Step 1: Standardize Timestamps ---
    try:
        candles_df['date'] = pd.to_datetime(candles_df['date'])
        news_df['date'] = pd.to_datetime(news_df['date'])
    except Exception as e:
        print(f"  [Error] Failed to parse dates: {e}")
        return None

    # --- Step 2: Aggregate candles_df to Daily Level ---
    candles_df.set_index('date', inplace=True)
    daily_candles = candles_df.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    daily_candles.dropna(subset=['open'], inplace=True)

    # --- Step 3: Aggregate news_df to Daily Level ---
    
    # --- FIX IS HERE ---
    # 1. Fix the FutureWarning by using a safe assignment
    # 2. Force both title and description to be strings with .astype(str)
    #    This prevents the TypeError during ' '.join
    news_df['description'] = news_df['description'].fillna('').astype(str)
    news_df['title'] = news_df['title'].astype(str)
    # --- END FIX ---
    
    news_df['full_text'] = news_df['title'] + ' ' + news_df['description']
    news_df.set_index('date', inplace=True)
    
    daily_news = news_df.resample('D').agg({
        'full_text': ' '.join,
        'title': 'count'
    })
    daily_news.rename(columns={'title': 'news_count'}, inplace=True)
    daily_news = daily_news[daily_news['news_count'] > 0]

    # --- Step 4: Merge the Aggregated DataFrames ---
    merged_df = daily_candles.join(daily_news, how='left')
    merged_df['news_count'].fillna(0, inplace=True)
    merged_df['full_text'].fillna('', inplace=True)
    merged_df['news_count'] = merged_df['news_count'].astype(int)
    merged_df.reset_index(inplace=True)
    
    return merged_df

def main():
    # --- Define Directory Paths ---
    stock_data_dir = "Stock Data"
    stock_news_dir = "Stock News"
    output_dir = "output"
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Find all candle files to process ---
    search_path = os.path.join(stock_data_dir, "*_candles.csv")
    candle_files = glob.glob(search_path)
    
    if not candle_files:
        print(f"No *_candles.csv files found in '{stock_data_dir}' folder.")
        return

    print(f"Found {len(candle_files)} candle files. Starting batch processing...")
    
    success_count = 0
    fail_count = 0

    # --- Loop through each file ---
    for candle_file in candle_files:
        basename = os.path.basename(candle_file)
        ticker = basename.split('_candles.csv')[0]
        news_file = os.path.join(stock_news_dir, f"{ticker}_news.csv")
        
        print(f"\n--- Processing: {ticker} ---")
        
        if not os.path.exists(news_file):
            print(f"  [Skipped] No matching news file found at: {news_file}")
            fail_count += 1
            continue
            
        print(f"  > Reading candles: {candle_file}")
        print(f"  > Reading news: {news_file}")
        merged_data = process_stock(candle_file, news_file)
        
        if merged_data is not None:
            output_filename = os.path.join(output_dir, f"{ticker}_daily_merged.csv")
            merged_data.to_csv(output_filename, index=False)
            print(f"  [Success] Saved to: {output_filename}")
            success_count += 1
        else:
            print(f"  [Failed] Could not process files for {ticker}.")
            fail_count += 1

    print(f"\n--- Batch Complete ---")
    print(f"Successfully processed: {success_count} stocks")
    print(f"Skipped/Failed: {fail_count} stocks")

# This makes the script runnable
if __name__ == "__main__":
    main()