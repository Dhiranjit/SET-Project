import pandas as pd
import os
import glob
from transformers import pipeline 
import torch                   
import sys

# Suppress a common but harmless warning from transformers
from transformers import logging
logging.set_verbosity_error()

def analyze_sentiment_finbert(input_file):
    """
    Loads a merged data file, runs sentiment analysis using FinBERT, 
    and returns a new DataFrame.
    """
    try:
        df = pd.read_csv(input_file)
        if 'full_text' not in df.columns:
            print(f"  [Error] 'full_text' column not found in {input_file}.")
            return None
    except Exception as e:
        print(f"  [Error] Could not read file {input_file}: {e}")
        return None

    # --- FinBERT Pipeline Setup ---
    # Try to use GPU (CUDA) if available, otherwise default to CPU
    # On a Mac, 'mps' (Apple Silicon) might be an option, but 'cpu' is safer.
    device_name = 'cpu'
    if torch.cuda.is_available():
        device_name = 'cuda'
    # Note: For M1/M2/M3 Macs, you could try 'mps', but it can be buggy.
    # Let's stick to 'cpu' for your Mac to ensure it runs.
    
    print(f"  > Initializing FinBERT pipeline... (Using device: {device_name})")
    
    # Load the pre-trained FinBERT model for sentiment analysis
    try:
        # Use a specific model version that is known to work well
        sentiment_pipeline = pipeline(
            "sentiment-analysis", 
            model="ProsusAI/finbert", 
            device=-1 if device_name == 'cpu' else 0  # -1 for CPU, 0 for CUDA
        )
    except Exception as e:
        print(f"  [Fatal Error] Could not load FinBERT model: {e}")
        print("  > Make sure you have a stable internet connection for the first download.")
        return None
        
    print("  > Pipeline loaded.")

    # --- Batch Processing ---
    # Create a list of all non-empty, unique texts to analyze
    # This is MUCH faster than processing one by one
    valid_texts = df['full_text'].replace(r'^\s*$', np.nan, regex=True).dropna()
    unique_texts = valid_texts.unique().tolist()

    if not unique_texts:
        print("  > No text to analyze.")
        df['finbert_label'] = 'N/A'
        df['finbert_score'] = 0.0
        return df

    print(f"  > Analyzing {len(unique_texts)} unique non-empty text entries...")
    
    # Run the analysis on all texts.
    # This is the step that will be slow on a CPU.
    try:
        # We tell the pipeline to truncate long texts
        results = sentiment_pipeline(unique_texts, truncation=True, batch_size=32)
    except Exception as e:
        print(f"  [Error] FinBERT analysis failed: {e}")
        print("  [Info] This can happen if texts are too long or you run out of memory.")
        return None
    
    print("  > Analysis complete. Mapping results back to DataFrame...")

    # Create a mapping from the original text to its sentiment
    sentiment_map = {text: result for text, result in zip(unique_texts, results)}

    # Map the results back to the original DataFrame
    df['finbert_result'] = df['full_text'].map(sentiment_map)
    
    # Split the result dictionary into two columns
    df['finbert_label'] = df['finbert_result'].apply(lambda x: x['label'] if isinstance(x, dict) else 'N/A')
    df['finbert_score'] = df['finbert_result'].apply(lambda x: x['score'] if isinstance(x, dict) else 0.0)
    
    # Clean up
    df.drop(columns=['finbert_result'], inplace=True)

    return df

def main():
    # --- Define Directory Paths ---
    input_dir = "output"        # Read from the folder you created
    output_dir = "analysis_output_finbert"  # Save to a new folder
    
    # Create the new output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Find all merged files to process ---
    search_path = os.path.join(input_dir, "*_daily_merged.csv")
    merged_files = glob.glob(search_path)
    
    if not merged_files:
        print(f"No *_daily_merged.csv files found in '{input_dir}' folder.")
        return

    print(f"Found {len(merged_files)} files. Starting FinBERT sentiment analysis...")
    print("="*40)
    print("!! IMPORTANT !!")
    print("This will be MUCH SLOWer than VADER, especially on a CPU.")
    print("The first time you run this, it will download the model (~400MB).")
    print("This may take several minutes per stock.")
    print("="*40)
    
    success_count = 0
    fail_count = 0

    # --- Loop through each file ---
    for input_file in merged_files:
        basename = os.path.basename(input_file)
        ticker = basename.split('_daily_merged.csv')[0]
        
        print(f"\n--- Processing: {ticker} ---")
        
        # Analyze the file
        analysis_data = analyze_sentiment_finbert(input_file)
        
        # Save the result to the new output folder
        if analysis_data is not None:
            output_filename = os.path.join(output_dir, f"{ticker}_with_finbert.csv")
            analysis_data.to_csv(output_filename, index=False)
            print(f"  [Success] Saved to: {output_filename}")
            success_count += 1
        else:
            print(f"  [Failed] Could not process file for {ticker}.")
            fail_count += 1

    print(f"\n--- Analysis Complete ---")
    print(f"Successfully processed: {success_count} stocks")
    print(f"Skipped/Failed: {fail_count} stocks")

# This makes the script runnable
if __name__ == "__main__":
    # Add this to help pandas work with the new libraries
    import numpy as np 
    main()