"""
preprocess_data.py
Script to preprocess the dataset and create TF-IDF features

Run this script ONCE before running the Streamlit app:
    python scripts/preprocess_data.py

This will:
1. Load raw dataset
2. Clean and preprocess text
3. Create TF-IDF features
4. Save processed data for the app
"""

import pandas as pd
import pickle
import os
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')

from preprocessing import TextPreprocessor
from feature_engineering import FeatureEngineer


def load_dataset(filepath):
    """
    Load the arXiv dataset
    
    Args:
        filepath (str): Path to CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    print(f"Loading dataset from {filepath}...")
    
    # Read CSV
    df = pd.read_csv(filepath)
    
    print(f"Dataset loaded: {len(df)} papers")
    print(f"Columns: {df.columns.tolist()}")
    
    return df


def handle_missing_values(df):
    """
    Handle missing values in the dataset
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    print("\nHandling missing values...")
    
    # Check for missing values
    missing_counts = df.isnull().sum()
    print("Missing values per column:")
    print(missing_counts[missing_counts > 0])
    
    # Fill missing abstracts with empty string
    df['abstract'] = df['abstract'].fillna('')
    
    # Fill missing titles with 'Untitled'
    df['title'] = df['title'].fillna('Untitled')
    
    # Drop rows where abstract is still empty after filling
    initial_len = len(df)
    df = df[df['abstract'].str.strip() != '']
    print(f"Removed {initial_len - len(df)} papers with empty abstracts")
    
    return df


def main():
    """Main preprocessing pipeline"""
    
    print("=" * 60)
    print("Research Paper Recommendation System - Data Preprocessing")
    print("=" * 60)
    
    # Create directories
    Path('data').mkdir(exist_ok=True)
    
    # Check if raw data exists
    raw_data_path = 'data/arxiv_sample.csv'
    
    if not os.path.exists(raw_data_path):
        print(f"\n❌ Dataset not found at {raw_data_path}")
        print("\nPlease download the arXiv dataset and save it as 'data/arxiv_sample.csv'")
        print("\nYou can download from:")
        print("  - Kaggle: https://www.kaggle.com/datasets/Cornell-University/arxiv")
        print("  - Or create a sample dataset with papers you're interested in")
        print("\nRequired columns: id, title, abstract, categories, authors")
        return
    
    # Step 1: Load dataset
    df = load_dataset(raw_data_path)
    
    # Step 2: Handle missing values
    df = handle_missing_values(df)
    
    # Step 3: Preprocess text
    print("\n" + "=" * 60)
    print("Step 3: Text Preprocessing")
    print("=" * 60)
    
    preprocessor = TextPreprocessor()
    df = preprocessor.preprocess_dataframe(df, 'abstract')
    
    # Optional: Limit dataset size for faster processing (comment out for full dataset)
    MAX_PAPERS = 10000  # Adjust based on your needs
    if len(df) > MAX_PAPERS:
        print(f"\nLimiting dataset to {MAX_PAPERS} papers for faster processing...")
        df = df.head(MAX_PAPERS)
    
    # Step 4: Create TF-IDF features
    print("\n" + "=" * 60)
    print("Step 4: Feature Engineering (TF-IDF)")
    print("=" * 60)
    
    feature_engineer = FeatureEngineer(
        max_features=5000,  # Adjust based on dataset size
        ngram_range=(1, 2)  # Unigrams and bigrams
    )
    
    tfidf_matrix = feature_engineer.fit_transform(df['abstract_processed'])
    
    # Step 5: Save processed data
    print("\n" + "=" * 60)
    print("Step 5: Saving Processed Data")
    print("=" * 60)
    
    # Save dataframe
    with open('data/processed_papers.pkl', 'wb') as f:
        pickle.dump(df, f)
    print("✅ Saved: data/processed_papers.pkl")
    
    # Save vectorizer
    feature_engineer.save_vectorizer('data/vectorizer.pkl')
    
    # Save TF-IDF matrix
    feature_engineer.save_matrix('data/tfidf_matrix.pkl')
    
    # Step 6: Display sample results
    print("\n" + "=" * 60)
    print("Sample Preprocessed Papers")
    print("=" * 60)
    
    for idx in range(min(3, len(df))):
        print(f"\n📄 Paper {idx + 1}:")
        print(f"Title: {df.iloc[idx]['title'][:100]}...")
        print(f"Original abstract (first 100 chars): {df.iloc[idx]['abstract'][:100]}...")
        print(f"Processed abstract (first 100 chars): {df.iloc[idx]['abstract_processed'][:100]}...")
        
        # Show top features
        top_features = feature_engineer.get_top_features(idx, top_n=5)
        print("Top features:")
        for term, score in top_features:
            print(f"  - {term}: {score:.4f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Preprocessing Complete! ✨")
    print("=" * 60)
    print(f"Total papers processed: {len(df)}")
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Feature vocabulary size: {len(feature_engineer.feature_names)}")
    print("\n✅ You can now run the Streamlit app:")
    print("   streamlit run app.py")


if __name__ == "__main__":
    main()