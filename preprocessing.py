"""
preprocessing.py
Text Preprocessing Module for Research Paper Recommendation System

This module handles all NLP preprocessing tasks:
- Text cleaning (lowercase, punctuation removal)
- Stopword removal
- Tokenization
- Lemmatization
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')


class TextPreprocessor:
    """
    A class to handle all text preprocessing operations
    """
    
    def __init__(self):
        """Initialize preprocessor with required NLP tools"""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Keep some important words that are usually stopwords in research context
        self.stop_words -= {'not', 'no', 'nor', 'against'}
    
    def clean_text(self, text):
        """
        Clean raw text by removing special characters and normalizing
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits (keep only letters and spaces)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """
        Tokenize text into words
        
        Args:
            text (str): Text to tokenize
            
        Returns:
            list: List of tokens
        """
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        """
        Remove stopwords from token list
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Filtered tokens
        """
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize(self, tokens):
        """
        Lemmatize tokens to their base form
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text):
        """
        Complete preprocessing pipeline
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Preprocessed text ready for vectorization
        """
        # Step 1: Clean text
        cleaned = self.clean_text(text)
        
        # Step 2: Tokenize
        tokens = self.tokenize(cleaned)
        
        # Step 3: Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Step 4: Lemmatize
        tokens = self.lemmatize(tokens)
        
        # Step 5: Join back to string
        preprocessed_text = ' '.join(tokens)
        
        return preprocessed_text
    
    def preprocess_dataframe(self, df, text_column):
        """
        Preprocess entire dataframe column
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Column name to preprocess
            
        Returns:
            pd.DataFrame: DataFrame with preprocessed text column
        """
        print(f"Preprocessing '{text_column}' column...")
        df[f'{text_column}_processed'] = df[text_column].apply(self.preprocess)
        
        # Remove rows with empty processed text
        df = df[df[f'{text_column}_processed'].str.len() > 0]
        
        print(f"Preprocessing complete! {len(df)} valid papers remaining.")
        return df


# Example usage
if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = TextPreprocessor()
    
    sample_text = """
    This paper presents a Novel Deep Learning approach for Natural Language Processing.
    We achieved 95% accuracy on the benchmark dataset! Visit http://example.com for more.
    """
    
    print("Original Text:")
    print(sample_text)
    print("\nPreprocessed Text:")
    print(preprocessor.preprocess(sample_text))