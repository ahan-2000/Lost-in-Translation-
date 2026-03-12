"""
Phase 0: Data Cleaning and Preprocessing
Refactored from Phase_0.ipynb
"""

import pandas as pd
import numpy as np
import unidecode


def standardize(text):
    """
    Standardize text by converting to lowercase, stripping whitespace, and removing diacritics.
    
    Args:
        text: Input text string
        
    Returns:
        Standardized text string
    """
    if pd.isna(text):
        return text
    return unidecode.unidecode(str(text).lower().strip())


def load_and_clean_data(input_file):
    """
    Load dataset and perform initial cleaning.
    
    Args:
        input_file: Path to input CSV file
        
    Returns:
        Cleaned DataFrame
    """
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows")
    
    # Data audit - missing values
    print("\nMissing values per column:")
    print(df.isnull().sum())
    
    # Remove duplicates
    print(f"\nTotal rows before removing duplicates: {len(df)}")
    dup_rows = df[df.duplicated(subset=['Lemma', 'Akk.-Sing.'])]
    print(f"Number of duplicate rows: {len(dup_rows)}")
    df = df.drop(dup_rows.index)
    print(f"Total rows after removing duplicates: {len(df)}")
    
    return df


def validate_gender_labels(df):
    """
    Validate gender labels in the dataset.
    
    Args:
        df: DataFrame with gender columns
        
    Returns:
        None (prints validation results)
    """
    print("\nLatin Genus_lat values:", df['Genus_lat'].unique())
    print("Occitan Genus_ok values:", df['Genus_ok'].unique())


def filter_neuter_nouns(df):
    """
    Filter dataset to include only neuter gender nouns.
    
    Args:
        df: DataFrame with gender information
        
    Returns:
        Filtered DataFrame containing only neuter nouns
    """
    df_n = df[df['Genus_lat'] == 'n'].copy()
    print(f"\nNumber of neuter nouns: {len(df_n)}")
    return df_n


def process_neuter_nouns(df_n):
    """
    Process neuter nouns: standardize orthography and generate lexeme IDs.
    
    Args:
        df_n: DataFrame containing neuter nouns
        
    Returns:
        Processed DataFrame with standardized columns and lexeme IDs
    """
    # Standardize orthography
    df_n['Lemma_std'] = df_n['Lemma'].apply(standardize)
    df_n['Akk_Sing_std'] = df_n['Akk.-Sing.'].apply(standardize)
    
    # Generate unique lexeme ID
    df_n['Lexeme_ID'] = pd.factorize(df_n['Lemma_std'])[0]
    
    return df_n


def save_cleaned_data(df_n, output_file="data_clean.csv"):
    """
    Save cleaned data to CSV file.
    
    Args:
        df_n: Processed DataFrame
        output_file: Output file path
        
    Returns:
        Cleaned DataFrame with selected columns
    """
    clean_df = df_n[['Lexeme_ID', 'Lemma_std', 'Akk_Sing_std', 'Genus_lat', 'Genus_ok']]
    clean_df.to_csv(output_file, index=False)
    print(f"\nSaved {output_file}")
    return clean_df


def print_summary_statistics(clean_df):
    """
    Print summary statistics of cleaned dataset.
    
    Args:
        clean_df: Cleaned DataFrame
    """
    print("\nCleaned dataset shape:", clean_df.shape)
    print("Unique lemmas:", clean_df['Lemma_std'].nunique())
    print("Unique lexeme IDs:", clean_df['Lexeme_ID'].nunique())
    print("Genus_lat value counts:\n", clean_df['Genus_lat'].value_counts())
    print("Genus_ok value counts:\n", clean_df['Genus_ok'].value_counts())


def main(input_file="/content/latin_occitan_with_variations.csv", output_file="data_clean.csv"):
    """
    Main execution function for Phase 0 data cleaning.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        
    Returns:
        Cleaned DataFrame
    """
    print("=" * 70)
    print("PHASE 0: DATA CLEANING")
    print("=" * 70)
    
    # Load and clean data
    df = load_and_clean_data(input_file)
    
    # Validate gender labels
    validate_gender_labels(df)
    
    # Filter neuter nouns
    df_n = filter_neuter_nouns(df)
    
    # Process neuter nouns
    df_n = process_neuter_nouns(df_n)
    
    # Save cleaned data
    clean_df = save_cleaned_data(df_n, output_file)
    
    # Print summary statistics
    print_summary_statistics(clean_df)
    
    return clean_df


if __name__ == "__main__":
    clean_df = main()

