import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda():
    print("Loading data...")
    # The dataset typically uses latin-1 encoding and has extra empty columns
    df = pd.read_csv('spam.csv', encoding='latin-1', usecols=[0, 1])
    df.columns = ['label', 'text']
    
    print(f"Original shape: {df.shape}")
    
    # Check class distribution before duplicate removal
    print("\nClass distribution (before dedup):")
    print(df['label'].value_counts(normalize=True))
    
    # Remove duplicates
    df = df.drop_duplicates()
    print(f"\nShape after dropping duplicates: {df.shape}")
    
    # Check class distribution after duplicate removal
    print("\nClass distribution (after dedup):")
    print(df['label'].value_counts(normalize=True))
    
    # Analyze message lengths
    df['length'] = df['text'].apply(len)
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Plot length distributions
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='length', hue='label', bins=50, kde=True)
    plt.title('Message Length Distribution (Spam vs Ham)')
    plt.xlim(0, 300)
    plt.savefig('plots/length_distribution.png')
    print("\nSaved length distribution plot to plots/length_distribution.png")

if __name__ == '__main__':
    run_eda()
