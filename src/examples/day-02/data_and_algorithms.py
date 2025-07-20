"""
Day 2 Example: The Building Blocks - Data and Algorithms
This script demonstrates loading, cleaning, and analyzing data with a simple algorithm.
"""
import pandas as pd
import numpy as np

def main():
    # Create a small dataset with missing values
    data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'Age': [25, np.nan, 30, 22, 28],
        'Score': [85, 90, np.nan, 88, 92]
    }
    df = pd.DataFrame(data)
    print("Original Data:")
    print(df)
    print("\nStep 1: Data Cleaning (fill missing values with mean)")
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Score'].fillna(df['Score'].mean(), inplace=True)
    print(df)
    print("\nStep 2: Simple Algorithm (Calculate average score)")
    avg_score = df['Score'].mean()
    print(f"Average Score: {avg_score:.2f}")

if __name__ == "__main__":
    main() 