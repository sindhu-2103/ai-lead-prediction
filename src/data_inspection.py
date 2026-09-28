import pandas as pd
import numpy as np


# Dataset path
DATASET_PATH = "data/raw/bank-full.csv"


# Load the dataset
df = pd.read_csv(DATASET_PATH, sep=";")


# Basic dataset information
print("=== DATASET INSPECTION ===")

print("\n1. Dataset Shape:")
print(df.shape)

print("\n2. Column Names:")
print(df.columns.tolist())

print("\n3. First 5 Rows:")
print(df.head())

print("\n4. Data Types:")
print(df.dtypes)

print("\n5. Missing Values:")
print(df.isnull().sum())

print("\n6. Duplicate Rows:")
print(df.duplicated().sum())

print("\n7. Target Column Distribution:")
print(df["y"].value_counts())

print("\n8. Numerical Summary:")
print(df.describe())