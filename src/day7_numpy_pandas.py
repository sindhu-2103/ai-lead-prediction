import numpy as np
import pandas as pd
from pathlib import Path


# Load the Bank Marketing dataset
file_path = Path("data/raw/bank-full.csv")
output_folder = Path("data/processed")
output_folder.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(file_path, sep=";")


# Display basic information
print("=== DATASET ===")
print("Rows and columns:", df.shape)
print("\nFirst five records:")
print(df.head())


# NumPy array operations
ages = df["age"].to_numpy()

print("\n=== NUMPY: AGE ===")
print("First ten ages:", ages[:10])
print("Average age:", round(ages.mean(), 2))
print("Minimum age:", ages.min())
print("Maximum age:", ages.max())


# Check missing values
print("\n=== MISSING VALUES ===")
missing_values = df.isnull().sum()

if missing_values.sum() == 0:
    print("No missing values found.")
else:
    print(missing_values[missing_values > 0])

print("Total missing values:", int(missing_values.sum()))


# Check and remove duplicate rows
print("\n=== DUPLICATES ===")
duplicate_count = int(df.duplicated().sum())

print("Duplicate rows found:", duplicate_count)

cleaned_df = df.drop_duplicates().copy()

print("Rows after duplicate removal:", len(cleaned_df))


# Filter student customers
students = cleaned_df.loc[cleaned_df["job"] == "student"]

print("\n=== FILTERING ===")
print("Student customer records:", len(students))
print(students.head())


# Sort customers by balance
sorted_customers = cleaned_df.sort_values(
    "balance", ascending=False
)

print("\n=== TOP FIVE BALANCES ===")
print(sorted_customers[["age", "job", "balance"]].head())


# Group customers by job
job_summary = cleaned_df.groupby("job").agg(
    customer_count=("job", "size"),
    average_balance=("balance", "mean"),
    average_age=("age", "mean"),
).sort_values("customer_count", ascending=False)

print("\n=== SUMMARY BY JOB ===")
print(job_summary.round(2))


# Save processed data
cleaned_df.to_csv(
    output_folder / "bank_cleaned.csv",
    index=False
)

job_summary.to_csv(output_folder / "job_summary.csv")

print("\nSaved processed data and job summary.")