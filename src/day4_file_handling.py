import pandas as pd
import json


# Read the raw data
file_path = "data/raw/bank-full.csv"
df = pd.read_csv(file_path, sep=";")

print("Raw data loaded successfully")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("First 5 records:")
print(df.head())


# Clean the data
duplicate_count = df.duplicated().sum()
cleaned_data = df.drop_duplicates()

print("\nData cleaning completed")
print("Duplicates found:", duplicate_count)
print("Rows after cleaning:", len(cleaned_data))


# Create a summary
summary = {
    "total_records": len(cleaned_data),
    "total_columns": len(cleaned_data.columns),
    "duplicate_records": int(duplicate_count),
    "unique_jobs": int(cleaned_data["job"].nunique()),
    "average_age": round(cleaned_data["age"].mean(), 2),
    "average_balance": round(cleaned_data["balance"].mean(), 2)
}


# Save the summary as JSON
output_file = "data/processed/summary.json"

with open(output_file, "w") as file:
    json.dump(summary, file, indent=4)

print("\nSummary JSON created successfully")
print("Saved to:", output_file)


# Read the JSON file
with open(output_file, "r") as file:
    saved_summary = json.load(file)

print("\nSummary:")
print(saved_summary)