import pandas as pd
DATASET_PATH = "data/raw/bank-full.csv"
df = pd.read_csv(DATASET_PATH, sep=";")
# Filter Customer Records
students = df[df["job"] == "student"]

print("\n--- 1. Filter Customer Records ---")
print("Student customer records:")
print(students.head())

print("\nNumber of student customers:")
print(len(students))

#  Find Unique Values

unique_jobs = df["job"].unique()

print("\n--- 2. Find Unique Values ---")
print("Unique job categories:")
print(unique_jobs)

print("\nNumber of unique job categories:")
print(df["job"].nunique())

#  Count Categories


job_counts = df["job"].value_counts()

print("\n--- 3. Count Categories ---")
print("Customer count by job category:")
print(job_counts)

#  Identify Duplicates


duplicate_count = df.duplicated().sum()

print("\n--- 4. Identify Duplicates ---")
print("Number of duplicate rows:")
print(duplicate_count)

if duplicate_count > 0:
    print("\nDuplicate records:")
    print(df[df.duplicated()].head())
else:
    print("No duplicate records found.")

#  Generate Basic Statistics


print("\n--- 5. Basic Statistics ---")

print("\nAge Statistics:")
print("Mean:", df["age"].mean())
print("Minimum:", df["age"].min())
print("Maximum:", df["age"].max())

print("\nBalance Statistics:")
print("Mean:", df["balance"].mean())
print("Minimum:", df["balance"].min())
print("Maximum:", df["balance"].max())

print("\nOverall Numerical Statistics:")
print(df.describe())


