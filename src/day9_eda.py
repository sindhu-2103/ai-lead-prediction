import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# Load the dataset
file_path = Path("data/raw/bank-full.csv")
output_folder = Path("reports/day9_eda")
output_folder.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(file_path, sep=";")


# Convert subscription outcome into a numerical column
df["converted"] = (df["y"] == "yes").astype(int)


# Descriptive statistics
print("=== DESCRIPTIVE STATISTICS ===")
print(df.describe(include="all").transpose())


# Check data quality
missing_count = int(df.isnull().sum().sum())
duplicate_count = int(df.duplicated().sum())

unknown_counts = (
    df.select_dtypes(include="object")
    .eq("unknown")
    .sum()
)

print("\n=== DATA QUALITY ===")
print("Missing values:", missing_count)
print("Duplicate records:", duplicate_count)
print("Unknown values by column:")
print(unknown_counts[unknown_counts > 0])


# Overall subscription rate
overall_rate = df["converted"].mean() * 100


# Subscription rate by contact method
contact_summary = df.groupby("contact").agg(
    customers=("converted", "size"),
    subscriptions=("converted", "sum"),
    conversion_rate=("converted", "mean"),
)

contact_summary["conversion_rate"] *= 100

contact_summary = contact_summary.sort_values(
    "conversion_rate", ascending=False
)

contact_summary.to_csv(
    output_folder / "conversion_by_contact.csv"
)

print("\n=== SUBSCRIPTION RATE BY CONTACT METHOD ===")
print(contact_summary.round(2))


# Subscription rate by job category
job_summary = df.groupby("job").agg(
    customers=("converted", "size"),
    subscriptions=("converted", "sum"),
    conversion_rate=("converted", "mean"),
)

job_summary["conversion_rate"] *= 100

job_summary = job_summary.sort_values(
    "conversion_rate", ascending=False
)

job_summary.to_csv(output_folder / "conversion_by_job.csv")

print("\n=== SUBSCRIPTION RATE BY JOB ===")
print(job_summary.round(2))


# Subscription rate by education
education_summary = df.groupby("education").agg(
    customers=("converted", "size"),
    subscriptions=("converted", "sum"),
    conversion_rate=("converted", "mean"),
)

education_summary["conversion_rate"] *= 100

education_summary = education_summary.sort_values(
    "conversion_rate", ascending=False
)

education_summary.to_csv(
    output_folder / "conversion_by_education.csv"
)


# Chart 1: Distribution of customer ages
plt.figure(figsize=(8, 5))
plt.hist(df["age"], bins=20, edgecolor="black")
plt.title("Distribution of Customer Ages")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.tight_layout()

plt.savefig(output_folder / "age_distribution.png", dpi=150)
plt.close()


# Chart 2: Subscription outcomes
outcome_counts = df["y"].value_counts()

plt.figure(figsize=(6, 5))
outcome_counts.plot(kind="bar")
plt.title("Customer Subscription Outcomes")
plt.xlabel("Subscribed")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    output_folder / "subscription_distribution.png",
    dpi=150
)
plt.close()


# Chart 3: Subscription rate by contact method
plt.figure(figsize=(7, 5))
contact_summary["conversion_rate"].plot(kind="bar")
plt.title("Subscription Rate by Contact Method")
plt.xlabel("Contact Method")
plt.ylabel("Subscription Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    output_folder / "conversion_by_contact.png",
    dpi=150
)
plt.close()


# Chart 4: Subscription rate by job
plt.figure(figsize=(10, 5))
job_summary["conversion_rate"].plot(kind="bar")
plt.title("Subscription Rate by Job Category")
plt.xlabel("Job Category")
plt.ylabel("Subscription Rate (%)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    output_folder / "conversion_by_job.png",
    dpi=150
)
plt.close()


# Correlation between numerical features
numeric_df = df.select_dtypes(include="number")
correlation = numeric_df.corr()

correlation.to_csv(output_folder / "correlation_matrix.csv")

plt.figure(figsize=(10, 7))
plt.imshow(
    correlation,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    aspect="auto"
)

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Correlation Between Numerical Features")
plt.tight_layout()

plt.savefig(
    output_folder / "correlation_matrix.png",
    dpi=150
)
plt.close()


# Save observations from the analysis
best_contact = contact_summary.index[0]
best_job = job_summary.index[0]
best_education = education_summary.index[0]

observations = [
    "DAY 9 - EXPLORATORY DATA ANALYSIS",
    "",
    f"Total customer records: {len(df)}",
    f"Total original columns: {len(df.columns) - 1}",
    f"Missing values: {missing_count}",
    f"Duplicate records: {duplicate_count}",
    f"Overall subscription rate: {overall_rate:.2f}%",
    (
        f"Highest observed contact-method rate: {best_contact} "
        f"({contact_summary.loc[best_contact, 'conversion_rate']:.2f}%)"
    ),
    (
        f"Highest observed job-category rate: {best_job} "
        f"({job_summary.loc[best_job, 'conversion_rate']:.2f}%)"
    ),
    (
        f"Highest observed education-category rate: {best_education} "
        f"({education_summary.loc[best_education, 'conversion_rate']:.2f}%)"
    ),
    "",
    "Interpretation notes:",
    "These results show associations, not proof of cause.",
    "The contact column records contact method, not a dedicated lead source.",
    "Check unknown categories before deciding how to treat them.",
    "",
    "Charts saved:",
    "age_distribution.png",
    "subscription_distribution.png",
    "conversion_by_contact.png",
    "conversion_by_job.png",
    "correlation_matrix.png",
]

report_path = output_folder / "eda_observations.txt"

report_path.write_text(
    "\n".join(observations),
    encoding="utf-8"
)

print("\nSaved charts, summaries, and observations to:", output_folder)