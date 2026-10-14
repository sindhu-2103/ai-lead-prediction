import pandas as pd


# Load the Bank Marketing dataset
df = pd.read_csv("data/raw/bank-full.csv", sep=";")

print("Dataset shape:", df.shape)
print("\nFirst five records:")
print(df.head())


# Separate features and target
X = df.drop(columns=["y", "duration"])
y = df["y"].map({"no": 0, "yes": 1})


# Display the features and target
print("\nFeature columns:")
print(X.columns.tolist())

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

print("\nSubscription counts:")
print(y.value_counts())

print("\nSubscription percentages:")
print((y.value_counts(normalize=True) * 100).round(2))