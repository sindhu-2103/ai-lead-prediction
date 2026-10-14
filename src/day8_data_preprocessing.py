import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Load the dataset
file_path = Path("data/raw/bank-full.csv")
output_folder = Path("data/processed")
output_folder.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(file_path, sep=";")


# Check data quality
print("=== DATA QUALITY ===")
print("Original rows:", len(df))
print("Missing values:", int(df.isnull().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))

print("\nData types:")
print(df.dtypes)


# Remove duplicate records
df = df.drop_duplicates().copy()


# Separate features and target
y = df["y"].map({"no": 0, "yes": 1})
X = df.drop(columns=["y"])


# Split before fitting preprocessing steps
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# Identify balance outliers using training data
Q1 = X_train["balance"].quantile(0.25)
Q3 = X_train["balance"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = (
    (X_train["balance"] < lower_limit) |
    (X_train["balance"] > upper_limit)
).sum()

print("\n=== OUTLIERS ===")
print("Balance outliers in training data:", int(outliers))


# Cap extreme balance values
X_train = X_train.copy()
X_test = X_test.copy()

X_train.loc[:, "balance"] = np.clip(
    X_train["balance"], lower_limit, upper_limit
)

X_test.loc[:, "balance"] = np.clip(
    X_test["balance"], lower_limit, upper_limit
)


# Identify numerical and categorical columns
numeric_columns = X_train.select_dtypes(
    include="number"
).columns.tolist()

categorical_columns = X_train.select_dtypes(
    include="object"
).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)


# Prepare numerical data
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])


# Prepare categorical data
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        drop="first"
    )),
])


# Combine preprocessing steps
preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns),
])


# Fit on training data and transform both datasets
X_train_array = preprocessor.fit_transform(X_train)
X_test_array = preprocessor.transform(X_test)


# Convert processed data into DataFrames
feature_names = preprocessor.get_feature_names_out()

if hasattr(X_train_array, "toarray"):
    X_train_array = X_train_array.toarray()

if hasattr(X_test_array, "toarray"):
    X_test_array = X_test_array.toarray()

X_train_ready = pd.DataFrame(
    X_train_array,
    columns=feature_names,
    index=X_train.index
)

X_test_ready = pd.DataFrame(
    X_test_array,
    columns=feature_names,
    index=X_test.index
)


# Save the training and testing datasets
X_train_ready.to_csv(
    output_folder / "X_train.csv",
    index=False
)

X_test_ready.to_csv(
    output_folder / "X_test.csv",
    index=False
)

y_train.to_frame(name="y").to_csv(
    output_folder / "y_train.csv",
    index=False
)

y_test.to_frame(name="y").to_csv(
    output_folder / "y_test.csv",
    index=False
)


# Display the results
print("\n=== ML-READY DATA ===")
print("Training feature shape:", X_train_ready.shape)
print("Testing feature shape:", X_test_ready.shape)
print("Training target rows:", len(y_train))
print("Testing target rows:", len(y_test))

print("\nSaved training and testing datasets in data/processed.")