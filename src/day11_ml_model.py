import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression


# Load the Bank Marketing dataset
file_path = Path("data/raw/bank-full.csv")
df = pd.read_csv(file_path, sep=";")

print("Dataset loaded")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# Separate input features and target
y = df["y"].map({"no": 0, "yes": 1})

# Keep only rows with a valid target
valid_rows = y.notna()
X = df.loc[valid_rows].drop(columns=["y", "duration"])
y = y.loc[valid_rows].astype(int)

print("\nTarget distribution:")
print(y.value_counts().rename(index={0: "no", 1: "yes"}))


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# Identify numerical and categorical columns
numeric_columns = X_train.select_dtypes(
    include="number"
).columns.tolist()

categorical_columns = X_train.select_dtypes(
    include="object"
).columns.tolist()


# Prepare numerical features
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# Prepare categorical features
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# Apply the preprocessing steps
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])


# Create the machine-learning model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])


# Train the model
model.fit(X_train, y_train)

print("\nModel training completed.")


# Generate predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\nFirst 10 predictions:")
print(predictions[:10])

print("\nFirst 10 predicted subscription probabilities:")
print(probabilities[:10].round(3))


# Save the model
model_folder = Path("models")
model_folder.mkdir(parents=True, exist_ok=True)

model_path = model_folder / "lead_conversion_model.joblib"
joblib.dump(model, model_path)


# Save the test predictions
output_folder = Path("data/processed")
output_folder.mkdir(parents=True, exist_ok=True)

prediction_results = pd.DataFrame({
    "actual_subscription": y_test.map({0: "no", 1: "yes"}),
    "predicted_subscription": pd.Series(
        predictions, index=X_test.index
    ).map({0: "no", 1: "yes"}),
    "predicted_probability": probabilities
}, index=X_test.index)

prediction_results.to_csv(
    output_folder / "day11_predictions.csv",
    index=False
)

print("\nSaved model to:", model_path)
print("Saved predictions to:", output_folder / "day11_predictions.csv")