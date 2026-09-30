import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path, sep=";")
    return df

def clean_data(df):
    df = df.drop_duplicates()
    return df

# Dataset path
DATASET_PATH = "data/raw/bank-full.csv"

# Load the dataset
df = load_data(DATASET_PATH)

print("Dataset loaded successfully.")
print("Shape before cleaning:", df.shape)

# Clean the loaded dataset
cleaned_df = clean_data(df)

print("\nCleaned dataset shape:")
print(cleaned_df.shape)

print("\nNumber of duplicate rows after cleaning:")
print(cleaned_df.duplicated().sum())

# Filter Data Function
def filter_data(df, column, value):
    filtered_df = df[df[column] == value]
    return filtered_df


# Filter customers whose job is student
student_customers = filter_data(df, "job", "student")

print("Filtered student customer records successfully.")
print("Number of student customers:", len(student_customers))

print("\nFirst 5 student customer records:")
print(student_customers.head())

# Calculate Statistics Function
def calculate_statistics(df, column):
    mean_value = df[column].mean()
    minimum_value = df[column].min()
    maximum_value = df[column].max()

    return mean_value, minimum_value, maximum_value


# Calculate statistics for customer age
age_mean, age_min, age_max = calculate_statistics(df, "age")

print("Customer age statistics calculated successfully.")
print("Mean age:", age_mean)
print("Minimum age:", age_min)
print("Maximum age:", age_max)

# Generate Customer Summary Function
def generate_customer_summary(df, customer_index):
    customer = df.iloc[customer_index]

    summary = {
        "age": customer["age"],
        "job": customer["job"],
        "marital": customer["marital"],
        "education": customer["education"],
        "balance": customer["balance"],
        "housing": customer["housing"],
        "loan": customer["loan"],
        "target": customer["y"]
    }

    return summary


# Generate summary for the first customer
customer_summary = generate_customer_summary(df, 0)

print("Customer summary generated successfully.")
print("\nCustomer Summary:")
print(customer_summary)