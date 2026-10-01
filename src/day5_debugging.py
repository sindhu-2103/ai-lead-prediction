customer_name = "Sindhu"

print("Customer Name:", customer_name) 

import pandas as pd

df = pd.read_csv("data/raw/bank-full.csv", sep=";")

age = df["age"].iloc[0]
job = df["job"].iloc[0]

result = str(age) + " " + job

print(result)