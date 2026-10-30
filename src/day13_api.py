import pandas as pd
from pathlib import Path

from fastapi import FastAPI, HTTPException


app = FastAPI(title="Bank Marketing API")


# Load Bank Marketing dataset
file_path = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw"
    / "bank-full.csv"
)

df = pd.read_csv(file_path, sep=";")


# Get customer details from the dataset
def get_customer_data(row, customer_id):

    return {
        "id": customer_id,
        "age": int(row["age"]),
        "job": row["job"],
        "marital": row["marital"],
        "education": row["education"],
        "balance": int(row["balance"]),
        "housing": row["housing"],
        "loan": row["loan"]
    }


# Home endpoint
@app.get("/")
def home():

    return {
        "message": "Bank Marketing API is running"
    }


# GET all customers
@app.get("/customers")
def get_customers(skip: int = 0, limit: int = 20):

    if skip < 0 or limit < 1 or limit > 100:
        raise HTTPException(
            status_code=400,
            detail="Invalid pagination values"
        )

    customers = []

    records = df.iloc[skip:skip + limit]

    for index, row in records.iterrows():

        customer = get_customer_data(
            row,
            int(index) + 1
        )

        customers.append(customer)

    return {
        "total": len(df),
        "customers": customers
    }


# GET customer by ID
@app.get("/customers/{customer_id}")
def get_customer(customer_id: int):

    if customer_id < 1 or customer_id > len(df):
        raise HTTPException(
            status_code=404,
            detail="Customer not found"
        )

    row = df.iloc[customer_id - 1]

    return get_customer_data(row, customer_id)


# POST prediction request
# Actual ML model integration will be done on Day 16.
@app.post("/predict", status_code=200)
def predict(customer: dict):

    required_fields = [
        "age",
        "job",
        "marital",
        "education",
        "default",
        "balance",
        "housing",
        "loan",
        "contact",
        "day",
        "month",
        "campaign",
        "pdays",
        "previous",
        "poutcome"
    ]

    missing_fields = [
        field for field in required_fields
        if field not in customer
    ]

    if missing_fields:
        raise HTTPException(
            status_code=422,
            detail={
                "missing_fields": missing_fields
            }
        )

    return {
        "message": "Prediction request received successfully",
        "status": "Waiting for ML model integration",
        "received_fields": len(required_fields)
    }