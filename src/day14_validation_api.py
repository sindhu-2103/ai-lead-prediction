import pandas as pd
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(title="Bank Marketing API - Validation")


# Load Bank Marketing dataset
file_path = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw"
    / "bank-full.csv"
)

df = pd.read_csv(file_path, sep=";")


# Request model
class BankCustomer(BaseModel):

    age: int = Field(ge=18, le=100)

    job: str = Field(min_length=2)

    marital: str

    education: str

    default: str

    balance: int

    housing: str

    loan: str

    contact: str

    day: int = Field(ge=1, le=31)

    month: str

    campaign: int = Field(ge=1)

    pdays: int = Field(ge=-1)

    previous: int = Field(ge=0)

    poutcome: str


# Response model
class ValidationResponse(BaseModel):

    message: str

    validated_features: BankCustomer


# GET customers
@app.get("/customers")
def get_customers(skip: int = 0, limit: int = 20):

    if skip < 0 or limit < 1 or limit > 100:
        raise HTTPException(
            status_code=400,
            detail="Invalid pagination values"
        )

    records = df.iloc[skip:skip + limit]

    customers = []

    for index, row in records.iterrows():

        customers.append({
            "id": int(index) + 1,
            "age": int(row["age"]),
            "job": row["job"],
            "marital": row["marital"],
            "education": row["education"],
            "balance": int(row["balance"])
        })

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

    return {
        "id": customer_id,
        "age": int(row["age"]),
        "job": row["job"],
        "marital": row["marital"],
        "education": row["education"],
        "balance": int(row["balance"])
    }


# POST prediction request with Pydantic validation
@app.post(
    "/predict",
    response_model=ValidationResponse
)
def validate_prediction(customer: BankCustomer):

    return ValidationResponse(
        message="Customer data is valid",
        validated_features=customer
    )