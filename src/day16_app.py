import json
import sqlite3
import os

from pathlib import Path
from contextlib import contextmanager
from typing import Optional

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field


# Project paths
PROJECT_FOLDER = Path(__file__).resolve().parent.parent

DATABASE_PATH = Path(
    os.getenv(
        "LEAD_APP_DB",
        str(PROJECT_FOLDER / "data" / "lead_app.db")
    )
)

MODEL_PATH = Path(
    os.getenv(
        "LEAD_MODEL_PATH",
        str(PROJECT_FOLDER / "models" / "lead_conversion_model.joblib")
    )
)


# Database tables
SCHEMA = """

CREATE TABLE IF NOT EXISTS customers (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    source_row_id INTEGER UNIQUE,

    age INTEGER NOT NULL,

    job TEXT,

    marital TEXT,

    education TEXT,

    default_status TEXT,

    balance INTEGER,

    housing TEXT,

    loan TEXT,

    contact TEXT,

    day INTEGER,

    month TEXT,

    campaign INTEGER,

    pdays INTEGER,

    previous INTEGER,

    poutcome TEXT,

    historical_subscription TEXT,

    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE IF NOT EXISTS leads (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    customer_id INTEGER,

    input_payload TEXT NOT NULL,

    created_at TEXT DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (customer_id)
        REFERENCES customers(id)
);


CREATE TABLE IF NOT EXISTS predictions (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    customer_id INTEGER,

    lead_id INTEGER,

    prediction TEXT NOT NULL,

    probability REAL NOT NULL,

    model_name TEXT NOT NULL,

    model_version TEXT NOT NULL,

    created_at TEXT DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (customer_id)
        REFERENCES customers(id),

    FOREIGN KEY (lead_id)
        REFERENCES leads(id)
);

"""


# Database connection
@contextmanager
def db_connection():

    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    connection = sqlite3.connect(DATABASE_PATH)

    connection.row_factory = sqlite3.Row

    connection.execute("PRAGMA foreign_keys = ON")

    try:

        connection.executescript(SCHEMA)

        yield connection

        connection.commit()

    except Exception:

        connection.rollback()

        raise

    finally:

        connection.close()


# Bank Marketing input fields
class BankFeatures(BaseModel):

    age: int = Field(ge=18, le=100)

    job: str

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


# Customer input
class CustomerCreate(BankFeatures):
    pass


# Prediction input
class PredictionRequest(BankFeatures):

    customer_id: Optional[int] = None


app = FastAPI(
    title="Bank Marketing Lead Prediction API",
    description="Customer conversion prediction using ML and SQLite",
    version="1.0"
)


# Connect the frontend
# Connect the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:5501",
        "http://localhost:5501"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# Load trained model
def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


# Home
@app.get("/")
def home():

    return {
        "message": "Bank Marketing Prediction API is running",
        "documentation": "/docs"
    }


# POST - Create customer
@app.post("/customers", status_code=201)
def create_customer(customer: CustomerCreate):

    data = customer.model_dump()

    db_data = data.copy()

    db_data["default_status"] = db_data.pop("default")

    columns = [
        "age",
        "job",
        "marital",
        "education",
        "default_status",
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

    values = [db_data[column] for column in columns]

    placeholders = ",".join(["?"] * len(columns))

    with db_connection() as connection:

        cursor = connection.execute(
            f"""
            INSERT INTO customers ({",".join(columns)})
            VALUES ({placeholders})
            """,
            values
        )

        customer_id = cursor.lastrowid

        row = connection.execute(
            "SELECT * FROM customers WHERE id = ?",
            (customer_id,)
        ).fetchone()

        return dict(row)


# GET - All customers
@app.get("/customers")
def get_customers(skip: int = 0, limit: int = 20):

    if skip < 0 or limit < 1 or limit > 100:
        raise HTTPException(
            status_code=400,
            detail="Invalid pagination values"
        )

    with db_connection() as connection:

        rows = connection.execute(
            """
            SELECT *
            FROM customers
            ORDER BY id
            LIMIT ? OFFSET ?
            """,
            (limit, skip)
        ).fetchall()

        return [dict(row) for row in rows]


# GET - Customer by ID
@app.get("/customers/{customer_id}")
def get_customer(customer_id: int):

    with db_connection() as connection:

        row = connection.execute(
            "SELECT * FROM customers WHERE id = ?",
            (customer_id,)
        ).fetchone()

        if row is None:

            raise HTTPException(
                status_code=404,
                detail="Customer not found"
            )

        return dict(row)


# POST - Generate actual ML prediction
@app.post("/predict")
def predict(request: PredictionRequest):

    # Check customer if an ID is provided
    if request.customer_id is not None:

        with db_connection() as connection:

            customer = connection.execute(
                "SELECT id FROM customers WHERE id = ?",
                (request.customer_id,)
            ).fetchone()

            if customer is None:

                raise HTTPException(
                    status_code=404,
                    detail="Customer not found"
                )

    # Load model
    try:

        model = load_model()

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=503,
            detail=str(error)
        ) from error

    # Prepare input for the model
    features = request.model_dump(
        exclude={"customer_id"}
    )

    input_data = pd.DataFrame([features])

    # Generate prediction
    predicted_value = int(
        model.predict(input_data)[0]
    )

    if hasattr(model, "named_steps"):
        classifier = model.named_steps["classifier"]
    else:
        classifier = model

    classes = list(classifier.classes_)

    positive_index = classes.index(1)

    probability = float(
        model.predict_proba(input_data)[0][positive_index]
    )

    prediction = (
        "yes" if predicted_value == 1 else "no"
    )

    model_name = "LogisticRegression"

    model_version = MODEL_PATH.name

    # Save lead and prediction
    with db_connection() as connection:

        cursor = connection.execute(
            """
            INSERT INTO leads
            (customer_id, input_payload)
            VALUES (?, ?)
            """,
            (
                request.customer_id,
                json.dumps(features)
            )
        )

        lead_id = cursor.lastrowid

        connection.execute(
            """
            INSERT INTO predictions (
                customer_id,
                lead_id,
                prediction,
                probability,
                model_name,
                model_version
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                request.customer_id,
                lead_id,
                prediction,
                probability,
                model_name,
                model_version
            )
        )

    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "model_name": model_name,
        "model_version": model_version
    }


# GET - Prediction history
@app.get("/predictions")
def get_predictions():

    with db_connection() as connection:

        rows = connection.execute(
            """
            SELECT *
            FROM predictions
            ORDER BY id DESC
            """
        ).fetchall()

        return [dict(row) for row in rows]