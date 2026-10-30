import sqlite3
import pandas as pd

from pathlib import Path


PROJECT_FOLDER = Path(__file__).resolve().parent.parent

CSV_PATH = PROJECT_FOLDER / "data" / "raw" / "bank-full.csv"

DATABASE_PATH = PROJECT_FOLDER / "data" / "lead_app.db"


# Database tables
CREATE_TABLES = """

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


# Load dataset
df = pd.read_csv(CSV_PATH, sep=";")


# Prepare columns for the database
df = df.copy()

df.insert(
    0,
    "source_row_id",
    range(1, len(df) + 1)
)

df = df.rename(columns={
    "default": "default_status",
    "y": "historical_subscription"
})


DATABASE_COLUMNS = [
    "source_row_id",
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
    "poutcome",
    "historical_subscription"
]


# Create database folder
DATABASE_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# Connect to SQLite
connection = sqlite3.connect(DATABASE_PATH)

connection.execute("PRAGMA foreign_keys = ON")

connection.executescript(CREATE_TABLES)


# Insert Bank Marketing customers
placeholders = ",".join(["?"] * len(DATABASE_COLUMNS))

query = f"""
INSERT OR IGNORE INTO customers
({",".join(DATABASE_COLUMNS)})
VALUES ({placeholders})
"""

records = list(
    df[DATABASE_COLUMNS].itertuples(
        index=False,
        name=None
    )
)

connection.executemany(query, records)

connection.commit()


# Check database records
customer_count = connection.execute(
    "SELECT COUNT(*) FROM customers"
).fetchone()[0]

lead_count = connection.execute(
    "SELECT COUNT(*) FROM leads"
).fetchone()[0]

prediction_count = connection.execute(
    "SELECT COUNT(*) FROM predictions"
).fetchone()[0]


print("=== DATABASE DETAILS ===")

print("Database:", DATABASE_PATH)

print("\nCustomers:", customer_count)

print("Leads:", lead_count)

print("Predictions:", prediction_count)


# Display sample customer records
print("\n=== SAMPLE CUSTOMERS ===")

rows = connection.execute(
    """
    SELECT
        id,
        age,
        job,
        balance,
        historical_subscription
    FROM customers
    LIMIT 5
    """
).fetchall()

for row in rows:
    print(row)


# Display tables
print("\n=== DATABASE TABLES ===")

tables = connection.execute(
    """
    SELECT name
    FROM sqlite_master
    WHERE type = 'table'
    ORDER BY name
    """
).fetchall()

for table in tables:
    print(table[0])


connection.close()