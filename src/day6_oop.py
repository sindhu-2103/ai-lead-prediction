import pandas as pd
from dataclasses import dataclass


# ============================================================
# Customer class
# ============================================================

class Customer:
    def __init__(self, age, job, balance):
        self.age = age
        self.job = job
        self.balance = balance

    def show_details(self):
        print("Age:", self.age)
        print("Job:", self.job)
        print("Balance:", self.balance)


# ============================================================
# Student Customer - Inheritance
# ============================================================

class StudentCustomer(Customer):
    def get_customer_type(self):
        return "Student Customer"


# ============================================================
# Dataclass
# ============================================================

@dataclass
class CustomerSummary:
    age: int
    job: str
    balance: float
    target: str


# ============================================================
# Load Bank Marketing dataset
# ============================================================

file_path = "data/raw/bank-full.csv"

df = pd.read_csv(file_path, sep=";")


# ============================================================
# Create Customer Object
# ============================================================

first_customer = df.iloc[0]

customer = Customer(
    first_customer["age"],
    first_customer["job"],
    first_customer["balance"]
)

print("=== CUSTOMER DETAILS ===")
customer.show_details()


# ============================================================
# Create Student Customer Object
# ============================================================

student = StudentCustomer(
    25,
    "student",
    1000
)

print("\n=== INHERITANCE ===")
print(student.get_customer_type())


# ============================================================
# Create Customer Summary
# ============================================================

summary = CustomerSummary(
    age=int(first_customer["age"]),
    job=first_customer["job"],
    balance=float(first_customer["balance"]),
    target=first_customer["y"]
)

print("\n=== CUSTOMER SUMMARY ===")
print(summary)