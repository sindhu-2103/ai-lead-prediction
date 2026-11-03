
import importlib

import pandas as pd
import pytest

from fastapi.testclient import TestClient

from src.day19_data_utils import (
    remove_duplicates,
    fill_missing_values,
    validate_required_columns
)


# --------------------------------------------------
# SAMPLE CUSTOMER DATA
# --------------------------------------------------

def sample_customer():

    return {
        "age": 35,
        "job": "technician",
        "marital": "married",
        "education": "secondary",
        "default": "no",
        "balance": 1500,
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "day": 15,
        "month": "may",
        "campaign": 1,
        "pdays": -1,
        "previous": 0,
        "poutcome": "unknown"
    }


# --------------------------------------------------
# DUMMY MODEL FOR PREDICTION TESTING
# --------------------------------------------------

class DummyClassifier:

    classes_ = [0, 1]


class DummyModel:

    named_steps = {
        "classifier": DummyClassifier()
    }

    def predict(self, input_data):

        return [1]

    def predict_proba(self, input_data):

        return [[0.20, 0.80]]


# --------------------------------------------------
# FIXTURE: CREATE A SEPARATE TEST DATABASE
# --------------------------------------------------

@pytest.fixture
def api_module(tmp_path, monkeypatch):

    database_path = tmp_path / "test_lead_app.db"

    monkeypatch.setenv(
        "LEAD_APP_DB",
        str(database_path)
    )

    module = importlib.import_module("src.day16_app")

    module = importlib.reload(module)

    monkeypatch.setattr(
        module,
        "load_model",
        lambda: DummyModel()
    )

    return module


@pytest.fixture
def client(api_module):

    with TestClient(api_module.app) as test_client:

        yield test_client


# ==================================================
# SECTION 1: DATA PROCESSING TESTS (6 TESTS)
# ==================================================


# TEST 1: Remove duplicate records
def test_remove_duplicates():

    data = pd.DataFrame({
        "age": [35, 35, 40],
        "job": ["technician", "technician", "admin"]
    })

    result = remove_duplicates(data)

    assert len(result) == 2


# TEST 2: Original data should remain unchanged
def test_remove_duplicates_preserves_original():

    data = pd.DataFrame({
        "age": [35, 35, 40],
        "job": ["technician", "technician", "admin"]
    })

    remove_duplicates(data)

    assert len(data) == 3


# TEST 3: Fill numerical missing values with median
def test_fill_numeric_missing_values():

    data = pd.DataFrame({
        "balance": [10, None, 30]
    })

    result = fill_missing_values(data)

    assert result.loc[1, "balance"] == 20


# TEST 4: Fill categorical missing values with mode
def test_fill_categorical_missing_values():

    data = pd.DataFrame({
        "housing": ["yes", None, "no", "yes"]
    })

    result = fill_missing_values(data)

    assert result.loc[1, "housing"] == "yes"


# TEST 5: Required columns are available
def test_required_columns_available():

    data = pd.DataFrame({
        "age": [35],
        "job": ["technician"],
        "balance": [1500]
    })

    result = validate_required_columns(
        data,
        ["age", "job", "balance"]
    )

    assert result is True


# TEST 6: Missing required columns
def test_missing_required_columns():

    data = pd.DataFrame({
        "age": [35],
        "job": ["technician"]
    })

    with pytest.raises(ValueError):

        validate_required_columns(
            data,
            ["age", "job", "balance"]
        )


# ==================================================
# SECTION 2: API AND DATABASE TESTS (5 TESTS)
# ==================================================


# TEST 7: API home endpoint
def test_home_endpoint(client):

    response = client.get("/")

    assert response.status_code == 200

    assert response.json()["message"] == (
        "Bank Marketing Prediction API is running"
    )


# TEST 8: Create customer
def test_create_customer(client):

    response = client.post(
        "/customers",
        json=sample_customer()
    )

    assert response.status_code == 201

    assert response.json()["id"] is not None


# TEST 9: Retrieve existing customer
def test_get_existing_customer(client):

    create_response = client.post(
        "/customers",
        json=sample_customer()
    )

    customer_id = create_response.json()["id"]

    response = client.get(
        f"/customers/{customer_id}"
    )

    assert response.status_code == 200

    assert response.json()["id"] == customer_id


# TEST 10: Retrieve customers with pagination
def test_get_customers_with_pagination(client):

    client.post("/customers", json=sample_customer())
    client.post("/customers", json=sample_customer())

    response = client.get(
        "/customers?skip=0&limit=1"
    )

    assert response.status_code == 200

    assert len(response.json()) == 1


# TEST 11: Customer not found
def test_customer_not_found(client):

    response = client.get("/customers/999999")

    assert response.status_code == 404

    assert response.json()["detail"] == "Customer not found"


# ==================================================
# SECTION 3: VALIDATION TESTS (6 TESTS)
# ==================================================


# TEST 12: Age below minimum
def test_invalid_age_below_minimum(client):

    data = sample_customer()

    data["age"] = 10

    response = client.post(
        "/customers",
        json=data
    )

    assert response.status_code == 422


# TEST 13: Age above maximum
def test_invalid_age_above_maximum(client):

    data = sample_customer()

    data["age"] = 101

    response = client.post(
        "/customers",
        json=data
    )

    assert response.status_code == 422


# TEST 14: Invalid day
def test_invalid_day(client):

    data = sample_customer()

    data["day"] = 40

    response = client.post(
        "/customers",
        json=data
    )

    assert response.status_code == 422


# TEST 15: Missing required field
def test_missing_required_field(client):

    data = sample_customer()

    data.pop("job")

    response = client.post(
        "/customers",
        json=data
    )

    assert response.status_code == 422


# TEST 16: Invalid data type
def test_invalid_age_data_type(client):

    data = sample_customer()

    data["age"] = "abc"

    response = client.post(
        "/customers",
        json=data
    )

    assert response.status_code == 422


# TEST 17: Invalid campaign value
def test_invalid_campaign(client):

    data = sample_customer()

    data["campaign"] = 0

    response = client.post(
        "/customers",
        json=data
    )

    assert response.status_code == 422


# ==================================================
# SECTION 4: PREDICTION AND ERROR HANDLING (3 TESTS)
# ==================================================


# TEST 18: Generate prediction and verify database history
def test_prediction_and_database_storage(client):

    # Create customer
    customer_response = client.post(
        "/customers",
        json=sample_customer()
    )

    assert customer_response.status_code == 201

    customer_id = customer_response.json()["id"]

    # Prepare prediction input
    prediction_data = sample_customer()

    prediction_data["customer_id"] = customer_id

    # Generate prediction
    response = client.post(
        "/predict",
        json=prediction_data
    )

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] == "yes"

    assert result["probability"] == 0.8

    assert result["model_name"] == "LogisticRegression"

    # Check prediction history
    history = client.get("/predictions")

    assert history.status_code == 200

    assert len(history.json()) == 1

    assert history.json()[0]["customer_id"] == customer_id


# TEST 19: Prediction with invalid customer ID
def test_prediction_invalid_customer(client):

    data = sample_customer()

    data["customer_id"] = 999999

    response = client.post(
        "/predict",
        json=data
    )

    assert response.status_code == 404

    assert response.json()["detail"] == "Customer not found"


# TEST 20: Model unavailable error
def test_model_unavailable(client, api_module, monkeypatch):

    def missing_model():

        raise FileNotFoundError("Trained model not found")

    monkeypatch.setattr(
        api_module,
        "load_model",
        missing_model
    )

    response = client.post(
        "/predict",
        json=sample_customer()
    )

    assert response.status_code == 503

    assert "Trained model not found" in response.json()["detail"]
