import math

import pytest


def test_health_endpoint(test_client):
    """Test the /health endpoint returns correct status."""
    response = test_client.get("/health")
    assert response.status_code == 200
    response_data = response.json()
    assert "status" in response_data
    assert response_data["status"] == "healthy"


def test_predict_endpoint_valid_input(test_client, sample_home_features):
    """Test the /predict endpoint with valid input."""
    response = test_client.post("/predict", json=sample_home_features)
    assert response.status_code == 200
    response_data = response.json()
    assert "predicted_price" in response_data
    assert math.isfinite(response_data["predicted_price"])
    # Captured from the original complete-input prediction path.
    assert response_data["predicted_price"] == pytest.approx(253980.0)


@pytest.mark.parametrize(
    "field",
    [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "sqft_above",
        "sqft_basement",
    ],
)
@pytest.mark.parametrize("missing_as", ["null", "omitted"])
def test_predict_endpoint_imputes_each_optional_field(
    test_client, sample_home_features, field, missing_as
):
    payload = {**sample_home_features}
    if missing_as == "null":
        payload[field] = None
    else:
        payload.pop(field)

    response = test_client.post("/predict", json=payload)

    assert response.status_code == 200
    assert math.isfinite(response.json()["predicted_price"])


def test_predict_endpoint_imputes_all_house_fields_from_zip_demographics(
    test_client,
):
    response = test_client.post("/predict", json={"zipcode": "98042"})

    assert response.status_code == 200
    assert math.isfinite(response.json()["predicted_price"])


@pytest.mark.parametrize("zipcode_value", ["omitted", None], ids=["omitted", "null"])
def test_predict_endpoint_requires_zipcode(
    test_client, sample_home_features, zipcode_value
):
    payload = {**sample_home_features}
    if zipcode_value is None:
        payload["zipcode"] = None
    else:
        payload.pop("zipcode")

    response = test_client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_endpoint_rejects_unknown_zipcode(test_client):
    response = test_client.post("/predict", json={"zipcode": "99999"})

    assert response.status_code == 422
    assert "Unknown zipcode '99999'" in response.json()["detail"]
