import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

router = APIRouter()

HOUSE_FIELDS = (
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "sqft_above",
    "sqft_basement",
)


class HomeFeatures(BaseModel):
    bedrooms: int | None = None
    bathrooms: float | None = None
    sqft_living: float | None = None
    sqft_lot: float | None = None
    floors: float | None = None
    sqft_above: float | None = None
    sqft_basement: float | None = None
    zipcode: str


@dataclass
class PredictionResources:
    model: object
    model_features: list[str]
    demographics: pd.DataFrame
    feature_scaler: StandardScaler
    imputer: KNNImputer


def load_prediction_resources() -> PredictionResources:
    """Load model inputs and fit the donor-based imputer once at startup."""
    # This resolves to /app when Docker copies `src` into /app and to the
    # repository's src directory when imported by tests.
    src_dir = Path(__file__).resolve().parents[1]
    with (src_dir / "model" / "model.pkl").open("rb") as model_file:
        model = pickle.load(model_file)
    with (src_dir / "model" / "model_features.json").open() as features_file:
        model_features = json.load(features_file)

    demographics = pd.read_csv(
        src_dir / "data" / "zipcode_demographics.csv", dtype={"zipcode": str}
    ).set_index("zipcode")
    historical_sales = pd.read_csv(
        src_dir / "data" / "kc_house_data.csv", dtype={"zipcode": str}
    )

    # Only the model's feature columns enter KNN. Sale price and zipcode are
    # deliberately excluded; zipcode is represented by its known demographics.
    imputation_features = [
        feature for feature in model_features if feature not in {"price", "zipcode"}
    ]
    donors = historical_sales.loc[:, [*HOUSE_FIELDS, "zipcode"]].merge(
        demographics.reset_index(), on="zipcode", how="left", validate="many_to_one"
    )
    donor_features = donors.loc[:, imputation_features]

    feature_scaler = StandardScaler()
    scaled_donors = feature_scaler.fit_transform(donor_features)
    imputer = KNNImputer(n_neighbors=5, weights="distance")
    imputer.fit(scaled_donors)

    return PredictionResources(
        model=model,
        model_features=model_features,
        demographics=demographics,
        feature_scaler=feature_scaler,
        imputer=imputer,
    )


@router.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    return {"status": "healthy"}


@router.post("/predict")
async def predict(home_features: HomeFeatures, request: Request):
    resources: PredictionResources = request.app.state.prediction_resources
    zipcode = home_features.zipcode
    if zipcode not in resources.demographics.index:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown zipcode '{zipcode}': no demographic data is available.",
        )

    house_values = home_features.dict(exclude={"zipcode"})
    demographic_values = resources.demographics.loc[zipcode].to_dict()
    input_data = pd.DataFrame(
        [{**house_values, **demographic_values}], columns=resources.model_features
    )

    # Preserve the original model input exactly when all seven home fields were
    # supplied. Missing values are imputed in standardized feature space.
    if input_data.loc[:, HOUSE_FIELDS].isna().to_numpy().any():
        scaled_input = resources.feature_scaler.transform(input_data)
        imputed_scaled_input = resources.imputer.transform(scaled_input)
        input_data = pd.DataFrame(
            resources.feature_scaler.inverse_transform(imputed_scaled_input),
            columns=resources.model_features,
        )

    prediction = resources.model.predict(input_data)
    return {"predicted_price": float(prediction[0])}
