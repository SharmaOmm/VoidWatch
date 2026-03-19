from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# Load model and features
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE_DIR, 'model', 'voidwatch_model.pkl'))
features = joblib.load(os.path.join(BASE_DIR, 'model', 'features.pkl'))

# Initialize app
app = FastAPI(
    title="VoidWatch API",
    description="Space Debris Collision Prediction System",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConjunctionInput(BaseModel):
    miss_distance: float
    mahalanobis_distance: float
    relative_position_r: float
    relative_position_t: float
    relative_position_n: float
    relative_speed: float
    time_to_tca: float
    c_sigma_r: float
    c_sigma_t: float
    c_sigma_n: float
    c_sigma_rdot: float
    c_sigma_tdot: float
    c_sigma_ndot: float
    t_sigma_r: float
    t_sigma_t: float
    t_sigma_n: float
    t_sigma_rdot: float
    t_sigma_tdot: float
    t_sigma_ndot: float
    t_j2k_inc: float
    t_j2k_sma: float
    t_j2k_ecc: float
    c_j2k_inc: float
    c_j2k_sma: float
    c_j2k_ecc: float
    t_h_apo: float
    t_h_per: float
    c_h_apo: float
    c_h_per: float
    t_obs_used: float
    c_obs_used: float
    t_weighted_rms: float
    c_weighted_rms: float
    relative_velocity_r: float
    relative_velocity_t: float
    relative_velocity_n: float
    F10: float
    AP: float

@app.get("/")
def home():
    return {
        "message": "VoidWatch API is live",
        "version": "1.0",
        "docs": "/docs"
    }

@app.post("/predict")
def predict(data: ConjunctionInput):
    input_dict = data.dict()
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[features]

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    classes = model.classes_

    feature_importances = pd.Series(
        model.feature_importances_,
        index=features
    ).sort_values(ascending=False)

    top_factors = feature_importances.head(3).index.tolist()
    top_factor_values = [
        f"{feat}: {round(input_dict[feat], 2)}"
        for feat in top_factors
    ]

    time_to_tca = input_dict['time_to_tca']
    if prediction == 'HIGH':
        urgency = "Act immediately"
        maneuver_window = f"{round(time_to_tca * 24 * 0.5, 1)}-{round(time_to_tca * 24 * 0.7, 1)} hours from now"
    elif prediction == 'MEDIUM':
        urgency = "Monitor closely"
        maneuver_window = f"{round(time_to_tca * 24 * 0.6, 1)}-{round(time_to_tca * 24 * 0.8, 1)} hours from now"
    else:
        urgency = "No action required"
        maneuver_window = "N/A"

    return {
        "risk_level": str(prediction),
        "probabilities": {str(k): float(v) for k, v in zip(classes, probabilities)},
        "confidence": float(round(max(probabilities), 3)),
        "top_risk_factors": top_factor_values,
        "recommended_action": {
            "urgency": str(urgency),
            "maneuver_window": str(maneuver_window)
        }
    }
