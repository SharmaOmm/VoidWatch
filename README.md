# VoidWatch 🛰️
### Space Debris Collision Prediction System

An agentic AI system that predicts collision risk between orbital objects using 
real ESA Conjunction Data Messages. Built with a Random Forest Classifier trained 
on 162,634 real conjunction events.

---

## Results

| Metric | Score |
|--------|-------|
| Overall Accuracy | 97% |
| HIGH Risk Precision | 93% |
| HIGH Risk Recall | 76% |
| Training Data | 162,634 ESA CDM events |

![Feature Importance](assets/feature_importance.png)
![Confusion Matrix](assets/confusion_matrix.png)

---

## Project Structure
```
VoidWatch/
├── data/raw/              # Raw ESA conjunction dataset
├── notebooks/
│   ├── 01_eda.ipynb       # Exploration, cleaning, feature selection
│   └── 02_model_training.ipynb  # Training and evaluation
├── model/                 # Saved feature list
├── api/                   # FastAPI app for local testing
└── assets/                # Visualizations
```

---

## Dataset

This project uses the ESA Collision Avoidance Challenge dataset from Kaggle.

Download here: https://www.kaggle.com/datasets/shadmanrohan/collisionavoidancechallenge

Place both files in `data/raw/` before running the notebooks.

---

## Download Trained Model

The trained model is 128MB and hosted on Google Drive.

**Download:** [voidwatch_model.pkl](https://drive.google.com/file/d/1xQcHpxMFGhou6qXqsDzvdm4Csdg0rXL9/view?usp=drive_link)

Place the downloaded file in the `model/` folder before running the API.

---

## How It Works

**Three layer pipeline:**

**1. Data Ingestion**
Raw Conjunction Data Messages from ESA containing orbital parameters
for two objects on a collision course — the target satellite and
the chaser object (debris or another satellite).

**2. Prediction Engine**
Random Forest Classifier trained on 38 orbital features including
miss distance, relative velocity, position uncertainty (sigma values),
and Mahalanobis distance. Outputs three risk classes:

- 🔴 HIGH — collision probability > 10^-5
- 🟡 MEDIUM — collision probability between 10^-10 and 10^-5
- 🟢 LOW — collision probability < 10^-10

**3. API Layer**
FastAPI endpoint that accepts conjunction features and returns
structured JSON with risk level, probabilities, top risk factors,
and maneuver recommendation.

---

## Key Features

- Trained on 162,634 real ESA conjunction events
- Handles class imbalance with balanced class weighting
- Explainable output — top risk factors per prediction
- Actionable recommendations — maneuver window and urgency level
- REST API ready for frontend integration

---

## Local Setup

**1. Clone the repo**
```bash
git clone https://github.com/SharmaOmm/VoidWatch.git
cd VoidWatch
```

**2. Download the trained model**
Download `voidwatch_model.pkl` from the link above and place it in the `model/` folder.

**3. Install dependencies**
```bash
pip install -r api/requirements.txt
```

**4. Run the API**
```bash
cd api
uvicorn main:app --reload
```

**5. Test with sample input**

A `sample_input.json` is included in the `api/` folder with a real 
conjunction event. Use it directly to test the endpoint.
```bash
cd api
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_input.json
```

Or once the API is running, open http://localhost:8000/docs 
in your browser for the interactive testing interface.

---

## Sample Output
```json
{
  "risk_level": "HIGH",
  "probabilities": {
    "HIGH": 0.71,
    "MEDIUM": 0.28,
    "LOW": 0.01
  },
  "confidence": 0.71,
  "top_risk_factors": [
    "miss_distance: 576.0",
    "mahalanobis_distance: 1.6",
    "relative_position_r: -9.6"
  ],
  "recommended_action": {
    "urgency": "Act immediately",
    "maneuver_window": "61.4-86.0 hours from now"
  }
}
```

---

## Tech Stack

- **ML:** scikit-learn, pandas, numpy
- **API:** FastAPI, uvicorn
- **Data:** ESA Collision Avoidance Challenge (Kaggle)

---

## Future Scope

- Live TLE data ingestion from Space-Track.org via SGP4 propagation
- Autonomous maneuver execution via ground station API integration
- Fleet-scale monitoring for large satellite constellations
- Risk-as-a-Service API for space insurance pricing

---
