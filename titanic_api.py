from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("grid_pipeline_XGB_model.pkl")

THRESHOLD = 0.385

# ✅ Define input schema
class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    Fare: float
    Embarked: str

@app.post("/predict")
def predict(data: Passenger):
    df = pd.DataFrame([data.dict()])
    
    prob = model.predict_proba(df)[0][1]
    pred = int(prob >= THRESHOLD)
    
    return {
        "prediction": pred,
        "probability": float(prob)
    }