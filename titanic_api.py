print("Starting Titanic API...")
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import shap
import numpy as np

app = FastAPI()

THRESHOLD = 0.5

# Load model once
import joblib

try:
    print("Loading model...")
    grid = joblib.load("titanic_final_model_v2.pkl")
    pipeline = grid.best_estimator_

    preprocessor = pipeline.named_steps["preprocessing"]
    model = pipeline.named_steps["model"]
    print("Model loaded successfully")

    explainer = shap.TreeExplainer(model)


except Exception as e:
    print("Model load failed:", e)
    model = None


# Input schema
from pydantic import BaseModel, Field, field_validator

class Passenger(BaseModel):
    Pclass: int = Field(..., ge=1, le=3)
    Sex: str
    Age: float = Field(..., ge=0, le=100)
    Fare: float = Field(..., ge=0)
    Embarked: str

    @field_validator("Sex")
    def validate_sex(cls, v):
        v = v.lower()
        if v not in ["male", "female"]:
            raise ValueError("Sex must be 'male' or 'female'")
        return v

    @field_validator("Embarked")
    def validate_embarked(cls, v):
        v = v.upper()
        if v not in ["S", "C", "Q"]:
            raise ValueError("Embarked must be 'S', 'C', or 'Q'")
        return v


# Health check (important for deployment)
@app.get("/")
def home():
    return {"message": "Titanic Prediction API is running"}


# Prediction API
@app.post("/predict")
def predict(data: Passenger):

    # Convert to DataFrame
    df = pd.DataFrame([data.dict()])

    # Fix input format (IMPORTANT)
    df["Sex"] = df["Sex"].str.lower()
    df["Embarked"] = df["Embarked"].str.upper()

    # Prediction
    prob = pipeline.predict_proba(df)[0][1]
    pred = int(prob >= THRESHOLD)

    # Better response
    return {
        "prediction": pred,
        "probability": round(float(prob), 4),
        "result": "Survived" if pred == 1 else "Not Survived"
    }

#SHAP API
@app.post("/shap")
def shap_explain(data: Passenger):

    df = pd.DataFrame([data.dict()])

    df["Sex"] = df["Sex"].str.lower()
    df["Embarked"] = df["Embarked"].str.upper()

    X_transformed = pipeline[:-1].transform(df)
    print("X_transformed:\n",X_transformed)
    X_transformed = X_transformed.astype(np.float32)
    print("X_transformed_asFloat:\n",X_transformed)
    shap_values = explainer.shap_values(X_transformed)
    print("shap_values:\n",shap_values)

    # ✅ Handle list output
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
        print("shap_values_AfterHandling_list:\n",shap_values)

    base_value = explainer.expected_value
    print("base_value:\n",base_value)

    # ✅ Handle list base value
    if isinstance(base_value, list):
        base_value = base_value[1]
        print("base_value_afterHandlingList:\n",base_value)
    return {
        "values": shap_values.tolist(),
        "base_values": [float(base_value)],
        "data": X_transformed.tolist(),
        "feature_names": pipeline[:-1].get_feature_names_out().tolist()
    }

# uvicorn <filename_without_.py>:<app_variable>
# uvicorn titanic_api:app --reload
#  After running Open:  API_URL = http://127.0.0.1:8000/docs

#deployed via render
# API_URL = https://ml-titanic-65cv.onrender.com/docs