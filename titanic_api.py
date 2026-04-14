print("Starting Titanic API...")
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


app = FastAPI()

THRESHOLD = 0.385

# Load model once
import joblib

try:
    print("Loading model...")
    model = joblib.load("titanic_final_model_v2.pkl")
    print("Model loaded successfully")
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
    prob = model.predict_proba(df)[0][1]
    pred = int(prob >= THRESHOLD)

    # Better response
    return {
        "prediction": pred,
        "probability": round(float(prob), 4),
        "result": "Survived" if pred == 1 else "Not Survived"
    }



# uvicorn <filename_without_.py>:<app_variable>
# uvicorn titanic_api:app --reload
#  After running Open:  http://127.0.0.1:8000/docs

#deployed via render
# https://ml-titanic-65cv.onrender.com