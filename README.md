# Titanic Survival Prediction with SHAP (End-to-End ML App)

This project predicts whether a passenger survives the Titanic disaster and explains the prediction using SHAP.

I built this as a complete ML system with:
- Model + Pipeline
- Explainability (SHAP)
- FastAPI backend
- Streamlit frontend
- Deployment

---

## Live Apps

### 1. Streamlit App (Direct Model)
https://shank-titanic-ml.streamlit.app/

- Model runs directly inside Streamlit
- Faster response
- Includes SHAP explanation

---

### 2. Streamlit App (API Version)
https://shank-titanic-ml-api.streamlit.app/

- Calls FastAPI backend
- Uses /predict and /shap APIs
- Shows real-world architecture
- SHAP explanation via API

---

### FastAPI Backend
https://ml-titanic-65cv.onrender.com/docs

- You can test APIs here
- Swagger UI available

---

## How it works

### Direct App
Streamlit → Pipeline → Prediction + SHAP

### API App
Streamlit → FastAPI → Pipeline → SHAP → Response

---

## Features

- Predict survival probability
- Full preprocessing pipeline (imputer + scaler + encoder)
- SHAP explanation (feature impact)
- Waterfall plot visualization
- API-based architecture
- Deployed apps

---

## Tech Stack

- Python
- Scikit-learn
- XGBoost
- SHAP
- FastAPI
- Streamlit
- Pandas / NumPy
- Joblib

---

## Model Details

- Model: XGBoost Classifier
- Pipeline includes:
  - Missing value handling
  - Standard scaling
  - One-hot encoding
- Hyperparameter tuning using GridSearchCV

---

## Screenshots
yet to add screenshots <=== Pending
### Prediction UI
![Prediction](images/prediction.png)

### SHAP Plot
![SHAP](images/shap.png)

---

## Run Locally

### Clone repo
git clone https://github.com/shankambur/ml-titanic.git
cd ml-titanic

### Install

pip install -r requirements.txt


### Run API

uvicorn titanic_api:app --reload


### Run Streamlit (API version)

streamlit run titanic_streamlit_app_api.py

## API Endpoints
API_URL = https://ml-titanic-65cv.onrender.com/docs
### POST /predict
Input:

{
"Pclass": 1,
"Sex": "female",
"Age": 25,
"Fare": 100,
"Embarked": "S"
}


Output:

{
"prediction": 1,
"probability": 0.89,
"result": "Survived"
}


---

### POST /shap

Returns:
- SHAP values
- feature names
- transformed data

Used for plotting explanation in Streamlit

---

## What I learned

- How to build ML pipeline properly
- Difference between model vs pipeline prediction
- How SHAP works with transformed features
- Handling JSON ↔ numpy issues in API
- Deploying ML app (Streamlit + FastAPI)
- Debugging real issues (shape mismatch, SHAP errors)

---

## Future improvements

- Better SHAP explanation text
- Global feature importance view
- Reduce API cold start time
- UI improvements

---

## Author

Shank Ambur