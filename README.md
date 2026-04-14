# 🚢 Titanic Survival Prediction (End-to-End ML Project)

## 📌 Overview

This project is an **end-to-end Machine Learning application** that predicts whether a passenger survived the Titanic disaster using structured data.

It covers:

* Data preprocessing & feature engineering
* Model training & hyperparameter tuning
* Pipeline-based ML workflow
* Model explainability (SHAP)
* Deployment using Streamlit Cloud

👉 **Live App: https://shank-titanic-ml.streamlit.app/
👉 **GitHub Repo:** https://github.com/shankambur/ml-titanic

---

## 🎯 Problem Statement

Build a machine learning model to predict passenger survival based on features like age, gender, ticket class, fare, and embarkation port.

---

## 📊 Dataset

* Titanic dataset (Kaggle)
* Target variable: `Survived` (0 = No, 1 = Yes)

---

## ⚙️ Feature Engineering

Key transformations:

* **FamilySize** = SibSp + Parch + 1
* **IsAlone** = FamilySize == 1
* **Title extraction** from Name (Mr, Mrs, Miss, Rare)
* Dropped features:

  * `Cabin` (too many missing values)
  * `Ticket`, `PassengerId` (low predictive value)

---

## 🧹 Data Preprocessing

### Numerical Features

* Age, Fare
* Missing values → Median imputation
* Scaling → StandardScaler

### Categorical Features

* Sex, Embarked, Pclass
* Missing values → Most frequent
* Encoding → OneHotEncoder

---

## 🏗️ ML Pipeline

```python
Pipeline:
    → ColumnTransformer (Preprocessing)
    → XGBoost Classifier (Model)
```

* Ensures consistent preprocessing during training and inference
* Prevents data leakage
* Simplifies deployment

---

## 🤖 Models Used

* Logistic Regression
* Random Forest
* XGBoost (**final selected model**)

---

## 🔍 Hyperparameter Tuning

Used `GridSearchCV` to tune:

* n_estimators
* max_depth
* learning_rate

---

## 📈 Model Performance

Evaluated using:

* Accuracy
* Confusion Matrix
* Classification Report

---

## 🧠 Model Explainability (SHAP)

* Implemented SHAP for **local prediction explanations**
* Displays:

  * Top contributing features
  * Feature impact direction (+ / -)
  * Waterfall plot visualization

---

## 🚀 Deployment

* Built interactive UI using **Streamlit**
* Deployed on **Streamlit Cloud**
* Supports real-time predictions

---

## ⚠️ Challenges & Solutions

### ❌ Issue: ColumnTransformer error in cloud

✅ Solution: Used full pipeline (`pipeline.predict()`) instead of manual transform

### ❌ Issue: Version mismatch (sklearn, shap, xgboost)

✅ Solution: Locked versions in `requirements.txt`

### ❌ Issue: Feature mismatch

✅ Solution: Used `pipeline.get_feature_names_out()`

---

## 🧪 Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* SHAP
* Streamlit
* Joblib

---

## 📂 Project Structure

```
ml-titanic/
│
├── titanic_FastTrack.py              # Multiple model experiments
├── titanic_FastTrack_final.py        # Final training pipeline
├── titanic_streamlit_app.py          # Streamlit app
├── titanic_final_model_v2.pkl        # Saved model
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## ▶️ How to Run Locally

```bash
git clone https://github.com/shankambur/ml-titanic
cd ml-titanic
pip install -r requirements.txt
streamlit run titanic_streamlit_app.py
```

---

## 🧾 Resume Highlight

Built and deployed an end-to-end Machine Learning application using Scikit-learn Pipeline and XGBoost to predict Titanic passenger survival. Implemented feature engineering, hyperparameter tuning, and SHAP-based explainability. Resolved real-world deployment challenges including preprocessing pipeline issues and dependency mismatches.

---

## 🚀 Future Improvements

* Add FastAPI backend
* Improve UI/UX
* Add global feature importance
* Dockerize the application
* Add monitoring/logging

---

## ⭐ Key Takeaways

* End-to-end ML pipeline development
* Real-world debugging & deployment experience
* Explainable AI integration
* Production-ready mindset

---

## 🙌 Acknowledgements

* Kaggle Titanic Dataset
* Scikit-learn, XGBoost, SHAP, Streamlit

---

## 📬 Contact

If you found this useful, feel free to shankambur@gmail.com
