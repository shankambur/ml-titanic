# Titanic Survival Prediction Project

## About this project

This is my end-to-end machine learning project where I tried to predict whether a passenger survived in Titanic or not.

In this project I worked on:

* Data cleaning
* Feature engineering
* Model building
* Deployment using Streamlit
* Model explainability using SHAP

---

## Problem statement

Given passenger details like age, gender, class, fare, etc., predict whether the passenger survived or not.

---

## Dataset

I used Titanic dataset.

Target column:

* Survived (0 = No, 1 = Yes)

---

## Feature Engineering

I created some useful features:

* FamilySize = SibSp + Parch + 1
* IsAlone → whether passenger is alone or not
* Title extracted from Name (Mr, Mrs, Miss, etc.)

Dropped columns:

* Cabin (too many missing values)
* Ticket, PassengerId (not useful for prediction)

---

## Preprocessing

Numerical columns:

* Age, Fare
* Filled missing values using median
* Used StandardScaler

Categorical columns:

* Sex, Embarked, Pclass
* Filled missing values using most frequent
* Used OneHotEncoder

---

## Model

I tried multiple models:

* Logistic Regression
* Random Forest
* XGBoost

Finally selected XGBoost because it gave better performance.

Used GridSearchCV for tuning.

---

## Pipeline

I used sklearn Pipeline with ColumnTransformer.

Reason:

* Keeps preprocessing and model together
* Helps during deployment
* Avoids mistakes between training and prediction

---

## Model Explainability

I used SHAP to explain predictions.

It shows:

* Which feature increased survival
* Which feature decreased survival

This helped me understand model behavior better.

---

## Deployment

I created a Streamlit app.

User can:

* Enter passenger details
* Get prediction
* See explanation (SHAP)

Deployed in Streamlit Cloud.

---

## Challenges I faced

1. ColumnTransformer error in cloud
   Fixed by using full pipeline instead of manual transform

2. Version mismatch issue
   Fixed by adding exact versions in requirements.txt

3. Feature mismatch issue
   Fixed by using pipeline.get_feature_names_out()

---

## Project files

* titanic_FastTrack.py → tried multiple models
* titanic_FastTrack_final.py → final training code
* titanic_streamlit_app.py → Streamlit app
* model.pkl → saved model

---

## How to run

```bash
git clone https://github.com/shankambur/ml-titanic
cd ml-titanic
pip install -r requirements.txt
streamlit run titanic_streamlit_app.py
```

---

## What I learned

* How to build end-to-end ML project
* How to use Pipeline and ColumnTransformer
* How to debug real deployment issues
* Importance of matching library versions
* Basics of explainable AI using SHAP

---

## Future improvements

* Improve UI
* Add more features
* Try other models
* Deploy using FastAPI


##Contact
If you found this useful, feel free to shankambur@gmail.com