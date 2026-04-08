# Titanic Survival Prediction

## Problem Statement

Predict whether a passenger survived the Titanic disaster using machine learning based on features like age, gender, class, and family size.

---

## Dataset

* Source: Kaggle Titanic Dataset
* Binary classification problem (Survived: 0 or 1)

---

## Approach

### 1. Data Preprocessing

* Handled missing values (Age, Embarked)
* Converted categorical features (Sex, Embarked)
* Feature engineering:

  * Created **IsAlone** feature
  * Dropped irrelevant columns (Name, Ticket, PassengerId)

---

### 2. Model Building

* Logistic Regression
* Decision Tree
* Random Forest (final model)

---

### 3. Model Evaluation

* Accuracy
* Confusion Matrix
* Classification Report (Precision, Recall, F1-score)

---

## Final Model

* **Random Forest Classifier**
* Selected based on better generalization performance

---

## Key Learnings

* Importance of feature engineering
* Handling missing data
* Comparing multiple models for best performance

---

##  How to Run

```bash
python titanic_FastTrack.py
```

---

##  Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn

