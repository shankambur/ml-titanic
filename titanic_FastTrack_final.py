#  Goal

# Turn raw Titanic data into smart features that improve prediction

# Why Feature Engineering Matters
# A model doesn’t “understand” data like humans.
# We convert raw columns into meaningful signals.

# Example:

# Name  → useless as-is
# Extract Title  → VERY powerful
# Dataset Columns (Quick Recall)
# PassengerId
# Survived (target)
# Pclass
# Name
# Sex
# Age
# SibSp
# Parch
# Ticket
# Fare
# Cabin
# Embarked


# What Each Column Means (Quick Clarity)

# Let’s simplify your columns:

# PassengerId → just ID (we will drop later)
# Survived →  target (0 = No, 1 = Yes)
# Pclass → ticket class (rich/poor)
# Name → useful for feature engineering (Title)
# Sex → gender (important)
# Age → has missing values
# SibSp → siblings/spouse
# Parch → parents/children
# Ticket → mostly useless
# Fare → ticket price
# Cabin → many missing values
# Embarked → port (categorical)


import pandas as pd


df = pd.read_csv('titanic.csv')

# 🚀 Now We Move Forward (IMPORTANT STEP)

# Before feature engineering, we must understand the data

print("Before Handling Misisng values, Feature Engineering and Encode df.head:\n",df.head())
print("Before Handling Misisng values, Feature Engineering and Encode df.shape\n:",df.shape)
# df.info()
print("df.isnull().sum()\n:",df.isnull().sum())

# output:
# df.isnull().sum(): PassengerId      0
# Survived         0
# Pclass           0
# Name             0
# Sex              0
# Age            177
# SibSp            0
# Parch            0
# Ticket           0
# Fare             0
# Cabin          687
# Embarked         2

#  Understand Missing Data (Very Important)

# Age → 177 missing (~20%)
# Cabin → 687 missing (~77%) ❗
# Embarked → 2 missing (very small)

# 👉 Decision thinking (this is what interviewers expect):
# | Column   | Action            | Reason            |
# | -------- | ----------------- | ----------------- |
# | Age      | Fill              | Important feature |
# | Cabin    | Drop or transform | Too many missing  |
# | Embarked | Fill              | Very few missing  |

#  Step 2: Handle Missing Values
#  we can handle misisng values using inmputer in pipeline

# Handle Cabin
# Too many missing → drop
df.drop('Cabin', axis=1, inplace=True)



# Feature Engineering (GAME CHANGER)

# Now we create powerful features 

# Create Family Size
# Why?
# People with family had higher survival chances
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1


# Extract Title from Name 
# 👉 Example:
# "Mr", "Mrs", "Miss", "Master"
df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

#  Simplify Titles
                               
df['Title'] = df['Title'].replace({
    'Mlle': 'Miss',
    'Ms': 'Miss',
    'Mme': 'Mrs',
    'the Countess': 'Countess'   # 👈 Fix added
})
df['Title'] = df['Title'].replace(
    ['Lady', 'Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],
    'Rare'
)
print(df['Title'].unique())

#  Drop Useless Columns
# No predictive value (noise)
df.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

# IsAlone based on Family Size:
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
print(df['IsAlone'].unique())

################################

print("df['Title'].isnull().sum():\n",df['Title'].isnull().sum())


# Split Data for Train and Test
X = df.drop('Survived', axis=1)
y = df['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(X.columns)
print("#########################  Started ColumnTransformer In pipeline with XGBoost")

# Titanic Features Setup
# Example features:
num_features = ['Age', 'Fare']
cat_features = ['Sex', 'Embarked', 'Pclass']

# Import Required Libraries
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
import joblib
from sklearn.metrics import accuracy_score

# Build Preprocessing Pipeline
# Numeric Pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),   # Handing Missing values
    ('scaler', StandardScaler())
])
# Categorical Pipeline
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')), # Handing Missing values
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
#ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])
# Full Pipeline 
pipeline_XGB = Pipeline([
    ('preprocessing', preprocessor),
    # ('smote', SMOTE(random_state=42)),
    ('model', XGBClassifier(random_state=42,))
])

#Key change: use model__ prefix
param_grid = {
    'model__n_estimators': [100],
    'model__max_depth': [4],
    'model__learning_rate': [0.1]
}

# 🔧 GridSearch with Pipeline
grid_pipeline_XGB = GridSearchCV(
    pipeline_XGB,
    param_grid,
    cv=2,
    verbose=1,
    n_jobs=1,
    error_score='raise'
)


print("df['Sex'].unique()\n",df['Sex'].unique())
print("df['Embarked'].unique():\n",df['Embarked'].unique())

# ▶️ Step 7: Train Model
print("Started training")
grid_pipeline_XGB.fit(X_train, y_train)
joblib.dump(grid_pipeline_XGB,"titanic_final_model_v2.pkl")
print("titanic_final_model_v2.pkl saved successfully")
grid_pipeline_XGB = joblib.load("titanic_final_model_v2.pkl")
print("Best_Params_grid_pipeline_XGB:", grid_pipeline_XGB.best_params_)
print("Best_Score_grid_pipeline_XGB:", grid_pipeline_XGB.best_score_)
#Evaluation

pipeline = grid_pipeline_XGB.best_estimator_
encoder = pipeline.named_steps['preprocessing']\
    .named_transformers_['cat']\
    .named_steps['encoder']

print("encoder::\n",encoder)

# Accuracy - grid
train_accuracy_grid_pipeline_XGB = grid_pipeline_XGB.score(X_train, y_train)
test_accuracy_grid_pipeline_XGB = grid_pipeline_XGB.score(X_test, y_test)
print("train_accuracy_grid_pipeline_XGB:", train_accuracy_grid_pipeline_XGB)
print("test_accuracy_grid_pipeline_XGB:", test_accuracy_grid_pipeline_XGB) 

# 📊 Step 8: Predict
y_pred_grid_pipeline_XGB = grid_pipeline_XGB.predict(X_test)

# ✅ Step 9: Accuracy
print("Accuracy_grid_pipeline_XGB:", accuracy_score(y_test, y_pred_grid_pipeline_XGB))

# ✅ Step 3: Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_grid_pipeline_XGB = confusion_matrix(y_test, y_pred_grid_pipeline_XGB)
print("Confusion Matrix_grid_pipeline_XGB:\n", cm_grid_pipeline_XGB)

# ✅ Step 4: Full Report
from sklearn.metrics import classification_report
cr_grid_pipeline_XGB = classification_report(y_test, y_pred_grid_pipeline_XGB)
print('classification_report_grid_pipeline_XGB:\n',cr_grid_pipeline_XGB)




print("#########################  Completed ColumnTransformer In pipeline with XGBoost")

# Production-ready

