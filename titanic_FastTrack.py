#Goal
# Turn raw Titanic data into smart features that improve prediction
import pandas as pd

df = pd.read_csv('titanic.csv')

#  understand the data

print("Before Handling Misisng values, Feature Engineering and Encode df.head:\n",df.head())
print("Before Handling Misisng values, Feature Engineering and Encode df.shape\n:",df.shape)
df.info()
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

# Age → 177 missing (~20%)
# Cabin → 687 missing (~77%) 
# Embarked → 2 missing (very small)


# Step 2: Handle Missing Values
# Handle Age (Smart way, not basic)

# Instead of simple mean, use median (more robust)
print('####### AGE ###########')
print("Before Handle Misisng values Mean:", df["Age"].mean())
print("Before Handle Misisng values Median:", df["Age"].median())
print("Before Handle Misisng values Mode:", df["Age"].mode())
df['Age'] = df['Age'].fillna(df['Age'].median())
# print(df['Age'])
print("After Handle Misisng values Mean:", df["Age"].mean())
print("After Handle Misisng values Median:", df["Age"].median())
print("After Handle Misisng values Mode:", df["Age"].mode())


# 2.2 Handle Embarked
# Only 2 missing → fill with mode
print('####### Embarked ###########')
print("Before Handle Misisng values Mode:", df["Embarked"].mode())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
# print(df["Embarked"])
print("After Handle Misisng values Mode:", df["Embarked"].mode())


#  2.3 Handle Cabin

# Too many missing → drop
df.drop('Cabin', axis=1, inplace=True)

print("df.isnull().sum():\n",df.isnull().sum())
# ZERO missing value 


#Step 3: Feature Engineering 


# 3.1 Create Family Size
# People with family had higher survival chances
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# 3.2 Extract Title from Name 
# Example:
# "Mr", "Mrs", "Miss", "Master"
df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

# 3.3 Simplify Titles
                               
df['Title'] = df['Title'].replace({
    'Mlle': 'Miss',
    'Ms': 'Miss',
    'Mme': 'Mrs',
    'the Countess': 'Countess'   
})
df['Title'] = df['Title'].replace(
    ['Lady', 'Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],
    'Rare'
)
print(df['Title'].unique())

# 3.4 Drop Useless Columns
df.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

# 3.5 IsAlone based on Family Size:
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
print(df['IsAlone'].unique())

# Step 4: Convert Categorical → Numeric
# ML models don’t understand text.

# Encode Sex
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Encode Embarked
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Encode Title
df['Title'] = df['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4})


print("df['Title'].isnull().sum():\n",df['Title'].isnull().sum())
df['Title'] = df['Title'].astype(int) # this will convert float to integer

print("After Handling Misisng values, Feature Engineering and Encode df.head:\n",df.head())
print("After Handling Misisng values, Feature Engineering and Encode df.shape:\n",df.shape)


# Split Data for Train and Test
X = df.drop('Survived', axis=1)
y = df['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train using Logistic Regression Model
print('############### Started Train using Logistic Regression Model')
from sklearn.linear_model import LogisticRegression
#I got below error when I run model_LR = LogisticRegression()
# /Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:406: ConvergenceWarning: lbfgs failed to converge after 100 iteration(s) (status=1):
# STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
# Increase the number of iterations to improve the convergence (max_iter=100).
# So, I have added (max_iter=1000)
model_LR = LogisticRegression(max_iter=1000)
model_LR.fit(X_train, y_train)
import joblib
# joblib.dump(model_LR,"titanic_model_LR.pkl")
# model_LR = joblib.load("titanic_model_LR.pkl") 

#Evaluation

# ✅ Step 1: Make Predictions
prediction_LR = model_LR.predict(X_test) 
print("prediction_LR:", prediction_LR)

# ✅ Step 2: Accuracy
from sklearn.metrics import accuracy_score
print("Accuracy_LR:", accuracy_score(y_test, prediction_LR))

# 6️⃣ Predict Probability (Very Important)
prob_LR = model_LR.predict_proba(X_test)
print("Probability_LP:", prob_LR)

# ✅ Step 3: Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_LR = confusion_matrix(y_test, prediction_LR)
print("Confusion Matrix_LR:\n", cm_LR)

# ✅ Step 4: Full Report
from sklearn.metrics import classification_report
cr_LR = classification_report(y_test, prediction_LR)
print('classification_report_LR:\n',cr_LR)
print('############### Completed Train using Logistic Regression Model')

# Train using DecisionTree Model
print('################## Started Train using DecisionTree Model')
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
model_DT = DecisionTreeClassifier()
model_DT.fit(X_train, y_train)
# joblib.dump(model_DT,"titanic_model_DT.pkl")
# model_DT = joblib.load("titanic_model_DT.pkl") 

#Evaluation

# ✅ Step 1: Make Predictions
prediction_DT = model_DT.predict(X_test)
# print("prediction_DT:", prediction_DT)

# ✅ Step 2: Accuracy
from sklearn.metrics import accuracy_score
print("Accuracy_DT:", accuracy_score(y_test, prediction_DT))

# 6️⃣ Predict Probability (Very Important)
prob_DT = model_DT.predict_proba(X_test)
# print("Probability_DT:", prob_DT).   ********** Pls try to understand this later *****

# ✅ Step 3: Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_DT = confusion_matrix(y_test, prediction_DT)
print("Confusion Matrix_DT:\n", cm_DT)

# ✅ Step 4: Full Report
from sklearn.metrics import classification_report
cr_DT = classification_report(y_test, prediction_DT)
print('classification_report_DT:\n',cr_DT)

# Accuracy - LR
train_accuracy_LR = model_LR.score(X_train, y_train)
test_accuracy_LR = model_LR.score(X_test, y_test)
print("Train Accuracy_LR:", train_accuracy_LR)
print("Test Accuracy_LR:", test_accuracy_LR) 

# Accuracy - DT
train_accuracy_DT = model_DT.score(X_train, y_train)
test_accuracy_DT = model_DT.score(X_test, y_test)
print("Train Accuracy_DT:", train_accuracy_DT)
print("Test Accuracy_DT:", test_accuracy_DT) 

plt.figure(figsize=(12,8))
plot_tree(model_DT, feature_names=X.columns, class_names=["No","Yes"], filled=True)
# plt.show().   <-   lot of splits 
print('################## Started Train using DecisionTree Model')


print('######################Train model with multiple max_depth with default min_samples_split and min_samples_leaf')
for depth in range(1, 11):
    model_with_max_depth = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model_with_max_depth.fit(X_train, y_train)
    
    train_acc = model_with_max_depth.score(X_train, y_train)
    test_acc = model_with_max_depth.score(X_test, y_test)
    
    print(f"Depth: {depth} | Train: {train_acc:.3f} | Test: {test_acc:.3f}")


# Results 
# Depth: 1 → Train: 0.782 | Test: 0.782
# Depth: 2 → Train: 0.796 | Test: 0.760
# Depth: 3 → Train: 0.837 | Test: 0.827
# Depth: 4 → Train: 0.843 | Test: 0.832  ✅
# Depth: 5 → Train: 0.857 | Test: 0.832  ✅
# Depth: 6 → Train: 0.883 | Test: 0.816
# ...
# Depth: 10 → Train: 0.933 | Test: 0.777

print('################ Train model with multiple max_depth with  min_samples_split=10 and min_samples_leaf=5')
for depth in range(1, 11):
    model_with_max_depth_min_samples_split_leaves = DecisionTreeClassifier(max_depth=depth,min_samples_split=10,
min_samples_leaf=5, random_state=42)
    model_with_max_depth_min_samples_split_leaves.fit(X_train, y_train)
    
    train_acc = model_with_max_depth_min_samples_split_leaves.score(X_train, y_train)
    test_acc = model_with_max_depth_min_samples_split_leaves.score(X_test, y_test)
    
    print(f"Depth: {depth} | Train: {train_acc:.3f} | Test: {test_acc:.3f}")

# Depth: 1 | Train: 0.782 | Test: 0.782
# Depth: 2 | Train: 0.796 | Test: 0.760
# Depth: 3 | Train: 0.834 | Test: 0.816
# Depth: 4 | Train: 0.840 | Test: 0.821
# Depth: 5 | Train: 0.851 | Test: 0.821
# Depth: 6 | Train: 0.868 | Test: 0.821
# Depth: 7 | Train: 0.876 | Test: 0.849
# Depth: 8 | Train: 0.872 | Test: 0.849
# Depth: 9 | Train: 0.876 | Test: 0.844
# Depth: 10 | Train: 0.881 | Test: 0.832



print('#################### Start FINAL DECISION TREE MODEL')
model_DT_Final = DecisionTreeClassifier(
    max_depth=7,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
model_DT_Final.fit(X_train, y_train)
joblib.dump(model_DT_Final,"titanic_model_DT_Final.pkl")
model_DT_Final = joblib.load("titanic_model_DT_Final.pkl")
#Evaluation

# Accuracy - DT_Final
train_accuracy_DT_Final = model_DT_Final.score(X_train, y_train)
test_accuracy_DT_Final = model_DT_Final.score(X_test, y_test)
print("Train Accuracy_DT_Final:", train_accuracy_DT_Final)
print("Test Accuracy_DT_Final:", test_accuracy_DT_Final) 

# ✅ Step 1: Make Predictions
prediction_DT_Final = model_DT_Final.predict(X_test)
# print("prediction_DT_Final:", prediction_DT_Final)

# ✅ Step 2: Accuracy
from sklearn.metrics import accuracy_score
print("Accuracy_DT_Final:", accuracy_score(y_test, prediction_DT_Final))

# 6️⃣ Predict Probability (Very Important)
prob_DT_Final = model_DT_Final.predict_proba(X_test)
# print("Probability_DT_Final:", model_DT_Final).   ********** Pls try to understand this later *****

# ✅ Step 3: Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_DT_Final = confusion_matrix(y_test, prediction_DT_Final)
print("Confusion Matrix_DT_Final:\n", cm_DT_Final)

# ✅ Step 4: Full Report
from sklearn.metrics import classification_report
cr_DT_Final = classification_report(y_test, prediction_DT_Final)
print('classification_report_DT_Final:\n',cr_DT_Final)

print('#################### Completed FINAL DECISION TREE MODEL') 


# ✅ 1. Random Forest (Baseline Model)
# Start simple before tuning.
print('#########  Start RandomForest Model - BaseLine')
from sklearn.ensemble import RandomForestClassifier
model_RF = RandomForestClassifier(random_state=42)
model_RF.fit(X_train, y_train)
# joblib.dump(model_RF,"titanic_model_RF.pkl")
# model_RF = joblib.load("titanic_model_RF.pkl")

#Evaluation

# Accuracy - model_RF
train_accuracy_RF = model_RF.score(X_train, y_train)
test_accuracy_RF = model_RF.score(X_test, y_test)
print("Train Accuracy_RF:", train_accuracy_RF)
print("Test Accuracy_RF:", test_accuracy_RF) 

# ✅ Step 1: Make Predictions
prediction_RF = model_RF.predict(X_test)
# print("prediction_RF:", prediction_RF)

# ✅ Step 2: Accuracy
from sklearn.metrics import accuracy_score
print("Accuracy_RF:", accuracy_score(y_test, prediction_RF))

# 6️⃣ Predict Probability (Very Important)
prob_RF = model_RF.predict_proba(X_test)
# print("Probability_RF:", model_RF).   ********** Pls try to understand this later *****

# ✅ Step 3: Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_RF = confusion_matrix(y_test, prediction_RF)
print("Confusion Matrix_RF:\n", cm_RF)

# ✅ Step 4: Full Report
from sklearn.metrics import classification_report
cr_RF = classification_report(y_test, prediction_RF)
print('classification_report_RF:\n',cr_RF)
print('#########  Completed RandomForest Model - BaseLine')

print('############# Start Train Model with GridSearch');
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, 10]
}
grid = GridSearchCV(RandomForestClassifier(),param_grid)
grid.fit(X_train, y_train)
# joblib.dump(grid,"titanic_grid.pkl")
# grid = joblib.load("titanic_grid.pkl")
print("Best Params:", grid.best_params_)
print("Best Score:", grid.best_score_)
#Evaluation

# Accuracy - grid
train_accuracy_grid = grid.score(X_train, y_train)
test_accuracy_grid = grid.score(X_test, y_test)
print("Train Accuracy_grid:", train_accuracy_grid)
print("Test Accuracy_grid:", test_accuracy_grid) 

# ✅ Step 1: Make Predictions
prediction_grid = grid.predict(X_test)
# print("prediction_grid:", prediction_grid)

# ✅ Step 2: Accuracy
from sklearn.metrics import accuracy_score
print("Accuracy_grid:", accuracy_score(y_test, prediction_grid))

# 6️⃣ Predict Probability (Very Important)
prob_grid = grid.predict_proba(X_test)
# print("Probability_grid:", grid).   ********** Pls try to understand this later *****

# ✅ Step 3: Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_grid = confusion_matrix(y_test, prediction_grid)
print("Confusion Matrix_grid:\n", cm_grid)

# ✅ Step 4: Full Report
from sklearn.metrics import classification_report
cr_grid = classification_report(y_test, prediction_grid)
print('classification_report_grid:\n',cr_grid)
print('############# Completed Train Model with GridSearch');

print('#################### Start GridSearch model with Tuned')
param_grid_Tuned = {
    'n_estimators': [100, 200],
    'max_depth': [5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
grid_Tuned = GridSearchCV(RandomForestClassifier(),param_grid_Tuned)
# grid_Tuned.fit(X_train, y_train)  #### skipped training due to poor performance, running for more than 7 minutes
print('took more than 7 mins to train the  GridSearch model with Tuned, poor performance Pls look at below to run this model with more tuned parameters which will run in parallel')
print('#################### Completed GridSearch model with Tuned')


print('#################### Start GridSearch model with Tuned to run in parallel')

grid_Tuned_RunInParallel = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid_Tuned,
    cv=5,
    n_jobs=-1   # parallel run 
    # verbose=2.   #  to see progress while training model
)

grid_Tuned_RunInParallel.fit(X_train, y_train) 
# joblib.dump(grid_Tuned_RunInParallel,"titanic_grid_Tuned_RunInParallel.pkl")
# grid_Tuned_RunInParallel = joblib.load("titanic_grid_Tuned_RunInParallel.pkl") 
print("Best Params_Grid_Tuned_RunInParallel:", grid_Tuned_RunInParallel.best_params_)
print("Best Score_Grid_Tuned_RunInParallel:", grid_Tuned_RunInParallel.best_score_)
#Evaluation

# Accuracy - grid
train_accuracy_grid_Tuned_RunInParallel = grid_Tuned_RunInParallel.score(X_train, y_train)
test_accuracy_grid_Tuned_RunInParallel = grid_Tuned_RunInParallel.score(X_test, y_test)
print("Train Accuracy_grid_Tuned_RunInParallel:", train_accuracy_grid_Tuned_RunInParallel)
print("Test Accuracy_grid_Tuned_RunInParallel:", test_accuracy_grid_Tuned_RunInParallel) 

# ✅ Step 1: Make Predictions
prediction_grid_Tuned_RunInParallel = grid_Tuned_RunInParallel.predict(X_test)
# print("prediction_grid_Tuned_RunInParallel:", prediction_grid_Tuned_RunInParallel)

# ✅ Step 2: Accuracy
from sklearn.metrics import accuracy_score
print("Accuracy_grid_Tuned_RunInParallel:", accuracy_score(y_test, prediction_grid_Tuned_RunInParallel))

# 6️⃣ Predict Probability (Very Important)
prob_grid_Tuned_RunInParallel = grid_Tuned_RunInParallel.predict_proba(X_test)
# print("Probability_grid_Tuned_RunInParallel:", prob_grid_Tuned_RunInParallel).   ********** Pls try to understand this later *****

# ✅ Step 3: Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_grid_Tuned_RunInParallel = confusion_matrix(y_test, prediction_grid_Tuned_RunInParallel)
print("Confusion Matrix_grid_Tuned_RunInParallel:\n", cm_grid_Tuned_RunInParallel)

# ✅ Step 4: Full Report
from sklearn.metrics import classification_report
cr_grid_Tuned_RunInParallel = classification_report(y_test, prediction_grid_Tuned_RunInParallel)
print('classification_report_grid_Tuned_RunInParallel:\n',cr_grid_Tuned_RunInParallel)
print('#################### Completed GridSearch model with Tuned to run in parallel')
