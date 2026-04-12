import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

import numpy as np

print("Start the streamlie******")
def clean_feature_name(name, value, input_data):
    print("clean_feature_name:Name:\n",name)
    print("clean_feature_name:Value:\n",value)
    name = name.replace("cat__", "").replace("num__", "")

    # 🎯 Numeric features → show original value
    if name in ["Age", "Fare"]:
     original_value = input_data[name].iloc[0]
     if name == "Age":
        if original_value < 13:
            return f"Child (Age {round(original_value, 2)})"
        elif original_value < 20:
            return f"Teen (Age {round(original_value, 2)})"
        elif original_value < 60:
            return f"Adult (Age {round(original_value, 2)})"
        else:
            return f"Senior (Age {round(original_value, 2)})"

     return f"{name} ({round(original_value, 2)})"

    # 🎯 Sex
    if "Sex_female" in name:
        return "Female" if value == 1 else "Male"

    if "Sex_male" in name:
        return "Male" if value == 1 else "Female"

    # 🎯 Pclass
    if "Pclass" in name and value == 1:
        return f"Class {name.split('_')[-1]}"

    # 🎯 Embarked
    if "Embarked" in name and value == 1:
        return f"Embarked {name.split('_')[-1]}"

    return None  # skip inactive features


@st.cache_resource
def load_model():
    print("Loading model only once...")
    
    grid = joblib.load("titanic_final_model.pkl")
    pipeline = grid.best_estimator_


    preprocessor = pipeline.named_steps['preprocessing']
    print("preprocessor:\n",preprocessor)
    model = pipeline.named_steps['model']
    encoder_cat = preprocessor.named_transformers_['cat'].named_steps['encoder']
    print("Encoder_Categories:", encoder_cat.categories_)
    scaler = preprocessor.named_transformers_['num'].named_steps['scaler']
    print("scaler:",scaler)
    print("Mean:", scaler.mean_)
    print("Scale:", scaler.scale_)
    feature_names = preprocessor.get_feature_names_out()
    print(" loading model : feature_names",feature_names)
    clean_names = [name.split("__")[1] for name in feature_names]
    print(" loading model : clean_names",clean_names)
    return pipeline, preprocessor, model, clean_names



pipeline, preprocessor, model, clean_names = load_model()

@st.cache_resource(hash_funcs={type(model): id})
def get_explainer(model):
    print("Creating SHAP explainer once...")
    return shap.TreeExplainer(model.get_booster())

explainer = get_explainer(model)
print("explainer:\n",explainer)

def get_feature_emoji(name):
    if "Age" in name or "Child" in name or "Adult" in name:
        return "👶"
    if "Fare" in name:
        return "💰"
    if "Male" in name:
        return "👨"
    if "Female" in name:
        return "👩"
    if "Class" in name:
        return "🎟️"
    if "Embarked" in name:
        return "🛳️"
    return "🔹"

THRESHOLD = 0.5

st.title("🚢 Titanic Survival Prediction App")

st.write("Adjust passenger details to predict survival probability")

# 🎛️ Inputs
pclass = st.selectbox("Passenger Class", [3, 2, 1])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 30)
fare = st.slider("Fare", 0.0, 300.0, 55.0)
embarked = st.selectbox("Embarked", ["Q", "C", "S"])

# 📦 Create DataFrame
input_data = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex,
    "Age": age,
    "Fare": fare,
    "Embarked": embarked
}])



if pclass == 3 and fare > 40:
    st.warning("⚠️ Unusual combination: High fare for 3rd class")

if pclass == 1 and fare < 20:
    st.warning("⚠️ Unusual: Very low fare for 1st class")

# 🔮 Prediction
if st.button("Predict"):
    # ✅ Transform input first
    print("Transforming input")
    print("input_data:\n",input_data)
    input_data["Sex"] = input_data["Sex"].str.lower()
    input_data["Embarked"] = input_data["Embarked"].str.upper()
    print("input_data_afterUC_LC:\n",input_data)
    X_transformed = preprocessor.transform(input_data)
    print("X_transformed:\n",X_transformed)
    X_transformed_df = pd.DataFrame(X_transformed,columns=clean_names)
    print("X_transformed_df:\n",X_transformed_df)
    

    # ✅ Prediction
    print("pipeline.predict_proba(input_data)\n",pipeline.predict_proba(input_data))
    prob = pipeline.predict_proba(input_data)[0][1]
    print("prob:\n",prob)
    pred = int(prob >= THRESHOLD)
    # st.write("Transformed Data:", X_transformed_df)
    st.subheader("Result")
    st.write(f"Survival Probability: **{prob:.2f}**")

    if prob > 0.7:
        st.success("High confidence prediction")
    elif prob < 0.3:
        st.info("Low survival probability")
    else:
        st.warning("⚠️ Model uncertain")

    if pred == 1:
        st.success("✅ Likely to Survive")
    else:
        st.error("❌ Not Likely to Survive")

    # 🧠 SHAP EXPLANATION
    st.subheader("🔍 Why this prediction? (SHAP)")


    shap_values = explainer(X_transformed_df)
    values = shap_values.values[0]
    feature_names = X_transformed_df.columns

    feature_impact = list(zip(feature_names, values))
# Sort by absolute importance
    feature_impact = sorted(feature_impact, key=lambda x: abs(x[1]), reverse=True)
    st.markdown("### 🧠 Top Factors Influencing Prediction")
    st.markdown("---")
    print("feature_impactAll:\n",feature_impact)
    print("feature_impact.shape:",np.array(feature_impact).shape)
    filtered_features = []
    for name, val in feature_impact:
      value = X_transformed_df[name].iloc[0]
      print("name :\n",name)
      print("val :\n",val)
      print("value:\n",value)
      # Skip inactive one-hot features
      if ("Sex_" in name or "Pclass_" in name or "Embarked_" in name) and value == 0:
        continue
      clean_name = clean_feature_name(name,value,input_data)
      if clean_name is not None:
        filtered_features.append((clean_name, val))

      print("filtered_features:\n",filtered_features)

    # 👉 Take top 3
    top_features = sorted(filtered_features, key=lambda x: abs(x[1]), reverse=True)[:3]

    print("top_features:\n",top_features)

    # 👉 Display
    for name, val in top_features:
        emoji = get_feature_emoji(name)

        strength = abs(val)

        if val > 0:
            if strength > 1:
                st.success(f"{emoji} **{name}** strongly increased survival")
            else:
                st.write(f"{emoji} ✅ **{name}** → increased survival probability")
        else:
            if strength > 1:
                st.error(f"{emoji} **{name}** strongly decreased survival")
            else:
                st.write(f"{emoji} ❌ **{name}** → decreased survival probability")

    # Plot
    fig, ax = plt.subplots()
    shap.plots.waterfall(
        shap_values[0],
        max_display=10,
        show=False
    )
    st.pyplot(fig)



# We’ll convert SHAP into plain English explanations

# 🧠 What We’re Building

# Instead of only chart:

# 👉 You’ll show:

# Top factors influencing prediction:

# 1. Female → increased survival probability
# 2. Age → decreased survival probability
# 3. Fare → decreased survival probability


# 🚀 Step-by-Step Implementation
# ✅ Step 1: Clean Feature Names


# ✅ Step 2: Extract SHAP Values

# After:
# shap_values = explainer(X_transformed_df)

# Add:

# values = shap_values.values[0]
# feature_names = X_transformed_df.columns