import streamlit as st
import requests
import shap
import matplotlib.pyplot as plt
import numpy as np

st.title("Titanic Prediction (API Version)")
st.info("⏳ First request may take ~20–30 seconds (API waking up on Render)")

import os

ENV = os.getenv("ENV", "local")
print("App running in ",ENV)
if ENV == "local":
    PREDICTION_API_URL = "http://127.0.0.1:8000/predict"
    SHAP_API_URL = "http://127.0.0.1:8000/shap"
else:
    PREDICTION_API_URL = "https://ml-titanic-65cv.onrender.com/predict"
    SHAP_API_URL = "https://ml-titanic-65cv.onrender.com/shap"

print("PREDICTION_API_URL:",PREDICTION_API_URL)
print("SHAP_API_URL:",SHAP_API_URL)



def clean_feature_name(name, value, input_data):
    print("**clean_feature_name:Name:\n",name)
    print("**clean_feature_name:Value:\n",value)
    name = name.replace("cat__", "").replace("num__", "")

    # 🎯 Numeric features → show original value
    if name in ["Age", "Fare"]:
     original_value = input_data[0]
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
    # 🎯 Sex
    if "Sex_male" in name and value == 1:
        return "Male"

    if "Sex_female" in name and value == 1:
        return "Female"

    # 🎯 Pclass
    if "Pclass_" in name and value == 1:
        return f"Class {name.split('_')[-1]}"

    # 🎯 Embarked
    if "Embarked_" in name and value == 1:
        return f"Embarked {name.split('_')[-1]}"

    return None  # skip inactive features

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


def call_predict_api():
       print("**Run PREDICTION_API_URL")
       with st.spinner("Getting prediction from API..."): response = requests.post(PREDICTION_API_URL, json=payload, timeout=10)
       print("**Got response:\n",response.text)
       if response.status_code != 200:
          st.error(f"API Error: {response.text}")
       else:
        print("**response:\n",response)
        resp_json = response.json()
        print("**resp_json\n",resp_json)
        prediction = resp_json["prediction"]
        probability = resp_json["probability"]
        
        st.subheader("Result")
        st.write(f"Survival Probability: **{probability:.2f}**")

        if probability > 0.7:
          st.success("High confidence prediction")
        elif probability < 0.3:
          st.info("Low survival probability")
        else:
          st.warning("⚠️ Model uncertain")

        if prediction == 1:
          st.success(f"✅ Survives ({probability:.2%})")
        else:
          st.error(f"❌ Not Survive ({probability:.2%})")


def call_shap_api():
        print("**Run SHAP_API_URL")
        # 🧠 SHAP EXPLANATION
        st.subheader("🔍 Why this prediction? (SHAP)")
        with st.spinner("Getting SHAP values from API..."): shap_response = requests.post(SHAP_API_URL, json=payload, timeout=10)
        print("**Got response:\n",shap_response.json())
        if shap_response.status_code != 200:
          st.error(f"SHAP API Error: {shap_response.text}")
        else:

          shap_data = shap_response.json()
          values = np.array(shap_data["values"])      # keep 2D
          data = np.array(shap_data["data"])          # keep 2D
          base_values = np.array(shap_data["base_values"])
          feature_names = shap_data["feature_names"]
          print("**shap_values:\n",values)
          print("**base_value:\n",base_values)
          print("**feature_names:\n",feature_names)
          print("**data\n",data)

          build_shap_plot(feature_names, values[0], data[0])




          shap_exp = shap.Explanation(
               values=values,
               base_values=base_values,
               data=data,
               feature_names=feature_names
           )

          fig, ax = plt.subplots()
          shap.plots.waterfall(shap_exp[0], max_display=10, show=False)
          st.pyplot(fig)
          print("##################Transaction Completed ##############")

def build_shap_plot(feature_names,shap_values,data):
      print("**build_shap_plot")
      print("**feature_names:\n",feature_names)
      feature_impact = list(zip(feature_names, shap_values))
      # Sort by absolute importance
      feature_impact = sorted(feature_impact, key=lambda x: abs(x[1]), reverse=True)
      st.markdown("### 🧠 Top Factors Influencing Prediction")
      st.markdown("---")
      print("**feature_impactAll:\n",feature_impact)
      print("**feature_impact.shape:",np.array(feature_impact).shape)
      filtered_features = []
      for name, val in feature_impact:
        value = data[feature_names.index(name)]
        print("**name :\n",name)
        print("**val :\n",val)
        print("**value:\n",value)
        # Skip inactive one-hot features
        if ("Sex_" in name or "Pclass_" in name or "Embarked_" in name) and value == 0:
          continue
        clean_name = clean_feature_name(name,value,data)
        if clean_name is not None:
          filtered_features.append((clean_name, val))

        print("**filtered_features:\n",filtered_features)

      # 👉 Take top 3
      top_features = sorted(filtered_features, key=lambda x: abs(x[1]), reverse=True)[:3]

      print("**top_features:\n",top_features)

      # 👉 Display
      for name, val in top_features:
          emoji = get_feature_emoji(name)

          strength =val

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

   
# Inputs
pclass = st.selectbox("Pclass", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 30)
fare = st.slider("Fare", 0.0, 300.0, 50.0)
embarked = st.selectbox("Embarked", ["S", "C", "Q"])

print("** Wait for user input**")
if st.button("Predict"):
    print("**Get prediction based on user input")
    payload = {
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "Fare": fare,
        "Embarked": embarked
    }

    try:
        call_predict_api()
        call_shap_api() 
    except requests.exceptions.Timeout:
        st.error("API is taking too long (Render may be waking up)")
    except Exception as e:
        st.error(f"Error: {e}")



# run app using below command
# streamlit run titanic_streamlit_app_api.py