import streamlit as st
import pandas as pd
import numpy as np
import joblib
import IPython
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components
from IPython.core.display import HTML

# Init SHAP JS
shap.initjs()

# Load assets
@st.cache_resource
def load_assets():
    scaler = joblib.load("scaler.pkl")
    xgb_model = joblib.load("xgb_model.pkl")
    lr_model = joblib.load("lr_model.pkl")
    features = joblib.load("features.pkl")
    return scaler, xgb_model, lr_model, features

scaler, xgb_model, lr_model, feature_names = load_assets()

# Feature tooltips
feature_tooltips = {
    'MedInc': "Median income in block group",
    'HouseAge': "Median house age",
    'AveRooms': "Average number of rooms",
    'AveBedrms': "Average number of bedrooms",
    'Population': "Population in block",
    'AveOccup': "Average house occupancy",
    'Latitude': "Latitude of location",
    'Longitude': "Longitude of location"
}

# Realistic input ranges for sliders
input_ranges = {
    'MedInc': (0.5, 15.0),
    'HouseAge': (1.0, 52.0),
    'AveRooms': (1.0, 10.0),
    'AveBedrms': (0.5, 5.0),
    'Population': (3, 35000),
    'AveOccup': (0.5, 7.0),
    'Latitude': (32.0, 42.0),
    'Longitude': (-125.0, -114.0)
}

# Sidebar
st.sidebar.title("⚙️ Settings")
model_choice = st.sidebar.selectbox("Choose Model", ("Linear Regression", "XGBoost"))

# Title
st.title("🏡 House Price Prediction with SHAP")
st.write("Predict California house prices and interpret predictions using SHAP and visual insights.")

# Input mode
input_mode = st.radio("Choose input mode", ("Manual Input", "CSV Upload"))

# Sample background data for SHAP
@st.cache_data
def get_background():
    df = pd.DataFrame(np.random.rand(100, len(feature_names)), columns=feature_names)
    return df

X_background = get_background()

if input_mode == "Manual Input":
    user_input = {}
    for feature in feature_names:
        min_val, max_val = input_ranges.get(feature, (0.0, 100.0))
        user_input[feature] = st.slider(
            f"{feature} ℹ️",
            min_value=float(min_val),
            max_value=float(max_val),
            value=float((min_val + max_val) / 2),
            step=0.1,
            help=feature_tooltips.get(feature, "")
        )

    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)

    if st.button("Predict"):
        # Common code for prediction
        model = lr_model if model_choice == "Linear Regression" else xgb_model
        prediction = model.predict(input_scaled)[0]
        st.success(f"💰 Predicted Median House Price: **${prediction * 100000:.2f}**")

        st.subheader("🔍 SHAP Explanation")

        # SHAP for Linear Regression
        if model_choice == "Linear Regression":
            explainer = shap.KernelExplainer(model.predict, X_background)
            shap_values = explainer.shap_values(input_df)
            force_plot = shap.force_plot(explainer.expected_value, shap_values[0], input_df, matplotlib=False)
            html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
            components.html(html, height=300, scrolling=True)

        # SHAP for XGBoost
        else:
            explainer = shap.Explainer(model, X_background)
            shap_values = explainer(input_df)

            # Summary plot
            st.write("Feature importance summary:")
            fig, ax = plt.subplots()
            shap.plots.bar(shap_values[0], show=False)
            st.pyplot(fig)

            # Force plot
            st.write("Feature impact on prediction:")
            force_plot = shap.force_plot(shap_values.base_values[0], shap_values[0].values, input_df, matplotlib=False)
            html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
            components.html(html, height=300, scrolling=True)

        # Additional visualization
        st.subheader("📊 Input Feature Distribution")
        fig2, ax2 = plt.subplots()
        sns.barplot(x=list(user_input.keys()), y=list(user_input.values()), ax=ax2)
        plt.xticks(rotation=45)
        st.pyplot(fig2)

else:
    uploaded_file = st.file_uploader("Upload CSV with all 8 features", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("📄 Uploaded Data Preview:", df.head())

        if all(col in df.columns for col in feature_names):
            X_scaled = scaler.transform(df[feature_names])
            model = lr_model if model_choice == "Linear Regression" else xgb_model
            predictions = model.predict(X_scaled)
            df["PredictedPrice"] = predictions * 100000
            st.dataframe(df)

            csv = df.to_csv(index=False).encode()
            st.download_button("⬇️ Download Predictions CSV", csv, "predictions.csv", "text/csv")
        else:
            st.error("CSV must contain these columns: " + ", ".join(feature_names))

