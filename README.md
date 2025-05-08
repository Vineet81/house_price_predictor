DEMO- Link:
[Clich-Here](https://housepricepredictor-mgkbixntf6jnhs3ppxayyv.streamlit.app/)

# House Price Prediction with SHAP

This Streamlit web app predicts the median house price in California based on various features and interprets predictions using SHAP (SHapley Additive exPlanations). It uses machine learning models, including Linear Regression and XGBoost, to provide predictions and visualizations.

## Features
- **Prediction**: Predict the median house price based on features like median income, house age, average rooms, and more.
- **Model Choice**: Choose between Linear Regression and XGBoost models.
- **SHAP Explanations**: Get feature importance and a force plot to understand how each feature impacts the prediction.
- **CSV Upload**: Upload a CSV file containing the 8 required features for bulk predictions.

## Technologies
- **Streamlit**: For building the web app.
- **SHAP**: For interpreting model predictions.
- **XGBoost**: For building the prediction model.
- **Scikit-learn**: For Linear Regression model.
- **Pandas**: For data manipulation.
- **Joblib**: For saving and loading models.

## Prerequisites

Before running this app, ensure you have the following installed:

- Python 3.x
- Streamlit
- SHAP
- XGBoost
- Pandas
- Joblib
- Matplotlib

You can install the necessary libraries using `pip`:

```bash
pip install streamlit shap xgboost pandas joblib matplotlib
## Prerequisites

Before running this app, ensure you have the following installed:

- Python 3.x
- Streamlit
- SHAP
- XGBoost
- Pandas
- Joblib
- Matplotlib

You can install the necessary libraries using `pip`:

```bash
pip install streamlit shap xgboost pandas joblib matplotlib

How to Use

    Clone this repository or download the code files.

    Install the required dependencies as mentioned above.

    Run the Streamlit app:

streamlit run app.py

    Access the web app in your browser at http://localhost:8501.

Input Modes

    Manual Input: Use sliders to input values for the features and get predictions for a single house.

    CSV Upload: Upload a CSV file containing the 8 required features for bulk predictions.

Models

    Linear Regression: A simple model based on linear relationships between the features and target variable.

    XGBoost: A gradient boosting model for better prediction accuracy.

SHAP Visualizations

    Feature Importance: Visualize which features have the most impact on the predictions.

    Force Plot: Understand how individual features contribute to the model’s prediction.

Model Files

The models (xgb_model.pkl, lr_model.pkl) and scaler (scaler.pkl) are pre-trained and can be found in the project directory. If you want to train the models yourself, you can modify the code to train the models and save them using joblib.
Example of CSV for Bulk Prediction

Ensure the uploaded CSV contains the following columns:

    MedInc: Median income in block group

    HouseAge: Median house age

    AveRooms: Average number of rooms

    AveBedrms: Average number of bedrooms

    Population: Population in block

    AveOccup: Average house occupancy

    Latitude: Latitude of location

    Longitude: Longitude of location
    
    Acknowledgments

    SHAP for model interpretability.

    XGBoost for powerful gradient boosting models.

    Streamlit for building interactive web applications easily.


### Key Sections:
- **Overview**: Describes the app's functionality and the models used.
- **Technologies**: Lists the libraries and frameworks used.
- **How to Use**: Step-by-step guide on how to run the app locally.
- **Input Modes**: Describes the manual input and CSV upload options.
- **Models**: Brief overview of the models used (Linear Regression and XGBoost).
- **SHAP Visualizations**: Explains the SHAP-related visualizations available in the app.
- **Model Files**: Mentions the pre-trained models used and how they can be modified.
# house_price_predictor
