import streamlit as st
import pandas as pd
import pickle

# Load cleaned dataset
data = pd.read_csv("cleaned_data.csv")

# Load trained model
model = pickle.load(open("RidgeModel.pkl", "rb"))

# App title
st.title("Bengaluru House Price Prediction")

st.write("Enter house details below")

# Location dropdown from dataset
location = st.selectbox(
    "Select Location",
    sorted(data["location"].unique())
)

# Dynamic limits from dataset
total_sqft = st.number_input(
    "Enter Total Square Feet",
    min_value=float(data["total_sqft"].min()),
    max_value=float(data["total_sqft"].max()),
    step=50.0
)

bath = st.number_input(
    "Enter Number of Bathrooms",
    min_value=int(data["bath"].min()),
    max_value=int(data["bath"].max()),
    step=1
)

bhk = st.number_input(
    "Enter BHK",
    min_value=int(data["bhk"].min()),
    max_value=int(data["bhk"].max()),
    step=1
)

# Predict button
if st.button("Predict Price"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "location": [location],
        "total_sqft": [total_sqft],
        "bath": [bath],
        "bhk": [bhk]
    })

    # Prediction
    prediction = model.predict(input_data)[0]

    # Avoid negative values
    prediction = max(0, prediction)

    # Show prediction
    st.success(
        f"Predicted House Price: ₹ {prediction:.2f} Lakhs"
    )