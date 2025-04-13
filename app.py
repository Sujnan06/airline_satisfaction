import streamlit as st
import pandas as pd
import joblib

# Theme function
def apply_theme(theme):
    if theme == "Dark":
        st.markdown(
            """
            <style>
            .main {
                background-color: #333;
                color: white;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <style>
            .main {
                background-color: #ffffff;
                color: black;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

# Sidebar
st.sidebar.header("Settings")
theme = st.sidebar.radio("Choose Theme", ("Light", "Dark"))
apply_theme(theme)

# Load model and tools
model = joblib.load("scripts/../models/best_ml_model.pkl")
label_encoders = joblib.load("scripts/../models/label_encoders.pkl")
scaler = joblib.load("scripts/../models/scaler.pkl")
feature_names = joblib.load("scripts/../models/feature_names.pkl")

# App title
st.title("✈️ Airline Customer Satisfaction Classifier")
st.write("Predict whether a customer is satisfied based on their travel details.")

# User input
st.sidebar.header("Customer Information")

def user_input_features():
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    customer_type = st.sidebar.selectbox("Customer Type", ["Loyal Customer", "Disloyal Customer"])
    travel_type = st.sidebar.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
    travel_class = st.sidebar.selectbox("Class", ["Eco", "Eco Plus", "Business"])
    
    features = {
        "Gender": gender,
        "Customer Type": customer_type,
        "Type of Travel": travel_type,
        "Class": travel_class,
        "Age": st.sidebar.slider("Age", 7, 85, 30),
        "Flight Distance": st.sidebar.slider("Flight Distance", 31, 5000, 500),
        "Seat comfort": st.sidebar.slider("Seat comfort", 0, 5, 3),
        "Departure/Arrival time convenient": st.sidebar.slider("Departure/Arrival time convenient", 0, 5, 3),
        "Food and drink": st.sidebar.slider("Food and drink", 0, 5, 3),
        "Gate location": st.sidebar.slider("Gate location", 0, 5, 3),
        "Inflight wifi service": st.sidebar.slider("Inflight wifi service", 0, 5, 3),
        "Inflight entertainment": st.sidebar.slider("Inflight entertainment", 0, 5, 3),
        "Online support": st.sidebar.slider("Online support", 0, 5, 3),
        "Ease of Online booking": st.sidebar.slider("Ease of Online booking", 0, 5, 3),
        "On-board service": st.sidebar.slider("On-board service", 0, 5, 3),
        "Leg room service": st.sidebar.slider("Leg room service", 0, 5, 3),
        "Baggage handling": st.sidebar.slider("Baggage handling", 0, 5, 3),
        "Checkin service": st.sidebar.slider("Checkin service", 0, 5, 3),
        "Cleanliness": st.sidebar.slider("Cleanliness", 0, 5, 3),
        "Online boarding": st.sidebar.slider("Online boarding", 0, 5, 3),
        "Departure Delay in Minutes": st.sidebar.slider("Departure Delay in Minutes", 0, 1200, 0),
        "Arrival Delay in Minutes": st.sidebar.slider("Arrival Delay in Minutes", 0, 1200, 0),
    }

    return pd.DataFrame([features])

input_df = user_input_features()

# Columns
categorical_columns = ["Gender", "Customer Type", "Type of Travel", "Class"]
numerical_columns = [
    "Age", "Flight Distance", "Seat comfort", "Departure/Arrival time convenient",
    "Food and drink", "Gate location", "Inflight wifi service", "Inflight entertainment",
    "Online support", "Ease of Online booking", "On-board service", "Leg room service",
    "Baggage handling", "Checkin service", "Cleanliness", "Online boarding",
    "Departure Delay in Minutes", "Arrival Delay in Minutes"
]

# Encode categorical variables
def safe_transform(encoder, value):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        return -1

for col in categorical_columns:
    input_df[col] = input_df[col].astype(str)
    input_df[col] = input_df[col].fillna('Unknown')
    input_df[col] = input_df[col].apply(lambda x: safe_transform(label_encoders[col], x))

# Scale numerical features
input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

# Final DataFrame in the same order as training
input_data_for_prediction = input_df[categorical_columns + numerical_columns]
input_data_for_prediction = input_data_for_prediction[feature_names]  # Match training feature order

# Predict
prediction = model.predict(input_data_for_prediction)
prediction_proba = model.predict_proba(input_data_for_prediction)

# Display result
st.subheader("Prediction")
st.write("✅ Satisfied" if prediction[0] == 1 else "❌ Not Satisfied")
st.subheader("Prediction Probability")
st.write(f"Not Satisfied: {prediction_proba[0][0]:.2f}")
st.write(f"Satisfied: {prediction_proba[0][1]:.2f}")
