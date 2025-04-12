import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

def get_cleaned_data():
    # Load dataset
    file_name = "../data/Invistico_Airline.csv"
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), file_name))
    df = pd.read_csv(file_path)
    
    # Drop missing values
    df.dropna(inplace=True)
    
    # Encode categorical columns
    label_encoders = {}
    categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Scale numerical columns
    numerical_columns = [
        "Age", "Flight Distance", "Seat comfort", "Departure/Arrival time convenient",
        "Food and drink", "Gate location", "Inflight wifi service", "Inflight entertainment",
        "Online support", "Ease of Online booking", "On-board service", "Leg room service",
        "Baggage handling", "Checkin service", "Cleanliness", "Online boarding",
        "Departure Delay in Minutes", "Arrival Delay in Minutes"
    ]
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    
    return df, label_encoders, scaler

# Only run this block when executing the script directly
if __name__ == "__main__":
    print("Script is running!")

    try:
        df, label_encoders, scaler = get_cleaned_data()
        print(f"Dataset loaded successfully! Shape: {df.shape}")
    except Exception as e:
        print(f"Error loading or processing dataset: {e}")
        exit()

    # Save processed data
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "processed_data.csv"), index=False)
    print("Processed data saved!")

    # Save encoders and scaler to models/
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models")
    os.makedirs(models_dir, exist_ok=True)
    
    with open(os.path.join(models_dir, 'label_encoders.pkl'), 'wb') as f:
        pickle.dump(label_encoders, f)
        print("Label encoders saved!")
    
    with open(os.path.join(models_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
        print("Scaler saved!")
