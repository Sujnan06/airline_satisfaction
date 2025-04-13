# ✈️ Airline Customer Satisfaction Predictor

A machine learning application for predicting airline passenger satisfaction based on travel experiences and service ratings.

## Project Overview

This project analyzes airline customer feedback data to identify factors influencing passenger satisfaction. Using various Random Forest, the system predicts whether a customer will be satisfied with their travel experience based on service quality metrics, flight details, and passenger demographics.

## Repository Structure

```
AIRLINE/
├── data/                            # Data files
│   ├── Invistico_Airline.csv        # Original dataset
│   └── processed_data.csv           # Preprocessed data
├── models/                          # Saved ML artifacts
│   ├── best_ml_model.pkl            # Optimized ML model
│   ├── label_encoders.pkl           # Categorical encoders
│   └── scaler.pkl                   # Feature scaler
├── results/                         # Analysis outputs
│   └── model_performance_comparison.png
└── scripts/                         # Core code modules
    ├── data_cleaning.py             # Data preprocessing
    ├── exploratory_data_analysis.py # EDA visualizations  
    ├── model_training.py            # Model selection & training
    ├── run_all.py                   # Pipeline orchestration
```

## Setup and Installation

```bash
# Clone repository
git clone 
cd airline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the complete pipeline
python scripts/run_all.py

# Launch the Streamlit prediction interface
streamlit run app.py
```

## Methodology

The project follows a standard data science workflow:

1. **Data Preprocessing**: Handling missing values, encoding categorical variables
2. **Exploratory Analysis**: Visualizing patterns and relationships in passenger feedback
3. **Model Development**: Training and evaluating multiple classifiers (Random Forest, Logistic Regression, etc.)
4. **Deployment**: Interactive web application for real-time predictions

## Key Results

The model successfully identifies the most significant factors affecting passenger satisfaction, achieving 96% prediction accuracy.



## Author
Sujnan J Kalmady

