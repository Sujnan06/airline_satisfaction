import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data_cleaning import get_cleaned_data

# Get the cleaned data from data_cleaning script
df, label_encoders, scaler = get_cleaned_data()

# Splitting Data into X (features) and y (target)
X = df.drop(columns=['satisfaction'])
y = label_encoders['satisfaction'].transform(df['satisfaction'])

# Train-Test Split (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models and their respective hyperparameters for tuning
models = {
    "Logistic Regression": {
        "model": LogisticRegression(),
        "params": {"C": [0.001, 0.01, 0.1, 1, 10], "solver": ['lbfgs', 'liblinear'], "max_iter": [100, 200, 300]}  # Fixed the params dictionary
    },
    "Random Forest": {
        "model": RandomForestClassifier(),
        "params": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20, 30]}
    },
    "K-Nearest Neighbors": {
        "model": KNeighborsClassifier(),
        "params": {"n_neighbors": [3, 5, 7], "weights": ['uniform', 'distance']}
    },
    "Naive Bayes": {
        "model": GaussianNB(),
        "params": {}  # No hyperparameters for Naive Bayes
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(),
        "params": {"max_depth": [None, 10, 20, 30], "min_samples_split": [2, 5, 10]}
    }
}

# Hyperparameter tuning and evaluation
best_model = None
best_accuracy = 0
accuracies = {}

for name, model_info in models.items():
    print(f"Tuning {name}...")
    
    # Perform GridSearchCV for hyperparameter tuning
    grid = GridSearchCV(model_info["model"], model_info["params"], cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    # Get the best model and parameters
    best_model_name = grid.best_estimator_
    best_params = grid.best_params_
    
    # Evaluate the best model
    test_pred = grid.predict(X_test)
    accuracy = accuracy_score(y_test, test_pred)
    accuracies[name] = accuracy
    print(f"Best Parameters: {best_params}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", classification_report(y_test, test_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, test_pred))
    
    # Save the best model if it's the best performing so far
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = grid.best_estimator_  # Fix the assignment to `best_estimator_`
        
        # Ensure the model directory exists
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models")
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, "best_ml_model.pkl")
        joblib.dump(best_model, model_path)
        print(f"Best performing model ({name}) saved as best_ml_model.pkl")
feature_names = X.columns.tolist()
joblib.dump(feature_names, "scripts/../models/feature_names.pkl")
# Plot accuracy comparison for all models
# Ensure the results directory exists
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results")
os.makedirs(results_dir, exist_ok=True)

plt.figure(figsize=(10, 6))
plt.bar(accuracies.keys(), accuracies.values(), color='skyblue')
plt.xlabel('Models')
plt.ylabel('Accuracy Score')
plt.title('Comparison of Machine Learning Models')
plt.xticks(rotation=45)
plt.savefig(os.path.join(results_dir, "model_performance_comparison.png"))
plt.show()

print(f"Final best model saved with accuracy: {best_accuracy:.4f}")
joblib.dump(best_model, 'models/best_ml_model.pkl', compress=3)
