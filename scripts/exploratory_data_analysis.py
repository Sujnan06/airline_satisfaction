# exploratory_data_analysis.py

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 📍 Load the dataset safely
file_name = "../data/Invistico_Airline.csv"
file_path = os.path.join(os.path.dirname(__file__), file_name)

try:
    df = pd.read_csv(file_path)
    print(f"✅ Dataset loaded from: {file_path}")
except FileNotFoundError:
    print(f"❌ File not found at: {file_path}")
    exit()

# 🧾 Basic Information
print("\n🔍 Dataset Overview:")
print(df.info())
print("\n📊 Summary Statistics:")
print(df.describe(include='all'))

# 🔎 Missing Values
print("\n❓ Missing Values:")
print(df.isnull().sum())

plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values in Dataset")
plt.tight_layout()
plt.show()

# 🎯 Target Variable Distribution
if 'satisfaction' in df.columns:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x='satisfaction', palette='coolwarm')
    plt.title("Customer Satisfaction Distribution")
    plt.xlabel("Satisfaction Level")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ 'satisfaction' column not found!")

# 🔗 Correlation Matrix
plt.figure(figsize=(12, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()

# 📦 Boxplots for Outlier Detection
num_columns = ['Age', 'Flight Distance', 'Seat comfort', 'Departure/Arrival time convenient']
for col in num_columns:
    if col in df.columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(y=df[col], palette='coolwarm')
        plt.title(f'Boxplot of {col}')
        plt.tight_layout()
        plt.show()

# 📉 Categorical Feature Distributions
categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
for col in categorical_columns:
    if col in df.columns:
        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x=col, palette='coolwarm')
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# 📊 Bivariate Analysis with Satisfaction
cross_features = ['Customer Type', 'Type of Travel', 'Class']
for col in cross_features:
    if col in df.columns and 'satisfaction' in df.columns:
        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x=col, hue='satisfaction', palette='coolwarm')
        plt.title(f'Satisfaction vs {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# 🔍 Numerical Feature Distribution by Satisfaction
for col in num_columns:
    if col in df.columns and 'satisfaction' in df.columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(data=df, x=col, hue='satisfaction', kde=True, element='step', palette='coolwarm')
        plt.title(f'{col} Distribution by Satisfaction')
        plt.tight_layout()
        plt.show()

# 📈 Grouped Summary Statistics
if 'satisfaction' in df.columns:
    print("\n📌 Grouped Mean Statistics by Satisfaction:")
    print(df.groupby('satisfaction')[['Age', 'Flight Distance']].mean())

# 🧬 Check for Duplicates
print("\n🗃️ Duplicate Rows:", df.duplicated().sum())

# 🧱 Constant Features
print("\n🧊 Constant Columns:")
for col in df.columns:
    if df[col].nunique() == 1:
        print(f"- {col}")

# 🧮 Feature Engineering: Flight Distance Buckets
if 'Flight Distance' in df.columns:
    df['Flight Bucket'] = pd.cut(df['Flight Distance'], bins=5)
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x='Flight Bucket', hue='satisfaction', palette='coolwarm')
    plt.title("Satisfaction by Flight Distance Bucket")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 🌀 PCA Projection (for numeric features only)
print("\n🎯 PCA Projection (2 Components)...")
numeric_df = df.select_dtypes(include=['int64', 'float64']).dropna()

# Remove target if present
if 'satisfaction' in numeric_df.columns:
    numeric_df = numeric_df.drop(columns=['satisfaction'])

# Standardize
X_scaled = StandardScaler().fit_transform(numeric_df)

# Run PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

# Plot PCA results
if 'satisfaction' in df.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=df['satisfaction'], palette='coolwarm')
    plt.title("PCA Projection of Numerical Features")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.tight_layout()
    plt.show()

print("\n✅ Exploratory Data Analysis Completed!")
