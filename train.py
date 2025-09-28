import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# --- 1. Load Data ---
try:
    file_path = 'C:/project/New folder/heart.csv'
    df = pd.read_csv(file_path)
    print(f"Successfully loaded data from {file_path}")

    # --- 2. Data Preprocessing ---
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Store feature names for later use in the prediction script
    feature_names = list(X.columns)

    # Split data. We don't need the test set here since we're training on all data
    # for the final model, but splitting is good practice for validation.
    # For the final saved model, we can train on the full dataset for better performance.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Data has been scaled.")

    # --- 3. Model Training ---
    # The concern about overfitting is addressed by using a RandomForestClassifier,
    # which is an ensemble method less prone to overfitting than single decision trees.
    # We train the final model on the entire dataset.
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    print("Model has been trained on the full dataset.")

    # --- 4. Save the Model and Scaler ---
    joblib.dump(model, 'model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(feature_names, 'feature_names.joblib') # Save feature names
    
    print("\nModel, scaler, and feature names have been saved successfully!")
    print("You can now run 'check_heart_disease.py' to make predictions.")

except FileNotFoundError:
    print(f"Error: The file was not found at the path '{file_path}'")
    print("Please make sure 'heart.csv' is in the correct directory.")
except Exception as e:
    print(f"An error occurred: {e}")