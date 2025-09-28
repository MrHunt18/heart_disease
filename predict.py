import joblib
import numpy as np

def get_user_input(feature_names):
    """
    Prompts the user to enter the 13 feature values.
    """
    print("--- Please Enter Patient Data ---")
    print("You will be asked for 13 medical features.")
    
    user_input = []
    
    # These descriptions will help the user understand what to input.
    feature_descriptions = {
        'age': 'Age in years',
        'sex': 'Sex (1 = male; 0 = female)',
        'cp': 'Chest pain type (0-3)',
        'trestbps': 'Resting blood pressure (in mm Hg)',
        'chol': 'Serum cholesterol in mg/dl',
        'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)',
        'restecg': 'Resting electrocardiographic results (0-2)',
        'thalach': 'Maximum heart rate achieved',
        'exang': 'Exercise induced angina (1 = yes; 0 = no)',
        'oldpeak': 'ST depression induced by exercise relative to rest',
        'slope': 'The slope of the peak exercise ST segment (0-2)',
        'ca': 'Number of major vessels (0-4) colored by fluoroscopy',
        'thal': 'Thalassemia (0 = normal; 1 = fixed defect; 2 = reversible defect)'
    }

    for feature in feature_names:
        while True:
            try:
                value = float(input(f"Enter value for {feature} ({feature_descriptions.get(feature, '')}): "))
                user_input.append(value)
                break
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    return np.array(user_input).reshape(1, -1)

def main():
    """
    Main function to load model and make prediction.
    """
    try:
        # --- 1. Load the Saved Model and Scaler ---
        model = joblib.load('model.joblib')
        scaler = joblib.load('scaler.joblib')
        feature_names = joblib.load('feature_names.joblib')
        print("Model and scaler loaded successfully.")

    except FileNotFoundError:
        print("Error: 'model.joblib' or 'scaler.joblib' not found.")
        print("Please run the 'train_model.py' script first to create these files.")
        return
    except Exception as e:
        print(f"An error occurred while loading files: {e}")
        return

    # --- 2. Get Input from User ---
    new_data = get_user_input(feature_names)

    # --- 3. Scale the Input and Predict ---
    # Scale the user's input using the loaded scaler
    new_data_scaled = scaler.transform(new_data)

    # Make the prediction
    prediction = model.predict(new_data_scaled)
    prediction_proba = model.predict_proba(new_data_scaled)

    # --- 4. Display the Result ---
    print("\n--- Prediction Result ---")
    
    if prediction[0] == 1:
        probability = prediction_proba[0][1]
        print(f"Prediction: The model predicts that this person is LIKELY to have heart disease.")
        print(f"Confidence: {probability:.2%}")
    else:
        probability = prediction_proba[0][0]
        print(f"Prediction: The model predicts that this person is NOT LIKELY to have heart disease.")
        print(f"Confidence: {probability:.2%}")

if __name__ == '__main__':
    main()