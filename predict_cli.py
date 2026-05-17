import joblib
import pandas as pd
import os

def predict_flood():
    model_path = 'flood_model_fixed.pkl'
    features_path = 'model_features.pkl'

    if not os.path.exists(model_path):
        print("Error: Model file not found. Please run 'python train_model.py' first.")
        return

    # Load model and feature list
    model = joblib.load(model_path)
    features = joblib.load(features_path)

    print("\n--- Flood Prediction CLI ---")
    print("Please enter the following details for prediction:")
    
    input_data = {}
    try:
        for feature in features:
            val = input(f"Enter {feature}: ")
            input_data[feature] = [float(val)]
        
        # Create DataFrame for prediction
        input_df = pd.DataFrame(input_data)
        
        # Predict
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        print("\n--- Prediction Result ---")
        if prediction == 1:
            print(f"RESULT: FLOOD LIKELY (Probability: {probability:.2%})")
        else:
            print(f"RESULT: NO FLOOD (Probability of Flood: {probability:.2%})")
            
    except ValueError:
        print("\nError: Please enter valid numerical values.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    predict_flood()
