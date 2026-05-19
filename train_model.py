import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def train_flood_model():
    # 1. Load Data
    file_path = 'datasets/active_training_data.csv'
    print(f"Reading dataset from {file_path}...")

    # The file might be CSV or Excel
    try:
        df = pd.read_csv(file_path)
    except:
        df = pd.read_excel(file_path)

    # 2. Dynamic Preprocessing (Robust Column Matching)
    col_map = {}
    for col in df.columns:
        c = str(col).lower().strip()
        if c == 'latitude' or c == 'lat': col_map[col] = 'Latitude'
        elif c == 'longitude' or c == 'lon': col_map[col] = 'Longitude'
        elif 'rain' in c or 'precip' in c or 'annual' in c: col_map[col] = 'Rainfall'
        elif 'elev' in c: col_map[col] = 'Elevation'
        elif 'slope' in c: col_map[col] = 'Slope'
        elif 'dist' in c: col_map[col] = 'distance'
        elif 'occur' in c or c == 'flood' or 'target' in c or 'class' in c or c == 'floods': col_map[col] = 'occured'
    
    df = df.rename(columns=col_map)
    
    features = ['Latitude', 'Longitude', 'Rainfall', 'Elevation', 'Slope', 'distance']
    target = 'occured'
    
    missing = [f for f in features + [target] if f not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns. Missing: {missing}. Found: {df.columns.tolist()}")

    # Drop rows with missing values
    df_clean = df[features + [target]].dropna()

    X = df_clean[features]
    y = df_clean[target]

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Model Training
    print("Training Balanced Random Forest model...")
    model = RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    # 5. Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # 6. Save Model
    model_save_path = 'flood_model_fixed.pkl'
    joblib.dump(model, model_save_path)
    joblib.dump(features, 'model_features.pkl')
    
    import json
    metrics = {
        'accuracy': float(acc),
        'rows': len(df_clean)
    }
    with open('model_metrics.json', 'w') as f:
        json.dump(metrics, f)
    
    return f"Model successfully retrained! New Accuracy: {acc:.2%} on {len(df_clean)} rows."

if __name__ == '__main__':
    print(train_flood_model())
