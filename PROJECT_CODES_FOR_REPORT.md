# 🌊 FloodGuard AI - Project Codes

This document contains all the current source code for the FloodGuard AI project, organized by file. You can copy these sections directly into your final report.

---

## 1. Backend Server (`app.py`)
The Flask application that handles routing, real-time satellite data fetching from Open-Meteo, and AI model serving.

```python
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os
import requests
from datetime import datetime

app = Flask(__name__)

# Load model and features
model_path = 'flood_model_fixed.pkl'
features_path = 'model_features.pkl'

if os.path.exists(model_path):
    model = joblib.load(model_path)
    features = joblib.load(features_path)
    # Load dataset for feature fetching
    df_full = pd.read_csv('flood_dataset_classification.xls')
else:
    model = None
    features = []
    df_full = None

@app.route('/')
def index():
    return render_template('index.html', features=features)

@app.route('/get_features', methods=['POST'])
def get_features():
    if df_full is None:
        return jsonify({'error': 'Dataset not loaded'}), 500
    
    try:
        data = request.json
        lat = float(data['Latitude'])
        lon = float(data['Longitude'])
        
        # 2. Get Terrain Data (Slope, Distance) from Nearest Point in Dataset
        distances = ((df_full['Latitude'] - lat)**2 + (df_full['Longitude'] - lon)**2)**0.5
        nearest_idx = distances.idxmin()
        nearest_point = df_full.iloc[nearest_idx]
        
        # 1. Fetch Real-time Data from Open-Meteo
        elev_url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=precipitation&current_weather=true"
        
        api_log = {"elevation": "Fetching...", "rainfall": "Fetching..."}
        
        try:
            elev_resp = requests.get(elev_url, timeout=5).json()
            elevation = float(elev_resp.get('elevation', [0])[0])
            api_log["elevation"] = f"{elevation}m (Source: Open-Meteo)"
            
            weather_resp = requests.get(weather_url, timeout=5).json()
            rainfall = float(weather_resp.get('current_weather', {}).get('precipitation', 0))
            if rainfall == 0 and 'hourly' in weather_resp:
                rainfall = float(weather_resp['hourly']['precipitation'][0])
            api_log["rainfall"] = f"{rainfall}mm (Source: Open-Meteo)"
        except Exception as e:
            api_log["error"] = str(e)
            # Fallback ONLY if API fails completely
            elevation = float(nearest_point['Elevation'])
            rainfall = float(nearest_point['Rainfall'])
        
        # 2. Assemble Features (STRICT REAL-TIME)
        result = {}
        for f in features:
            if f == 'Latitude': result[f] = lat
            elif f == 'Longitude': result[f] = lon
            elif f == 'Rainfall': result[f] = rainfall
            elif f == 'Elevation': result[f] = elevation
            elif f == 'time': result[f] = float(datetime.now().year)
            else:
                result[f] = float(nearest_point[f]) # Terrain only
        
        return jsonify({
            'features': result, 
            'api_log': api_log,
            'location_context': f"Nearest historical match was {distances.min():.4f} degrees away."
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        input_data = {f: [float(data[f])] for f in features}
        input_df = pd.DataFrame(input_data)
        
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])
        
        return jsonify({
            'prediction': prediction,
            'probability': probability,
            'result_text': "FLOOD LIKELY" if prediction == 1 else "NO FLOOD"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

---

## 2. Model Training (`train_model.py`)
The script used to train the Random Forest Classifier with balanced class weights to handle historical flood data.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# 1. Load Data
file_path = 'flood_dataset_classification.xls'
df = pd.read_csv(file_path)

# 2. Preprocessing
features = ['Latitude', 'Longitude', 'Rainfall', 'Elevation', 'Slope', 'distance', 'time']
target = 'occured'

df_clean = df[features + [target]].dropna()
X = df_clean[features]
y = df_clean[target]

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Training
model = RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# 5. Evaluation
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# 6. Save Model
joblib.dump(model, 'flood_model_fixed.pkl')
joblib.dump(features, 'model_features.pkl')
```

---

## 3. Frontend UI (`index.html`)
The main user interface featuring the interactive map and the real-time intelligence report panel.

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FloodGuard | Intelligent Prediction</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
</head>
<body>
    <div class="container">
        <div class="glass-card">
            <header>
                <h1>FloodGuard AI</h1>
                <p class="subtitle">Next-generation flood prediction using advanced machine learning</p>
            </header>
            <form id="predict-form">
                <section class="form-section">
                    <div id="map"></div>
                    <div class="input-grid">
                        <input type="number" step="any" id="Latitude" name="Latitude" placeholder="Latitude" required>
                        <input type="number" step="any" id="Longitude" name="Longitude" placeholder="Longitude" required>
                    </div>
                </section>
                <button type="submit" id="submit-btn">Generate Risk Analysis</button>
            </form>
            <div id="result-container">
                <div id="result-text" class="result-val">---</div>
                <div class="transparency-note">
                    <h3>Live AI Intelligence Report</h3>
                </div>
            </div>
        </div>
    </div>
    <script src="{{ url_for('static', filename='js/main.js') }}"></script>
</body>
</html>
```

---

## 4. Map & API Logic (`main.js`)
The JavaScript code that handles Leaflet map interactions and coordinates communication with the Flask backend.

```javascript
// Initialize Map
const map = L.map('map').setView([20, 0], 2);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);
let marker = L.marker([20, 0]).addTo(map);

const latInput = document.getElementById('Latitude');
const lonInput = document.getElementById('Longitude');

map.on('click', async (e) => {
    const { lat, lng } = e.latlng;
    latInput.value = lat.toFixed(6);
    lonInput.value = lng.toFixed(6);
    marker.setLatLng([lat, lng]);
    await fetchEnvironmentalData(lat, lng);
});

async function fetchEnvironmentalData(lat, lng) {
    const response = await fetch('/get_features', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ Latitude: lat, Longitude: lng })
    });
    const data = await response.json();
    if (response.ok) {
        for (const [key, value] of Object.entries(data.features)) {
            const input = document.getElementById(key);
            if (input) input.value = value.toFixed(2);
        }
        document.getElementById('predict-form').dispatchEvent(new Event('submit'));
    }
}
```

---

## 5. Modern Styling (`style.css`)
The CSS file implementing the premium Glassmorphism design and responsive layout.

```css
:root {
    --primary: #6366f1;
    --bg-gradient: radial-gradient(circle at top left, #0f172a, #1e1b4b);
    --glass: rgba(255, 255, 255, 0.05);
    --text-main: #f8fafc;
}
body {
    background: var(--bg-gradient);
    color: var(--text-main);
    font-family: 'Inter', sans-serif;
}
.glass-card {
    background: var(--glass);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 24px;
    padding: 3rem;
}
#map {
    height: 350px;
    border-radius: 20px;
    margin-bottom: 1.5rem;
}
```

---

## 6. Command Line Interface (`predict_cli.py`)
A standalone CLI tool for quick risk assessments without using the web interface.

```python
import joblib
import pandas as pd
import os

def predict_flood():
    model_path = 'flood_model_fixed.pkl'
    features_path = 'model_features.pkl'

    if not os.path.exists(model_path):
        print("Error: Model file not found.")
        return

    model = joblib.load(model_path)
    features = joblib.load(features_path)

    input_data = {}
    for feature in features:
        val = input(f"Enter {feature}: ")
        input_data[feature] = [float(val)]
    
    input_df = pd.DataFrame(input_data)
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    print(f"\nRESULT: {'FLOOD LIKELY' if prediction == 1 else 'NO FLOOD'}")
    print(f"Confidence: {probability:.2%}")

if __name__ == "__main__":
    predict_flood()
```
