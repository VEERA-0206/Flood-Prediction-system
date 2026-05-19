from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from functools import wraps
import joblib
import pandas as pd
import os
import requests
from datetime import datetime
from train_model import train_flood_model

app = Flask(__name__)
app.secret_key = 'master_admin_secret_key_123'

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Load model and features
model_path = 'flood_model_fixed.pkl'
features_path = 'model_features.pkl'

def load_resources():
    global model, features, df_full, model_metrics
    import json
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        features = joblib.load(features_path)
        try:
            df_full = pd.read_csv('datasets/active_training_data.csv')
        except FileNotFoundError:
            # Fallback if no active dataset has been uploaded yet
            df_full = pd.read_csv('datasets/flood_dataset_classification.xls')
        try:
            with open('model_metrics.json', 'r') as f:
                model_metrics = json.load(f)
        except:
            model_metrics = {'accuracy': 0.0, 'rows': len(df_full)}
    else:
        model = None
        features = []
        df_full = None
        model_metrics = None

load_resources()

@app.route('/')
def role_selection():
    return render_template('role_selection.html')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form.get('username') == 'admin' and request.form.get('password') == 'admin123':
            session['logged_in'] = True
            return redirect(url_for('admin'))
        else:
            error = 'Invalid credentials. Master access denied.'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('role_selection'))

@app.route('/predict_page')
def predict_page():
    return render_template('predict.html', features=features)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/admin', methods=['GET', 'POST'])
@login_required
def admin():
    message = None
    error = None
    if request.method == 'POST':
        if 'dataset' not in request.files:
            error = "No file part"
        else:
            file = request.files['dataset']
            if file.filename == '':
                error = "No selected file"
            elif file:
                try:
                    # Save the uploaded file as the active training dataset
                    file.save('datasets/active_training_data.csv')
                    
                    # Trigger retraining
                    from train_model import train_flood_model
                    message = train_flood_model()
                    
                    # Reload the model into memory
                    load_resources()
                except Exception as e:
                    error = f"Error during retraining: {str(e)}"
                    
    feature_importances = []
    feature_names = []
    if model is not None:
        feature_importances = list(model.feature_importances_)
        feature_names = list(features)
        
    return render_template('admin.html', 
                           message=message, 
                           error=error, 
                           metrics=model_metrics, 
                           feature_names=feature_names, 
                           feature_importances=feature_importances)

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
            api_log["elevation"] = f"{elevation}m"
            
            weather_resp = requests.get(weather_url, timeout=5).json()
            rainfall = float(weather_resp.get('current_weather', {}).get('precipitation', 0))
            if rainfall == 0 and 'hourly' in weather_resp:
                rainfall = float(weather_resp['hourly']['precipitation'][0])
            api_log["rainfall"] = f"{rainfall}mm"
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
            elif f == 'Rainfall': result[f] = rainfall # NO FALLBACK to historical if 0
            elif f == 'Elevation': result[f] = elevation
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
        # Convert input to DataFrame
        input_data = {f: [float(data[f])] for f in features}
        input_df = pd.DataFrame(input_data)
        
        # Prediction
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
