# FloodGuard: Real-Time Flood Prediction System 🌊

A machine learning-powered web application that predicts flood risks dynamically using coordinates (latitude and longitude). The application retrieves real-time weather and elevation data from Open-Meteo and predicts the flood probability using a Balanced Random Forest classifier.

### 🔗 Live Application URL
**[https://flood-prediction-system-p3op.onrender.com/](https://flood-prediction-system-p3op.onrender.com/)**

---

## 🚀 Key Features

* **Real-Time Data Integration**: Fetches current rainfall precipitation and elevation data automatically using the Open-Meteo API based on inputted geographic coordinates.
* **Balanced Random Forest Classifier**: Uses an ensemble machine learning model trained on historical spatial and meteorological datasets to classify flood probability.
* **Interactive Prediction Panel**: A simple, user-friendly interface for public users to estimate flood risks instantly.
* **Secure Admin Dashboard**: Allows administrators to:
  * Upload new training datasets (`.csv` format).
  * Retrain the machine learning model dynamically on the fly.
  * Monitor model evaluation metrics (e.g., accuracy, total trained dataset rows).
  * Visualize feature importances (e.g., Rainfall, Elevation, Slope, distance).
* **CLI Interface**: Includes a command-line tool (`predict_cli.py`) for quick console predictions.

---

## 🛠️ Technology Stack

* **Core Backend**: Python, Flask
* **Machine Learning**: Scikit-learn, Joblib, NumPy, Pandas
* **API Integrations**: Open-Meteo API (Elevation & Weather Forecast APIs)
* **Frontend**: HTML5, Vanilla CSS, JavaScript
* **Production Web Server**: Gunicorn
* **Hosting**: Render

---

## 💻 Local Setup & Development

### 1. Prerequisites
Make sure you have Python 3.10+ installed.

### 2. Install Dependencies
Clone the repository and install the required packages:
```bash
git clone https://github.com/VEERA-0206/Flood-Prediction-system.git
cd Flood-Prediction-system
pip install -r requirements.txt
```

### 3. Run the Web Server
Launch the Flask development server:
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000`.

### 4. Admin Credentials
To access the admin dashboard panel:
* **Username**: `admin`
* **Password**: `admin123`

---

## 📊 Model & Dataset Details

The model evaluates predictions using the following features:
1. **Latitude** & **Longitude**: Spatial location coordinates.
2. **Rainfall**: Real-time precipitation (mm) fetched dynamically.
3. **Elevation**: Altitude (meters) fetched dynamically.
4. **Slope** & **distance**: Terrain characteristics matched from the closest historical dataset records.
