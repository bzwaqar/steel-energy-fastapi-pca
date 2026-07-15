# ⚡ Steel Industry Energy Predictive Analytics Dashboard

An end-to-end Machine Learning web application built using **FastAPI**, **Scikit-Learn**, **Principal Component Analysis (PCA)**, and **XGBoost** to predict electrical energy consumption (`Usage_kWh`) in a steel industry environment.

The application combines automated feature engineering, dimensionality reduction, model inference, and interactive visualization into a single production-ready deployment pipeline.

---

## 🚀 Project Overview

This project predicts industrial energy consumption using historical operational data collected from a steel manufacturing facility.

The complete machine learning workflow includes:

- Data Preprocessing
- Automated Feature Engineering
- Categorical Encoding
- Feature Scaling
- Principal Component Analysis (PCA)
- Hyperparameter-Tuned XGBoost Regression
- FastAPI Deployment
- Interactive Prediction Interface
- Analytics Dashboard

The final trained model is serialized into a single `joblib` pipeline and loaded directly by the FastAPI application.

---

## ✨ Key Features

### Machine Learning Pipeline

- Automated Feature Engineering
- One-Hot Encoding
- Standard Scaling
- PCA Dimensionality Reduction
- Tuned XGBoost Regressor
- Single Serialized Production Pipeline
- Leakage-Free Training Workflow

### FastAPI Application

- Modern FastAPI Backend
- Dynamic Prediction Form
- Real-Time Energy Consumption Prediction
- Analytics Dashboard
- Responsive User Interface
- Production-Ready Routing Structure
- Reusable Model Loading Architecture

---

## 🧠 Machine Learning Workflow

```text
Raw Data
    ↓
Feature Engineering
    ↓
Categorical Encoding
    ↓
Feature Scaling
    ↓
Principal Component Analysis (PCA)
    ↓
XGBoost Regressor
    ↓
Predicted Usage_kWh
```

The entire workflow is encapsulated inside a single pipeline:

```python
joblib.dump(model_pipeline, "model.joblib")
```

This ensures that all preprocessing steps used during training are automatically applied during prediction.

---

## 📁 Project Structure

```text
fastapi_app/
│
├── main.py
├── feature_engineering.py
├── train_steel_model.py
├── model.joblib
├── requirements.txt
├── requirements_colab.txt
├── README.md
│
├── templates/
│   ├── index.html
│   ├── dashboard.html
│   └── predict.html
│
├── static/
│   ├── style.css
│   ├── energy_by_hour.png
│   ├── energy_by_load_type.png
│   └── correlation_heatmap.png
│
└── venv/
```

---

## 📊 Dashboard Analytics

The dashboard provides visual insights into industrial energy consumption patterns.

### Included Visualizations

#### Energy Consumption by Hour

Displays average electricity usage across different hours of the day.

#### Energy Consumption by Load Type

Shows energy usage distribution across:

- Light Load
- Medium Load
- Maximum Load

#### Correlation Heatmap

Visualizes relationships between operational variables and energy consumption.

---

## 🖥️ Application Routes

| Route | Method | Description |
|---------|---------|-------------|
| `/` | GET | Home Page |
| `/dashboard` | GET | Analytics Dashboard |
| `/predict` | GET | Prediction Form |
| `/predict` | POST | Generate Energy Prediction |

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd fastapi_app
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### 3. Install Dependencies

Recommended:

```bash
pip install -r requirements_colab.txt
```

or

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Application

Default Port:

```bash
python -m uvicorn main:app --reload
```

Custom Port:

```bash
python -m uvicorn main:app --reload --port 8001
```

---

## 🌐 Access Application

### Home Page

```text
http://127.0.0.1:8000
```

### Dashboard

```text
http://127.0.0.1:8000/dashboard
```

### Prediction Form

```text
http://127.0.0.1:8000/predict
```

---

## 🔮 Example Prediction Inputs

The prediction form accepts:

- Date & Time
- Lagging Reactive Power
- Leading Reactive Power
- CO₂ Emissions
- Lagging Power Factor
- Leading Power Factor
- NSM
- Load Type
- Week Status
- Day of Week

The model returns:

```text
Predicted Usage_kWh
```

---

## 🛠️ Recovery & Retraining

If the model becomes unavailable or needs retraining:

```bash
python train_steel_model.py
```

This script will:

- Retrain the PCA + XGBoost Pipeline
- Generate a New `model.joblib`
- Create Dashboard Visualizations
- Refresh Deployment Assets

---

## 📦 Technology Stack

### Backend

- FastAPI
- Starlette
- Jinja2

### Machine Learning

- Scikit-Learn
- PCA
- XGBoost
- Joblib

### Data Processing

- Pandas
- NumPy
- SciPy

### Visualization

- Matplotlib
- Seaborn

---

## 🎯 Project Objective

To accurately predict industrial electricity consumption (`Usage_kWh`) while reducing feature dimensionality through PCA and maintaining high predictive performance using XGBoost.

---

## 👨‍💻 Author

**Waqar Khan**

Machine Learning Engineer | Computer Vision Enthusiast

COMSATS University Islamabad

LinkedIn: www.linkedin.com/in/waqar-khan-9a7016321

GitHub: github.com/bbzwaqar

---

## 📄 License

This project is intended for educational, research, and portfolio purposes.
