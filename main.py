"""
Steel Industry Energy Consumption — Week 3 FastAPI Dashboard
--------------------------------------------------------------
Loads the model.joblib pipeline saved by week3_pca.ipynb (feature engineering ->
one-hot + scaling -> PCA -> tuned XGBoost, all in one sklearn Pipeline) and serves:

  GET  /            Home page with nav bar
  GET  /dashboard   Static EDA charts (PNGs generated in Week 2/3 notebooks)
  GET  /predict     Prediction form (raw inputs)
  POST /predict     Runs the raw form input through the pipeline, returns Usage_kWh

Run with:  uvicorn main:app --reload
"""

from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Required import: model.joblib's pipeline references engineer_features by module
# path ("feature_engineering.engineer_features"). This import must succeed BEFORE
# joblib.load() runs below, or unpickling the pipeline will fail with:
#   AttributeError: Can't get attribute 'engineer_features' on <module '__main__' ...>
# feature_engineering.py must sit in this same folder, next to main.py.
from feature_engineering import engineer_features  # noqa: F401  (imported for pickle resolution)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"

app = FastAPI(title="Steel Industry Energy Consumption Dashboard")

# Mount static assets (charts, CSS)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Jinja2 templates
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# The pipeline is loaded once at startup and reused across requests.
model_pipeline = None

# Dropdown choices — must match the exact string values used to train the pipeline
# (i.e. the raw category labels found in the original dataset columns).
LOAD_TYPE_OPTIONS = ["Light_Load", "Medium_Load", "Maximum_Load"]
WEEK_STATUS_OPTIONS = ["Weekday", "Weekend"]
DAY_OF_WEEK_OPTIONS = ["Monday", "Tuesday", "Wednesday", "Thursday",
                        "Friday", "Saturday", "Sunday"]


@app.on_event("startup")
def load_model():
    """Load the trained pipeline once when the app starts."""
    global model_pipeline
    if not MODEL_PATH.exists():
        # App still starts so /dashboard and the empty form remain usable,
        # but /predict will report a clear error until model.joblib is placed here.
        print(f"[WARN] {MODEL_PATH} not found. Copy model.joblib from week3_pca.ipynb "
              f"into this app's folder before using /predict.")
        model_pipeline = None
        return
    model_pipeline = joblib.load(MODEL_PATH)
    print("Model pipeline loaded from", MODEL_PATH)


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request, "home.html", {"request": request})


@app.get("/dashboard")
def dashboard(request: Request):
    """
    Renders 3 static EDA charts as PNGs from static/.

    To generate these, add a `plt.savefig(...)` call at the end of the relevant
    Week 2 EDA cells (in eda.ipynb / steel_energy_full_pipeline notebook), e.g.:

        # Energy by Hour
        fig, ax = plt.subplots(figsize=(10, 5))
        df.groupby('hour')['Usage_kWh'].mean().plot(kind='bar', ax=ax, color='teal')
        ax.set_title('Average Usage_kWh by Hour')
        plt.tight_layout()
        plt.savefig('static/energy_by_hour.png', dpi=150)
        plt.show()

        # Energy by Load Type
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, x='Load_Type', y='Usage_kWh', ax=ax)
        ax.set_title('Usage_kWh by Load_Type')
        plt.tight_layout()
        plt.savefig('static/energy_by_load_type.png', dpi=150)
        plt.show()

        # Correlation Heatmap
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('static/correlation_heatmap.png', dpi=150)
        plt.show()

    Then copy the 3 PNGs into this app's static/ folder — this route just references
    the filenames, it doesn't generate them.
    """
    charts = [
        {"filename": "energy_by_hour.png", "title": "Average Usage by Hour"},
        {"filename": "energy_by_load_type.png", "title": "Usage by Load Type"},
        {"filename": "correlation_heatmap.png", "title": "Feature Correlation Heatmap"},
    ]
    return templates.TemplateResponse(
        request, "dashboard.html", {"request": request, "charts": charts}
    )


@app.get("/predict")
def predict_form(request: Request):
    return templates.TemplateResponse(
        request,
        "predict.html",
        {
            "request": request,
            "load_type_options": LOAD_TYPE_OPTIONS,
            "week_status_options": WEEK_STATUS_OPTIONS,
            "day_of_week_options": DAY_OF_WEEK_OPTIONS,
            "prediction": None,
            "error": None,
            "form_values": {},
        },
    )


@app.post("/predict")
def predict_submit(
    request: Request,
    date_time: str = Form(..., description="Datetime picker value, e.g. 2018-03-14T13:30"),
    lagging_reactive_power: float = Form(...),
    leading_reactive_power: float = Form(...),
    co2: float = Form(...),
    lagging_power_factor: float = Form(...),
    leading_power_factor: float = Form(...),
    nsm: float = Form(...),
    load_type: str = Form(...),
    week_status: str = Form(...),
    day_of_week: str = Form(...),
):
    form_values = {
        "date_time": date_time,
        "lagging_reactive_power": lagging_reactive_power,
        "leading_reactive_power": leading_reactive_power,
        "co2": co2,
        "lagging_power_factor": lagging_power_factor,
        "leading_power_factor": leading_power_factor,
        "nsm": nsm,
        "load_type": load_type,
        "week_status": week_status,
        "day_of_week": day_of_week,
    }

    context = {
        "request": request,
        "load_type_options": LOAD_TYPE_OPTIONS,
        "week_status_options": WEEK_STATUS_OPTIONS,
        "day_of_week_options": DAY_OF_WEEK_OPTIONS,
        "prediction": None,
        "error": None,
        "form_values": form_values,
    }

    if model_pipeline is None:
        context["error"] = ("model.joblib not found next to main.py — run week3_pca.ipynb "
                             "and copy the saved file into this app's folder.")
        return templates.TemplateResponse(request, "predict.html", context)

    try:
        parsed_date = datetime.fromisoformat(date_time)
    except ValueError:
        context["error"] = "Couldn't parse the date/time — please use the picker."
        return templates.TemplateResponse(request, "predict.html", context)

    # Build a single-row raw dataframe with the EXACT raw column names the pipeline
    # was trained on. The pipeline's own feature-engineering step (from week3_pca.ipynb)
    # derives hour/day/month/weekday/cyclical features from 'date' internally.
    raw_row = pd.DataFrame([{
        "date": parsed_date,
        "Lagging_Current_Reactive.Power_kVarh": lagging_reactive_power,
        "Leading_Current_Reactive_Power_kVarh": leading_reactive_power,
        "CO2(tCO2)": co2,
        "Lagging_Current_Power_Factor": lagging_power_factor,
        "Leading_Current_Power_Factor": leading_power_factor,
        "NSM": nsm,
        "Load_Type": load_type,
        "WeekStatus": week_status,
        "Day_of_week": day_of_week,
    }])

    try:
        pred = model_pipeline.predict(raw_row)[0]
        context["prediction"] = round(float(pred), 3)
    except Exception as exc:  # surfaced to the page rather than a raw 500
        context["error"] = f"Prediction failed: {exc}"

    return templates.TemplateResponse(request, "predict.html", context)
