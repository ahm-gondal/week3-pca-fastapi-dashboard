"""
Week 3, Part 2 — FastAPI Dashboard for Steel Plant Energy Prediction
Run with:  uvicorn main:app --reload
Then open: http://127.0.0.1:8000
"""

import joblib
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Steel Plant Energy Dashboard")

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the saved pipeline (StandardScaler + PCA + Random Forest) and the
# exact feature-name list it was trained on — saved together in Week 3 Part 1.
bundle = joblib.load("model.joblib")
pipeline = bundle["pipeline"]
FEATURES = bundle["features"]

LOAD_TYPES = ["Light_Load", "Medium_Load", "Maximum_Load"]
DAYS = ["Friday", "Monday", "Saturday", "Sunday", "Thursday", "Tuesday", "Wednesday"]


def build_feature_row(numeric: dict, load_type: str, day_of_week: str) -> pd.DataFrame:
    """Build a single-row DataFrame with exactly the columns the model expects."""
    row = {f: 0.0 for f in FEATURES}
    row.update(numeric)
    # One-hot columns (drop_first=True at training dropped Light_Load and Friday,
    # so those are represented by all zeros)
    lt_col = f"Load_Type_{load_type}"
    if lt_col in row:
        row[lt_col] = 1.0
    day_col = f"Day_of_week_{day_of_week}"
    if day_col in row:
        row[day_col] = 1.0
    return pd.DataFrame([row], columns=FEATURES)


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/dashboard")
def dashboard(request: Request):
    charts = [
        ("energy_by_hour.png", "Average Energy Usage by Hour of Day"),
        ("energy_by_load_type.png", "Average Energy Consumption by Load Type"),
        ("correlation_heatmap.png", "Correlation Heatmap of Numerical Features"),
        ("cumulative_variance.png", "PCA — Cumulative Explained Variance (95% at 10 components)"),
    ]
    return templates.TemplateResponse(
        "dashboard.html", {"request": request, "charts": charts}
    )


@app.get("/predict")
def predict_form(request: Request):
    return templates.TemplateResponse(
        "predict.html",
        {"request": request, "load_types": LOAD_TYPES, "days": DAYS, "prediction": None},
    )


@app.post("/predict")
def predict(
    request: Request,
    lagging_reactive: float = Form(...),
    leading_reactive: float = Form(...),
    lagging_pf: float = Form(...),
    leading_pf: float = Form(...),
    nsm: float = Form(...),
    hour: int = Form(...),
    day_num: int = Form(...),
    month: int = Form(...),
    is_weekend: int = Form(...),
    pf_ratio: float = Form(...),
    load_type: str = Form(...),
    day_of_week: str = Form(...),
):
    numeric = {
        "Lagging_Current_Reactive.Power_kVarh": lagging_reactive,
        "Leading_Current_Reactive_Power_kVarh": leading_reactive,
        "Lagging_Current_Power_Factor": lagging_pf,
        "Leading_Current_Power_Factor": leading_pf,
        "NSM": nsm,
        "Hour": hour,
        "Day_Num": day_num,
        "Month": month,
        "Is_Weekend": is_weekend,
        "Power_Factor_Ratio": pf_ratio,
    }
    X = build_feature_row(numeric, load_type, day_of_week)
    prediction = float(pipeline.predict(X)[0])
    return templates.TemplateResponse(
        "predict.html",
        {
            "request": request,
            "load_types": LOAD_TYPES,
            "days": DAYS,
            "prediction": round(prediction, 2),
            "submitted": {**numeric, "Load_Type": load_type, "Day_of_week": day_of_week},
        },
    )
