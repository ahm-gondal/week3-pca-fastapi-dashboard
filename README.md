# Week 3 — Dimensionality Reduction (PCA) & FastAPI Dashboard

**Data Analysis & Business Understanding — Internship Task 3**

## Project Overview

Two parts, reflecting what ML engineers do before and after building a model:

- **Part 1 (`notebooks/week3_pca.ipynb`):** apply PCA to the Week 2 engineered steel-plant energy dataset, compare accuracy at 3 components vs the 95%-variance level vs the original features, and save the full trained pipeline (scaler + PCA + Random Forest) with joblib.
- **Part 2 (FastAPI app):** deploy that pipeline behind a web dashboard with EDA visualizations and a real-time energy-consumption prediction form.

## Dataset & Model

- **Dataset:** Steel Industry Energy Consumption ([UCI](https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption)) — 35,040 readings (every 15 min, 2018), engineered to 18 features in Week 2.
- **Model:** Random Forest — Week 2's best model by cross-validation RMSE.
- **Saved pipeline:** `model.joblib` = StandardScaler → PCA (10 components) → Random Forest, plus the exact training feature list. Nothing about preprocessing is lost at load time.

## PCA Results (Part 1)

| Version | RMSE (kWh) | R² |
|---|---|---|
| Original (18 features) | 0.877 | 0.9993 |
| PCA — 3 components (52.5% var) | 11.026 | 0.8930 |
| PCA — 10 components (95% var) | 4.605 | 0.9813 |
| Deployed pipeline (10 comp + compact forest) | 5.069 | 0.9774 |

Key findings: 10 of 18 components hold 95% of the variance (8 dimensions removable at a ~2% R² cost); 3 components lose too much; PCA is recommended for memory-constrained deployment — the compressed pipeline is only **5 MB**. Scaler and PCA were fitted on the training set only (no leakage). Full report inside the notebook.

## Environment Setup & Running the App

```bash
git clone https://github.com/ahm-gondal/week3-pca-fastapi-dashboard.git
cd week3-pca-fastapi-dashboard
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open **http://127.0.0.1:8000** in your browser.

## Routes

| Route | What it does |
|---|---|
| `/` | Welcome page with navigation bar |
| `/dashboard` | 4 EDA/PCA visualizations served from `static/` |
| `/predict` (GET) | Prediction form — one input per model feature |
| `/predict` (POST) | Returns the predicted energy consumption (kWh) on the same page |

## Project Structure

```
├── main.py                      # FastAPI app (loads model.joblib, 3 routes)
├── model.joblib                 # Saved pipeline: scaler + PCA + Random Forest
├── templates/                   # Jinja2 HTML: index, dashboard, predict
├── static/                      # Chart PNGs generated in the notebook
├── notebooks/
│   └── week3_pca.ipynb          # Part 1 — executed, all outputs visible
├── README.md
└── requirements.txt
```

## What I Learned

PCA trades accuracy for compactness in a controllable way; fitting preprocessing on training data only is non-negotiable; and saving the *whole pipeline* (not just the model) is what makes deployment safe — the FastAPI app never needs to re-implement scaling or PCA.

## Tools

Python 3 · scikit-learn · pandas · FastAPI · Uvicorn · Jinja2 · joblib · Matplotlib · Seaborn
