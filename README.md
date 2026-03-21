# Property assessment valuator

## Problem statement
Property assessments determine tax obligations for homeowners and businesses across Calgary. Understanding what drives assessed values helps owners appeal unfair assessments and helps real estate professionals price listings accurately. This project predicts 500K+ property values with XGBoost and uses SHAP to explain what drives each individual valuation.

## Approach
- Fetched 617,000+ property assessments from Calgary Open Data (Socrata API)
- Cleaned, de-duplicated, and log-transformed the right-skewed value distribution
- Engineered community-level aggregates (avg/median value, property counts) and land-use frequencies
- Trained Ridge, Random Forest, Gradient Boosting, and XGBoost regressors
- Applied SHAP TreeExplainer for global feature importance and per-prediction waterfall plots
- Built a Streamlit dashboard with a property valuator and SHAP explainer

## Key results

| Metric | XGBoost | Gradient Boosting | Random Forest |
|--------|---------|-------------------|---------------|
| R-squared | **~0.77** | ~0.75 | ~0.70 |
| MAE ($) | ~42,000 | ~45,000 | ~50,000 |
| MAPE (%) | ~16% | ~18% | ~20% |

## How to run
```bash
pip install -r requirements.txt
python src/data_loader.py    # fetch assessment data
streamlit run app.py         # launch dashboard
```

## Project structure
```
project_12_property_assessment_valuator/
├── app.py                  # Streamlit dashboard
├── requirements.txt
├── README.md
├── data/                   # Cached CSV data
├── models/                 # Saved model artifacts
├── notebooks/
│   └── 01_eda.ipynb        # Exploratory data analysis
└── src/
    ├── __init__.py
    ├── data_loader.py      # Data fetching & feature engineering
    └── model.py            # Model training, evaluation & SHAP
```

## Technical stack
pandas, NumPy, scikit-learn, XGBoost, SHAP, Plotly, Streamlit, sodapy
