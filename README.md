<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1e3a5f,100:2d8cf0&height=220&section=header&text=Calgary%20Property%20Assessment%20Valuator&fontSize=35&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=XGBoost%20%2B%20SHAP%20explainability%20for%20500K%2B%20property%20assessments&descSize=16&descAlignY=55&descColor=c8ddf0" width="100%" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/status-complete-2ea44f?style=for-the-badge" />
  <img src="https://img.shields.io/badge/license-MIT-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/XGBoost-regression-FF6600?style=for-the-badge" />
  <img src="https://img.shields.io/badge/SHAP-explainability-blueviolet?style=for-the-badge" />
  <img src="https://img.shields.io/badge/streamlit-dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
</p>

---

## Table of contents

- [Overview](#overview)
- [Results](#results)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Quickstart](#quickstart)
- [Dataset](#dataset)
- [Tech stack](#tech-stack)
- [Methodology](#methodology)
- [Acknowledgements](#acknowledgements)

---

## Overview

**Problem** -- Property assessments determine tax obligations for homeowners and businesses across Calgary. Understanding what drives assessed values is critical for owners who want to appeal unfair assessments and for real estate professionals who need to price listings accurately, yet the assessment methodology remains opaque to most stakeholders.

**Solution** -- This project predicts 500,000+ property assessment values using XGBoost regression and applies SHAP (SHapley Additive exPlanations) to provide transparent, per-prediction explanations of what drives each individual valuation, including community-level aggregates, land-use type, and property characteristics.

**Impact** -- Achieves an R-squared of 0.77 with a mean absolute error of $42,000, giving property owners and real estate professionals both accurate valuations and interpretable explanations of the key factors influencing each assessment.

---

## Results

| Metric | XGBoost | Gradient Boosting | Random Forest |
|--------|---------|-------------------|---------------|
| R-squared | **0.77** | 0.75 | 0.70 |
| MAE ($) | 42,000 | 45,000 | 50,000 |
| MAPE (%) | 16% | 18% | 20% |

---

## Architecture

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Calgary Open     │────▶│  Feature          │────▶│  XGBoost          │
│  Data (Socrata)   │     │  Engineering      │     │  Regression       │
│  617K+ records    │     │  - Log transform  │     └────────┬─────────┘
└──────────────────┘     │  - Community aggs │              │
                         └──────────────────┘     ┌────────▼─────────┐
                                                  │  SHAP             │
                                                  │  TreeExplainer    │
                                                  └────────┬─────────┘
                                                           │
                                                  ┌────────▼─────────┐
                                                  │  Streamlit        │
                                                  │  Dashboard        │
                                                  └──────────────────┘
```

---

<details>
<summary><strong>Project structure</strong></summary>

```
project_12_property_assessment_valuator/
├── app.py                  # Streamlit dashboard
├── requirements.txt        # Python dependencies
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

</details>

---

## Quickstart

```bash
# Clone the repository
git clone https://github.com/guydev42/calgary-property-valuator.git
cd calgary-property-valuator

# Install dependencies
pip install -r requirements.txt

# Fetch assessment data
python src/data_loader.py

# Launch the dashboard
streamlit run app.py
```

---

## Dataset

| Dataset | Source | Records | Key fields |
|---------|--------|---------|------------|
| Property assessments | Calgary Open Data | 617,000+ | Assessed value, land use, community, property type |

---

## Tech stack

<p align="center">
  <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-FF6600?style=flat-square" />
  <img src="https://img.shields.io/badge/SHAP-blueviolet?style=flat-square" />
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/sodapy-API-blue?style=flat-square" />
</p>

---

## Methodology

1. **Data collection** -- Fetched 617,000+ property assessment records from Calgary Open Data via Socrata API, including assessed values, land-use designations, community identifiers, and property types.
2. **Data cleaning** -- Removed duplicates, handled missing values, and applied log transformation to the right-skewed assessed value distribution to improve model performance.
3. **Feature engineering** -- Computed community-level aggregates (average and median assessed values, property counts), encoded land-use type frequencies, and created interaction features between property characteristics.
4. **Model training** -- Trained and compared Ridge Regression, Random Forest, Gradient Boosting, and XGBoost regressors, with XGBoost achieving the best R-squared of 0.77.
5. **Explainability** -- Applied SHAP TreeExplainer to generate global feature importance rankings and per-prediction waterfall plots showing how each feature contributes to an individual property's valuation.
6. **Dashboard** -- Built a Streamlit application with a property valuator tool, SHAP explanation visualizations, and community-level assessment comparisons.

---

## Acknowledgements

Data provided by the [City of Calgary Open Data Portal](https://data.calgary.ca/). This project was developed as part of a municipal data analytics portfolio.

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1e3a5f,100:2d8cf0&height=120&section=footer" width="100%" />
</p>

<p align="center">
  Built by <a href="https://github.com/guydev42">Ola K.</a>
</p>
