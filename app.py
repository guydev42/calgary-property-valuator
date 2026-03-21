"""
Calgary Property Assessment Valuator
Streamlit application for predicting property assessed values
with SHAP explainability using Calgary Open Data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))
from src.data_loader import load_or_fetch_data, preprocess_data, engineer_features
from src.model import (
    prepare_model_data, train_models, get_feature_importance,
    save_model, load_model,
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES,
)

# ── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Calgary Property Assessment Valuator",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom Styling ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6B7B8D;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 0.8rem;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Data Loading ────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


@st.cache_data(show_spinner="Loading property assessment data...")
def load_data():
    """Load and preprocess property assessment data."""
    df = load_or_fetch_data(DATA_DIR, limit=100000)
    df = preprocess_data(df)
    df = engineer_features(df)
    return df


@st.cache_resource(show_spinner="Training ML models...")
def train_all_models(data_hash):
    """Train models and cache results."""
    df = load_data()
    X, y, label_encoders, feature_names = prepare_model_data(df)
    trained_models, results, scaler, X_test, y_test = train_models(X, y)
    # Save best model (XGBoost)
    save_model(
        trained_models["XGBoost"], scaler, label_encoders, feature_names, MODEL_DIR
    )
    return trained_models, results, scaler, label_encoders, feature_names


# ── Main App ────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">Calgary Property Assessment Valuator</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">'
    "Predict property assessed values with SHAP explainability using ML on 617K+ assessments from Calgary Open Data"
    "</p>",
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Assessment Dashboard", "Property Valuator", "SHAP Explainer", "Model Performance", "About"],
)

# Load data
try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Ensure you have internet access to download the dataset, or place 'property_assessments.csv' in the data/ folder.")
    st.stop()

# ── Page: Assessment Dashboard ──────────────────────────────────────────────
if page == "Assessment Dashboard":
    st.header("Assessment Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Properties", f"{len(df):,}")
    with col2:
        st.metric("Avg Value", f"${df['assessed_value'].mean():,.0f}")
    with col3:
        st.metric("Median Value", f"${df['assessed_value'].median():,.0f}")
    with col4:
        st.metric("Communities", f"{df['community'].nunique()}")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Value Distribution", "Top Communities", "Raw Data"])

    with tab1:
        col_left, col_right = st.columns(2)

        with col_left:
            fig = px.histogram(
                df, x="assessed_value", nbins=50,
                title="Distribution of Assessed Values",
                labels={"assessed_value": "Assessed Value ($)"},
                color_discrete_sequence=["#667eea"],
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            fig = px.histogram(
                df, x="log_value", nbins=50,
                title="Distribution of Log-Transformed Values",
                labels={"log_value": "Log(Value + 1)"},
                color_discrete_sequence=["#764ba2"],
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

        if "property_class" in df.columns:
            fig = px.box(
                df, x="property_class", y="assessed_value",
                title="Value Distribution by Property Class",
                labels={"property_class": "Property Class", "assessed_value": "Assessed Value ($)"},
                color="property_class",
            )
            fig.update_layout(showlegend=False, height=450)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if "community" in df.columns:
            community_stats = df.groupby("community").agg(
                total_properties=("assessed_value", "count"),
                avg_value=("assessed_value", "mean"),
                median_value=("assessed_value", "median"),
            ).reset_index().sort_values("avg_value", ascending=False)

            top_n = st.slider("Show top N communities", 10, 50, 20)

            fig = px.bar(
                community_stats.head(top_n),
                x="community", y="avg_value",
                title=f"Top {top_n} Communities by Average Assessed Value",
                labels={"community": "Community", "avg_value": "Avg Value ($)"},
                color="total_properties",
                color_continuous_scale="Viridis",
            )
            fig.update_layout(xaxis_tickangle=-45, height=500)
            st.plotly_chart(fig, use_container_width=True)

            fig = px.bar(
                community_stats.sort_values("total_properties", ascending=False).head(top_n),
                x="community", y="total_properties",
                title=f"Top {top_n} Communities by Property Count",
                labels={"community": "Community", "total_properties": "Property Count"},
                color="avg_value",
                color_continuous_scale="Plasma",
            )
            fig.update_layout(xaxis_tickangle=-45, height=500)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Raw Data Sample")
        show_n = st.slider("Number of rows to display", 10, 500, 50)
        st.dataframe(df.head(show_n), use_container_width=True)

        csv = df.head(1000).to_csv(index=False)
        st.download_button(
            "Download Sample (1000 rows)",
            csv, "property_assessments_sample.csv", "text/csv",
        )

# ── Page: Property Valuator ─────────────────────────────────────────────────
elif page == "Property Valuator":
    st.header("Predict Property Value")
    st.markdown("Enter property details below to get an estimated assessed value.")

    try:
        trained_models, results, scaler, label_encoders, feature_names = train_all_models(
            str(len(df))
        )
    except Exception as e:
        st.error(f"Error training models: {e}")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        property_classes = sorted(df["property_class"].dropna().unique()) if "property_class" in df.columns else ["Residential"]
        selected_property_class = st.selectbox("Property Class", property_classes)

        communities = sorted(df["community"].dropna().unique()) if "community" in df.columns else []
        selected_community = st.selectbox("Community", communities) if communities else "Unknown"

    with col2:
        land_uses = sorted(df["land_use_designation"].dropna().unique()) if "land_use_designation" in df.columns else ["R-1"]
        selected_land_use = st.selectbox("Land Use Designation", land_uses)

    if st.button("Predict Value", type="primary", use_container_width=True):
        # Build input data
        input_data = {}

        # Encode categorical features
        cat_mapping = {
            "property_class": selected_property_class,
            "community": selected_community,
            "land_use_designation": selected_land_use,
        }
        for col in CATEGORICAL_FEATURES:
            if col in feature_names:
                val = cat_mapping.get(col, "Unknown")
                if col in label_encoders:
                    le = label_encoders[col]
                    if val in le.classes_:
                        input_data[col] = le.transform([val])[0]
                    else:
                        input_data[col] = 0
                else:
                    input_data[col] = 0

        # Numerical features
        num_defaults = {}
        if "community" in df.columns and selected_community:
            comm_data = df[df["community"] == selected_community]
            if len(comm_data) > 0:
                num_defaults["community_avg_value"] = comm_data["assessed_value"].mean()
                num_defaults["community_median_value"] = comm_data["assessed_value"].median()
                num_defaults["community_property_count"] = len(comm_data)
        if "land_use_designation" in df.columns:
            lu_freq = df["land_use_designation"].value_counts()
            num_defaults["land_use_frequency"] = lu_freq.get(selected_land_use, 0)

        for col in NUMERICAL_FEATURES:
            if col in feature_names:
                input_data[col] = num_defaults.get(col, 0)

        # Create DataFrame with correct column order
        input_df = pd.DataFrame([input_data])[feature_names]

        # Predict with XGBoost
        model = trained_models["XGBoost"]
        log_prediction = model.predict(input_df)[0]
        prediction = np.expm1(log_prediction)

        st.markdown("---")
        st.subheader("Valuation Results")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Estimated Value", f"${prediction:,.0f}")
        with col_b:
            low = prediction * 0.85
            high = prediction * 1.20
            st.metric("Likely Range", f"${low:,.0f} - ${high:,.0f}")
        with col_c:
            if selected_community and "community" in df.columns:
                comm_median = df[df["community"] == selected_community]["assessed_value"].median()
                if pd.notna(comm_median) and comm_median > 0:
                    diff_pct = ((prediction - comm_median) / comm_median) * 100
                    st.metric("vs Community Median", f"{diff_pct:+.1f}%")

        # Show all model predictions
        st.markdown("#### All Model Predictions")
        model_preds = {}
        for name, model_obj in trained_models.items():
            if name == "Ridge Regression":
                pred = np.expm1(model_obj.predict(scaler.transform(input_df))[0])
            else:
                pred = np.expm1(model_obj.predict(input_df)[0])
            model_preds[name] = pred

        pred_df = pd.DataFrame(
            [{"Model": k, "Predicted Value": f"${v:,.0f}"} for k, v in model_preds.items()]
        )
        st.dataframe(pred_df, use_container_width=True, hide_index=True)

# ── Page: SHAP Explainer ────────────────────────────────────────────────────
elif page == "SHAP Explainer":
    st.header("SHAP Model Explainability")
    st.markdown("Understand what drives property valuations using SHAP (SHapley Additive exPlanations).")

    try:
        trained_models, results, scaler, label_encoders, feature_names = train_all_models(
            str(len(df))
        )
    except Exception as e:
        st.error(f"Error training models: {e}")
        st.stop()

    model = trained_models["XGBoost"]

    # Prepare data for SHAP
    X, y, _, _ = prepare_model_data(df.copy())

    try:
        import shap

        st.subheader("Feature Contribution Analysis")

        # Compute SHAP values on a sample
        sample_size = min(500, len(X))
        X_sample = X.sample(sample_size, random_state=42)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # Summary bar plot
        st.subheader("Global Feature Importance (SHAP)")
        shap_importance = pd.DataFrame({
            "Feature": feature_names,
            "Mean |SHAP|": np.abs(shap_values).mean(axis=0),
        }).sort_values("Mean |SHAP|", ascending=False)

        fig = px.bar(
            shap_importance, x="Mean |SHAP|", y="Feature",
            orientation="h",
            title="Mean Absolute SHAP Values",
            color="Mean |SHAP|",
            color_continuous_scale="Blues",
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Beeswarm-style scatter plot
        st.subheader("SHAP Beeswarm Plot")
        for i, feat in enumerate(shap_importance["Feature"].head(8)):
            feat_idx = feature_names.index(feat)
            fig_data = pd.DataFrame({
                "SHAP Value": shap_values[:, feat_idx],
                "Feature Value": X_sample[feat].values,
            })
            fig = px.scatter(
                fig_data, x="SHAP Value", y=[feat] * len(fig_data),
                color="Feature Value",
                color_continuous_scale="RdBu_r",
                title=f"SHAP Values for {feat}",
            )
            fig.update_layout(height=150, showlegend=False, yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

        # Individual prediction explanation
        st.subheader("Individual Prediction Explanation")
        idx = st.number_input("Select sample index", 0, sample_size - 1, 0)
        sample_shap = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": shap_values[idx],
            "Feature Value": X_sample.iloc[idx].values,
        }).sort_values("SHAP Value", key=abs, ascending=False)

        fig = px.bar(
            sample_shap, x="SHAP Value", y="Feature",
            orientation="h",
            title=f"SHAP Waterfall for Sample #{idx}",
            color="SHAP Value",
            color_continuous_scale="RdBu_r",
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), height=500)
        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        st.warning("SHAP library not installed. Run `pip install shap` to enable explainability features.")
    except Exception as e:
        st.error(f"Error computing SHAP values: {e}")
        st.info("SHAP analysis requires a trained tree-based model. Ensure models are trained first.")

# ── Page: Model Performance ─────────────────────────────────────────────────
elif page == "Model Performance":
    st.header("Model Performance Comparison")

    try:
        trained_models, results, scaler, label_encoders, feature_names = train_all_models(
            str(len(df))
        )
    except Exception as e:
        st.error(f"Error training models: {e}")
        st.stop()

    # Results table
    results_df = pd.DataFrame(results).T
    results_df = results_df.round(2)
    results_df.columns = ["MAE ($)", "RMSE ($)", "R-squared", "MAPE (%)"]

    st.subheader("Model Comparison")
    st.dataframe(results_df, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            results_df.reset_index(),
            x="index", y="R-squared",
            title="R-squared Score by Model",
            labels={"index": "Model", "R-squared": "R-squared"},
            color="R-squared",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            results_df.reset_index(),
            x="index", y="MAE ($)",
            title="Mean Absolute Error by Model",
            labels={"index": "Model", "MAE ($)": "MAE ($)"},
            color="MAE ($)",
            color_continuous_scale="Reds_r",
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Actual vs Predicted scatter
    st.subheader("Actual vs Predicted (XGBoost)")
    X, y, _, _ = prepare_model_data(df.copy())
    _, X_t, _, y_t = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = trained_models["XGBoost"].predict(X_t)

    scatter_df = pd.DataFrame({
        "Actual (log)": y_t.values,
        "Predicted (log)": y_pred,
    })
    fig = px.scatter(
        scatter_df, x="Actual (log)", y="Predicted (log)",
        title="Actual vs Predicted Values (Log Scale)",
        opacity=0.3,
    )
    fig.add_trace(
        go.Scatter(
            x=[y_t.min(), y_t.max()],
            y=[y_t.min(), y_t.max()],
            mode="lines",
            name="Perfect Prediction",
            line=dict(color="red", dash="dash"),
        )
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    st.subheader("Feature Importance (XGBoost)")
    importance = get_feature_importance(
        trained_models["XGBoost"], feature_names, "XGBoost"
    )
    if not importance.empty:
        fig = px.bar(
            importance.head(15),
            x="Importance", y="Feature",
            orientation="h",
            title="Top Feature Importances",
            color="Importance",
            color_continuous_scale="Blues",
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), height=500)
        st.plotly_chart(fig, use_container_width=True)

# ── Page: About ─────────────────────────────────────────────────────────────
elif page == "About":
    st.header("About This Project")

    st.markdown("""
    ### Problem Statement
    Property assessments determine tax obligations for homeowners and businesses across
    Calgary. Understanding what drives assessed values helps property owners, real estate
    professionals, and city planners make informed decisions. This application uses machine
    learning to predict property values and SHAP to explain the key drivers behind each
    valuation.

    ### Dataset
    - **Source:** [Calgary Open Data - Property Assessments](https://data.calgary.ca/Government/Property-Assessments/6zp6-pxei)
    - **Dataset ID:** `6zp6-pxei`
    - **Records:** 617,000+ property assessments
    - **Features:** Assessed value, property class, community, land use designation

    ### Methodology
    1. **Data Preprocessing:** Converted values to numeric, removed zero/missing values,
       removed outliers (top/bottom 1%)
    2. **Feature Engineering:** Log-transformed values, created community-level aggregates
       (average, median, count), land use frequency
    3. **Models Trained:**
       - Ridge Regression (baseline)
       - Random Forest Regressor
       - Gradient Boosting Regressor
       - XGBoost Regressor (best performer)
    4. **Explainability:** SHAP TreeExplainer for feature contribution analysis
    5. **Evaluation:** MAE, RMSE, R-squared, MAPE

    ### Technical Stack
    - **Data Processing:** pandas, NumPy
    - **ML:** scikit-learn, XGBoost
    - **Explainability:** SHAP
    - **Visualization:** Plotly
    - **Web App:** Streamlit
    - **Data Access:** Socrata API (sodapy)

    ### Data Source & License
    Contains information licensed under the Open Government License - City of Calgary.
    Data accessed from [data.calgary.ca](https://data.calgary.ca/).
    """)

    st.markdown("---")
    st.markdown(
        "Built as part of the "
        "[Calgary Open Data ML/DS Portfolio](https://github.com/guydev42/calgary-data-portfolio)"
    )
