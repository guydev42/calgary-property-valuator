"""ML model training and evaluation for Property Assessment Valuation."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os


CATEGORICAL_FEATURES = [
    "property_class", "community", "land_use_designation",
]

NUMERICAL_FEATURES = [
    "community_avg_value", "community_median_value",
    "community_property_count", "land_use_frequency",
]

TARGET = "log_value"


def prepare_model_data(df):
    """Prepare feature matrix and target vector for modeling."""
    df = df.copy()

    # Encode categorical features
    label_encoders = {}
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = df[col].fillna("Unknown")
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    # Select available features
    available_features = [
        c for c in CATEGORICAL_FEATURES + NUMERICAL_FEATURES if c in df.columns
    ]

    X = df[available_features].copy()
    y = df[TARGET].copy()

    # Fill missing numerical values with median
    for col in X.select_dtypes(include=[np.number]).columns:
        X[col] = X[col].fillna(X[col].median())

    # Remove any remaining NaN rows
    valid_mask = X.notna().all(axis=1) & y.notna()
    X = X[valid_mask]
    y = y[valid_mask]

    return X, y, label_encoders, available_features


def train_models(X, y, random_state=42):
    """Train multiple regression models and return results."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=10,
            random_state=random_state, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=random_state,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=7, learning_rate=0.1,
            random_state=random_state, n_jobs=-1,
        ),
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        if name == "Ridge Regression":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        # Clip predictions to training data range
        y_pred_clipped = np.clip(y_pred, y_train.min(), y_train.max())

        # Convert from log space back to original for interpretable metrics
        y_test_original = np.expm1(y_test)
        y_pred_original = np.expm1(y_pred_clipped)

        results[name] = {
            "MAE": mean_absolute_error(y_test_original, y_pred_original),
            "RMSE": np.sqrt(mean_squared_error(y_test_original, y_pred_original)),
            "R2": r2_score(y_test, y_pred),
            "MAPE": np.mean(
                np.abs((y_test_original - y_pred_original) / y_test_original)
            ) * 100,
        }
        trained_models[name] = model

    return trained_models, results, scaler, X_test, y_test


def explain_prediction(model, X_sample, feature_names=None):
    """Generate SHAP explanation for a prediction using TreeExplainer."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        return explainer, shap_values
    except Exception:
        return None, None


def get_feature_importance(model, feature_names, model_name="XGBoost"):
    """Extract feature importance from tree-based models."""
    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=False)
        return importance
    return pd.DataFrame()


def save_model(model, scaler, label_encoders, feature_names, model_dir):
    """Save trained model and preprocessing artifacts."""
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "best_model.joblib"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))
    joblib.dump(label_encoders, os.path.join(model_dir, "label_encoders.joblib"))
    joblib.dump(feature_names, os.path.join(model_dir, "feature_names.joblib"))


def load_model(model_dir):
    """Load trained model and preprocessing artifacts."""
    model = joblib.load(os.path.join(model_dir, "best_model.joblib"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    label_encoders = joblib.load(os.path.join(model_dir, "label_encoders.joblib"))
    feature_names = joblib.load(os.path.join(model_dir, "feature_names.joblib"))
    return model, scaler, label_encoders, feature_names
