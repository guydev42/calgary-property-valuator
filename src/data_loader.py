"""Data loading and preprocessing for Calgary Property Assessments dataset."""

import os
import logging
import pandas as pd
import numpy as np
from sodapy import Socrata

logger = logging.getLogger(__name__)

# The original dataset 6zp6-pxei now requires authentication.
# Use the public "Current Year Property Assessments (Parcel)" dataset instead.
DATASET_ID = "4bsw-nn7w"
DOMAIN = "data.calgary.ca"

COLUMNS_TO_KEEP = [
    "assessed_value", "property_class", "community",
    "land_use_designation",
]

# Mapping from API column names to expected column names.
API_COLUMN_RENAMES = {
    "assessment_class_description": "property_class",
    "comm_name": "community",
}


def fetch_property_assessments(limit=100000):
    """Fetch property assessment data from Calgary Open Data API."""
    logger.info("Fetching property assessment data from Socrata API (dataset %s)...", DATASET_ID)
    try:
        client = Socrata(DOMAIN, None, timeout=60)
        results = client.get(DATASET_ID, limit=limit)
        client.close()
        logger.info("Fetched %d records from API.", len(results))
        df = pd.DataFrame.from_records(results)
        # Rename API columns to the names expected by our pipeline
        df = df.rename(columns=API_COLUMN_RENAMES)
        return df
    except Exception as exc:
        logger.error("Failed to fetch property assessment data from Socrata API: %s", exc)
        raise


def _generate_sample_data(n=10000):
    """Generate realistic sample property assessment data for demo purposes.

    This is used as a fallback when the API is unavailable (e.g., auth required).
    """
    logger.warning("Generating sample property assessment data for demo purposes.")
    rng = np.random.RandomState(42)

    communities = [
        "BELTLINE", "DOWNTOWN COMMERCIAL CORE", "SUNALTA", "KENSINGTON",
        "HILLHURST", "BRIDGELAND/RIVERSIDE", "INGLEWOOD", "MISSION",
        "ALTADORE", "MOUNT ROYAL", "SIGNAL HILL", "TUSCANY",
        "PANORAMA HILLS", "CRANSTON", "MCKENZIE TOWNE", "AUBURN BAY",
        "MAHOGANY", "EVANSTON", "LEGACY", "SETON",
    ]
    land_uses = [
        "R-C1", "R-C2", "M-C1", "M-C2", "M-H1", "C-C1", "C-C2",
        "DC", "S-CI", "S-R", "R-CG", "M-U1",
    ]
    property_classes = ["Residential", "Non Residential"]

    community_col = rng.choice(communities, size=n)
    # Base value varies by community
    community_base = {c: rng.uniform(200000, 900000) for c in communities}
    base_values = np.array([community_base[c] for c in community_col])
    noise = rng.normal(1.0, 0.3, size=n).clip(0.3, 3.0)
    values = (base_values * noise).round(0)

    df = pd.DataFrame({
        "assessed_value": values,
        "property_class": rng.choice(property_classes, size=n, p=[0.85, 0.15]),
        "community": community_col,
        "land_use_designation": rng.choice(land_uses, size=n),
    })
    return df


def load_or_fetch_data(data_dir, limit=100000, force_refresh=False):
    """Load data from local CSV or fetch from API if not available."""
    csv_path = os.path.join(data_dir, "property_assessments.csv")
    if os.path.exists(csv_path) and not force_refresh:
        logger.info("Loading cached property assessment data from %s", csv_path)
        df = pd.read_csv(csv_path, low_memory=False)
        logger.info("Loaded %d records from cache.", len(df))
        return df

    try:
        df = fetch_property_assessments(limit=limit)
        os.makedirs(data_dir, exist_ok=True)
        df.to_csv(csv_path, index=False)
        logger.info("Cached %d records to %s", len(df), csv_path)
    except Exception as exc:
        logger.error("API fetch failed: %s", exc)
        if os.path.exists(csv_path):
            logger.warning("Falling back to cached property assessment data.")
            return pd.read_csv(csv_path, low_memory=False)
        # Generate sample data as last resort
        logger.warning(
            "No cached data available. Generating sample data for demo. "
            "Error was: %s", exc
        )
        df = _generate_sample_data(n=min(limit, 10000))
        os.makedirs(data_dir, exist_ok=True)
        df.to_csv(csv_path, index=False)
        logger.info("Saved sample data (%d records) to %s", len(df), csv_path)
    return df


def preprocess_data(df):
    """Clean and preprocess property assessment data for modeling."""
    df = df.copy()

    # Keep relevant columns that exist
    available_cols = [c for c in COLUMNS_TO_KEEP if c in df.columns]
    df = df[available_cols]

    # Convert assessed_value to numeric
    if "assessed_value" in df.columns:
        df["assessed_value"] = pd.to_numeric(df["assessed_value"], errors="coerce")

    # Remove rows without value or with zero/negative value
    df = df.dropna(subset=["assessed_value"])
    df = df[df["assessed_value"] > 0]

    # Remove extreme outliers (top/bottom 1%)
    lower = df["assessed_value"].quantile(0.01)
    upper = df["assessed_value"].quantile(0.99)
    df = df[(df["assessed_value"] >= lower) & (df["assessed_value"] <= upper)]

    # Fill missing categorical values
    for col in ["property_class", "community", "land_use_designation"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    return df


def engineer_features(df):
    """Create features for ML modeling."""
    df = df.copy()

    # Log transform of target variable
    df["log_value"] = np.log1p(df["assessed_value"])

    # Community-level aggregate features
    if "community" in df.columns:
        community_stats = df.groupby("community")["assessed_value"].agg(
            ["mean", "median", "count"]
        )
        community_stats.columns = [
            "community_avg_value",
            "community_median_value",
            "community_property_count",
        ]
        df = df.merge(community_stats, left_on="community", right_index=True, how="left")

    # Property class encoding is handled in model.py via LabelEncoder

    # Land use designation frequency
    if "land_use_designation" in df.columns:
        lu_freq = df["land_use_designation"].value_counts()
        df["land_use_frequency"] = df["land_use_designation"].map(lu_freq)

    return df
