from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "telco_churn.csv"

TARGET = "Churn"

NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

CAT_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]


def load_data(path=DATA_PATH):
    """Load the CSV and do the basic cleaning."""
    df = pd.read_csv(path)

    # customerID is just an identifier, not useful for prediction
    df = df.drop(columns=["customerID"], errors="ignore")

    # TotalCharges has blank strings (" ") for new customers -> convert to NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # SeniorCitizen is 0/1 -> make it Yes/No like the other categorical columns
    if df["SeniorCitizen"].dtype != object:
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # Target: Yes -> 1 , No -> 0
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})

    return df


def split_xy(df):
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    return X, y


def build_preprocessor():
    """
    Numeric columns     -> fill missing with median, then scale
    Categorical columns -> fill missing with most frequent, then one-hot encode

    Everything is inside a Pipeline, so the same steps are applied
    to train, test, and to new data coming from the app (no data leakage).
    """
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", num_pipe, NUM_COLS),
        ("cat", cat_pipe, CAT_COLS),
    ])
