
import pandas as pd
import numpy as np
import warnings
import os
import joblib

warnings.filterwarnings("ignore")

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    FunctionTransformer
)

from sklearn.compose import ColumnTransformer


# ==========================================
# 1. Cyclic Encoding Functions
# ==========================================

def month_encoding(x):
    return np.hstack([
        np.sin(2 * np.pi * x / 12),
        np.cos(2 * np.pi * x / 12)
    ])


def day_encoding(x):
    return np.hstack([
        np.sin(2 * np.pi * x / 7),
        np.cos(2 * np.pi * x / 7)
    ])


# ==========================================
# 2. Preprocessing Function
# ==========================================

def preprocess_data(df):

    df=df.sort_values("Arrival_Date",ascending=True)

    # Chronological Train-Test Split
    split = int(len(df) * 0.8)

    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()

    # Separate Features and Target
    X_train = train_df.drop(
        columns=["Modal_Price", "Arrival_Date"]
    )

    y_train = train_df["Modal_Price"]

    X_test = test_df.drop(
        columns=["Modal_Price", "Arrival_Date"]
    )

    y_test = test_df["Modal_Price"]

    # Display Shapes
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")

    # Identify Numerical Columns
    train_num = (
        X_train.drop(columns=["Month", "Day_of_week"])
        .select_dtypes(include=[np.number])
        .columns.tolist()
    )

    # Identify Categorical Columns
    train_cat = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # ======================================
    # 3. Create Column Transformer
    # ======================================

    preprocessor = ColumnTransformer(
        transformers=[

            ("num_cols",
             StandardScaler(),
             train_num),

            ("cat_cols",
             OneHotEncoder(
                 drop="first",
                 handle_unknown="ignore"
             ),
             train_cat),

            ("month_cyclic",
             FunctionTransformer(
                 month_encoding,
                 validate=False
             ),
             ["Month"]),

            ("day_of_week",
             FunctionTransformer(
                 day_encoding,
                 validate=False
             ),
             ["Day_of_week"])

        ],
        remainder="drop"
    )

    # save the preprocessor
    joblib.dump(preprocessor,"models/preprocessor.pkl")

    return X_train,y_train,X_test,y_test

print("\n Data Preprocesing Completed")
