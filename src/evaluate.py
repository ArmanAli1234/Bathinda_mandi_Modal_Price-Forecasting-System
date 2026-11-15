import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from data_cleaning import clean_df
from sklearn.pipeline import Pipeline
from feature_engineering import create_feature
from data_preprocessing import preprocess_data
from train import train_model
from sklearn.linear_model import SGDRegressor
import joblib
from sklearn.metrics import mean_absolute_error,mean_squared_error,root_mean_squared_error,r2_score

PATH= "./data/proceed/clean_data.csv"
# load data
df = clean_df(PATH)

# create features
df = create_feature(df)

# apply preprocessing
X_train, y_train, X_test, y_test = preprocess_data(df)

# load preprocessor
preprocessor = joblib.load("models/preprocessor.pkl")

# create pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', SGDRegressor(
        loss="squared_error",
        penalty=None,
        learning_rate="constant",
        eta0=0.001,
        max_iter=20000,
        tol=1e-6,
        random_state=42
    ))
])

# train pipeline
pipeline.fit(X_train, y_train)

# prediction
y_pred = pipeline.predict(X_test)

# evaluation
print(f"MSE:- {mean_squared_error(y_test, y_pred)}")
print(f"MAE:- {mean_absolute_error(y_test, y_pred)}")
print(f"RMSE:- {root_mean_squared_error(y_test, y_pred)}")
print(f"r2_Score:- {r2_score(y_test, y_pred)}")

# save pipeline
joblib.dump(pipeline, "models/model.pkl")