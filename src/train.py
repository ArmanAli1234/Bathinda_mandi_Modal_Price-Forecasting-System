import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDRegressor
import joblib


# train the model

def train_model(X_train,y_train):

    model = SGDRegressor(
    loss="squared_error",
    penalty=None,
    learning_rate="constant",
    eta0=0.001,          # Learning Rate
    max_iter=20000,      # Epochs
    tol=1e-6,
    random_state=42
    )

    # fit the model
    model.fit(X_train,y_train)

    # save the model

    joblib.dump(model,"models/sgd_model.pkl")
    print("Save Model Completly")

    return model

