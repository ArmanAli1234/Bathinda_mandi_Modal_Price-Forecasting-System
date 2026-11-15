import pandas as pd
import numpy as np

# create features

def create_feature(df):


    group_columns=["Commodity","Market"]

    # lag features
    df["lag1"]=df.groupby(group_columns)["Modal_Price"].shift(1)
    df["lag7"]=df.groupby(group_columns)["Modal_Price"].shift(7)
    df["lag_30"]=df.groupby(group_columns)["Modal_Price"].shift(30)

    # rolling mean features

    df["rolling_mean7"]=df.groupby(group_columns)["Modal_Price"].transform(lambda x: x.shift(1).rolling(window=7).mean())

    df["rolling_mean30"]=df.groupby(group_columns)["Modal_Price"].transform(lambda x: x.shift(1).rolling(window=30).mean())

    # rolling std of last 7 days
    df["rolling_std7"]=df.groupby(group_columns)["Modal_Price"].transform(lambda x: x.shift(1).rolling(window=7).std())

    df.dropna(inplace=True)

    print("Feature Engineering Completed")

    return df


