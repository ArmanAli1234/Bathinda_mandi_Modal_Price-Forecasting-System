# import libraries
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# load the clean data
def load(file_path):

    df=pd.read_csv(file_path)

    return df

# data cleaning

def data_cleaning(df):
    df=df.copy()

    # change the datatype of date into datetimes

    df["Arrival_Date"]=pd.to_datetime(
        df["Arrival_Date"],
        errors="coerce"
    )
     # info about data
    print(df.info())

    print(df.shape)    

    # check null value
    print(f"Null value:- ",np.sum(df.isnull().sum()))

    # remove all duplicates
    print(f"Duplicated report:- {df.duplicated().sum()}")
    df.drop_duplicates(inplace=True)

    # select not-null data
    df=df[
    (df['Max_Price']>=0) & (df["Min_Price"]>=0) & (df["Modal_Price"]>=0)
    ]

    # sort value on basis of date
    df=df.sort_values("Arrival_Date",ascending=True)

    # drop unwanted features
    df=df.drop(columns=["Max_Price","Min_Price","Day"])

    print(df.columns)
    return df


# final pipeline

def clean_df(file_path):
    df=load(file_path)
    df=data_cleaning(df)

    print("Date Cleaning completed")

    return df