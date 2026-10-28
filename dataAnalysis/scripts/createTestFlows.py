## createTestFlows.py

# load data
import datetime as dt
import pandas as pd
import os
import numpy as np

from sklearn.model_selection import TimeSeriesSplit 
from sklearn.preprocessing import StandardScaler
from scripts.imputation import impute

FULL_RANGE = pd.date_range(
    start="2022-01-01 00:00:00",
    end="2024-12-31 23:00:00",
    freq="h",
)

def create_test_set(data_dir: str,
    dataset_name: str,
    full_range=FULL_RANGE,
    noiseFac: float = 0.3,
    test_ratio: float = 0.15):

    '''Create test set to post to API.'''

    ## load traffic
    df = pd.read_csv(os.path.join(data_dir, dataset_name), sep=';')

    ## drop columns
    sublanco = df["sublanco"].iloc[0] if "sublanco" in df.columns else None
    sentido = df["sentido"].iloc[0] if "sentido" in df.columns else None
    df = df.drop(columns=["sublanco", "sentido"], errors="ignore")

    ## timestamp as index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').set_index('timestamp')

    df = df.reindex(full_range)
    df.index.name = 'timestamp'
    df = df.sort_index()

    df1 = impute(df, noiseFac)

    df1 = df1.iloc[168:]
    # features
    X = df1.drop(columns=["flows"])
    # target
    y = df1["flows"]

    # split into train and test
    n = len(X)
    test_start = int(n * (1-test_ratio)) -168

    # test and 168 h before test
    test_timestamps = X.index[test_start:]

    y_test = y.iloc[test_start:].reset_index(drop=True)

    return y_test, test_timestamps, sublanco, sentido