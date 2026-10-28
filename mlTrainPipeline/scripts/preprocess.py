## prepocess.py
## data preprocessing and cv + test split script

# load data
import datetime as dt
import pandas as pd
import os
import numpy as np

from sklearn.model_selection import TimeSeriesSplit 
from sklearn.preprocessing import StandardScaler

from scripts.impute import impute

def scale_fold(train_x, val_x, train_y, val_y):
    """
    StandardScaler
    """
    # features
    feat_cols = ["t","lag_1", "lag_24", "lag_168", "Avg_Temperature", "Avg_Precipitation"]

    cols_present = [c for c in feat_cols if c in train_x.columns]
 
    feat_scaler = StandardScaler()
    train_x = train_x.copy()
    val_x   = val_x.copy()
    train_x[cols_present] = feat_scaler.fit_transform(train_x[cols_present])
    val_x[cols_present]   = feat_scaler.transform(val_x[cols_present])
 
    # target
    flows_scaler = StandardScaler()
    train_y = pd.Series(
        flows_scaler.fit_transform(train_y.values.reshape(-1, 1)).flatten(),
        name="flows")

    val_y = pd.Series(
        flows_scaler.transform(val_y.values.reshape(-1, 1)).flatten(),
        name="flows")
 
    return train_x, val_x, train_y, val_y, feat_scaler, flows_scaler

def preprocess(data_dir: str,
    weather_dir: str,
    dataset_name: str,
    full_range,
    noiseFac: float = 0.3,
    test_ratio: float = 0.15):

    '''Preprocess dataset for model training. Imputation, feature engineering, scaling, cv split.'''

    ## load traffic
    df = pd.read_csv(os.path.join(data_dir, dataset_name), sep=';')

    ## drop columns
    df = df.drop(columns=["sublanco", "sentido"], errors="ignore")

    ## timestamp as index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').set_index('timestamp')

    df = df.reindex(full_range)
    df.index.name = 'timestamp'
    df = df.sort_index()

    ## imputation
    df1 = impute(df, noiseFac)

    ## feature engineering
    start = df1.index.min()
    df1["t"] = ((df1.index - start).total_seconds() / 3600).astype(int)

    df1['hour'] = df1.index.hour
    df1['day'] = df1.index.day
    df1['month'] = df1.index.month
    df1['year'] = df1.index.year

    df1['weekday'] = df1.index.weekday # 0=Monday, 6=Sunday

    df1["hour_sin"] = np.sin(2 * np.pi * df1["hour"] / 24)
    df1["hour_cos"] = np.cos(2 * np.pi * df1["hour"] / 24)

    df1["weekday_sin"] = np.sin(2 * np.pi * df1["weekday"] / 7)
    df1["weekday_cos"] = np.cos(2 * np.pi * df1["weekday"] / 7)

    df1["month_sin"] = np.sin(2 * np.pi * df1["month"] / 12)
    df1["month_cos"] = np.cos(2 * np.pi * df1["month"] / 12)

    df1['is_weekend']=(df1.index.weekday >= 5).astype(int)
    # # group weekday and hour into a variable from 0-167 values
    # df1['weekday_hour'] = df1['weekday'] * 24 + df1['hour']

    # holiday 2022-2024
    holidays_list = [
        dt.date(2022, 1, 1),   # ano novo
        dt.date(2022, 3, 1),   # carnaval
        dt.date(2022, 4, 15),  # 6º santa
        dt.date(2022, 4, 17),  # pascoa
        dt.date(2022, 4, 25),  # liberdade
        dt.date(2022, 5, 1),   # trabalhador
        dt.date(2022, 6, 10),   # dia portugal
        dt.date(2022, 6, 16),   # corpo deus
        dt.date(2022, 6, 24),   # sao joao
        dt.date(2022, 8, 15),   # assunçao
        dt.date(2022, 10, 5),   # implantação
        dt.date(2022, 11, 1),   # dia todos santos
        dt.date(2022, 12, 1), # restauração
        dt.date(2022, 12, 8),  # imaculada conceição
        dt.date(2022, 12, 25), # natal

        dt.date(2023, 1, 1),   # ano novo
        dt.date(2023, 2, 21),   # carnaval
        dt.date(2023, 4, 7),  # 6º santa
        dt.date(2023, 4, 9),  # pascoa
        dt.date(2023, 4, 25),  # liberdade
        dt.date(2023, 5, 1),   # trabalhador
        dt.date(2023, 6, 10),   # dia portugal
        dt.date(2023, 6, 8),   # corpo deus
        dt.date(2023, 6, 24),   # sao joao
        dt.date(2023, 8, 15),   # assunçao
        dt.date(2023, 10, 5),   # implantação
        dt.date(2023, 11, 1),   # dia todos santos
        dt.date(2023, 12, 1), # restauração
        dt.date(2023, 12, 8),  # imaculada conceição
        dt.date(2023, 12, 25), # natal

        dt.date(2024, 1, 1),   # ano novo
        dt.date(2024, 2, 13),   # carnaval
        dt.date(2024, 3, 29),  # 6º santa
        dt.date(2024, 3, 31),  # pascoa
        dt.date(2024, 4, 25),  # liberdade
        dt.date(2024, 5, 1),   # trabalhador
        dt.date(2024, 6, 10),   # dia portugal
        dt.date(2024, 5, 30),   # corpo deus
        dt.date(2024, 6, 24),   # sao joao
        dt.date(2024, 8, 15),   # assunçao
        dt.date(2024, 10, 5),   # implantação
        dt.date(2024, 11, 1),   # dia todos santos
        dt.date(2024, 12, 1), # restauração
        dt.date(2024, 12, 8),  # imaculada conceição
        dt.date(2024, 12, 25), # natal
    ]

    df1['is_holiday'] = df1.index.normalize().isin(pd.to_datetime(holidays_list)).astype(int)
    
    # lags 1, 24 and 168 (1 hour, 1 day and 1 week)
    df1['lag_1'] = df1['flows'].shift(1)
    df1['lag_24'] = df1['flows'].shift(24)
    df1['lag_168'] = df1['flows'].shift(168)
    # loose the 168 rows that have nan
    df1 = df1.dropna()
    
    ## add temperature and precipitation values
    df_weather = pd.read_csv(os.path.join(weather_dir, 'weather_data.csv'))
    
    df1 = df1.reset_index().merge(
        df_weather[['year', 'month', 'day', 'Avg_Temperature', 'Avg_Precipitation']],
        on=['year', 'month', 'day'],
        how='left').set_index('timestamp')

    # for sarima
    series = df1["flows"].copy()
    series = series.asfreq("h")

    df1 = df1.drop(columns=["hour","month","weekday"])
    # features
    X = df1.drop(columns=["flows"])
    # target
    y = df1["flows"]

    ## split into 5 folds and a final test

    # split into train and test
    n = len(X)
    test_start = int(n * (1-test_ratio))

    test_timestamps = X.index[test_start:]

    X_test = X.iloc[test_start:].reset_index(drop=True)
    y_test = y.iloc[test_start:].reset_index(drop=True)
    series_test = series.iloc[test_start:].reset_index(drop=True)

    X_cv = X.iloc[:test_start]
    y_cv = y.iloc[:test_start]
    series_cv = series.iloc[:test_start].reset_index(drop=True)

    # CV 5 folds
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)

    cv_folds = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_cv)):
        
        # not scaled
        ns_train_x = X_cv.iloc[train_idx].reset_index(drop=True)
        ns_val_x = X_cv.iloc[val_idx].reset_index(drop=True)
        ns_train_y = y_cv.iloc[train_idx].reset_index(drop=True)
        ns_val_y = y_cv.iloc[val_idx].reset_index(drop=True)

        # scaled folds, feature and target scaler
        train_x, val_x, train_y, val_y, feat_scaler, flows_scaler = scale_fold(ns_train_x, ns_val_x, ns_train_y, ns_val_y)
        
        cv_folds.append({
            "fold": fold,
            # full features
            "train_x":   train_x,
            "val_x":     val_x,
            "train_y":   train_y,
            "val_y":     val_y,
            # scaler
            "feat_scaler": feat_scaler,
            "tgt_scaler":  flows_scaler,
            # sarima series (not scaled!)
            "train_series": series_cv.iloc[train_idx].reset_index(drop=True),
            "val_series": series_cv.iloc[val_idx].reset_index(drop=True)})
    
    # test scaling
    feat_cols    = ["t", "lag_1", "lag_24", "lag_168", "Avg_Temperature", "Avg_Precipitation"]
    cols_present = [c for c in feat_cols if c in X_cv.columns]
 
    final_feat_scaler  = StandardScaler().fit(X_cv[cols_present])
    final_flows_scaler = StandardScaler().fit(y_cv.values.reshape(-1, 1))
 
    X_test_scaled = X_test.copy()
    X_test_scaled[cols_present] = final_feat_scaler.transform(X_test[cols_present])
 
    y_test_scaled = pd.Series(
        final_flows_scaler.transform(y_test.values.reshape(-1, 1)).flatten(),
        name="flows",
    )
 
    test_split = {
        "test_x": X_test_scaled,
        "test_y": y_test_scaled,
        "test_series": series_test,         # not scaled
        "feat_scaler": final_feat_scaler,   
        "tgt_scaler": final_flows_scaler,
        "test_timestamps": test_timestamps,
    }
 
    print(
        f"[preprocess] {dataset_name} — "
        f"cv={len(X_cv)}  test={len(X_test)}  folds={len(cv_folds)}"
    )

    return df1, series, cv_folds, test_split