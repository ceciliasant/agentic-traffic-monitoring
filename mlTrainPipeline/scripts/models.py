## models.py
## model training script
## sarima, random forests (scikit-learn), eXtreme Gradient Boosting, lstm (keras)

from fastapi import params
import mlflow

import math
import warnings

import numpy as np

import optuna
# model libraries
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

EPOCHS = 50
def regression_metrics(true_y, preds):
    """Dict with mse, rmse, mae, smape, r²."""
    true_y = np.asarray(true_y)
    preds  = np.asarray(preds)

    mse  = mean_squared_error(true_y, preds)
    rmse = math.sqrt(mse)
    mae  = mean_absolute_error(true_y, preds)

    denominator = (np.abs(true_y) + np.abs(preds) + 1e-16)
    smape = np.mean(2.0 * np.abs(preds - true_y) / denominator)

    r2 = r2_score(true_y,preds)
   
    return {"mse": mse, "rmse": rmse, "mae": mae, "smape": smape, "r2": r2}

# ========================= XGboost =========================
 
def make_xgboost_objective(cv_folds):
    def objective(trial):
        """
        adapted from mlflow tutorial https://mlflow.org/docs/latest/ml/traditional-ml/tutorials/hyperparameter-tuning/notebooks/hyperparameter-tuning-with-child-runs/

        Tunes booster type, regularisation, tree depth, learning rate, gamma and grow policy.
        """
        try:
            with mlflow.start_run(nested=True):
                # hyperparameters
                params = {
                    "objective": "reg:squarederror",
                    "eval_metric": "rmse",
                    "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
                    # L2 reg
                    "lambda": trial.suggest_float("lambda", 0.01, 1.0, log=True),
                    # L1 reg
                    "alpha": trial.suggest_float("alpha", 0.01, 1.0, log=True),
                    # learning rate - prevent overfitting
                    "eta": trial.suggest_float("eta", 0.01, 1.0, log=True)
                }

                # tree based models (not gblinear)
                if params["booster"] == "gbtree" or params["booster"] == "dart":
                    # depth of tree
                    params["max_depth"] = trial.suggest_int("max_depth", 1, 9)
                    # minimum loss reduction
                    params["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
                    # way new nodes are added to the tree
                    params["grow_policy"] = trial.suggest_categorical(
                        "grow_policy", ["depthwise", "lossguide"]
                    )

                # Train XGBoost model
                fold_results = {metric: [] for metric in ["mse", "rmse", "mae", "smape", "r2"]}
                for fold in cv_folds:
                    dtrain = xgb.DMatrix(fold["train_x"], label=fold["train_y"])
                    dval   = xgb.DMatrix(fold["val_x"],   label=fold["val_y"])

                    bst = xgb.train(params, dtrain, num_boost_round=200, evals=[(dval, "val")], early_stopping_rounds=20,verbose_eval=False)  
                    preds = bst.predict(dval)

                    fold_metrics = regression_metrics(fold["val_y"], preds)
                    for metric, value in fold_metrics.items():
                        fold_results[metric].append(value)

                # average over folds            
                mean_rmse = float(np.mean(fold_results["rmse"]))
                
                means = {f"cv_mean_{m}": float(np.mean(fold_results[m])) for m in fold_results}
                stds  = {f"cv_std_{m}":  float(np.std(fold_results[m]))  for m in fold_results}

                mlflow.log_params(params)
                mlflow.log_metrics(means)
                mlflow.log_metrics(stds)

                for m in fold_results:
                    trial.set_user_attr(f"cv_mean_{m}", means[f"cv_mean_{m}"])
                    trial.set_user_attr(f"cv_std_{m}", stds[f"cv_std_{m}"])
                return mean_rmse
            
        except Exception:
            raise optuna.TrialPruned()
    return objective

def train_xgboost_final(best_params, dtrain, dtest, test_y):
    """Train XGBoost with best params, evaluate on test."""
    print(f" [XGBoost] training final model …")
    model = xgb.train(best_params, dtrain, num_boost_round=200, verbose_eval=False)
    preds = model.predict(dtest)
    return model, preds, regression_metrics(test_y, preds)
 
# ========================= Random Forest =========================

def make_rf_objective(cv_folds):
    """
    Optuna objective for sklearn RandomForestRegressor.
    Tunes n_estimators, max_depth, min_samples_split/leaf, max_features.
    """
    def objective(trial):
        with mlflow.start_run(nested=True):
            params = {
                # no. trees in forest
                "n_estimators":      trial.suggest_int("n_estimators", 50, 500, step=50),
                # max depth of each tree 
                "max_depth":         trial.suggest_int("max_depth", 2, 20),
                # minimum number of samples required to split an internal node
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                # minimum number of samples required to be at a leaf node
                "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
                # number of features to consider when looking for the best split
                "max_features":      trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", None]
                ),
                ## all processors
                "n_jobs": -1,
                "random_state": 42,
            }

            fold_results = {metric: [] for metric in ["mse", "rmse", "mae", "smape", "r2"]}
            for fold in cv_folds:
                model = RandomForestRegressor(**params)
                model.fit(fold["train_x"], fold["train_y"])
                preds = model.predict(fold["val_x"])

                fold_metrics = regression_metrics(fold["val_y"], preds)
                for metric, value in fold_metrics.items():
                    fold_results[metric].append(value)

            # average over folds            
            mean_rmse = float(np.mean(fold_results["rmse"]))
            
            means = {f"cv_mean_{m}": float(np.mean(fold_results[m])) for m in fold_results}
            stds  = {f"cv_std_{m}":  float(np.std(fold_results[m]))  for m in fold_results}

            mlflow.log_params(params)
            mlflow.log_metrics(means)
            mlflow.log_metrics(stds)

            for m in fold_results:
                trial.set_user_attr(f"cv_mean_{m}", means[f"cv_mean_{m}"])
                trial.set_user_attr(f"cv_std_{m}", stds[f"cv_std_{m}"])

            return mean_rmse
    return objective
 
 
def train_rf_final(best_params, train_x, test_x, train_y, test_y):
    """Train Random Forest with best params, evaluate on test."""
    
    print(f" [Random Forest] training final model …")
    best_params = {**best_params, "n_jobs": -1, "random_state": 42}
    model = RandomForestRegressor(**best_params)
    model.fit(train_x, train_y)
    preds = model.predict(test_x)
    return model, preds, regression_metrics(test_y, preds)
 
 
# ========================= SARIMA =========================

def make_sarima_objective(cv_folds):
    """
    Optuna objective for SARIMA.
    SARIMA(p, d, q)(P, D, Q, s)
    p: Autoregressive order
    d: Number of non-seasonal differences
    q: Moving average order
    P: Seasonal autoregressive order
    D: Seasonal differencing order
    Q: Seasonal moving average order
    s: Length of the seasonal period
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
 
    def objective(trial):
        with mlflow.start_run(nested=True):
            p = trial.suggest_int("p", 0, 3)
            d = 0 # stationary series
            q = trial.suggest_int("q", 0, 3)
            P = trial.suggest_int("P", 0, 1)
            D = 0
            Q = trial.suggest_int("Q", 0, 1)
            s = 24

            fold_results = {metric: [] for metric in ["mse", "rmse", "mae", "smape", "r2"]}
            for fold in cv_folds:
                train_s = fold["train_series"].values
                val_s = fold["val_series"].values
                try:
                    model = SARIMAX(
                        train_s,
                        order=(p, d, q),
                        seasonal_order=(P, D, Q, s),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(maxiter=50, disp=False)
                    preds = model.forecast(steps=len(val_s))
                    fold_metrics = regression_metrics(val_s, preds)
                    for metric, value in fold_metrics.items():
                        fold_results[metric].append(value)
                except Exception:
                    raise optuna.TrialPruned()

                
            # average over folds            
            mean_rmse = float(np.mean(fold_results["rmse"]))
            
            means = {f"cv_mean_{m}": float(np.mean(fold_results[m])) for m in fold_results}
            stds  = {f"cv_std_{m}":  float(np.std(fold_results[m]))  for m in fold_results}

            params_logged = {"p": p, "d": d, "q": q, "P": P, "D": D, "Q": Q, "s": s}
            mlflow.log_params(params_logged)
            mlflow.log_metrics(means)
            mlflow.log_metrics(stds)

            for m in fold_results:
                trial.set_user_attr(f"cv_mean_{m}", means[f"cv_mean_{m}"])
                trial.set_user_attr(f"cv_std_{m}", stds[f"cv_std_{m}"])


            return mean_rmse
    return objective
 
 
def train_sarima_final(best_params, train_series, test_series):
    """
    Best Sarima, evaluate on test.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    
    print(f" [SARIMA] training final model …")
    p = best_params["p"]
    d = best_params.get("d", 0) 
    q = best_params["q"]
    
    P = best_params["P"]
    D = best_params.get("D", 0)
    Q = best_params["Q"]
    s = best_params.get("s", 24)

    order = (p, d, q)
    seasonal_order = (P, D, Q, s)

    model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False).fit(maxiter=50, disp=False)

    preds = model.forecast(steps=len(test_series))
    return model, preds, regression_metrics(test_series, preds)
 
 
# ========================= LSTM =========================
# adapted from https://www.tensorflow.org/tutorials/structured_data/time_series#data_windowing

class WindowGenerator():
    def __init__(self, input_width, label_col_idx, train_df, val_df, batch_size=32, shift=1):

        import tensorflow as tf
        
        # window parameters
        self.input_width = input_width
        self.label_col_idx = label_col_idx
        self.batch_size = batch_size
        self.shift = shift

        self.total_window_size = input_width + shift

        self._train_ds = self.make_dataset(train_df.values.astype("float32"), shuffle=True)
        self._val_ds = self.make_dataset(val_df.values.astype("float32"), shuffle=False)
 
    def make_dataset(self, data, shuffle=False):
        import tensorflow as tf
 
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=shuffle,
            batch_size=self.batch_size,
        )
        return ds.map(self.split_window)
 
    def split_window(self, window):
        import tensorflow as tf
        inputs = window[:, :self.input_width, :]
        labels = window[:, self.shift:, self.label_col_idx]
        # (batch, input_width) - (batch, input_width, 1)
        labels = tf.expand_dims(labels, axis=-1)
 
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.input_width, 1])
        return inputs, labels
 
    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    @property
    def train(self):
        return self._train_ds
 
    @property
    def val(self):
        return self._val_ds
 
    def make_test_dataset(self, test_df):
        """test tf.data.Dataset, same window"""
        return self.make_dataset(test_df.values.astype("float32"), shuffle=False)

def build_lstm(input_width, n_features, units_1, units_2, dropout, lr):
    """
    Two-layer LSTM, w return_sequences=True
    """
    from tensorflow import keras
 
    model = keras.Sequential([
        # layer 1
        keras.layers.LSTM(units_1, return_sequences=True, input_shape=(input_width, n_features)),
        keras.layers.Dropout(dropout),
        # layer 2
        keras.layers.LSTM(units_2, return_sequences=True),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(1)])
    
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError()])
    
    return model

def make_lstm_objective(cv_folds):
    """
    Optuna objective for a Keras LSTM.
    Tunes look_back window, units in 2 LSTM layers, dropout, learning rate, batch size.
    """
    import tensorflow as tf
    from tensorflow import keras
 
    def objective(trial):
        with mlflow.start_run(nested=True):
            input_width = trial.suggest_int("input_width", 24, 168, step=24)
            units_1 = trial.suggest_int("units_1", 32, 256, step=32)
            units_2 = trial.suggest_int("units_2", 16, 128, step=16)
            dropout = trial.suggest_float("dropout", 0.0, 0.5)
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

            fold_results = {metric: [] for metric in ["mse", "rmse", "mae", "smape", "r2"]}
            for fold in cv_folds:
                
                # X and y in DataFrame for WindowGenerator
                import pandas as pd
                train_df = fold["train_x"].copy()
                train_df["flows"] = fold["train_y"].values
                val_df   = fold["val_x"].copy()
                val_df["flows"]   = fold["val_y"].values
 
                if len(train_df) <= input_width or len(val_df) <= input_width:
                    fold_results["rmse"].append(float("inf"))
                    for m in ["mse", "mae", "smape", "r2"]:
                        fold_results[m].append(None)
                    continue
 
                n_features    = train_df.shape[1]
                # flows in last column
                label_col_idx = n_features - 1   
 
                win = WindowGenerator(
                    input_width=input_width,
                    label_col_idx=label_col_idx,
                    train_df=train_df,
                    val_df=val_df,
                    batch_size=batch_size,
                )
 
                model = build_lstm(input_width, n_features, units_1, units_2, dropout, lr)

                history = model.fit(
                    win.train,
                    validation_data=win.val,
                    epochs=EPOCHS,
                    verbose=0,
                    callbacks=[keras.callbacks.EarlyStopping(
                        monitor="val_loss", patience=3,
                        restore_best_weights=True,
                    )],
                )

                for epoch, val_loss in enumerate(history.history["val_loss"]):
                    trial.report(val_loss, epoch)

                    if trial.should_prune():
                        raise optuna.TrialPruned()
 
        
                val_preds, val_true = [], []
                for x_batch, y_batch in win.val:
                    val_preds.append(model.predict(x_batch, verbose=0))
                    val_true.append(y_batch.numpy())
 
                # val_preds = np.concatenate(val_preds).flatten()
                # val_true  = np.concatenate(val_true).flatten()

                val_preds_all = np.concatenate(val_preds, axis=0)  # (N, input_width, 1)
                val_true_all  = np.concatenate(val_true,  axis=0)  # (N, input_width, 1)

                val_preds_flat = val_preds_all[:, -1, 0].flatten()
                val_true_flat  = val_true_all[:, -1, 0].flatten()

                fold_metrics = regression_metrics(val_true_flat, val_preds_flat)
                # fold_metrics = regression_metrics(val_true, val_preds)
                for metric, value in fold_metrics.items():
                    fold_results[metric].append(value)
            
            valid_rmse = [r for r in fold_results["rmse"] if math.isfinite(r)]
            if not valid_rmse:
                return float("inf")
            # average over folds            
            mean_rmse = float(np.mean(valid_rmse))
            
            means = {}
            stds = {}
            for m in fold_results:
                vals = [v for v in fold_results[m] if v is not None and math.isfinite(v)]
                means[f"cv_mean_{m}"] = float(np.mean(vals)) if vals else None
                stds[f"cv_std_{m}"]  = float(np.std(vals)) if vals else None

            params_logged = {
                "input_width": input_width, "units_1": units_1,
                "units_2": units_2, "dropout": dropout,
                "lr": lr, "batch_size": batch_size,
            }
            mlflow.log_params(params_logged)

            if math.isfinite(mean_rmse):
                clean_means = {k: v for k, v in means.items() if v is not None}
                clean_stds = {k: v for k, v in stds.items() if v is not None}
                
                mlflow.log_metrics(clean_means)
                mlflow.log_metrics(clean_stds)

                for m in fold_results:
                    if means[f"cv_mean_{m}"] is not None:
                        trial.set_user_attr(f"cv_mean_{m}", means[f"cv_mean_{m}"])
                    if stds[f"cv_std_{m}"] is not None:
                        trial.set_user_attr(f"cv_std_{m}", stds[f"cv_std_{m}"])

                mlflow.log_metric("cv_mean_rmse", mean_rmse)
                mlflow.log_metric("cv_std_rmse", float(np.std(valid_rmse)))

            return mean_rmse
    return objective
 
def train_lstm_final(best_params, full_x, test_x, full_y, test_y):
    """
    train with best params on full test data
    """
    import pandas as pd
    from tensorflow import keras
 
    print(f" [LSTM] training final model …")
    input_width = best_params["input_width"]
    units_1     = best_params["units_1"]
    units_2     = best_params["units_2"]
    dropout     = best_params["dropout"]
    lr          = best_params["lr"]
    batch_size  = best_params["batch_size"]
 
    # X + y
    # flows in last column
    train_df = full_x.copy()
    train_df["flows"] = full_y.values
    test_df  = test_x.copy()
    test_df["flows"]  = test_y.values
 
    n_features    = train_df.shape[1]
    label_col_idx = n_features - 1
 
    # window
    win = WindowGenerator(
        input_width=input_width,
        label_col_idx=label_col_idx,
        train_df=train_df,
        val_df=test_df,      
        batch_size=batch_size,
    )
 
    model = build_lstm(input_width, n_features, units_1, units_2, dropout, lr)
    history = model.fit(
        win.train,
        validation_data=win.val,
        epochs=EPOCHS,
        verbose=1,
        callbacks=[keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
        )],
    )
 
    test_ds = win.make_test_dataset(test_df)
    x_batches, y_batches = [], []
    for x_batch, y_batch in test_ds:
        x_batches.append(x_batch.numpy())
        y_batches.append(y_batch.numpy())

    import numpy as np
    x_all   = np.concatenate(x_batches, axis=0)
    true_y  = np.concatenate(y_batches, axis=0)  #(N, input_width, 1)
    # preds = model.predict(x_all, batch_size=batch_size, verbose=0).flatten()
 
    raw_preds = model.predict(x_all, batch_size=batch_size, verbose=0) #(N, input_width, 1)
    # (Samples,)
    preds = raw_preds[:, -1, 0].flatten() # last timestep prediction
    true_y = true_y[:, -1, 0].flatten() # last timestep true
    return model, preds, regression_metrics(true_y, preds), true_y, history.history
 
