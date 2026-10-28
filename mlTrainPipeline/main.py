## main.py
## main script that calls preprocess, models, train, plots

# arima, random forests (scikit-learn), eXtreme Gradient Boosting (xgboost), lstm (keras)

## ============== LIBRARIES ==============

import os
import sys
import math

import joblib

import numpy as np
import mlflow
import optuna
import pandas as pd
import xgboost as xgb

## ============== FUNCTIONS ==============
from scripts.models import (
    make_xgboost_objective, train_xgboost_final,
    make_rf_objective, train_rf_final,
    make_sarima_objective, train_sarima_final,
    make_lstm_objective, train_lstm_final,
)
from scripts.plots import (
    plot_predictions, plot_residuals,
    plot_residual_diagnostics, plot_error_by_hour, plot_residual_qq,
    plot_lstm_learning_curve, plot_model_shap
)

from scripts.preprocess import preprocess
from scripts.train import champion_callback, get_or_create_experiment
from scripts.save_artefacts import save_detector_artefacts
from scripts.save_metrics import MetricsStore

## ============== PATHS/variables ==============
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR",    "/app/data")
WEATHER_DIR = os.environ.get("WEATHER_DIR", "/app/data/weather")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR",  "/app/output")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")

# jan 2022 - dec 2024
FULL_RANGE = pd.date_range(
    start="2022-01-01 00:00:00",
    end="2024-12-31 23:00:00",
    freq="h",
)
 
NOISE_FAC = 0.3
 
N_TRIALS = {
    "xgboost": 50,
    "random_forest":      50,
    "gradient_boosting":  50,
    "sarima":             10, 
    "lstm":               20,
}
 
optuna.logging.set_verbosity(optuna.logging.WARNING)
mlflow.set_tracking_uri(MLFLOW_URI)

## ============== UTILS ==============
def make_output_dirs(dataset_stem: str, model_name: str) -> str:
    """output/<dataset>/<model>/ directory."""
    path = os.path.join(OUTPUT_DIR, dataset_stem, model_name)
    os.makedirs(path, exist_ok=True)
    return path
 
 
def _full_cv_x(cv_folds) -> pd.DataFrame:
    """features every fold train + last fold vl"""
    frames = [f["train_x"] for f in cv_folds] + [cv_folds[-1]["val_x"]]
    return pd.concat(frames, ignore_index=True)
 
 
def _full_cv_y(cv_folds) -> pd.Series:
    """target """
    frames = [f["train_y"] for f in cv_folds] + [cv_folds[-1]["val_y"]]
    return pd.concat(frames, ignore_index=True)
 
 
def _full_cv_series(cv_folds) -> np.ndarray:
    """full CV series for SARIMA"""
    frames = [f["train_series"] for f in cv_folds] + [cv_folds[-1]["val_series"]]
    return pd.concat(frames, ignore_index=True).values

def inverse_metrics(preds, test_y, tgt_scaler, model_metrics):
    """
    regression metrics (inverse-transformed)
    """
    from scripts.models import regression_metrics
    preds_real  = tgt_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    true_real   = tgt_scaler.inverse_transform(
                      np.asarray(test_y).reshape(-1, 1)).flatten()
    return regression_metrics(true_real, preds_real)

## ============== MODELS ==============

## ============== XGBOOST ==============

def run_xgboost(cv_folds, test_split, experiment_id, dataset_stem, out_dir, store: MetricsStore):    
    print(" [XGBoost] tuning …")
    test_x = test_split["test_x"]
    test_y = test_split["test_y"]
 
    with mlflow.start_run(experiment_id=experiment_id,
                          run_name=f"{dataset_stem}_xgboost"):
 
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=5,
                interval_steps=1
            )
        )

        study.optimize(
            make_xgboost_objective(cv_folds),
            n_trials=N_TRIALS["xgboost"],
            callbacks=[champion_callback],
        )
        store.add_cv("xgboost", study, scaled=True)
        mlflow.log_metric("cv_mean_rmse", study.best_value)

        best    = study.best_params
        full_x  = _full_cv_x(cv_folds)
        full_y  = _full_cv_y(cv_folds)
 
        dtrain = xgb.DMatrix(full_x, label=full_y)
        dtest  = xgb.DMatrix(test_x, label=test_y)
 
        model, preds, metrics = train_xgboost_final(best, dtrain, dtest, test_y)
        metrics_real = inverse_metrics(preds, test_y, test_split["tgt_scaler"], metrics)
            
        store.add_test("xgboost", metrics, metrics_real)
        
        true_real = test_split["tgt_scaler"].inverse_transform(np.asarray(test_y).reshape(-1, 1)).flatten()
        preds_real = test_split["tgt_scaler"].inverse_transform(preds.reshape(-1, 1)).flatten()
        
        # ACF/PACF
        acf_path = os.path.join(out_dir, "xgboost_residual_acf_pacf.png")
        plot_residual_diagnostics(preds_real, true_real, lags=48, save_path=acf_path)
        mlflow.log_artifact(acf_path)
        
        # Q-Q Plot
        qq_path = os.path.join(out_dir, "xgboost_residual_qq.png")
        plot_residual_qq(preds_real, true_real, save_path=qq_path)
        mlflow.log_artifact(qq_path)
        
        # SHAP
        shap_path = os.path.join(out_dir, "xgboost_shap_summary.png")
        plot_model_shap(model, test_x, save_path=shap_path) 
        mlflow.log_artifact(shap_path)
        
        # Hourly Error Boxplot
        if "test_timestamps" in test_split:
            hourly_err_path = os.path.join(out_dir, "xgboost_hourly_error.png")
            plot_error_by_hour(preds_real, true_real, test_split["test_timestamps"], save_path=hourly_err_path)
            mlflow.log_artifact(hourly_err_path)

        mlflow.log_params(best)
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        mlflow.log_metrics({f"test_real_{k}": v for k, v in metrics_real.items()})

        mlflow.set_tags({
            "project": "Traffic Flow Forecasting",
            "optimizer_engine": "optuna",
            "model_family": "xgboost",
        })
 
        # fig_res  = plot_residuals(preds, test_y,
        #                           f"- {dataset_stem}/XGBoost",
        #                           save_path=os.path.join(out_dir, "residuals.png"))
        # fig_pred = plot_predictions(preds, test_y,
        #                             f"- {dataset_stem}/XGBoost",
        #                             save_path=os.path.join(out_dir, "predictions.png"))
        # fig_imp  = plot_xgb_feature_importance(model, best.get("booster", "gbtree"), f"- {dataset_stem}/XGBoost",
        #                             save_path=os.path.join(out_dir, "feature_importance.png"))
        # mlflow.log_figure(fig_res,  "residuals.png")
        # mlflow.log_figure(fig_pred, "predictions.png")
        # mlflow.log_figure(fig_imp,  "feature_importance.png")
 
        # model.save_model(os.path.join(out_dir, "model.ubj"))
        save_detector_artefacts(
            det_id= f"{dataset_stem}_xgboost", # detector name
            model=model,
            feat_scaler=test_split.get("feat_scaler"),
            tgt_scaler=test_split.get("tgt_scaler"),
            fmt="xgboost",
            feature_cols=list(full_x.columns),
            models_dir=os.path.abspath(out_dir)
        )
        
        mlflow.xgboost.log_model(model, name="model",
                                 input_example=full_x.iloc[[0]],
                                 model_format="ubj",
                                 metadata={"model_data_version": 1})
 
        print(f" [XGBoost] RMSE={metrics['rmse']:.4f}  "
              f"MAE={metrics['mae']:.4f}  R²={metrics['r2']:.4f}")
    return metrics
 
## ============== RANDOM FOREST ==============

def run_random_forest(cv_folds, test_split, experiment_id, dataset_stem, out_dir, store: MetricsStore):
    print(" [Random Forest] tuning …")
    test_x = test_split["test_x"]
    test_y = test_split["test_y"]
 
    with mlflow.start_run(experiment_id=experiment_id,
                          run_name=f"{dataset_stem}_random_forest"):
 
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=5,
                interval_steps=1
            )
        )

        study.optimize(
            make_rf_objective(cv_folds),
            n_trials=N_TRIALS["random_forest"],
            callbacks=[champion_callback],
        )
        store.add_cv("random_forest", study, scaled=True)

        mlflow.log_metric("cv_mean_rmse", study.best_value)

        best   = study.best_params
        full_x = _full_cv_x(cv_folds)
        full_y = _full_cv_y(cv_folds)
 
        model, preds, metrics = train_rf_final(best, full_x, test_x, full_y, test_y)
        metrics_real = inverse_metrics(preds, test_y, test_split["tgt_scaler"], metrics)

        store.add_test("random_forest", metrics, metrics_real)

        true_real = test_split["tgt_scaler"].inverse_transform(np.asarray(test_y).reshape(-1, 1)).flatten()
        preds_real = test_split["tgt_scaler"].inverse_transform(preds.reshape(-1, 1)).flatten()
        
        # ACF/PACF
        acf_path = os.path.join(out_dir, "rf_residual_acf_pacf.png")
        plot_residual_diagnostics(preds_real, true_real, lags=48, save_path=acf_path)
        mlflow.log_artifact(acf_path)
        
        # Q-Q Plot
        qq_path = os.path.join(out_dir, "rf_residual_qq.png")
        plot_residual_qq(preds_real, true_real, save_path=qq_path)
        mlflow.log_artifact(qq_path)
        
        # SHAP
        shap_path = os.path.join(out_dir, "rf_shap_summary.png")
        plot_model_shap(model, test_x, save_path=shap_path) 
        mlflow.log_artifact(shap_path)
        
        # Hourly Error Boxplot
        if "test_timestamps" in test_split:
            hourly_err_path = os.path.join(out_dir, "rf_hourly_error.png")
            plot_error_by_hour(preds_real, true_real, test_split["test_timestamps"], save_path=hourly_err_path)
            mlflow.log_artifact(hourly_err_path)

        mlflow.log_params(best)
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        mlflow.log_metrics({f"test_real_{k}": v for k, v in metrics_real.items()})

        mlflow.set_tags({
            "project": "Traffic Flow Forecasting",
            "optimizer_engine": "optuna",
            "model_family": "random_forest",
        })
 
        # fig_res  = plot_residuals(preds, test_y,
        #                           f"- {dataset_stem}/RF",
        #                           save_path=os.path.join(out_dir, "residuals.png"))
        # fig_pred = plot_predictions(preds, test_y,
        #                             f"- {dataset_stem}/RF",
        #                             save_path=os.path.join(out_dir, "predictions.png"))
        # mlflow.log_figure(fig_res,  "residuals.png")
        # mlflow.log_figure(fig_pred, "predictions.png")
 
        # joblib.dump(model, os.path.join(out_dir, "model.joblib"))

        save_detector_artefacts(
            det_id=f"{dataset_stem}_random_forest",
            model=model,
            feat_scaler=test_split.get("feat_scaler"),
            tgt_scaler=test_split.get("tgt_scaler"),
            fmt="sklearn",
            feature_cols=list(full_x.columns),
            models_dir=os.path.abspath(out_dir)
        )
        
        mlflow.sklearn.log_model(model, name="model",
                                 input_example=full_x.iloc[[0]])
 
        print(f" [Random Forest] RMSE={metrics['rmse']:.4f}  "
              f"MAE={metrics['mae']:.4f}  R²={metrics['r2']:.4f}")
    return metrics
 
## ============== SARIMA ==============

def run_sarima(cv_folds, test_split, experiment_id, dataset_stem, out_dir, store: MetricsStore):
    print(" [SARIMA] tuning …")
    test_series = test_split["test_series"].values
 
    with mlflow.start_run(experiment_id=experiment_id,
                          run_name=f"{dataset_stem}_sarima"):
 
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=5,
                interval_steps=1
            )
        )
        study.optimize(
            make_sarima_objective(cv_folds),
            n_trials=N_TRIALS["sarima"],
            callbacks=[champion_callback],
        )
        store.add_cv("sarima", study, scaled=False, note = "no scaler")

        mlflow.log_metric("cv_mean_rmse", study.best_value)

        best = study.best_params
        full_series = _full_cv_series(cv_folds)
 
        model, preds, metrics = train_sarima_final(best, full_series, test_series)
        store.add_test("sarima", metrics, metrics_real = None)

        true_real = (np.asarray(test_series).reshape(-1, 1)).flatten()
        preds_real = (preds.reshape(-1, 1)).flatten()
        
        # ACF/PACF
        acf_path = os.path.join(out_dir, "sarima_residual_acf_pacf.png")
        plot_residual_diagnostics(preds_real, true_real, lags=48, save_path=acf_path)
        mlflow.log_artifact(acf_path)
        
        # Q-Q Plot
        qq_path = os.path.join(out_dir, "sarima_residual_qq.png")
        plot_residual_qq(preds_real, true_real, save_path=qq_path)
        mlflow.log_artifact(qq_path)
        
        # Hourly Error Boxplot
        if "test_timestamps" in test_split:
            hourly_err_path = os.path.join(out_dir, "sarima_hourly_error.png")
            plot_error_by_hour(preds_real, true_real, test_split["test_timestamps"], save_path=hourly_err_path)
            mlflow.log_artifact(hourly_err_path)

        mlflow.log_params(best)
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        mlflow.set_tags({
            "project": "Traffic Flow Forecasting",
            "optimizer_engine": "optuna",
            "model_family": "sarima",
        })
 
        fig_res  = plot_residuals(preds, test_series,
                                  f"- {dataset_stem}/SARIMA",
                                  save_path=os.path.join(out_dir, "residuals.png"))
        fig_pred = plot_predictions(preds, test_series,
                                    f"- {dataset_stem}/SARIMA",
                                    save_path=os.path.join(out_dir, "predictions.png"))
        mlflow.log_figure(fig_res,  "residuals.png")
        mlflow.log_figure(fig_pred, "predictions.png")
 
        # joblib.dump(model, os.path.join(out_dir, "model.pkl"))

        save_detector_artefacts(
            det_id=f"{dataset_stem}_sarima",
            model=model,
            feat_scaler=None,
            tgt_scaler=None,
            fmt="sarima",
            feature_cols=[],
            models_dir=os.path.abspath(out_dir)
        )
 
        print(f" [SARIMA] ({best['p']},0,{best['q']})"
              f"({best['P']},0,{best['Q']},24)  "
              f"RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}")
    return metrics
 
## ============== LSTM ==============

def run_lstm(cv_folds, test_split, experiment_id, dataset_stem, out_dir, store: MetricsStore):
    print(" [LSTM] tuning …")
    test_x = test_split["test_x"]
    test_y = test_split["test_y"]
 
    with mlflow.start_run(experiment_id=experiment_id,
                          run_name=f"{dataset_stem}_lstm"):
 
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=5,
                interval_steps=1
            )
        )
    
        study.optimize(
            make_lstm_objective(cv_folds),
            n_trials=N_TRIALS["lstm"],
            callbacks=[champion_callback],
        )
        store.add_cv("lstm", study, scaled=True)

        mlflow.log_metric("cv_mean_rmse", study.best_value)

        best   = study.best_params
        full_x = _full_cv_x(cv_folds)
        full_y = _full_cv_y(cv_folds)
 
        model, preds, metrics, true_y, history_dict= train_lstm_final(best, full_x, test_x, full_y, test_y)
        metrics_real = inverse_metrics(preds, true_y, test_split["tgt_scaler"], metrics)

        store.add_test("lstm", metrics, metrics_real)

        # true_real = test_split["tgt_scaler"].inverse_transform(np.asarray(test_y).reshape(-1, 1)).flatten()
        preds_real = test_split["tgt_scaler"].inverse_transform(preds.reshape(-1, 1)).flatten()
        true_real  = test_split["tgt_scaler"].inverse_transform(true_y.reshape(-1, 1)).flatten()            
        
        # lstm learning curve
        lstm_curve_path = os.path.join(out_dir, "lstm_learning_curve.png")
        plot_lstm_learning_curve(history_dict, save_path=lstm_curve_path)
        mlflow.log_artifact(lstm_curve_path)

        # ACF/PACF
        acf_path = os.path.join(out_dir, "lstm_residual_acf_pacf.png")
        plot_residual_diagnostics(preds_real, true_real, lags=48, save_path=acf_path)
        mlflow.log_artifact(acf_path)
        
        # Q-Q Plot
        qq_path = os.path.join(out_dir, "lstm_residual_qq.png")
        plot_residual_qq(preds_real, true_real, save_path=qq_path)
        mlflow.log_artifact(qq_path)
        
        # Hourly Error Boxplot
        if "test_timestamps" in test_split:
            timestamps = test_split["test_timestamps"]
            timestamps = timestamps[-len(true_real):]

            hourly_err_path = os.path.join(out_dir, "lstm_hourly_error.png")
            plot_error_by_hour(preds_real, true_real, timestamps, save_path=hourly_err_path)
            mlflow.log_artifact(hourly_err_path)

        mlflow.log_params(best)
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        mlflow.log_metrics({f"test_real_{k}": v for k, v in metrics_real.items()})
        mlflow.set_tags({
            "project": "Traffic Flow Forecasting",
            "optimizer_engine": "optuna",
            "model_family": "lstm",
        })
 
        fig_res  = plot_residuals(preds, true_y,
                                  f"- {dataset_stem}/LSTM",
                                  save_path=os.path.join(out_dir, "residuals.png"))
        fig_pred = plot_predictions(preds, true_y,
                                    f"- {dataset_stem}/LSTM",
                                    save_path=os.path.join(out_dir, "predictions.png"))
        mlflow.log_figure(fig_res,  "residuals.png")
        mlflow.log_figure(fig_pred, "predictions.png")
 
        # model.save(os.path.join(out_dir, "model.keras"))
        save_detector_artefacts(
            det_id=f"{dataset_stem}_lstm",
            model=model,
            feat_scaler=test_split.get("feat_scaler"),
            tgt_scaler=test_split.get("tgt_scaler"),
            fmt="keras",
            feature_cols=list(full_x.columns),
            models_dir=os.path.abspath(out_dir)
        )

        mlflow.tensorflow.log_model(model, name="model")
 
        print(f" [LSTM] input_width={best['input_width']}  "
              f"RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  "
              f"R²={metrics['r2']:.4f}")
    return metrics

## ============== MAIN ==============

def main():
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if not csv_files:
        print(f"No CSV files found in {DATA_DIR}. Exiting.")
        sys.exit(1)
 
    print(f"Found {len(csv_files)} CSV file(s): {sorted(csv_files)}")
 
    for i,csv_file in enumerate(sorted(csv_files)):
        

        dataset_stem = os.path.splitext(csv_file)[0]
        print(f"\n{'='*60}")
        print(f"Dataset: {csv_file}")
        print(f"{'='*60}")

        # PREPROCESS
        df, series, cv_folds, test_split = preprocess(
            DATA_DIR, WEATHER_DIR, csv_file, FULL_RANGE, NOISE_FAC
        )

        # MLFLOW EXPERIMENT - one for each dataset
        experiment_id = get_or_create_experiment(f"traffic_flow_{dataset_stem}")
        mlflow.set_experiment(experiment_id=experiment_id)

        # OUTPUTS
        dataset_out = os.path.join(OUTPUT_DIR, dataset_stem)
        os.makedirs(dataset_out, exist_ok=True)
        store = MetricsStore(dataset=dataset_stem, output_dir=dataset_out)

        # fig_corr = plot_correlation(
        #     df, save_path=os.path.join(dataset_out, "correlation.png")
        # )

        # TRAINING
        all_metrics = {}

        all_metrics["xgboost"] = run_xgboost(
            cv_folds, test_split, experiment_id, dataset_stem,
            make_output_dirs(dataset_stem, "xgboost"), store
        )
        all_metrics["random_forest"] = run_random_forest(
            cv_folds, test_split, experiment_id, dataset_stem,
            make_output_dirs(dataset_stem, "random_forest"), store  
        )
        all_metrics["sarima"] = run_sarima(
            cv_folds, test_split, experiment_id, dataset_stem,
            make_output_dirs(dataset_stem, "sarima"), store 
        )
        all_metrics["lstm"] = run_lstm(
            cv_folds, test_split, experiment_id, dataset_stem,
            make_output_dirs(dataset_stem, "lstm"), store
        )

        # SUMMARY
        df_metrics = pd.DataFrame.from_dict(all_metrics, orient='index')

        summary_csv_path = os.path.join(dataset_out, "all_metrics.csv")
        df_metrics.to_csv(summary_csv_path)

        print(all_metrics)
        
        # fig_summary = plot_metrics_comparison(
        #     {k: {"test_rmse": v["rmse"]} for k, v in all_metrics.items()},
        #     dataset_stem,
        #     save_path=os.path.join(dataset_out, "model_comparison.png"),
        # )

        with mlflow.start_run(experiment_id=experiment_id,
                            run_name=f"{dataset_stem}_summary"):
            # mlflow.log_figure(fig_corr, "correlation.png")
            # mlflow.log_figure(fig_summary, "model_comparison.png")
            metrics_csv = store.save() # output/<dataset>/metrics.csv
            mlflow.log_artifact(metrics_csv, artifact_path="metrics")
            store.log_to_mlflow()    
                
        # # BEST MODEL
        # print(f"\n  {'-'*56}")
        # print(f"  Results for {dataset_stem}:")
        # print(f"  {'Model':<22} {'RMSE':>8} {'MAE':>8} {'SMAPE':>8} {'R²':>7}")
        # print(f"  {'-'*56}")
        # for name, m in sorted(all_metrics.items(), key=lambda x: x[1]["rmse"]):
        #     print(f"  {name:<22} {m['rmse']:>8.4f} {m['mae']:>8.4f} "
        #         f"{m['smape']:>8.4f} {m['r2']:>7.4f}")
        # best_model = min(all_metrics, key=lambda k: all_metrics[k]["rmse"])
        # print(f"\n Best model: {best_model}")
    
    print("\nAll datasets done")
 
 
if __name__ == "__main__":
    main()