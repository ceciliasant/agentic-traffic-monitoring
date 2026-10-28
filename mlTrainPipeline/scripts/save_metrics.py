# scripts/save_metrics.py
"""
Collects CV and test metrics across all models for one detector/dataset
and writes a structured CSV at the end.
  dataset        : detector / dataset name
  model          : xgboost | random_forest | sarima | lstm
  split          : cv | test
  scaled         : true | false
  metric         : rmse | mae | mse | smape | r2
  value          : float
  cv_std         : std across folds (CV rows only, else NaN)
"""

import os
import math
import numpy as np
import pandas as pd

class MetricsStore:

    def __init__(self, dataset: str, output_dir: str):
        self.dataset    = dataset
        self.output_dir = output_dir
        self._rows: list[dict] = []
 
    def _add(self, model, split, scaled, metrics: dict,
             cv_std: dict | None = None, note: str = ""):
        for metric, value in metrics.items():
            self._rows.append({
                "dataset": self.dataset,
                "model":   model,
                "split":   split,
                "scaled":  scaled,
                "metric":  metric,
                "value":   round(float(value), 6) if value is not None and math.isfinite(float(value)) else None,
                "cv_std":  round(float(cv_std[metric]), 6) if cv_std and metric in cv_std else None,
                "note":    note,
            })
  
    def add_cv(self, model: str, study, scaled: bool = True, note: str = ""):
        """
        cv_mean metrics
        """
        best_trial = study.best_trial
        metric_keys = ["mse", "rmse", "mae", "smape", "r2"]
        
        cv_metrics = {}
        cv_stds = {}
        
        for m in metric_keys:
            mean_val = best_trial.user_attrs.get(f"cv_mean_{m}")
            std_val  = best_trial.user_attrs.get(f"cv_std_{m}")
            
            # Fallback for standard baseline configurations
            if m == "rmse" and mean_val is None:
                mean_val = best_trial.value
                std_val  = best_trial.user_attrs.get("cv_std_rmse")
                
            if mean_val is not None:
                cv_metrics[m] = mean_val
            if std_val is not None:
                cv_stds[m] = std_val
 
        self._add(model, "cv", scaled,
                  metrics=cv_metrics,
                  cv_std=cv_stds if cv_stds else None,
                  note=note)
 
    def add_test(self, model: str, metrics_scaled: dict,
                 metrics_real: dict | None = None):
        """
        Pass in the dicts already returned by regression_metrics().
        metrics_real is None for SARIMA (already in real units).
        """
        self._add(model, "test", scaled=True,  metrics=metrics_scaled)
        if metrics_real is not None:
            self._add(model, "test", scaled=False, metrics=metrics_real)
        else:
            # SARIMA: real == scaled, flag it clearly
            self._add(model, "test", scaled=False, metrics=metrics_scaled,
                      note="raw flows (no scaler)")
 
    def save(self, filename: str = "metrics.csv") -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, filename)
        pd.DataFrame(self._rows).to_csv(path, index=False)
        print(f"[metrics_store] {len(self._rows)} rows → {path}")
        return path
 
    def summary(self) -> pd.DataFrame:
        """Wide pivot: rows=model, cols=split_scaled_metric."""
        df = pd.DataFrame(self._rows)
        if df.empty:
            return df
        df["col"] = (df["split"] + "_"
                     + df["scaled"].map({True: "scaled", False: "real"})
                     + "_" + df["metric"])
        pivot = df.pivot_table(index="model", columns="col",
                               values="value", aggfunc="first").reset_index()
        pivot.columns.name = None
        return pivot
 
    def log_to_mlflow(self):
        """Log every metric as <model>_<split>_<scaled|real>_<metric>."""
        import mlflow
        for row in self._rows:
            if row["value"] is None:
                continue
            key = (f"{row['model']}_{row['split']}_"
                   f"{'scaled' if row['scaled'] else 'real'}_{row['metric']}")
            mlflow.log_metric(key, row["value"])
 
