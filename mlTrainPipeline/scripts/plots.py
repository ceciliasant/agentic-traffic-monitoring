## plots.py
## plotting functions for model evaluation and feature importance and correlation

import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import numpy as np
import pandas as pd

## functions to plot residuals and feature importance
def plot_residuals(preds, true_y, title_suffix: str = "", save_path=None):
    """
    Plots the residuals of the model predictions against the true values.

    Args:
    - preds: model predictions.
    - true_y: true values.
    - title_suffix: string plot title (like dataset name).
    - save_path (str, optional): Path to save the generated plot. If not specified, plot won't be saved.

    Returns:
    - None (Displays the residuals plot on a Jupyter window)

    adapted from mlflow tutorial https://mlflow.org/docs/latest/ml/traditional-ml/tutorials/hyperparameter-tuning/notebooks/hyperparameter-tuning-with-child-runs/
    """

    # Calculate residuals
    residuals = np.asarray(true_y) - np.asarray(preds)

    # Set Seaborn style
    sns.set_style("whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5})

    # Create scatter plot
    fig = plt.figure(figsize=(12, 8))
    plt.scatter(true_y, residuals, color="blue", alpha=0.5)
    plt.axhline(y=0, color="r", linestyle="-")

    # Set labels, title and other plot properties
    plt.title(f"Residuals vs True Values {title_suffix}", fontsize=18)
    plt.xlabel("True Values", fontsize=16)
    plt.ylabel("Residuals", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis="y")

    plt.tight_layout()

    # Save the plot if save_path is specified
    if save_path:
        plt.savefig(save_path, format="png", dpi=600)

    # Show the plot
    plt.close(fig)

    return fig

def plot_predictions(preds, true_y, title_suffix: str = "", save_path: str = None):
    """
    Line plot comparing predicted vs actual values over the test set.
    Useful for spotting systematic biases or lag.
 
    Args:
        preds:        model predictions.
        true_y:       true values.
        title_suffix: Optional string appended to the plot title.
        save_path:    If given, saves the figure as PNG (600 dpi).
 
    Returns:
        matplotlib Figure.
    """

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.set_style("whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5})

    ax.plot(np.asarray(true_y), label="Actual", linewidth=1.2, color="steelblue")
    ax.plot(np.asarray(preds),  label="Predicted", linewidth=1.2, color="orange", alpha=0.8)
    ax.set_title(f"Predicted vs Actual - Test Set {title_suffix}", fontsize=18)
    ax.set_xlabel("Sample Index", fontsize=16)
    ax.set_ylabel("Flows", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=14)
    plt.tight_layout()
 
    if save_path:
        fig.savefig(save_path, format="png", dpi=600)
    plt.close(fig)
    return fig

def plot_xgb_feature_importance(model, booster, title_suffix: str = "", save_path: str = None):
    """
    Plots feature importance for an XGBoost model.

    Args:
    - model: A trained XGBoost model

    Returns:
    - fig: The matplotlib figure object

    from mlflow tutorial https://mlflow.org/docs/latest/ml/traditional-ml/tutorials/hyperparameter-tuning/notebooks/hyperparameter-tuning-with-child-runs/
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    importance_type = "weight" if booster == "gblinear" else "gain"
    xgb.plot_importance(
        model,
        importance_type=importance_type,
        ax=ax,
        title=f"Feature Importance based on {importance_type} - {title_suffix}",
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, format="png", dpi=600)
    plt.close(fig)
    return fig

# def plot_rf_feature_importance(model, feature_names, title: str = "", save_path=None):

#     importances = model.feature_importances_
#     indices = np.argsort(importances)
#     names = [feature_names[i] for i in indices]
#     values = importances[indices]

#     fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.35)))
#     sns.set_style("whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5})

#     ax.barh(names, values, color=colors)
#     ax.set_title(f"Feature Importance {title}", fontsize=18)
#     ax.set_xlabel("Importance (Mean Decrease in Impurity)", fontsize=14)
#     ax.tick_params(labelsize=12)
#     plt.tight_layout()
 
#     if save_path:
#         fig.savefig(save_path, format="png", dpi=600)
#     plt.close(fig)
#     return fig

def plot_correlation(df, save_path=None):
    """
    Plots the correlation of each variable in the dataframe with the 'flows' column.

    Args:
    - df (pd.DataFrame): DataFrame containing the data, including a 'flows' column.
    - save_path (str, optional): Path to save the generated plot. If not specified, plot won't be saved.

    Returns:
    - None (Displays the plot on a Jupyter window)
    adapted from mlflow tutorial https://mlflow.org/docs/latest/ml/traditional-ml/tutorials/hyperparameter-tuning/notebooks/hyperparameter-tuning-with-child-runs/
    """

    # Compute correlations between all variables and 'demand'
    correlations = df.corr()["flows"].drop("flows").sort_values()

    # Generate a color palette from red to green
    colors = sns.diverging_palette(10, 130, as_cmap=True)
    color_mapped = correlations.map(colors)

    # Set Seaborn style
    sns.set_style(
        "whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5}
    )  # Light grey background and thicker grid lines

    # Create bar plot
    fig = plt.figure(figsize=(12, 8))
    plt.barh(correlations.index, correlations.values, color=color_mapped)

    # Set labels and title with increased font size
    plt.title("Correlation with Flows", fontsize=18)
    plt.xlabel("Correlation Coefficient", fontsize=16)
    plt.ylabel("Variable", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis="x")

    plt.tight_layout()

    # Save the plot if save_path is specified
    if save_path:
        plt.savefig(save_path, format="png", dpi=600)

    # prevent matplotlib from displaying the chart every time we call this function
    plt.close(fig)

    return fig

# def plot_metrics_comparison(results: dict, dataset_name: str, save_path: str = None):
#     """
#     Bar chart comparing RMSE across all trained model types for one dataset.
 
#     Args:
#         results:      Dict of {model_name: {"rmse": float, "mae": float, ...}}.
#         dataset_name: Used in the plot title.
#         save_path:    If given, saves the figure as PNG (600 dpi).
 
#     Returns:
#         matplotlib Figure.
#     """
#     sns.set_style("whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5})
    
#     names  = list(results.keys())
#     rmses  = [results[m]["test_rmse"] for m in names]
#     colors = sns.color_palette("muted", len(names))
 
#     fig, ax = plt.subplots(figsize=(10, 6))
#     bars = ax.bar(names, rmses, color=colors)
#     ax.bar_label(bars, fmt="%.2f", fontsize=12, padding=3)
#     ax.set_title(f"Test RMSE by Model - {dataset_name}", fontsize=18)
#     ax.set_xlabel("Model", fontsize=16)
#     ax.set_ylabel("RMSE", fontsize=16)
#     ax.tick_params(labelsize=14)
#     plt.tight_layout()
 
#     if save_path:
#         fig.savefig(save_path, format="png", dpi=600)
#     plt.close(fig)
#     return fig
 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def plot_residual_diagnostics(preds, true_y, lags=48, save_path=None):
    residuals = np.asarray(true_y) - np.asarray(preds)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    plot_acf(residuals, lags=lags, ax=axes[0], title="Residuals ACF")
    plot_pacf(residuals, lags=lags, ax=axes[1], title="Residuals PACF")
    
    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return fig

def plot_error_by_hour(preds, true_y, timestamps, save_path=None):
    """timestamps should match the test_y index"""
    absolute_errors = np.abs(np.asarray(true_y) - np.asarray(preds))
    df_err = pd.DataFrame({"Error": absolute_errors, "Hour": timestamps.hour})
    
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(x="Hour", y="Error", data=df_err, ax=ax, palette="vlag")
    ax.set_title("Absolute Error Distribution Across Hours of the Day")
    
    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return fig

import scipy.stats as stats

def plot_residual_qq(preds, true_y, save_path=None):
    residuals = np.asarray(true_y) - np.asarray(preds)
    fig, ax = plt.subplots(figsize=(6, 6))
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title("Normal Q-Q Plot of Residuals")
    
    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return fig

def plot_lstm_learning_curve(history_dict, save_path=None):
    """Pass history.history from your LSTM training loop"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history_dict['loss'], label='Train Loss')
    ax.plot(history_dict['val_loss'], label='Val Loss')
    ax.set_title("LSTM Loss Convergence")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss (MSE)")
    ax.legend()
    
    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return fig

import shap

def plot_model_shap(model, test_x, save_path=None):
    # tree models
    # https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Basic%20SHAP%20Interaction%20Value%20Example%20in%20XGBoost.html
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(test_x)
    
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, test_x, show=False)
    plt.title("SHAP Feature Impact Summary")
    
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300)
    plt.close(fig)
    return fig