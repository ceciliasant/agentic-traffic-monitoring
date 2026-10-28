# Traffic Flow Forecasting

A machine learning pipeline for forecasting hourly traffic flow using different forecasting models, hyperparameter optimization, experiment tracking with MLflow, and Docker execution.

## Features

* Forecast hourly traffic flow from historical traffic and weather data
* Automatic preprocessing and missing value imputation
* Forecasting models:

  * XGBoost
  * Random Forest
  * SARIMA
  * LSTM (TensorFlow/Keras)
* Hyperparameter optimization using Optuna
* Experiment tracking with MLflow
* Model evaluation plots
* Docker and Docker Compose
* Model artifact saving for deployment

---

## Project Structure

```text
.
├── data/
│   ├── weather/
│   └── *.csv
├── models/                    # Exported trained models
├── output/                    # Metrics and evaluation plots
├── scripts/
│   ├── impute.py              # Missing value imputation
│   ├── preprocess.py          # Feature engineering and dataset preparation
│   ├── models.py              # Model definitions and training
│   ├── train.py               # MLflow experiment management
│   ├── plots.py               # Evaluation and diagnostic plots
│   ├── save_artefacts.py      # Model and scaler serialization
│   └── save_metrics.py        # Metrics aggregation and MLflow logging
├── Dockerfile
├── docker-compose.yaml
├── main.py                    # Main training pipeline
├── requirements.txt
└── README.md
```

## Pipeline Overview

The complete workflow executed by `main.py` is:

```text
Traffic CSV(s)
        │
        ▼
Load Dataset
        │
        ▼
Missing Value Imputation
        │
        ▼
Feature Engineering
        │
        ▼
Weather Data Integration
        │
        ▼
Train/Test Split
        │
        ▼
Time Series Cross Validation
        │
        ▼
Feature & Target Scaling
        │
        ▼
Hyperparameter Optimization (Optuna)
        │
        ▼
Train Final Model
        │
        ▼
Model Evaluation
        │
        ▼
Diagnostic Plots
        │
        ▼
Save Models + Scalers
        │
        ▼
Log Everything to MLflow
```

---

## Models

The project compares four forecasting approaches.

| Model         | Library            |
| ------------- | ------------------ |
| XGBoost       | XGBoost            |
| Random Forest | Scikit-learn       |
| SARIMA        | Statsmodels        |
| LSTM          | TensorFlow / Keras |

Each model is independently tuned using Optuna before being retrained on the complete training data.

---

## Hyperparameter Optimization

Hyperparameter tuning is performed using **Optuna**.

The optimization uses:

* Median Pruner
* Cross-validation
* Early stopping
* MLflow nested runs for every trial

Number of optimization trials:

| Model         | Trials |
| ------------- | ------ |
| XGBoost       | 50     |
| Random Forest | 50     |
| SARIMA        | 10     |
| LSTM          | 20     |

---

## Experiment Tracking

The project uses **MLflow** to log:

* hyperparameters
* cross-validation metrics
* test metrics
* trained models
* plots
* model artifacts

Each dataset receives its own MLflow experiment.

---

## Evaluation Metrics

Models are evaluated using:

* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Error (MAE)
* Symmetric Mean Absolute Percentage Error (SMAPE)
* R² Score

For scaled models, metrics are also computed after inverse scaling to obtain results in the original traffic flow units.

---

## Generated Outputs

For every trained model the pipeline generates:

* Prediction plots
* Residual ACF/PACF
* Residual Q-Q plots
* Hourly error distributions
* SHAP summary plots (for tree models)
* LSTM learning curves

Additionally:

* CSV summary of model metrics
* Serialized trained models
* Feature scalers
* Target scalers

---

## Data Preprocessing

The preprocessing pipeline prepares every traffic dataset before model training.

### Missing Value Imputation

Missing traffic flow observations are filled using a hierarchical strategy:

1. Historical mean for the same weekday and hour
2. Historical mean for the same hour
3. Previous available observation

Gaussian noise is optionally added to preserve realistic variability.

### Feature Engineering

The following features are generated automatically:

**Temporal features**

* Hour
* Day
* Month
* Year
* Weekday
* Time index

**Cyclical encodings**

* Hour (sin/cos)
* Weekday (sin/cos)
* Month (sin/cos)

**Calendar features**

* Weekend indicator
* Public holiday indicator (2022–2024)

**Lag features**

* Previous hour (`lag_1`)
* Previous day (`lag_24`)
* Previous week (`lag_168`)

**Weather features**

Daily weather observations are merged with traffic data:

* Average temperature
* Average precipitation

### Data Scaling

Tree-based models and the LSTM use independent `StandardScaler` objects for:

* Input features
* Target variable (traffic flow)

SARIMA is trained directly on the original traffic flow values.

### Validation Strategy

The data is divided into:

* 85% training/cross-validation
* 15% final hold-out test set

The training portion is evaluated using a 5-fold expanding-window `TimeSeriesSplit` to preserve temporal ordering and prevent data leakage.

---
## Model Export

After training, every model is exported together with its preprocessing artefacts.

Each exported detector contains:

```text
models/
└── detector_name/
    ├── model.*
    ├── feat_scaler.pkl
    ├── tgt_scaler.pkl
    └── meta.json
```

The metadata file records:

* detector name
* model type
* model filename
* model size
* feature names
* scaler availability

Supported export formats are:

| Model         | Format    |
| ------------- | --------- |
| XGBoost       | `.ubj`    |
| Random Forest | `.joblib` |
| SARIMA        | `.pkl`    |
| LSTM          | `.keras`  |

## Metrics Storage

For every dataset, a consolidated `metrics.csv` file is generated containing:

* Dataset name
* Model name
* Cross-validation metrics
* Test metrics
* Scaled and original-unit results
* Cross-validation standard deviations
* Notes (example: SARIMA uses unscaled targets)

These metrics are also logged automatically to MLflow for experiment comparison.

---

## Requirements

Python dependencies:

```
pandas
numpy
scikit-learn
statsmodels
xgboost
tensorflow
keras
optuna
mlflow
matplotlib
seaborn
shap
joblib
```

---

## Running with Docker

Build and start the project:

```bash
docker compose up --build
```

This launches:

* MLflow tracking server
* Training container

The MLflow UI is available at:

```
http://localhost:5000
```


---

## Data

The pipeline expects one or more CSV files inside:

```
data/
```

Weather data should be located in:

```
data/weather/
```

Each dataset is processed independently.


## Output Structure

```
output/
└── dataset_name/
    ├── all_metrics.csv
    ├── metrics.csv
    ├── xgboost/
    ├── random_forest/
    ├── sarima/
    └── lstm/
```

Each model directory contains:

* saved model
* evaluation plots 

---

## Technologies

* Python 3.12
* TensorFlow / Keras
* XGBoost
* Scikit-learn
* Statsmodels
* Optuna
* MLflow
* Docker
* Docker Compose
