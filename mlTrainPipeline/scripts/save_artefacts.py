# save_artefacts.py
 
import json
import os
import joblib
from sklearn.preprocessing import StandardScaler
 
model_format = {
    "xgboost": "ubj",
    "sklearn":  "joblib",
    "sarima":   "pkl",
    "keras":    "keras",
}

def save_detector_artefacts(
    det_id: str,
    model,
    feat_scaler=None,   
    tgt_scaler=None,    
    fmt: str | None = None,
    feature_cols: list[str] | None = None,
    models_dir: str = "models",
) -> str:
    """
    saves and returns path to model + scalers for one detector 
    """
    
    out_dir = os.path.join(models_dir, det_id)
    os.makedirs(out_dir, exist_ok=True)
    
    # save model
    model_filename = f"model.{model_format[fmt]}"
    model_path = os.path.join(out_dir, model_filename)
    
    if fmt == "xgboost":
        model.save_model(model_path)
 
    elif fmt == "sklearn":
        joblib.dump(model, model_path)
 
    elif fmt == "sarima":
        model.save(model_path, remove_data=False)

    elif fmt == "keras":
        model.save(model_path)
 

    model_kb = os.path.getsize(model_path) / 1024

    # scalers
    feat_path = tgt_path = None
 
    if feat_scaler is not None:
        feat_path = os.path.join(out_dir, "feat_scaler.pkl")
        joblib.dump(feat_scaler, feat_path)
 
    if tgt_scaler is not None:
        tgt_path = os.path.join(out_dir, "tgt_scaler.pkl")
        joblib.dump(tgt_scaler, tgt_path)

 
    meta = {
    "det_id":        det_id,
    "model_type":    fmt,
    "model_file":    model_filename,
    "has_feat_scaler": feat_scaler is not None,
    "has_tgt_scaler":  tgt_scaler  is not None,
    "feature_cols":  feature_cols or [],
    "model_size_kb": round(model_kb, 2),
    }

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return out_dir
 
def save_all(detector_results: list[dict], models_dir: str = "models"):
    """
    detector_results is a list of dicts, one per detector
    """
    for d in detector_results:
        save_detector_artefacts(
            det_id       = d["det_id"],
            model        = d["model"],
            feat_scaler  = d.get("feat_scaler"),
            tgt_scaler   = d.get("tgt_scaler"),
            fmt          = d.get("fmt"),
            feature_cols = d.get("feature_cols"),
            models_dir   = models_dir,
        )
 
