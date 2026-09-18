import os
import sys
import logging
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, status, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------
# Logging Configuration
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ids_backend")

# --------------------------------------------------
# Robust Paths & Model Loading
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "ids_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
DEPLOYMENT_ENGINES_DIR = os.path.join(BASE_DIR, "deployment_engines")

# Fixed whitelist mapping: GAN Augmentation -> IDS Model -> Deployment Engine ZIP filename
DEPLOYMENT_ENGINES: Dict[str, Dict[str, str]] = {
    "CTGAN": {
        "Decision Tree": "DT_CTGAN_Deployment_Engine.zip",
        "MLP": "MLP_CTGAN_Deployment_Engine.zip",
        "Random Forest": "RF_CTGAN_Deployment_Engine.zip",
        "XGBoost": "XGB_CTGAN_Deployment_Engine.zip",
        "TabNet": "TabNet_CTGAN_Deployment_Engine.zip",
    },
    "CWGAN": {
        "Decision Tree": "DT_CWGAN_Deployment_Engine.zip",
        "MLP": "MLP_CWGAN_Deployment_Engine.zip",
        "Random Forest": "RF_CWGAN_Deployment_Engine.zip",
        "XGBoost": "XGB_CWGAN_Deployment_Engine.zip",
        "TabNet": "TabNet_CWGAN_Deployment_Engine.zip",
    },
}

ids_model = None
scaler = None
model_load_error: Optional[str] = None
scaler_load_error: Optional[str] = None

try:
    if os.path.exists(MODEL_PATH):
        ids_model = joblib.load(MODEL_PATH)
        logger.info(f"Model successfully loaded from {MODEL_PATH}")
    else:
        model_load_error = f"Model file not found at {MODEL_PATH}"
        logger.warning(model_load_error)
except Exception as e:
    model_load_error = f"Failed to load model: {str(e)}"
    logger.error(model_load_error, exc_info=True)

try:
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        logger.info(f"Scaler successfully loaded from {SCALER_PATH}")
    else:
        scaler_load_error = f"Scaler file not found at {SCALER_PATH}"
        logger.warning(scaler_load_error)
except Exception as e:
    scaler_load_error = f"Failed to load scaler: {str(e)}"
    logger.error(scaler_load_error, exc_info=True)


def get_model_metadata() -> Dict[str, Any]:
    """Inspect and extract detailed metadata from the loaded model and scaler."""
    meta = {
        "model_loaded": ids_model is not None,
        "scaler_loaded": scaler is not None,
        "model_type": type(ids_model).__name__ if ids_model is not None else None,
        "scaler_type": type(scaler).__name__ if scaler is not None else None,
        "expected_features": None,
        "feature_names_in": None,
        "classes": None,
        "has_predict_proba": hasattr(ids_model, "predict_proba") if ids_model is not None else False,
        "model_load_error": model_load_error,
        "scaler_load_error": scaler_load_error,
    }

    if scaler is not None:
        if hasattr(scaler, "n_features_in_"):
            meta["expected_features"] = int(scaler.n_features_in_)
        elif hasattr(scaler, "mean_"):
            meta["expected_features"] = int(len(scaler.mean_))
        if hasattr(scaler, "feature_names_in_"):
            meta["feature_names_in"] = [str(f) for f in scaler.feature_names_in_]
    elif ids_model is not None:
        if hasattr(ids_model, "n_features_in_"):
            meta["expected_features"] = int(ids_model.n_features_in_)
        if hasattr(ids_model, "feature_names_in_"):
            meta["feature_names_in"] = [str(f) for f in ids_model.feature_names_in_]

    if ids_model is not None and hasattr(ids_model, "classes_"):
        meta["classes"] = [str(c) for c in ids_model.classes_]

    return meta


# Log startup diagnostics
startup_meta = get_model_metadata()
logger.info(f"Startup Model Status: Loaded={startup_meta['model_loaded']}, Type={startup_meta['model_type']}, Classes={startup_meta['classes']}")
logger.info(f"Startup Scaler Status: Loaded={startup_meta['scaler_loaded']}, Type={startup_meta['scaler_type']}, Expected Features={startup_meta['expected_features']}")

# --------------------------------------------------
# FastAPI App Initialization
# --------------------------------------------------
app = FastAPI(
    title="GAN-Based Ag-IoT IDS Backend API",
    description=(
        "Research Prototype Backend for 'GAN-Based Data Augmentation for Multiclass "
        "Intrusion Detection in Ag-IoT' (FarmFlow Dataset). Supports DEMO ground-truth "
        "evaluation and live machine learning model inference (IDS mode)."
    ),
    version="3.0.0"
)


# Known target / label / metadata columns to exclude from feature matrix
KNOWN_TARGET_COLUMNS = [
    "traffic", "is_attack", "label", "Attack", "target", "class", "Class", "traffic_type"
]

# Non-numeric network identifier columns in FarmFlow
IDENTIFIER_COLUMNS = [
    "id.orig_h", "id.resp_h", "proto", "service", "conn_state",
    "local_orig", "local_resp", "history", "tunnel_parents"
]


@app.get(
    "/",
    summary="API Health and Model Status",
    response_description="Returns system status, active model information, and configuration"
)
def root():
    """Health check endpoint providing backend readiness and model metadata."""
    meta = get_model_metadata()
    return {
        "status": "Backend running",
        "message": "GAN-based Ag-IoT IDS backend is active",
        "model_metadata": meta
    }


@app.api_route(
    "/download-model",
    methods=["GET", "HEAD"],
    summary="Download Deployment Engine ZIP Package",
    response_description="Streams the deployment engine ZIP package directly as a downloadable attachment"
)
def download_model(
    gan: str = Query(..., description="GAN augmentation method (CTGAN or CWGAN)"),
    model: str = Query(..., description="IDS model (Decision Tree, MLP, Random Forest, XGBoost, TabNet)")
):
    """
    Download the pre-packaged deployment engine ZIP file for the selected GAN and IDS model.

    - Validates GAN against fixed whitelist (CTGAN, CWGAN)
    - Validates IDS model against fixed whitelist (Decision Tree, MLP, Random Forest, XGBoost, TabNet)
    - Uses fixed dictionary mapping to resolve the exact deployment ZIP filename
    - Verifies the ZIP file exists on disk
    - Returns the ZIP file using FastAPI FileResponse
    - Returns 404 if the ZIP file is missing
    - Never constructs paths directly from unvalidated user input
    """
    gan_clean = gan.strip()
    model_clean = model.strip()

    if gan_clean not in DEPLOYMENT_ENGINES or model_clean not in DEPLOYMENT_ENGINES[gan_clean]:
        valid_gans = list(DEPLOYMENT_ENGINES.keys())
        valid_models = list(DEPLOYMENT_ENGINES["CTGAN"].keys())
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid selection: gan='{gan}', model='{model}'. "
                f"Valid GANs: {valid_gans}, Valid Models: {valid_models}."
            )
        )

    filename = DEPLOYMENT_ENGINES[gan_clean][model_clean]
    file_path = os.path.join(DEPLOYMENT_ENGINES_DIR, gan_clean, filename)

    # Primary check against mapped filename, with disk fallback for alternate naming if present (e.g. XGB_ vs XGBoost_)
    if not os.path.isfile(file_path):
        alt_filename = filename.replace("XGB_", "XGBoost_") if "XGB_" in filename else filename.replace("XGBoost_", "XGB_")
        alt_path = os.path.join(DEPLOYMENT_ENGINES_DIR, gan_clean, alt_filename)
        if os.path.isfile(alt_path):
            file_path = alt_path
        else:
            logger.warning(f"Deployment engine ZIP not found: {file_path}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Deployment engine ZIP '{filename}' not found on backend server."
            )

    logger.info(f"Serving deployment engine: {file_path} as attachment '{filename}'")
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/zip"
    )



@app.post(
    "/predict",
    summary="Perform Intrusion Detection Prediction or Ground-Truth Evaluation",
    response_description="Multiclass detection summary, record count, and prediction details"
)
async def predict(
    file: UploadFile = File(..., description="Network traffic flow dataset in CSV format"),
    mode: str = Form("DEMO", description="Detection mode: 'DEMO' for dataset ground-truth summary, 'IDS' for ML model prediction")
):
    """
    Process an uploaded CSV dataset for Ag-IoT Intrusion Detection.

    - **DEMO Mode**: Analyzes ground-truth traffic labels from the dataset (e.g. `traffic` or `is_attack` column).
    - **IDS Mode**: Extracts numeric flow features, applies the standard scaler, and generates multiclass predictions using the saved IDS model.
    """
    try:
        # 1. Read CSV File
        try:
            df = pd.read_csv(file.file, low_memory=False)
        except Exception as csv_err:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error": "CSV Parse Error",
                    "details": f"Failed to parse CSV file: {str(csv_err)}"
                }
            )

        if df.empty:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error": "Empty Dataset",
                    "details": "The uploaded CSV file contains no data rows."
                }
            )

        selected_mode = mode.strip().upper()

        # ==================================================
        # DEMO MODE (Ground-Truth Dataset Evaluation)
        # ==================================================
        if selected_mode == "DEMO":
            label_col = None
            if "traffic" in df.columns:
                label_col = "traffic"
                labels = df["traffic"].astype(str)
            elif "is_attack" in df.columns:
                label_col = "is_attack"
                labels = df["is_attack"].map({0: "Normal (0)", 1: "Attack (1)"}).fillna(df["is_attack"].astype(str))
            else:
                for alt_col in ["label", "Attack", "target", "class", "Class"]:
                    if alt_col in df.columns:
                        label_col = alt_col
                        labels = df[alt_col].astype(str)
                        break

            if label_col is None:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "error": "Ground-Truth Column Not Found",
                        "details": (
                            "DEMO mode requires a ground-truth label column (such as 'traffic' or 'is_attack') "
                            "in the uploaded CSV. For unlabeled data inference, select IDS mode."
                        )
                    }
                )

            summary = labels.value_counts().to_dict()
            total_records = len(labels)

            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "mode": "DEMO (Ground Truth – FarmFlow)",
                    "label_column_used": label_col,
                    "total_records": total_records,
                    "attack_summary": summary,
                    "classes_detected": list(summary.keys()),
                    "sample_predictions": labels.head(50).tolist(),
                    "predictions": labels.tolist() if total_records <= 500000 else None,
                    "note": f"Evaluated using dataset ground-truth column '{label_col}' (DEMO mode)."
                }
            )

        # ==================================================
        # IDS MODE (Machine Learning Model Prediction)
        # ==================================================
        elif selected_mode == "IDS":
            if ids_model is None or scaler is None:
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content={
                        "error": "IDS Model Not Available",
                        "details": f"Model or Scaler is not loaded on backend: Model Error: '{model_load_error}', Scaler Error: '{scaler_load_error}'."
                    }
                )

            # Drop known target/label columns
            cols_to_drop = [c for c in KNOWN_TARGET_COLUMNS if c in df.columns]
            df_features = df.drop(columns=cols_to_drop, errors="ignore")

            # Check if model/scaler has defined feature names
            if hasattr(scaler, "feature_names_in_") and scaler.feature_names_in_ is not None:
                expected_names = list(scaler.feature_names_in_)
                missing = [col for col in expected_names if col not in df_features.columns]
                if missing:
                    return JSONResponse(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        content={
                            "error": "Feature Mismatch",
                            "details": f"Dataset is missing {len(missing)} required features: {missing[:10]}"
                        }
                    )
                X_df = df_features[expected_names].apply(pd.to_numeric, errors="coerce").fillna(0)
                X = X_df.values

            elif hasattr(ids_model, "feature_names_in_") and ids_model.feature_names_in_ is not None:
                expected_names = list(ids_model.feature_names_in_)
                missing = [col for col in expected_names if col not in df_features.columns]
                if missing:
                    return JSONResponse(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        content={
                            "error": "Feature Mismatch",
                            "details": f"Dataset is missing {len(missing)} required features: {missing[:10]}"
                        }
                    )
                X_df = df_features[expected_names].apply(pd.to_numeric, errors="coerce").fillna(0)
                X = X_df.values

            else:
                # No feature names stored: extract numeric flow features excluding identifier columns
                non_feature_cols = [c for c in IDENTIFIER_COLUMNS if c in df_features.columns]
                df_candidate_features = df_features.drop(columns=non_feature_cols, errors="ignore")

                # Convert to numeric (handling string numbers like in duration, orig_bytes)
                df_numeric = df_candidate_features.apply(pd.to_numeric, errors="coerce")
                
                # Keep columns that have valid numeric entries
                numeric_cols = [c for c in df_numeric.columns if df_numeric[c].notna().any()]
                df_numeric = df_numeric[numeric_cols].fillna(0)

                expected_features = getattr(scaler, "n_features_in_", 10)
                received_features = df_numeric.shape[1]

                if received_features < expected_features:
                    return JSONResponse(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        content={
                            "error": "Feature Mismatch",
                            "details": (
                                f"Model/Scaler expects {expected_features} numeric features, "
                                f"but only {received_features} numeric columns were extracted from the dataset."
                            )
                        }
                    )

                # Use the expected number of features
                X = df_numeric.iloc[:, :expected_features].values

            # Scale and Predict
            X_scaled = scaler.transform(X)
            predictions = ids_model.predict(X_scaled)

            attack_labels = [str(p) for p in predictions]
            summary = pd.Series(attack_labels).value_counts().to_dict()
            total_records = len(attack_labels)

            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "mode": "IDS (Model Prediction)",
                    "model_type": type(ids_model).__name__,
                    "total_records": total_records,
                    "attack_summary": summary,
                    "classes": [str(c) for c in getattr(ids_model, "classes_", [])],
                    "sample_predictions": attack_labels[:50],
                    "predictions": attack_labels if total_records <= 500000 else None,
                    "note": "Multiclass predictions generated by loaded Ag-IoT IDS model."
                }
            )

        else:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error": "Invalid Mode",
                    "details": f"Unknown detection mode '{mode}'. Supported modes are 'DEMO' and 'IDS'."
                }
            )

    except Exception as e:
        logger.error(f"Prediction endpoint error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Prediction Processing Error",
                "details": str(e)
            }
        )

