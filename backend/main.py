from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import joblib
import os

# --------------------------------------------------
# Initialize FastAPI app
# --------------------------------------------------
app = FastAPI(
    title="GAN-based Ag-IoT IDS Backend",
    description="Backend API for Multiclass Intrusion Detection System (FarmFlow)",
    version="3.0"
)

# --------------------------------------------------
# Load TEMP IDS Model & Scaler (placeholder)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "ids_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")

try:
    ids_model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception:
    ids_model = None
    scaler = None

# --------------------------------------------------
# Health check
# --------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "Backend running",
        "message": "GAN-based Ag-IoT IDS backend is active"
    }

# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------
from fastapi import Form

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    mode: str = Form("DEMO")
):

    try:
        # -----------------------------
        # Read CSV
        # -----------------------------
        df = pd.read_csv(file.file)

        if df.empty:
            return JSONResponse(
                status_code=400,
                content={"error": "Uploaded CSV file is empty"}
            )

        # ==================================================
        # ✅ DEMO MODE (FarmFlow Ground Truth)
        # ==================================================
        # FarmFlow multiclass label column = "traffic"
        # ==================================================
# ==================================================
# DEMO MODE (explicit user-selected)
# ==================================================
        if mode == "DEMO" and "traffic" in df.columns:
            labels = df["traffic"].astype(str)

            summary = labels.value_counts().to_dict()

            return JSONResponse(
                status_code=200,
                content={
                    "mode": "DEMO (Ground Truth – FarmFlow)",
                    "label_column_used": "traffic",
                    "total_records": len(labels),
                    "attack_summary": summary,
                    "note": "User-selected DEMO mode using dataset labels"
                }
            )


        # ==================================================
        # IDS MODE (Dummy / Real Model – later)
        # ==================================================
        if ids_model is None or scaler is None:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "IDS model not available",
                    "note": "Real IDS model will be integrated after training"
                }
            )

        df_numeric = df.select_dtypes(include=[np.number])

        if df_numeric.empty:
            return JSONResponse(
                status_code=400,
                content={"error": "No numeric features found"}
            )

        X = df_numeric.values

        expected_features = scaler.mean_.shape[0]
        received_features = X.shape[1]

        if received_features > expected_features:
            X = X[:, :expected_features]
        elif received_features < expected_features:
            padding = np.zeros((X.shape[0], expected_features - received_features))
            X = np.hstack((X, padding))

        X_scaled = scaler.transform(X)
        predictions = ids_model.predict(X_scaled)

        attack_labels = [str(p) for p in predictions]
        summary = pd.Series(attack_labels).value_counts().to_dict()

        return JSONResponse(
            status_code=200,
            content={
                "mode": "IDS (Model Prediction)",
                "total_records": len(attack_labels),
                "attack_summary": summary
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
