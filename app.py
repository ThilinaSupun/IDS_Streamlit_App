import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

# ------------------ UI CONFIG ------------------
st.set_page_config(page_title="Intrusion Detection System", layout="wide")

st.title("Intrusion Detection System (IDS)")
st.subheader("Network Traffic Attack Detection using Machine Learning")

# ------------------ LOAD MODEL ------------------
try:
    model = joblib.load("ids_model.pkl")
    scaler = joblib.load("scaler.pkl")
    st.success("Model and scaler loaded successfully")
except:
    st.error("❌ Model or scaler not found")
    st.stop()

# ------------------ CSV UPLOAD ------------------
st.header("Upload Network Traffic CSV File")
uploaded_file = st.file_uploader("Upload CSV (< 500 MB)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, low_memory=False)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # Separate features
    if "is_attack" in df.columns:
        X = df.drop("is_attack", axis=1)
        y_true = df["is_attack"]
        has_labels = True
    else:
        X = df.copy()
        has_labels = False

    # Preprocessing
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0)

    # Feature check
    expected_features = scaler.n_features_in_
    if X.shape[1] != expected_features:
        st.error(
            f"❌ Feature mismatch: model expects {expected_features} features, "
            f"but CSV has {X.shape[1]}"
        )
        st.stop()

    # ------------------ PREDICTION ------------------
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)

    # Add readable predictions
    df["Prediction"] = pd.Series(predictions).map({0: "Benign", 1: "Attack"})

    st.subheader("Prediction Results")
    st.dataframe(df.head(20))

    # ------------------ ATTACK PERCENTAGE CHART ------------------
    st.subheader("📊 Attack Percentage Distribution")

    attack_counts = df["Prediction"].value_counts(normalize=True) * 100
    attack_df = attack_counts.reset_index()
    attack_df.columns = ["Traffic Type", "Percentage"]

    st.bar_chart(attack_df.set_index("Traffic Type"))

    st.write(attack_df)

    # ------------------ MODEL ACCURACY (IF LABELS EXIST) ------------------
    if has_labels:
        st.subheader("📈 Model Performance Metrics")

        accuracy = accuracy_score(y_true, predictions)
        st.metric(label="Accuracy", value=f"{accuracy * 100:.2f}%")

        report = classification_report(
            y_true,
            predictions,
            target_names=["Benign", "Attack"],
            output_dict=True
        )

        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

    else:
        st.info("ℹ️ Ground truth labels (`is_attack`) not found. Accuracy metrics not available.")

    # ------------------ DOWNLOAD RESULTS ------------------
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Prediction Results",
        csv,
        "ids_predictions.csv",
        "text/csv"
    )
