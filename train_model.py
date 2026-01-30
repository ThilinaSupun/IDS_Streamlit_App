import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv("Farm-Flows.csv", low_memory=False)

# Separate features and label
X = data.drop("is_attack", axis=1)
y = data["is_attack"]

# Convert all feature columns to numeric
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(0)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
model.fit(X_scaled, y)

# Save trained model and scaler
joblib.dump(model, "ids_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Training completed successfully.")
