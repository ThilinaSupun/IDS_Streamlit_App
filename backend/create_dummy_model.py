import numpy as np
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Create dummy dataset
# -----------------------------
X_dummy = np.random.rand(100, 10)   # 10 features (temporary)
y_dummy = np.random.randint(0, 7, 100)  # 7 classes (multiclass IDS)

# -----------------------------
# Create dummy scaler
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_dummy)

# -----------------------------
# Create dummy IDS model
# -----------------------------
model = DummyClassifier(strategy="most_frequent")
model.fit(X_scaled, y_dummy)

# -----------------------------
# Save files
# -----------------------------
joblib.dump(model, "model/ids_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("✅ Dummy IDS model and scaler created successfully")
