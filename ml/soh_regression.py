import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("../dataset/new_battery_dataset.csv")

# Input Features
X = df[
    [
        "Cycle",
        "Voltage",
        "Current",
        "Temperature",
        "ChargeTime",
        "DischargeTime",
        "InternalResistance",
        "Capacity",
        "AmbientHumidity",
        "C_Rate",
    ]
]

# Target
y = df["SOH"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

# Model
model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n========== SOH Regression Results ==========\n")

print(f"MAE      : {mae:.4f}")
print(f"RMSE     : {rmse:.4f}")
print(f"R² Score : {r2:.4f}")

# Save model
joblib.dump(model, "../models/soh_model.pkl")

print("\nSOH model saved successfully!")