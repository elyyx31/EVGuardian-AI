import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("../dataset/new_battery_dataset.csv")

# Sort according to cycle
df = df.sort_values("Cycle").reset_index(drop=True)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================
# 2. INPUT FEATURES
# ==========================================

feature_columns = [
    "Cycle",
    "Voltage",
    "Current",
    "Temperature",
    "ChargeTime",
    "DischargeTime",
    "InternalResistance",
    "Capacity",
    "AmbientHumidity",
    "C_Rate"
]

X = df[feature_columns]
y = df["SOH"]


# ==========================================
# 3. TIME / CYCLE-BASED SPLIT
# ==========================================

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print("\n========== DATA SPLIT ==========")

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))

print(
    "Training cycle range:",
    df.iloc[:split_index]["Cycle"].min(),
    "to",
    df.iloc[:split_index]["Cycle"].max()
)

print(
    "Testing cycle range:",
    df.iloc[split_index:]["Cycle"].min(),
    "to",
    df.iloc[split_index:]["Cycle"].max()
)


# ==========================================
# 4. GRADIENT BOOSTING MODEL
# ==========================================

model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)


# ==========================================
# 5. TRAIN
# ==========================================

print("\nTraining Gradient Boosting model...")

model.fit(X_train, y_train)

print("Model trained successfully!")


# ==========================================
# 6. PREDICTION
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 7. EVALUATION
# ==========================================

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)


print("\n========== SOH VALIDATION RESULTS ==========")

print(f"MAE      : {mae:.4f}")
print(f"RMSE     : {rmse:.4f}")
print(f"R² Score : {r2:.4f}")


# ==========================================
# 8. FIRST 10 PREDICTIONS
# ==========================================

print("\n========== FIRST 10 PREDICTIONS ==========")

for actual, predicted in zip(
    y_test.iloc[:10],
    y_pred[:10]
):

    print(
        f"Actual SOH: {actual:.2f} "
        f"| Predicted SOH: {predicted:.2f}"
    )


# ==========================================
# 9. FEATURE IMPORTANCE
# ==========================================

print("\n========== FEATURE IMPORTANCE ==========")

importance = pd.DataFrame({

    "Feature": feature_columns,

    "Importance": model.feature_importances_

})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance.to_string(index=False))