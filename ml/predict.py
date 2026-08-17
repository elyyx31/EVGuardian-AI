from pathlib import Path
import joblib
import pandas as pd

# Handle imports whether called directly or from another package
try:
    from rul_calculator import calculate_rul
    from battery_health import battery_health
    from recommendation import get_recommendation
except ImportError:
    from ml.rul_calculator import calculate_rul
    from ml.battery_health import battery_health
    from ml.recommendation import get_recommendation

# Load trained model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "soh_model.pkl"

model = joblib.load(MODEL_PATH)

FEATURE_COLUMNS = [
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


def predict_battery(features):
    """
    Predict SOH, derived RUL, Battery Health category, and actionable Recommendation.

    Parameters:
        features: list, tuple, or dict containing the 10 battery parameters.

    Returns:
        dict with SOH (float), RUL (int), Health (str), and Recommendation (str).
    """
    if isinstance(features, dict):
        data = pd.DataFrame([features])[FEATURE_COLUMNS]
    else:
        data = pd.DataFrame([features], columns=FEATURE_COLUMNS)

    # Gradient Boosting SOH Regressor prediction
    predicted_soh = float(model.predict(data)[0])

    # Extract cycle and temperature by column name for safety
    cycle_val = float(data["Cycle"].iloc[0])
    temp_val = float(data["Temperature"].iloc[0])

    # Derived RUL estimation to EOL (SOH = 80%)
    rul = calculate_rul(predicted_soh, cycle_val)

    # Health status classification
    health = battery_health(predicted_soh)

    # Maintenance recommendation aligned with SOH and temperature safety
    recommendation = get_recommendation(
        predicted_soh,
        rul,
        temp_val
    )

    return {
        "SOH": round(predicted_soh, 2),
        "RUL": rul,
        "Health": health,
        "Recommendation": recommendation
    }


if __name__ == "__main__":
    # Representative sample from active dataset (Cycle 200)
    sample = [
        200,      # Cycle
        3.70,     # Voltage (V)
        1.00,     # Current (A)
        30.0,     # Temperature (°C)
        75.0,     # ChargeTime
        75.0,     # DischargeTime
        0.052,    # InternalResistance
        2.48,     # Capacity (Ah)
        50.0,     # AmbientHumidity (%)
        1.00      # C_Rate
    ]

    result = predict_battery(sample)
    print("Inference Test Result:")
    for k, v in result.items():
        print(f"  {k}: {v}")