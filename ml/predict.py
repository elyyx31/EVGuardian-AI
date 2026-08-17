import os
import joblib
import pandas as pd

from rul_calculator import calculate_rul
from battery_health import battery_health
from recommendation import get_recommendation

# Load trained model


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "soh_model.pkl")

model = joblib.load(MODEL_PATH)


def predict_battery(features):
    """
    Predict SOH, RUL, Battery Health and Recommendation
    """

    columns = [
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

    data = pd.DataFrame([features], columns=columns)

    predicted_soh = float(model.predict(data)[0])

    rul = calculate_rul(predicted_soh, features[0])

    health = battery_health(predicted_soh)

    recommendation = get_recommendation(
        predicted_soh,
        rul,
        features[3]   # Temperature
    )

    return {
        "SOH": round(predicted_soh, 2),
        "RUL": rul,
        "Health": health,
        "Recommendation": recommendation
    }


if __name__ == "__main__":

    sample = [
        450,      # Cycle
        3.72,     # Voltage
        1.50,     # Current
        32,       # Temperature
        1.8,      # ChargeTime
        1.7,      # DischargeTime
        0.048,    # InternalResistance
        2.10,     # Capacity
        52,       # AmbientHumidity
        1.0       # C_Rate
    ]

    result = predict_battery(sample)

    print(result)