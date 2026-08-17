from flask import Flask, render_template, request
import sys
import os
import pandas as pd

# Add ml folder to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "ml"))

from predict import predict_battery

app = Flask(__name__)
# Dataset path
DATASET_PATH = os.path.join(
    os.path.dirname(__file__),
    "dataset",
    "new_battery_dataset.csv"
)

def get_battery_history():

    df = pd.read_csv(DATASET_PATH)

    # Keep only the required columns
    history = df[["Cycle", "SOH"]].dropna()

    return history.to_dict(orient="records")


@app.route("/")
def home():

    history = get_battery_history()

    return render_template(
        "dashboard.html",
        result=None,
        inputs=None,
        history=history
    )

@app.route("/predict", methods=["POST"])
def predict():

    # Read all input values
    cycle = float(request.form["Cycle"])
    voltage = float(request.form["Voltage"])
    current = float(request.form["Current"])
    temperature = float(request.form["Temperature"])
    charge_time = float(request.form["ChargeTime"])
    discharge_time = float(request.form["DischargeTime"])
    internal_resistance = float(request.form["InternalResistance"])
    capacity = float(request.form["Capacity"])
    humidity = float(request.form["AmbientHumidity"])
    c_rate = float(request.form["C_Rate"])

    # Feature list for prediction
    features = [
        cycle,
        voltage,
        current,
        temperature,
        charge_time,
        discharge_time,
        internal_resistance,
        capacity,
        humidity,
        c_rate
    ]

    # Predict
    result = predict_battery(features)

    # Send input values to HTML
    inputs = {
        "Cycle": cycle,
        "Voltage": voltage,
        "Current": current,
        "Temperature": temperature,
        "ChargeTime": charge_time,
        "DischargeTime": discharge_time,
        "InternalResistance": internal_resistance,
        "Capacity": capacity,
        "AmbientHumidity": humidity,
        "C_Rate": c_rate
    }

    history = get_battery_history()

    return render_template(
        "dashboard.html",
        result=result,
        inputs=inputs,
        history=history
    )


if __name__ == "__main__":
    app.run(debug=True)