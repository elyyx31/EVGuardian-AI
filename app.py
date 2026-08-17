from pathlib import Path
import os
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Ensure ml package is in path
BASE_DIR = Path(__file__).resolve().parent
ML_DIR = BASE_DIR / "ml"

from ml.predict import predict_battery, FEATURE_COLUMNS

app = Flask(__name__)

# Dataset path
DATASET_PATH = BASE_DIR / "dataset" / "new_battery_dataset.csv"

# Cache dataset history in memory on startup
_CACHED_HISTORY = None


def get_battery_history():
    """Load and cache the historical cycle-SOH degradation dataset once."""
    global _CACHED_HISTORY
    if _CACHED_HISTORY is None:
        try:
            df = pd.read_csv(DATASET_PATH)
            _CACHED_HISTORY = df[["Cycle", "SOH"]].dropna().to_dict(orient="records")
        except Exception as e:
            app.logger.error(f"Failed to load dataset history: {e}")
            _CACHED_HISTORY = []
    return _CACHED_HISTORY


# Exact empirical training distribution ranges from dataset/new_battery_dataset.csv
TRAINING_RANGES = {
    "Cycle": (1.0, 2000.0, "cycles"),
    "Voltage": (3.00, 4.20, "V"),
    "Current": (0.50, 2.00, "A"),
    "Temperature": (9.90, 40.97, "°C"),
    "ChargeTime": (30.00, 119.88, ""),
    "DischargeTime": (30.02, 119.97, ""),
    "InternalResistance": (0.0465, 0.2519, "Ω"),
    "Capacity": (1.460, 2.521, "Ah"),
    "AmbientHumidity": (30.03, 69.97, "%"),
    "C_Rate": (0.50, 2.00, "C")
}

# Physical plausibility boundaries to reject impossible or hazardous inputs
PHYSICAL_BOUNDS = {
    "Cycle": (0.0, 5000.0, "Cycle cannot be negative."),
    "Voltage": (2.0, 5.5, "Voltage must be between 2.0 V and 5.5 V."),
    "Current": (0.01, 10.0, "Current must be positive and <= 10.0 A."),
    "Temperature": (-30.0, 100.0, "Temperature must be physically plausible (-30°C to 100°C)."),
    "ChargeTime": (0.1, 300.0, "Charge Time must be > 0."),
    "DischargeTime": (0.1, 300.0, "Discharge Time must be > 0."),
    "InternalResistance": (0.001, 2.0, "Internal Resistance must be > 0 and <= 2.0 Ω."),
    "Capacity": (0.1, 10.0, "Capacity must be positive and <= 10.0 Ah."),
    "AmbientHumidity": (0.0, 100.0, "Humidity must be between 0% and 100%."),
    "C_Rate": (0.01, 5.0, "C-Rate must be > 0 and <= 5.0 C.")
}


def validate_and_parse_inputs(source_dict):
    """
    Validate inputs against physical boundaries and detect out-of-distribution samples.

    Returns:
        (parsed_features_dict, error_message, warning_message)
    """
    parsed = {}
    warnings = []

    for col in FEATURE_COLUMNS:
        raw_val = source_dict.get(col)
        if raw_val is None or str(raw_val).strip() == "":
            return None, f"Missing required parameter: {col}.", None
        try:
            val = float(raw_val)
        except (ValueError, TypeError):
            return None, f"Invalid non-numeric value for {col}: '{raw_val}'.", None

        # 1. Hard physical boundary check
        p_min, p_max, p_err = PHYSICAL_BOUNDS[col]
        if not (p_min <= val <= p_max):
            return None, f"Physical limit violation for {col}: {val}. {p_err}", None

        # 2. Training distribution domain detection
        t_min, t_max, t_unit = TRAINING_RANGES[col]
        if not (t_min <= val <= t_max):
            unit_str = f" {t_unit}" if t_unit else ""
            warnings.append(f"{col} ({val}{unit_str}) outside training range [{t_min:.2f}–{t_max:.2f}{unit_str}]")

        parsed[col] = val

    warning_msg = None
    if warnings:
        warning_msg = f"Input is outside the training-data range ({'; '.join(warnings)}); prediction may be unreliable."

    return parsed, None, warning_msg


@app.route("/")
def home():
    history = get_battery_history()
    return render_template(
        "dashboard.html",
        result=None,
        inputs=None,
        history=history,
        error=None,
        warning=None
    )


@app.route("/predict", methods=["POST"])
def predict():
    history = get_battery_history()
    raw_inputs = request.form.to_dict()

    parsed_inputs, error, warning = validate_and_parse_inputs(raw_inputs)
    if error:
        return render_template(
            "dashboard.html",
            result=None,
            inputs=raw_inputs,
            history=history,
            error=error,
            warning=None
        ), 400

    try:
        features_list = [parsed_inputs[col] for col in FEATURE_COLUMNS]
        result = predict_battery(features_list)

        return render_template(
            "dashboard.html",
            result=result,
            inputs=parsed_inputs,
            history=history,
            error=None,
            warning=warning
        )
    except Exception as e:
        app.logger.error(f"Prediction pipeline error: {e}")
        return render_template(
            "dashboard.html",
            result=None,
            inputs=parsed_inputs,
            history=history,
            error=f"Prediction failed: {str(e)}",
            warning=None
        ), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST JSON API endpoint for battery prognostics."""
    if not request.is_json:
        return jsonify({"success": False, "error": "Request payload must be JSON"}), 400

    payload = request.get_json()
    parsed_inputs, error, warning = validate_and_parse_inputs(payload)
    if error:
        return jsonify({"success": False, "error": error}), 400

    try:
        features_list = [parsed_inputs[col] for col in FEATURE_COLUMNS]
        result = predict_battery(features_list)
        response_data = {
            "success": True,
            "inputs": parsed_inputs,
            "prediction": result
        }
        if warning:
            response_data["warning"] = warning
        return jsonify(response_data)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    # Ensure memory cache is warmed up
    get_battery_history()

    # Configurable debug mode (defaults to False for production safety)
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1", "t")
    app.run(debug=debug_mode, host="127.0.0.1", port=5000)
