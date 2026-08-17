def battery_health(predicted_soh):
    """
    Determine battery health based on predicted SOH.
    """

    if predicted_soh >= 90:
        return "Healthy"

    elif predicted_soh >= 80:
        return "Moderate"

    else:
        return "Poor"

