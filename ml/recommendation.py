def get_recommendation(soh, rul, temperature):
    """
    Generate maintenance recommendation based on
    SOH, RUL and battery temperature.
    """

    if temperature > 45:
        return "High battery temperature detected. Allow the battery to cool before charging."

    elif soh >= 90 and rul >= 500:
        return "Battery is operating normally. No maintenance required."

    elif soh >= 80 and rul >= 200:
        return "Battery health is moderate. Schedule preventive maintenance."

    else:
        return "Battery nearing end of life. Battery replacement is recommended."

