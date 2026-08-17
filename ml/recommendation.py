def get_recommendation(soh, rul, temperature):
    """
    Generate maintenance recommendations based on SOH, RUL, and thermal safety.

    Priority:
    1. Thermal safety warning (temperature > 45°C)
    2. SOH Health category:
       - SOH >= 90: Healthy / Normal operation
       - 80 <= SOH < 90: Moderate / Preventive maintenance
       - SOH < 80: Poor / End-of-life replacement
    RUL is included as supporting contextual metric and never contradicts health status.
    """
    if temperature > 45:
        return f"CRITICAL: High battery temperature detected ({temperature:.1f}°C). Allow the battery to cool down before charging or operating under high load."

    if soh >= 90:
        return f"Battery is operating normally with healthy State of Health ({soh:.1f}%). Estimated ~{int(rul)} cycles remaining to EOL threshold."

    elif soh >= 80:
        return f"Battery health is moderate ({soh:.1f}%). Schedule preventive maintenance and monitor cell degradation (~{int(rul)} cycles remaining to EOL)."

    else:
        return f"Battery has reached End-of-Life threshold (SOH: {soh:.1f}% < 80%). Battery replacement is recommended to maintain vehicle reliability."


