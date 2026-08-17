# Calibrated empirical SOH degradation model parameters:
# SOH(Cycle) = SOH_INTERCEPT - DEGRADATION_SLOPE * Cycle
# SOH(Cycle) = 93.5244 - 0.032399 * Cycle
SOH_INTERCEPT = 93.5244
DEGRADATION_SLOPE = 0.032399

# End-of-Life (EOL) SOH threshold defined in industrial battery literature
EOL_SOH_THRESHOLD = 80.0

# Calibrated EOL Cycle where SOH reaches 80.0%:
# EOL_CYCLE = (93.5244 - 80.0) / 0.032399 ≈ 417.43 cycles
EOL_CYCLE = (SOH_INTERCEPT - EOL_SOH_THRESHOLD) / DEGRADATION_SLOPE


def calculate_rul(predicted_soh, current_cycle):
    """
    Estimate Remaining Useful Life (RUL) in charging cycles until
    the battery reaches the End-of-Life (EOL) threshold (SOH = 80%).

    Prognostic Methodology:
        RUL is a model-derived prognostic estimation based on the calibrated
        global SOH degradation model:
            SOH = 93.5244 - 0.032399 * Cycle
        Setting SOH = 80.0% yields the empirical EOL boundary:
            EOL_cycle = (93.5244 - 80.0) / 0.032399 ≈ 417.43 cycles

        RUL is calculated as:
            - If SOH <= 80.0%: RUL = 0 (Battery has reached/exceeded EOL)
            - If SOH > 80.0%:  RUL = max(0, round(EOL_cycle - current_cycle))

    Note:
        This is a model-derived prognostic estimation, NOT an independently trained ML RUL model.
    """
    current_cycle = float(current_cycle)
    predicted_soh = float(predicted_soh)

    # If SOH is already at or below the 80% EOL threshold, remaining life is 0
    if predicted_soh <= EOL_SOH_THRESHOLD:
        return 0

    # Derived RUL relative to calibrated EOL cycle boundary
    estimated_rul = EOL_CYCLE - current_cycle

    return max(0, round(estimated_rul))


