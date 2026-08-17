MAX_CYCLE_LIFE = 1000

def calculate_rul(predicted_soh, current_cycle):
    """
    Estimate Remaining Useful Life (RUL) in charging cycles.
    """

    remaining_cycles = MAX_CYCLE_LIFE - current_cycle

    estimated_rul = (predicted_soh / 100) * remaining_cycles

    return max(0, round(estimated_rul))
