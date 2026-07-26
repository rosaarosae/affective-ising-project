"""Reusable metrics for analysing Affective Ising Model trajectories."""

import numpy as np


def classify_basin(y1, y2):
    """Classify the basin of attraction based on the values of y1 and y2."""
    if y1 > y2:
        return "positive"
    elif y2 > y1:
        return "negative"
    else:
        return "neutral"


def does_trajectory_end_in_negative_basin(trajectory):
    last_y1, last_y2 = trajectory[-1]
    return classify_basin(last_y1, last_y2) == "negative"


def deepest_m_into_negative(trajectory):
    y1 = trajectory[:, 0]
    y2 = trajectory[:, 1]
    m = y1-y2
    return np.min(m)


def deepest_m_into_positive(trajectory):
    y1 = trajectory[:, 0]
    y2 = trajectory[:, 1]
    m = y1-y2
    return np.max(m)


def recovery_time_after_negative_input(times, trajectory, start, duration, window=1000):
    input_end_time = start + duration
    # Find the index where the input ends
    end_index = np.searchsorted(times, input_end_time)
    # Search for the first period where y1 > y2 for 'window' consecutive steps
    for i in range(end_index, len(trajectory) - window + 1):
        if np.all(trajectory[i:i + window, 0] > trajectory[i:i + window, 1]):
            return times[i] - input_end_time  # Recovery time is relative to input end
    return np.nan  # Return NaN if recovery does not occur within the simulation


def classify_recovery_outcome(times, trajectory, start, duration, window=1000):
    recovery_time = recovery_time_after_negative_input(
        times,
        trajectory,
        start,
        duration,
        window
    )

    if np.isnan(recovery_time):
        return "never recovered"

    final_window_is_positive = np.all(
        trajectory[-window:, 0]
        >
        trajectory[-window:, 1]
    )

    if final_window_is_positive:
        return "persistent recovery"
    else:
        return "temporary recovery"
