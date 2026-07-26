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

if __name__ == "__main__":
    # tests
    trajectory = np.array([[0.9, 0.1], [0.8, 0.2]])
    print(does_trajectory_end_in_negative_basin(trajectory))  # False
    print(deepest_m_into_negative(trajectory))  # 0.6
    print(deepest_m_into_positive(trajectory))  # 0.8
    trajectory = np.array([[0.1, 0.9], [0.2, 0.8], [0.9, 0.1]])
    print(does_trajectory_end_in_negative_basin(trajectory))  # True
    print(deepest_m_into_negative(trajectory))
    print(deepest_m_into_positive(trajectory))

     
# Recovery time represents how long the simulated affective state needs to
# return to the positive-affect basin after the negative input has ended.
# The input ends at t = start + duration. From that moment, we search for
# the first period in which positive affect y1 remains greater than negative
# affect y2 for several consecutive steps. We use a window because a single
# point with y1 > y2 could only be a short random fluctuation. If the system
# does not return during the simulation, the recovery time is recorded as NaN.
# This is an abstract model measure and does not represent hours or days.
def recovery_time_after_negative_input(times, trajectory, start, duration, window=100):
    input_end_time = start + duration
    # Find the index where the input ends
    end_index = np.searchsorted(times, input_end_time)
    # Search for the first period where y1 > y2 for 'window' consecutive steps
    for i in range(end_index, len(trajectory) - window + 1):
        if np.all(trajectory[i:i + window, 0] > trajectory[i:i + window, 1]):
            return times[i] - input_end_time  # Recovery time is relative to input end
    return np.nan  # Return NaN if recovery does not occur within the simulation        
