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
