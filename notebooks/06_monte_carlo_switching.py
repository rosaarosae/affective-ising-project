#%%
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))
import numpy as np
from aim.simulation import euler_maruyama
from aim.metrics import classify_basin, does_trajectory_end_in_negative_basin, deepest_m_into_negative

#%%
times1, trajectory1 = euler_maruyama(
    y0=np.array([0.96, 0.04]),
    dt=0.0005,
    T=100,      
    D=0.01,
    seed=42
)
times2, trajectory2 = euler_maruyama(
    y0=np.array([0.04, 0.96]),
    dt=0.0005,
    T=100,      
    D=0.01,
    seed=42
)
#%%
print("Trajectory 1 ends in negative basin:", does_trajectory_end_in_negative_basin(trajectory1))
print("Trajectory 2 ends in negative basin:", does_trajectory_end_in_negative_basin(trajectory2))
print("Deepest m into negative for trajectory 1:", deepest_m_into_negative(trajectory1))
print("Deepest m into negative for trajectory 2:", deepest_m_into_negative(trajectory2))
# %%
# now for N runs
N_runs = 100
ending_in_negative = []
deepest_m_values = []
for i in range(N_runs):
    print(f"Run {i+1}/{N_runs}")
    times, trajectory = euler_maruyama(
        y0=np.array([0.96, 0.04]),
        dt=0.0005,
        T=100,      
        D=0.01,
        seed=i
    )
    ending_in_negative.append(does_trajectory_end_in_negative_basin(trajectory))
    deepest_m_values.append(deepest_m_into_negative(trajectory))
    
# %%
print(f"Fraction of trajectories ending in negative basin: {np.sum(ending_in_negative) / N_runs}")
# %%
plt.hist(deepest_m_values, bins=20)
plt.title("Histogram of deepest m into negative basin")
plt.xlabel("Deepest m value")
plt.ylabel("Frequency")
# %%
fraction_below_threshold = []
for threshold in np.linspace(-1, 1, 500):
    fraction_below_threshold.append(
        np.sum(np.array(deepest_m_values) < threshold) / N_runs
    )
plt.plot(np.linspace(-1, 1, 500), fraction_below_threshold, marker='o', ms=2, ls='')
plt.title("Fraction of trajectories with deepest m below threshold")
plt.xlabel("Threshold")
plt.ylabel("Fraction")
    
# %%