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
from aim.metrics import classify_basin, does_trajectory_end_in_negative_basin, deepest_m_into_negative, recovery_time_after_negative_input, classify_recovery_outcome
from aim.inputs import rectangular_negative_input

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
N_runs = 20
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
plt.figure()
plt.plot(np.linspace(-1, 1, 500), fraction_below_threshold, marker='o', ms=2, ls='')
plt.title("Fraction of trajectories with deepest m below threshold")
plt.xlabel("Threshold")
plt.ylabel("Fraction")
plt.ylim(0, 1)
    
# %%
# now we compare the control, weak input and strong input with several random seeds

N_runs = 20 #we first use 20 runs as a pilot experiment

initial_state = np.array([0.96, 0.04])
dt = 0.0005
final_time = 50
D = 0.01

#we define the two negative inputs
def weak_input(t):
    return rectangular_negative_input(
        t,
        amplitude=25,
        start=2,
        duration=3
    )


def strong_input(t):
    return rectangular_negative_input(
        t,
        amplitude=100,
        start=2,
        duration=3
    )


#we create three lists to save if every trajectory ends in the negative basin
control_switches = []
weak_switches = []
strong_switches = []
control_recovery_times = []
weak_recovery_times = []
strong_recovery_times = []
strong_recovery_outcomes = []

#we use the same seed for the three conditions in every run
for i in range(N_runs):
    print("Run:", i + 1, "of", N_runs)

    times_control, trajectory_control = euler_maruyama(
        y0=initial_state,
        dt=dt,
        T=final_time,
        D=D,
        seed=i,
        negative_input=None
    )

    times_weak, trajectory_weak = euler_maruyama(
        y0=initial_state,
        dt=dt,
        T=final_time,
        D=D,
        seed=i,
        negative_input=weak_input
    )

    times_strong, trajectory_strong = euler_maruyama(
        y0=initial_state,
        dt=dt,
        T=final_time,
        D=D,
        seed=i,
        negative_input=strong_input
    )

    control_switches.append(
        does_trajectory_end_in_negative_basin(
            trajectory_control
        )
    )

    weak_switches.append(
        does_trajectory_end_in_negative_basin(
            trajectory_weak
        )
    )

    strong_switches.append(
        does_trajectory_end_in_negative_basin(
            trajectory_strong
        )
    )

    control_recovery_times.append(
        recovery_time_after_negative_input(
            times_control,
            trajectory_control,
            start=2,
            duration=3,
            window=1000
        )
    )

    weak_recovery_times.append(
        recovery_time_after_negative_input(
            times_weak,
            trajectory_weak,
            start=2,
            duration=3,
            window=1000
        )
    )

    strong_recovery_times.append(
        recovery_time_after_negative_input(
            times_strong,
            trajectory_strong,
            start=2,
            duration=3,
            window=1000
        )
    )

    strong_recovery_outcomes.append(
        classify_recovery_outcome(
            times_strong,
            trajectory_strong,
            start=2,
            duration=3,
            window=1000
        )
    )


# %%
#we calculate the proportion of trajectories that end in the negative basin

control_probability = np.sum(control_switches) / N_runs
weak_probability = np.sum(weak_switches) / N_runs
strong_probability = np.sum(strong_switches) / N_runs

print("Control switching probability:", control_probability)
print("Weak input switching probability:", weak_probability)
print("Strong input switching probability:", strong_probability)
print("Mean control recovery time:", np.nanmean(control_recovery_times))
print("Mean weak-input recovery time:", np.nanmean(weak_recovery_times))
print("Mean observed strong-input recovery time:", np.nanmean(strong_recovery_times))

number_recovered = np.sum(
    ~np.isnan(strong_recovery_times)
)

number_not_recovered = np.sum(
    np.isnan(strong_recovery_times)
)

number_persistent_recovery = np.sum(
    np.array(strong_recovery_outcomes)
    ==
    "persistent recovery"
)

number_temporary_recovery = np.sum(
    np.array(strong_recovery_outcomes)
    ==
    "temporary recovery"
)

number_never_recovered = np.sum(
    np.array(strong_recovery_outcomes)
    ==
    "never recovered"
)

print("Strong-input trajectories recovered:", number_recovered)
print("Strong-input trajectories not recovered:", number_not_recovered)
print("Persistent recovery:", number_persistent_recovery)
print("Temporary recovery:", number_temporary_recovery)
print("Never recovered:", number_never_recovered)


# %%
#we compare the three switching probabilities in a bar plot

condition_names = [
    "Control",
    "Weak input",
    "Strong input"
]

switching_probabilities = [
    control_probability,
    weak_probability,
    strong_probability
]

plt.figure(figsize=(7, 5))

plt.bar(
    condition_names,
    switching_probabilities,
    color=["grey", "green", "red"]
)

plt.ylabel("Switching probability")
plt.title("Probability of ending in the negative-affect basin")
plt.ylim(0, 1)

plt.savefig(
    PROJECT_ROOT / "results/figures/monte_carlo_switching.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
