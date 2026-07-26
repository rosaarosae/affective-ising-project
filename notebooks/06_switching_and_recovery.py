#%%
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))
from aim.simulation import euler_maruyama
from aim.metrics import classify_basin, does_trajectory_end_in_negative_basin, deepest_m_into_negative, recovery_time_after_negative_input, classify_recovery_outcome
from aim.inputs import rectangular_negative_input

#we use the same simulation parameters throughout this file
initial_state = np.array([0.96, 0.04])
dt = 0.0005
final_time = 50
D = 0.01

#%%
times1, trajectory1 = euler_maruyama(
    y0=np.array([0.96, 0.04]),
    dt=dt,
    T=final_time,
    D=D,
    seed=42
)
times2, trajectory2 = euler_maruyama(
    y0=np.array([0.04, 0.96]),
    dt=dt,
    T=final_time,
    D=D,
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


#we create lists to save the results from the three conditions
ending_in_negative = []
deepest_m_values = []
weak_switches = []
strong_switches = []
control_recovery_times = []
weak_recovery_times = []
strong_recovery_times = []
strong_recovery_outcomes = []

#we use the same seed for the three conditions in every run
for i in range(N_runs):
    print(f"Run {i+1}/{N_runs}")

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

    ending_in_negative.append(
        does_trajectory_end_in_negative_basin(
            trajectory_control
        )
    )

    deepest_m_values.append(
        deepest_m_into_negative(
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
print(f"Control fraction ending in negative basin: {np.sum(ending_in_negative) / N_runs}")
# %%
plt.figure(figsize=(7, 5))
plt.hist(deepest_m_values, bins=20)
plt.title("Histogram of deepest m into negative basin")
plt.xlabel("Deepest m value")
plt.ylabel("Frequency")
plt.show()
plt.close()
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
plt.show()
plt.close()

# %%
#we calculate the proportion of trajectories that end in the negative basin

control_probability = np.sum(ending_in_negative) / N_runs
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
# %%
#we draw the recovery outcomes for the strong negative input

recovery_names = [
    "Persistent\nrecovery",
    "Temporary\nrecovery",
    "Never\nrecovered"
]

recovery_counts = [
    number_persistent_recovery,
    number_temporary_recovery,
    number_never_recovered
]

plt.figure(figsize=(7, 5))

bars = plt.bar(
    recovery_names,
    recovery_counts,
    color=["green", "orange", "red"]
)

#we write the number of trajectories above each bar
for bar, count in zip(bars, recovery_counts):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.2,
        str(count),
        ha="center"
    )

plt.ylabel("Number of trajectories")
plt.title("Recovery outcomes after the strong negative input")
plt.ylim(0, N_runs + 2)

plt.savefig(
    PROJECT_ROOT / "results/figures/recovery_outcomes.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()
# %%
#finally, we calculate the confidence intervals for the switching probabilities using the Wilson score interval
from statsmodels.stats.proportion import proportion_confint
control_switch_count=np.sum(ending_in_negative) #these values just count the number of trajectories that ended in the negative basin for each condition
weak_switch_count=np.sum(weak_switches)
strong_switch_count=np.sum(strong_switches)

#now we calculate the confidence intervals
#now we calculate the 95% confidence intervals
control_ci = proportion_confint(control_switch_count, N_runs, 0.05, method='wilson')
weak_ci = proportion_confint(weak_switch_count, N_runs, 0.05, method='wilson')
strong_ci = proportion_confint(strong_switch_count, N_runs, 0.05, method='wilson')

print("Control 95% confidence interval:", control_ci)
print("Weak-input 95% confidence interval:", weak_ci)
print("Strong-input 95% confidence interval:", strong_ci)

#we prepare the results for printing
confidence_intervals = [
    control_ci,
    weak_ci,
    strong_ci
]

switching_probabilities = [
    control_probability,
    weak_probability,
    strong_probability
]

lower_errors = []
upper_errors = []

for probability, interval in zip(
    switching_probabilities,
    confidence_intervals
):
    lower, upper = interval

    lower_errors.append(
        probability - lower
    )

    upper_errors.append(
        upper - probability
    )

confidence_errors = np.array([
    lower_errors,
    upper_errors
])
#now we plot the switching probabilities with error bars
plt.figure(figsize=(6.5, 4.5))
bars = plt.bar(
    condition_names,
    switching_probabilities,
    color=["grey", "green", "red"],
    yerr=confidence_errors,
    capsize=6,
    width=0.55,
    edgecolor="black",
    linewidth=0.8
)

for bar, probability, interval in zip(
    bars,
    switching_probabilities,
    confidence_intervals
):
    upper_limit = interval[1]
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        upper_limit + 0.025,
        f"{probability:.2f}",
        ha="center",
        va="bottom"
    )

plt.ylabel("Switching probability")
plt.title("Switching probability by input condition")
plt.ylim(0, 1)
plt.grid(axis="y", alpha=0.25)

plt.savefig(
    PROJECT_ROOT
    / "results"
    / "figures"
    / "monte_carlo_switching.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()
