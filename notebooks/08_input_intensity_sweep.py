#08_input_intensity_sweep
#we only have tried 25 and 100 but we are gonna try some more to see ho it chsnges when there is less difference


# %%
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from aim.inputs import rectangular_negative_input
from aim.simulation import euler_maruyama
from aim.metrics import does_trajectory_end_in_negative_basin
from statsmodels.stats.proportion import proportion_confint
#%%
amplitudes = [0, 25, 50, 70, 75, 80, 85, 90, 95, 100]
N_runs = 100
initial_state = np.array([0.96, 0.04])
dt = 0.0005
final_time = 50
D = 0.01
switching_probabilities = [] #we create an empty list to save one switching probability for every amplitude
switching_counts = [] #we also save the number of switches to calculate the confidence intervals
#%%
#now we see the effect of the amplitude of the negative input on the probability of switching to the negative basin

for amplitude in amplitudes:

    # The counter starts again from zero for every amplitude.
    negative_basin_count = 0

    # This function creates the rectangular input for the amplitude
    # currently being studied. Euler-Maruyama will evaluate it at
    # every simulation step, so the input modifies the dynamics while
    # the trajectory is generated, not after the simulation.
    def input_function(t):
        return rectangular_negative_input(
            t,
            amplitude=amplitude,
            start=2,
            duration=3
        )

    # We generate several independent stochastic trajectories.
    for run in range(N_runs):
        times, trajectory = euler_maruyama(
            y0=initial_state,
            dt=dt,
            T=final_time,
            D=D,
            seed=run,  # A different noise realization for each run
            negative_input=input_function
        )

        # The result is True when the trajectory ends with negative
        # affect y2 greater than positive affect y1.
        ends_in_negative_basin = (
            does_trajectory_end_in_negative_basin(
                trajectory
            )
        )

        if ends_in_negative_basin:
            negative_basin_count += 1

    # Monte Carlo estimate:
    # number of negative final states divided by total simulations.
    probability_of_switching = (
        negative_basin_count / N_runs
    )

    # We save one probability for each amplitude so that we can
    # draw the intensity-response curve afterwards.
    switching_probabilities.append(
        probability_of_switching
    )
    switching_counts.append(
        negative_basin_count
    )

    print(
        f"Amplitude: {amplitude}, "
        f"switching probability: "
        f"{probability_of_switching:.2f}"
    )
# %%
# We plot the estimated switching probability for every input amplitude.
# This allows us to identify the range where switching becomes more likely.

#we calculate a 95% Wilson confidence interval for every amplitude
confidence_intervals = []

for switching_count in switching_counts:
    confidence_interval = proportion_confint(
        switching_count,
        N_runs,
        0.05,
        method="wilson"
    )
    confidence_intervals.append(
        confidence_interval
    )

#matplotlib needs the distance from the probability to the lower and upper limits
lower_errors = []
upper_errors = []

for probability, interval in zip(
    switching_probabilities,
    confidence_intervals
):
    lower, upper = interval
    lower_errors.append(probability - lower)
    upper_errors.append(upper - probability)

confidence_errors = np.array([
    lower_errors,
    upper_errors
])

plt.figure(figsize=(7, 5))

plt.errorbar(
    amplitudes,
    switching_probabilities,
    yerr=confidence_errors,
    marker="o",
    linewidth=2,
    capsize=4
)

plt.xlabel("Negative input amplitude")
plt.ylabel("Switching probability")
plt.title("Effect of input intensity on switching probability")
plt.ylim(0, 1)
plt.grid(alpha=0.25)

plt.savefig(
    PROJECT_ROOT / "results/figures/input_intensity_sweep.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()
