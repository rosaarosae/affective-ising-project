#09_Equal_Area_Inputs
#
# The original Affective Ising Model includes time-dependent external
# inputs B1(t) and B2(t). B1(t) acts on positive affect and B2(t) acts
# on negative affect.
#
# The original paper does not study these inputs because the datasets
# do not contain enough information about external events. However,
# the authors suggest that controlled stimuli can be represented using
# boxcar functions. A boxcar function is constant and nonzero during
# a specific time interval and zero outside that interval.
#
# In this experiment, we study whether the temporal structure of a
# negative input changes the response of the AIM. We compare:
#
# 1. Acute input:
#    One strong negative event that lasts for a short time.
#
# 2. Prolonged input:
#    A weaker negative influence that lasts for a longer time.
#
# 3. Fragmented input:
#    Three short negative events separated by periods without input.
#
# The three inputs have the same total area. The area represents the
# accumulated magnitude of the negative input:
#
# area = amplitude * duration
#
# Keeping the area equal makes the comparison controlled. If one input
# had a larger area, we would not know whether a difference in switching
# probability was caused by temporal structure or simply by receiving
# more total input.

# %%
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# We find the main project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# We add the src folder to Python's import path.
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

# We reuse the rectangular input function that is already in src.
from aim.inputs import rectangular_negative_input
from aim.simulation import euler_maruyama
from aim.metrics import does_trajectory_end_in_negative_basin
from statsmodels.stats.proportion import proportion_confint

#we start by defining the initial conditions
input_start=2

#first input
#the first input is high amplitud and short duration
acute_amplitude = 100
acute_duration = 3

#second input
#the second one is low but prolonged on time
# Prolonged input:
# lower amplitude and longer duration.
prolonged_amplitude = 50
prolonged_duration = 6

# third input:
# three short pulses with pauses between them
fragmented_amplitude = 100
fragmented_duration = 1
fragmented_starts = [2, 4, 6]
# %%
#now we define the different functions to define the diff. type of pulses
def acute_input(t):
    return rectangular_negative_input(
        t,
        amplitude=acute_amplitude,
        start=input_start,
        duration=acute_duration
    )

def prolonged_input(t):
    return rectangular_negative_input(
        t,
        amplitude=prolonged_amplitude,
        start=input_start,
        duration=prolonged_duration
    )
def fragmented_input(t):
    total_input = 0.0

    for pulse_start in fragmented_starts:
        pulse_value = rectangular_negative_input(
            t,
            amplitude=fragmented_amplitude,
            start=pulse_start,
            duration=fragmented_duration
        )

        total_input += pulse_value

    return total_input
# %%
#now we calculate the theorical area of the different inputs
acute_area = (
    acute_amplitude
    * acute_duration
)

prolonged_area = (
    prolonged_amplitude
    * prolonged_duration
)
#for the fragmented pulse we have to multiply the area of one pulse by
# the number of pulses.
fragmented_area = (
    fragmented_amplitude
    * fragmented_duration
    * len(fragmented_starts)
)

print(
    "Acute input area:",
    acute_area
)

print(
    "Prolonged input area:",
    prolonged_area
)

print(
    "Fragmented input area:",
    fragmented_area
)


# %%
# We check that the three areas are equal.
# If one comparison is incorrect, the program stops at the assert.

assert np.isclose(
    acute_area,
    prolonged_area
)

assert np.isclose(
    acute_area,
    fragmented_area
)

print(
    "All input areas are equal"
)
# %%
# We create a time vector to evaluate and draw the input functions.

input_times = np.linspace(
    0,
    10,
    1000
)

# We create empty lists where we will save the input value at every time.

acute_values = []
prolonged_values = []
fragmented_values = []


# %%
# We evaluate the three input functions at every point of the time vector.

for t in input_times:

    acute_values.append(
        acute_input(t)
    )

    prolonged_values.append(
        prolonged_input(t)
    )

    fragmented_values.append(
        fragmented_input(t)
    )


# %%
# We plot each input separately because some lines have the same value
# at the same time and would cover each other in a single plot.

plt.figure(figsize=(9, 7))

plt.subplot(3, 1, 1)
plt.plot(
    input_times,
    acute_values,
    color="blue",
    linewidth=2
)
plt.title("Acute input")
plt.ylabel("B2(t)")
plt.ylim(-5, 110)
plt.grid(alpha=0.25)

plt.subplot(3, 1, 2)
plt.plot(
    input_times,
    prolonged_values,
    color="orange",
    linewidth=2
)
plt.title("Prolonged input")
plt.ylabel("B2(t)")
plt.ylim(-5, 110)
plt.grid(alpha=0.25)

plt.subplot(3, 1, 3)
plt.plot(
    input_times,
    fragmented_values,
    color="green",
    linewidth=2
)
plt.title("Fragmented input")
plt.xlabel("Time")
plt.ylabel("B2(t)")
plt.ylim(-5, 110)
plt.grid(alpha=0.25)

plt.suptitle("Negative inputs with equal total area")
plt.tight_layout()

plt.savefig(
    PROJECT_ROOT
    / "results"
    / "figures"
    / "equal_area_inputs.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()

# %%
# The model contains random noise, so we repeat the four conditions with
# different seeds and calculate the proportion of trajectories that end
# in the negative-affect basin.

initial_state = np.array([0.96, 0.04])
dt = 0.0005
final_time = 50
D = 0.01
seed = 42

N_runs = 100

control_switches = 0
acute_switches = 0
prolonged_switches = 0
fragmented_switches = 0

for run in range(N_runs):
    print("Run:", run + 1, "of", N_runs)

    times_control, trajectory_control = euler_maruyama(
        y0=initial_state,
        dt=dt,
        T=final_time,
        D=D,
        seed=run,
        negative_input=None
    )

    times_acute, trajectory_acute = euler_maruyama(
        y0=initial_state,
        dt=dt,
        T=final_time,
        D=D,
        seed=run,
        negative_input=acute_input
    )

    times_prolonged, trajectory_prolonged = euler_maruyama(
        y0=initial_state,
        dt=dt,
        T=final_time,
        D=D,
        seed=run,
        negative_input=prolonged_input
    )

    times_fragmented, trajectory_fragmented = euler_maruyama(
        y0=initial_state,
        dt=dt,
        T=final_time,
        D=D,
        seed=run,
        negative_input=fragmented_input
    )

    if does_trajectory_end_in_negative_basin(trajectory_control):
        control_switches += 1

    if does_trajectory_end_in_negative_basin(trajectory_acute):
        acute_switches += 1

    if does_trajectory_end_in_negative_basin(trajectory_prolonged):
        prolonged_switches += 1

    if does_trajectory_end_in_negative_basin(trajectory_fragmented):
        fragmented_switches += 1

# %%
# The Monte Carlo switching probability is the number of switches
# divided by the total number of simulated trajectories.

switching_counts = [
    control_switches,
    acute_switches,
    prolonged_switches,
    fragmented_switches
]

switching_probabilities = []

for count in switching_counts:
    switching_probabilities.append(
        count / N_runs
    )

condition_names = [
    "Control",
    "Acute",
    "Prolonged",
    "Fragmented"
]

for condition, probability in zip(
    condition_names,
    switching_probabilities
):
    print(
        condition,
        "switching probability:",
        probability
    )

# %%
# We calculate Wilson 95% confidence intervals because the switching
# result is binary: every trajectory either switches or does not switch.

confidence_intervals = []

for count in switching_counts:
    interval = proportion_confint(
        count,
        N_runs,
        0.05,
        method="wilson"
    )
    confidence_intervals.append(interval)

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

# %%
# Finally, we compare the four switching probabilities. The three
# negative inputs have the same total area, so differences between them
# are associated with their temporal organization.

plt.figure(figsize=(8, 5))

plt.bar(
    condition_names,
    switching_probabilities,
    color=["grey", "blue", "orange", "green"],
    edgecolor="black"
)

plt.errorbar(
    condition_names,
    switching_probabilities,
    yerr=confidence_errors,
    fmt="none",
    color="black",
    capsize=5
)

for position, probability in enumerate(
    switching_probabilities
):
    plt.text(
        position,
        probability + upper_errors[position] + 0.03,
        f"{probability:.2f}",
        ha="center"
    )

plt.ylabel("Switching probability")
plt.title("Switching probability for equal-area negative inputs")
plt.ylim(0, 1.1)
plt.grid(axis="y", alpha=0.25)

plt.savefig(
    PROJECT_ROOT
    / "results"
    / "figures"
    / "equal_area_switching.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()

# %%
# Now we simulate one trajectory for every input shape.
# We use the same initial state, parameters and random seed in the four
# conditions. This means that the noise is the same, so the difference
# between trajectories comes from the external input.

times_control, trajectory_control = euler_maruyama(
    y0=initial_state,
    dt=dt,
    T=final_time,
    D=D,
    seed=seed,
    negative_input=None
)

times_acute, trajectory_acute = euler_maruyama(
    y0=initial_state,
    dt=dt,
    T=final_time,
    D=D,
    seed=seed,
    negative_input=acute_input
)

times_prolonged, trajectory_prolonged = euler_maruyama(
    y0=initial_state,
    dt=dt,
    T=final_time,
    D=D,
    seed=seed,
    negative_input=prolonged_input
)

times_fragmented, trajectory_fragmented = euler_maruyama(
    y0=initial_state,
    dt=dt,
    T=final_time,
    D=D,
    seed=seed,
    negative_input=fragmented_input
)

# %%
# We draw the four trajectories to see if the same accumulated input
# produces a different response when its temporal structure changes.

plt.figure(figsize=(10, 10))

plt.subplot(4, 1, 1)
plt.plot(times_control, trajectory_control[:, 0], label="Positive affect y1")
plt.plot(times_control, trajectory_control[:, 1], label="Negative affect y2")
plt.title("Control: no external input")
plt.ylabel("Affective state")
plt.ylim(0, 1)
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(times_acute, trajectory_acute[:, 0], label="Positive affect y1")
plt.plot(times_acute, trajectory_acute[:, 1], label="Negative affect y2")
plt.title("Acute negative input")
plt.ylabel("Affective state")
plt.ylim(0, 1)

plt.subplot(4, 1, 3)
plt.plot(times_prolonged, trajectory_prolonged[:, 0], label="Positive affect y1")
plt.plot(times_prolonged, trajectory_prolonged[:, 1], label="Negative affect y2")
plt.title("Prolonged negative input")
plt.ylabel("Affective state")
plt.ylim(0, 1)

plt.subplot(4, 1, 4)
plt.plot(times_fragmented, trajectory_fragmented[:, 0], label="Positive affect y1")
plt.plot(times_fragmented, trajectory_fragmented[:, 1], label="Negative affect y2")
plt.title("Fragmented negative input")
plt.xlabel("Time")
plt.ylabel("Affective state")
plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    PROJECT_ROOT
    / "results"
    / "figures"
    / "equal_area_trajectories.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()
