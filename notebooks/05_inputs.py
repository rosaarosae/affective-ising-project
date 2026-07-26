# 05_inputs.py
# The original AIM introduces time-dependent external inputs B1(t) and B2(t),
# but it does not specify their exact temporal form. Here, we use a rectangular
# negative input as a simple controlled approximation. During a fixed interval,
# the external event tilts the energy landscape towards higher negative affect.
# Outside this interval, the input is zero.
#
# Mathematically:
# B2(t) = amplitude, if start <= t < start + duration
# B2(t) = 0, otherwise

# %%
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))


from aim.simulation import euler_maruyama
# %%
def rectangular_negative_input(t, amplitude, start, duration):
    """Return the intensity of a rectangular negative input at time t."""
    end=start + duration
    if start <= t < end:
        return amplitude
    else:
        return 0.0
#test to see if the function works
print("Before:", rectangular_negative_input(1, 4, 2, 3))
print("Start:", rectangular_negative_input(2, 4, 2, 3))
print("During:", rectangular_negative_input(4, 4, 2, 3))
print("End:", rectangular_negative_input(5, 4, 2, 3))
#we define the parameters for the rectangular input, this is a test
initial_state = np.array([0.95, 0.05])  # Start in the positive-affect homebase, because we want to see if the negative input can switch the system to the negative homebase
dt = 0.0005
final_time = 20
D = 0.01
seed = 42
#we prepare a function that only is dependent on time, so we can use it in the euler_maruyama function
def input_function(t):  
    return rectangular_negative_input(t, amplitude=25, start=2, duration=3) #these are not from the paper
#we simulate the trajectory with the rectangular negative input
times_control, trajectory_control = euler_maruyama(
    y0=initial_state,
    dt=dt,
    T=final_time,
    D=D,
    seed=seed,
    negative_input=None
)
# %%
#we simulate the trajectory with the rectangular negative input
times_input, trajectory_input = euler_maruyama(
    y0=initial_state,
    dt=dt,
    T=final_time,
    D=D,
    seed=seed,
    negative_input=input_function
)

#we compare the control trajectory and the trajectory with input
print(
    "Final control state:",
    trajectory_control[-1]
)

print(
    "Final state with input:",
    trajectory_input[-1]
)

print(
    "Maximum negative affect without input:",
    np.max(trajectory_control[:, 1])
)

print(
    "Maximum negative affect with input:",
    np.max(trajectory_input[:, 1])
)

print(
    "Maximum difference between trajectories:",
    np.max(
        np.abs(
            trajectory_input
            - trajectory_control
        )
    )
)
# %%
#we define a strong rectangular input. The amplitude is a synthetic pilot value,
#not a parameter obtained from the paper. It is the smallest tested amplitude
#that produces switching for this trajectory.
def strong_input_function(t):
    return rectangular_negative_input(
        t,
        amplitude=100,
        start=2,
        duration=3
    )

#we simulate the trajectory with the strong rectangular negative input
times_strong, trajectory_strong = euler_maruyama(
    y0=initial_state,
    dt=dt,
    T=final_time,
    D=D,
    seed=seed,
    negative_input=strong_input_function
)

#we check the effect of the strong input
print(
    "Final state with strong input:",
    trajectory_strong[-1]
)

print(
    "Maximum negative affect with strong input:",
    np.max(trajectory_strong[:, 1])
)

strong_switch = np.any(
    trajectory_strong[:, 1]
    > trajectory_strong[:, 0]
)

print(
    "Switching occurred with strong input:",
    strong_switch
)
#now, we are gonna create a figure to visualize the different trajectories
# %%
#first, we calculate the value of the two inputs at every time step so we can draw them
weak_input_values = np.zeros(len(times_control))
strong_input_values = np.zeros(len(times_control))

for i, time in enumerate(times_control):
    weak_input_values[i] = input_function(time)
    strong_input_values[i] = strong_input_function(time)

#we create four plots, one for every trajectory and one for the inputs
figure, axes = plt.subplots(
    4,
    1,
    figsize=(10, 10),
    sharex=True
)

#control trajectory
axes[0].plot(
    times_control,
    trajectory_control[:, 0],
    label="Positive affect y1",
    color="blue"
)

axes[0].plot(
    times_control,
    trajectory_control[:, 1],
    label="Negative affect y2",
    color="orange"
)

axes[0].set_title("Control: no external input")
axes[0].set_ylabel("Affective state")
axes[0].set_ylim(0, 1)
axes[0].legend()

#trajectory with the weak input
axes[1].plot(
    times_input,
    trajectory_input[:, 0],
    label="Positive affect y1",
    color="blue"
)

axes[1].plot(
    times_input,
    trajectory_input[:, 1],
    label="Negative affect y2",
    color="orange"
)

axes[1].set_title("Weak negative input")
axes[1].set_ylabel("Affective state")
axes[1].set_ylim(0, 1)
axes[1].legend()

#trajectory with the strong input
axes[2].plot(
    times_strong,
    trajectory_strong[:, 0],
    label="Positive affect y1",
    color="blue"
)

axes[2].plot(
    times_strong,
    trajectory_strong[:, 1],
    label="Negative affect y2",
    color="orange"
)

axes[2].set_title("Strong negative input")
axes[2].set_ylabel("Affective state")
axes[2].set_ylim(0, 1)
axes[2].legend()

#we draw the two rectangular inputs
axes[3].plot(
    times_control,
    weak_input_values,
    label="Weak input",
    color="green"
)

axes[3].plot(
    times_control,
    strong_input_values,
    label="Strong input",
    color="red"
)

axes[3].set_title("Negative external inputs")
axes[3].set_xlabel("Time")
axes[3].set_ylabel("B2(t)")
axes[3].legend()

plt.tight_layout()

figure_path = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "negative_input_trajectories_t20.png"
)

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
