# 04_stationary_validation.py
# %%
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from aim.model import BETA, F, grad_F
from aim.simulation import euler_maruyama   #we have moved the function to src so we can go faster
from aim.density import calculate_stationary_density

#we simulate a long trajrctory
times, trajectory = euler_maruyama(
    y0=np.array([0.5, 0.5]),
    dt=0.0005,
    T=100,      
    D=0.01,
    seed=42
)
print("Initial state:", trajectory[0])
print("Final state:", trajectory[-1])
positive_region = trajectory[:, 0] > trajectory[:, 1]
negative_region = trajectory[:, 1] > trajectory[:, 0]

number_positive = np.sum(positive_region)
number_negative = np.sum(negative_region)

print("Points in positive region:", number_positive)
print("Points in negative region:", number_negative)
# %%
burn_in_fraction = 0.10 #we discard the first 10% of the trajectory as burn-in because it is not stationary yet


burn_in_index = int( #this is the index where we start considering the trajectory as stationary
    burn_in_fraction * len(trajectory)
)

stationary_samples = trajectory[burn_in_index:] #we take the samples after the burn-in period
print("Burn-in index:", burn_in_index)
print("Stationary samples shape:", stationary_samples.shape)

positive_after_burn_in = ( #we do this to check how many points are in the positive region after burn-in
    stationary_samples[:, 0]
    > stationary_samples[:, 1]
)

negative_after_burn_in = (
    stationary_samples[:, 1]
    > stationary_samples[:, 0]
)

print(
    "Positive points after burn-in:",
    np.sum(positive_after_burn_in)
)

print(
    "Negative points after burn-in:",
    np.sum(negative_after_burn_in)
)
negative_samples = stationary_samples
# The trajectory starts at the unstable central barrier and initially visits
# the positive region. It then falls into the negative-affect homebase.
# After removing the burn-in, all samples remain in the negative basin,
# so the initial positive points are part of the transient, not a switch.
# We now simulate a second trajectory starting in the positive homebase.
# %%
timespositive, trajectory_positive = euler_maruyama(
    y0=np.array([0.95, 0.05]),  # Start in the positive-affect homebase
    dt=0.0005,
    T=100,
    D=0.01, 
    seed=43 #we change the seed so we get a different trajectory
)           
burn_in_positive = int(
    0.10 * len(trajectory_positive)
)

positive_samples = trajectory_positive[
    burn_in_positive:
]
positive_region_check = (
    positive_samples[:, 0]
    > positive_samples[:, 1]
)

negative_region_check = (
    positive_samples[:, 1]
    > positive_samples[:, 0]
)

print(
    "Positive-basin points:",
    np.sum(positive_region_check)
)

print(
    "Negative-basin points:",
    np.sum(negative_region_check)
)
# %%
#we now try to reduce the auto-correlation of the samples by sub-sampling them
def subsample(samples, step):   
    """Subsample the given samples by taking every 'step'-th sample."""
    return samples[::step]              
# We reduce temporal autocorrelation by keeping only one sample every
# 100 simulation steps. Consecutive states are very similar because the
# process evolves continuously, so using all of them would provide a large
# number of highly related observations. We apply the same subsampling
# procedure to the trajectories from both affective homebases.
# %%
# %%
positive_samples_thinned = subsample(
    positive_samples,
    100
)

negative_samples_thinned = subsample(
    negative_samples,
    100
)

print(
    "Positive thinned samples shape:",
    positive_samples_thinned.shape
)

print(
    "Negative thinned samples shape:",
    negative_samples_thinned.shape
)
#we combine the thinned samples from both homebases to get a more balanced dataset
combined_samples = np.vstack(
    (positive_samples_thinned, negative_samples_thinned)    
)               
print(
    "Combined thinned samples shape:",
    combined_samples.shape
)  
#now we calculate the density, is theorical and we reuse the function we created in 02_stationary_density.py

x, y, X, Y, den_theorical, z = calculate_stationary_density(
    points=100
)
#we draw the simulation and theorical density in a 2D histogram
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist2d(
    combined_samples[:, 0],
    combined_samples[:, 1],
    bins=50,
    density=True,
    cmap="viridis",
    range=[[0, 1], [0, 1]]
)
plt.title("Simulated stationary distribution")
plt.xlabel("x")
plt.ylabel("y")
plt.show

