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
from aim.density import calculate_stationary_density
from aim.simulation import euler_maruyama   #we have moved the function to src so we can go faster

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
# %%
#we combine the thinned samples from both homebases to get a more balanced dataset
#The two basins were sampled separately and equally weighted because no inter-basin transitions were observed.
combined_samples = np.vstack(
    (positive_samples_thinned, negative_samples_thinned)    
)               
print(
    "Combined thinned samples shape:",
    combined_samples.shape
)
# %%
#we reuse the same theoretical stationary density calculation from src/aim/density.py
x, y, X, Y, den_theorical, z = calculate_stationary_density(
    points=100
)
# %%
#we draw the simulation and theorical density in a 2D histogram
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
empirical_plot = plt.hist2d(
    combined_samples[:, 0],
    combined_samples[:, 1],
    bins=50,
    density=True,
    cmap="viridis",
    range=[[0, 1], [0, 1]]
)
plt.title("Simulated stationary distribution")
plt.xlabel("Positive affect y1")
plt.ylabel("Negative affect y2")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().set_aspect("equal")
plt.colorbar(
    empirical_plot[3],
    label="Probability density"
)

plt.subplot(1, 2, 2)
theoretical_plot = plt.contourf(
    X,
    Y,
    den_theorical,
    levels=30,
    cmap="viridis"
)
plt.title("Theoretical stationary distribution")
plt.xlabel("Positive affect y1")
plt.ylabel("Negative affect y2")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().set_aspect("equal")
plt.colorbar(
    theoretical_plot,
    label="Probability density"
)

plt.tight_layout()

plt.show()
# %%
#our next step is try to fin the empirical maximum, is symmetric so we can just take the maximum of the first column and the first row, and then we will compare it with the theoretical maximum
matrix_of_density = empirical_plot[0]
y1border = empirical_plot[1]
y2border = empirical_plot[2]
#for each interval we calculate its center using the left and right borders
y1_centers = (y1border[:-1] + y1border[1:]) / 2
y2_centers = (y2border[:-1] + y2border[1:]) / 2
# %%
#we will creat a matrix and we will see every value and always keep the bigger so we inicialize with really small values
positive_maximum_density = -np.inf
negative_maximum_density = -np.inf

positive_maximum_coordinates = None
negative_maximum_coordinates = None

#now we  will see every value of the 2d hiatograma
for i in range(len(y1_centers)):
    for j in range(len(y2_centers)):

        y1_value = y1_centers[i]
        y2_value = y2_centers[j]
        density_value = matrix_of_density[i, j]

        #Positive-affect basin
        if y1_value > y2_value:
            if density_value > positive_maximum_density:
                positive_maximum_density = density_value
                positive_maximum_coordinates = (
                    y1_value,
                    y2_value
                )

        #Negative-affect basin
        if y2_value > y1_value:
            if density_value > negative_maximum_density:
                negative_maximum_density = density_value
                negative_maximum_coordinates = (
                    y1_value,
                    y2_value
                )

print(
    "Empirical positive maximum:",
    positive_maximum_coordinates
)

print(
    "Empirical negative maximum:",
    negative_maximum_coordinates
)
# %%
 # %%
#now we find the theoretical maximum in each affective basin
theoretical_positive_maximum_density = -np.inf
theoretical_negative_maximum_density = -np.inf

theoretical_positive_maximum_coordinates = None
theoretical_negative_maximum_coordinates = None

#we visit every value of the theoretical density
for i in range(den_theorical.shape[0]):
    for j in range(den_theorical.shape[1]):

        y1_value = X[i, j]
        y2_value = Y[i, j]
        density_value = den_theorical[i, j]

        #Positive-affect basin
        if y1_value > y2_value:
            if density_value > theoretical_positive_maximum_density:
                theoretical_positive_maximum_density = density_value
                theoretical_positive_maximum_coordinates = (
                    y1_value,
                    y2_value
                )

        #Negative-affect basin
        if y2_value > y1_value:
            if density_value > theoretical_negative_maximum_density:
                theoretical_negative_maximum_density = density_value
                theoretical_negative_maximum_coordinates = (
                    y1_value,
                    y2_value
                )

print(
    "Theoretical positive maximum:",
    theoretical_positive_maximum_coordinates
)

print(
    "Theoretical negative maximum:",
    theoretical_negative_maximum_coordinates
)

# %%
#now we calculate the distance between the empirical and theoretical maxima is an euclidyan distance, we will use the formula sqrt((x2-x1)^2 + (y2-y1)^2)
positive_maximum_error = np.sqrt(
    (
        positive_maximum_coordinates[0]
        - theoretical_positive_maximum_coordinates[0]
    ) ** 2
    +
    (
        positive_maximum_coordinates[1]
        - theoretical_positive_maximum_coordinates[1]
    ) ** 2
)

negative_maximum_error = np.sqrt(
    (
        negative_maximum_coordinates[0]
        - theoretical_negative_maximum_coordinates[0]
    ) ** 2
    +
    (
        negative_maximum_coordinates[1]
        - theoretical_negative_maximum_coordinates[1]
    ) ** 2
)

print(
    "Positive maximum error:",
    positive_maximum_error
)

print(
    "Negative maximum error:",
    negative_maximum_error
)
