# 02_stationary_density.py 
# %%
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# This is done in order to find the main project folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# We add the src folder to Python's import path
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

# Import the reusable model
from aim.model import BETA, F
from aim.density import calculate_stationary_density
#print("Beta:", BETA).    Test to see if the model is imported correctly
#print("Example energy:", F(0.3, 0.7))

#%%
# We start creating the grid, is 2d because we have two variables
points = 100
x, y, X, Y, stationary_density, z = calculate_stationary_density(
    points
)
#we integrate as the stationary density is a function of two variables, we integrate first over x and then over y to get the total probability
# %%
integral_check_x = np.trapezoid(
    stationary_density,
    x=x,
    axis=1
)

total_probability = np.trapezoid(
    integral_check_x,
    x=y,
    axis=0
)
# %%
#We plot the stationary density as a heatmap
plt.figure(figsize=(8, 6))

heatmap = plt.imshow(
    stationary_density,
    extent=(
        x.min(),
        x.max(),
        y.min(),
        y.max()
    ),
    origin="lower",
    cmap="magma",
    aspect="equal",
    interpolation="bilinear"
)

plt.colorbar(
    heatmap,
    label="Probability density"
)

plt.title(
    "Theoretical stationary density"
)

plt.xlabel(
    "Positive affect $y_1$"
)

plt.ylabel(
    "Negative affect $y_2$"
)



plt.show()  

#Tests that I make to see if the stationary density is normalized correctly. The total probability should be 1.
print("Total probability:", total_probability)  
max_index = np.unravel_index(
    np.argmax(stationary_density),
    stationary_density.shape
)

max_y_index, max_x_index = max_index

print(
    "Maximum density location:",
    x[max_x_index],
    y[max_y_index]
)

print(
    "Maximum density:",
    stationary_density[max_index]
)
symmetry_error = np.max(
    np.abs(
        stationary_density
        - stationary_density.T
    )
)

print("Symmetry error:", symmetry_error)
