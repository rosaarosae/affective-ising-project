"""Reusable stationary-density calculation for the Affective Ising Model."""

import numpy as np

from .model import BETA, F


def calculate_stationary_density(points=100):
    # We start creating the grid, is 2d because we have two variables
    x = np.linspace(0.00001, 0.99999, points) #We avoid 0 and 1 because
    y = np.linspace(0.00001, 0.99999, points) #we have log(y) and log(1-y)
    X, Y = np.meshgrid(x, y)
    energy = F(X, Y) #This calculates the energy of the 100x100 grid of points.

    min_energy = np.min(energy) #This will be used to normalize and avoid numerical issues
    unnormalized_density =np.exp(-BETA * (energy - min_energy)) #This is the Boltzmann distribution
    #We integrate the function Z as the paper does, we use the trapezoidal rule to integrate over the grid.
    #We integrate over the x axis first, then over the y axis.
    # Integrate first over x (positive affect)
    integral_over_x = np.trapezoid(
        unnormalized_density,
        x=x,
        axis=1
    )
    # Integrate the previous result over y (negative affect), in order to get the partition function Z
    z= np.trapezoid(
        integral_over_x,
        x=y,
        axis=0
    )
    stationary_density = unnormalized_density / z

    return x, y, X, Y, stationary_density, z
