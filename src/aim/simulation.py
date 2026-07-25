"""Reusable simulation functions for the Affective Ising Model."""

import numpy as np

from .model import BETA, grad_F


def euler_maruyama(y0, dt, T, D, seed=None, negative_input=None):
    """Simulate the SDE using the Euler-Maruyama method."""

    random_number_generator = np.random.default_rng(seed) #We create a random number generator with the given seed for reproducibility
    n_steps = int(T / dt) #We calculate the number of steps needed to reach the final time T 
    print(n_steps)
    times=np.linspace(0, T, n_steps+1) #function to create the time vector

    #we need now to create amatrix to store the values of y1 and y2 at each time step
    traj=np.zeros((n_steps+1, 2)) #We create a matrix of zeros with n_steps rows and 2 columns (for y1 and y2), the +1 is because we include the initial condition
    print("Number of steps:", n_steps)
    print("Times shape:", times.shape)
    print("Trajectory shape:", traj.shape)
    #we set the initial condition
    traj[0, :] = y0
    #for every time step, we calculate the new value of y1 and y2 using the Euler-Maruyama method
    for i in range(n_steps):
        y1, y2 = traj[i, :]
        current_time = times[i]

        # We calculate the usual gradient of the free-energy landscape
        gradient = np.array(grad_F(y1, y2))

        # If no external input is provided, its value is zero and the simulation
        # behaves exactly as before. Otherwise, we evaluate the input function
        # at the current simulation time.
        if negative_input is None:
            input_value = 0.0
        else:
            input_value = negative_input(current_time)

        # According to the AIM, a negative external input B2(t) changes the
        # linear energy term from Theta2*y2 to (Theta2 - B2(t))*y2.
        # Therefore, B2(t) is subtracted only from the y2 gradient component.
        gradient[1] = gradient[1] - input_value

        drift = -BETA*D*gradient #We calculate the drift term using the gradient of the free energy landscape
        noise = np.sqrt(2 * D * dt) * random_number_generator.normal(size=2) #We calculate the noise term using a normal distribution with mean 0 and variance 1
        new_state = traj[i, :] + drift * dt + noise #We update the values of y1 and y2 using the Euler-Maruyama method
        traj[i + 1, :] = np.clip(new_state, 0.00001, 0.99999) #We force the value to be betwwen 0 and 1

    return times, traj
