import numpy as np
import matplotlib.pyplot as plt
import math

# Parameters
num_steps = 1000 # Number of steps
D = 1.0  # Diffusion constant
noise_strength = np.sqrt(2 * D)  # Strength of the noise term
mean = 0
std = 1
num_particles = 3
uby = 10 # Vertical Upper Boundary
lby = -10 # Vertical Lower Boundary
lbx = 0 # Horizontal Left Boundary
init_shift = 5 # It aggregates the initial positions of the particles around the centre of the domain

# Initialize arrays to store position data
x = [np.zeros(num_steps) for _ in range(num_particles)]
y0 = np.linspace(lby+init_shift, uby-init_shift, num_particles)
y = [np.zeros(num_steps) for _ in range(num_particles)]
for index, array in enumerate(y):
    array[0] = y0[index]

# Simulate Langevin dynamics
for n, position in enumerate(x):
    for i in range(1, num_steps):
        # Generate random forces (Gaussian white noise)
        eta_x = np.random.normal(mean, std)
        eta_y = np.random.normal(mean, std)
        
        x[n][i] = x[n][i - 1] + noise_strength*eta_x
        y[n][i] = y[n][i - 1] + noise_strength*eta_y

        if x[n][i] < lbx:
            x[n][i] = -0.1*x[n][i]
        if y[n][i] > uby or y[n][i] < lby:
            y[n][i] = -0.1*y[n][i]

# Plot the trajectory
plt.figure(figsize=(8, 8))
for i in range(num_particles):
    plt.plot(x[i], y[i], lw=0.5)
plt.title("2D Diffusion Process (Langevin Equation)")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.grid(True)
plt.show()