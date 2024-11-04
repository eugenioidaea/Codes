import numpy as np
import matplotlib.pyplot as plt
import math

# Parameters
num_steps = 300 # Number of steps
D = 1.0  # Diffusion constant
noise_strength = np.sqrt(2 * D)  # Strength of the noise term
mean = 0
std = 1
num_particles = 100
uby = 10 # Vertical Upper Boundary
lby = -10 # Vertical Lower Boundary
lbx = 0 # Horizontal Left Boundary
rbx = 100 # Horizontal Right Boundary
init_shift = 1 # It aggregates the initial positions of the particles around the centre of the domain

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
        
        x[n][i] = x[n][i-1] + noise_strength*eta_x
        y[n][i] = y[n][i-1] + noise_strength*eta_y

        if x[n][i] < lbx:
            x[n][i] = x[n][i-1] - noise_strength*eta_x
        if y[n][i] > uby or y[n][i] < lby:
            y[n][i] = y[n][i-1] - noise_strength*eta_y
        if x[n][i] > rbx:
            x[n] = x[n][:i]
            y[n] = y[n][:i]
            break

bc_time = [len(value)/num_steps for index, value in enumerate(x) if len(value)<num_steps]
bc_time.sort()
cum_part = [index/num_particles for index, value in enumerate(bc_time)]

# Plot the trajectory
plt.figure(figsize=(8, 8))
for i in range(num_particles):
    plt.plot(x[i], y[i], lw=0.5)
plt.axhline(y=uby, color='r', linestyle='--', linewidth=2)
plt.axhline(y=lby, color='r', linestyle='--', linewidth=2)
plt.title("2D Diffusion Process (Langevin Equation)")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.grid(True)
plt.show()

# Plot Breakthrough curve
plt.figure(figsize=(8, 8))
plt.plot(bc_time, cum_part, lw=0.5)
plt.title("Breakthorugh curve")
plt.xlabel("Time step")
plt.ylabel("CDF")
plt.grid(True)
plt.show()