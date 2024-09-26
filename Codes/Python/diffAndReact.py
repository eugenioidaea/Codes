""" import numpy as np
import matplotlib.pyplot as plt

# Define the range for x and y
x = np.linspace(0, 1, 10)
y = np.linspace(0, 1, 10)

# Generate the meshgrid
X, Y = np.meshgrid(x, y)

# Plot the 2D mesh
plt.figure(figsize=(5, 5))
plt.plot(X, Y, marker='.', color='k', linestyle='none')  # Mesh points
plt.grid(True)
plt.title("2D Grid Mesh")
plt.xlabel("X")
plt.ylabel("Y")
plt.show() """

import numpy as np
import matplotlib.pyplot as plt
import math

# Parameters
num_steps = 1000  # Number of steps
D = 1.0  # Diffusion constant
noise_strength = np.sqrt(2 * D)  # Strength of the noise term
mean = 0
std = 1
num_particles = 10
k = 0.01 # Reflection efficiency

react_dist = [math.exp(-k*t)*k for t in range(1, num_steps)]
psi = [react/sum(react_dist) for react in react_dist]
samples = np.random.choice(range(1, num_steps), size=num_particles, p=psi)

# Initialize arrays to store position data
x = [np.zeros(num_steps) for n in range (num_particles)]
y = [np.zeros(num_steps) for n in range (num_particles)]

# Simulate Langevin dynamics
for n, particle in enumerate(x):
    for i in range(1, num_steps):
        # Generate random forces (Gaussian white noise)
        eta_x = np.random.normal(mean, std)
        eta_y = np.random.normal(mean, std)
        
        # Update positions
        if i < samples[n]:
            x[n][i] = x[n][i - 1] + noise_strength*eta_x
            y[n][i] = 0
        else:
            x[n][i] = x[n][i - 1] + noise_strength*eta_x
            y[n][i] = y[n][i - 1] + noise_strength*eta_y

# Plot reflection probability
plt.figure(figsize=(8, 8))
plt.plot(range(1, num_steps), psi)

# Plot the trajectory
plt.figure(figsize=(8, 8))
for i in range(num_particles):
    plt.plot(x[i], y[i], lw=0.5)
plt.title("2D Diffusion Process (Langevin Equation)")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.grid(True)
plt.show()