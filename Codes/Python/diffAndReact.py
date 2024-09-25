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

# Parameters
num_steps = 10000  # Number of steps
D = 1.0  # Diffusion constant
noise_strength = np.sqrt(2 * D)  # Strength of the noise term
mean = 0
std = 1

# Initialize arrays to store position data
x = np.zeros(num_steps)
y = np.zeros(num_steps)

# Simulate Langevin dynamics
for i in range(1, num_steps):
    # Generate random forces (Gaussian white noise)
    eta_x = np.random.normal(mean, std)
    eta_y = np.random.normal(mean, std)
    
    # Update positions
    x[i] = x[i - 1] + noise_strength*eta_x
    y[i] = y[i - 1] + noise_strength*eta_y

# Plot the trajectory
plt.figure(figsize=(8, 8))
plt.plot(x, y, lw=0.5)
plt.title("2D Diffusion Process (Langevin Equation)")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.grid(True)
plt.show()