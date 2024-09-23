import numpy as np
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
plt.show()