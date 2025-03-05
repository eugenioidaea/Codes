import openpnm as op
import numpy as np

# Create a project
proj = op.Project()

# Number of fractures
num_fractures = 500  

# Define domain size
domain_size = [1.0, 1.0, 0.1]  # [Lx, Ly, Lz]

# Sample fracture lengths from a distribution (e.g., lognormal)
mean_length = 0.1
std_length = 0.02
fracture_lengths = np.random.lognormal(mean=np.log(mean_length), sigma=std_length, size=num_fractures)

# Generate random fracture orientations in 2D
angles = np.random.uniform(0, np.pi, size=num_fractures)  # Random angles in radians

# Generate fracture start points randomly inside the domain
start_x = np.random.uniform(0, domain_size[0], size=num_fractures)
start_y = np.random.uniform(0, domain_size[1], size=num_fractures)

# Compute end points based on length and orientation
end_x = start_x + fracture_lengths * np.cos(angles)
end_y = start_y + fracture_lengths * np.sin(angles)

# Clip end points to stay within the domain
end_x = np.clip(end_x, 0, domain_size[0])
end_y = np.clip(end_y, 0, domain_size[1])

# Collect pore coordinates (fracture intersections)
pore_coords = np.vstack((np.column_stack((start_x, start_y, np.zeros(num_fractures))),
                         np.column_stack((end_x, end_y, np.zeros(num_fractures)))))

# Define connections (throats) between start and end points of each fracture
throat_conns = np.column_stack((np.arange(num_fractures), np.arange(num_fractures, 2 * num_fractures)))

# Create an OpenPNM network
net = op.network.GenericNetwork(project=proj)
net.update({'pore.coords': pore_coords, 'throat.conns': throat_conns})

# Visualize the network
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([start_x, end_x], [start_y, end_y], 'k-', alpha=0.5)
ax.set_aspect('equal')
plt.show()
