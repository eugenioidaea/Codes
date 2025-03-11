import openpnm as op
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as spst
from shapely.geometry import LineString, Point, Polygon

# Create a project
proj = op.Project()

# Number of fractures
num_fractures = 30  

# Define domain size
domain_size = [0.0, 0.0, 1.0, 1.0] # [Xmin, Ymin, Xmax, Ymax]

# Sample fracture lengths from a distribution:
# A) Lognormal
# mean_length = 0.5
# std_length = 0.02
# fracture_lengths = np.random.lognormal(mean=np.log(mean_length), sigma=std_length, size=num_fractures)
# B) Powerlaw
l_min = 1e-1 # Minimum fracture length
l_max = 1 # maximum fracture length
exponent = -2.58  # Power law exponent
b = abs(exponent) - 1  # SciPy uses (b+1) where b is positive
fracture_lengths = spst.powerlaw.rvs(b, size=num_fractures) # Generate power-law samples
fracture_lengths = l_min + (l_max - l_min) * fracture_lengths # Rescale to the desired range [l_min, l_max]

# Generate random fracture orientations in 2D
angles = np.random.uniform(0, 2*np.pi, size=num_fractures)  # Random angles in radians

# Generate fracture start points randomly inside the domain
start_x = np.random.uniform(domain_size[0], domain_size[2], size=num_fractures)
start_y = np.random.uniform(domain_size[1], domain_size[3], size=num_fractures)

# Compute end points based on length and orientation
end_x = start_x + fracture_lengths * np.cos(angles)
end_y = start_y + fracture_lengths * np.sin(angles)

# Clip end points to stay within the domain
end_x = np.clip(end_x, domain_size[0], domain_size[2])
end_y = np.clip(end_y, domain_size[1], domain_size[3])

# Collect pore coordinates (fracture intersections)
pore_coords = np.vstack((np.column_stack((start_x, start_y, np.zeros(num_fractures))),
                         np.column_stack((end_x, end_y, np.zeros(num_fractures)))))

# Define connections (throats) between start and end points of each fracture
throat_conns = np.column_stack((np.arange(num_fractures), np.arange(num_fractures, 2 * num_fractures)))

fractureNetwork = plt.figure(figsize=(8, 8))
for i in range(len(start_x)):
    plt.plot([start_x[i], end_x[i]], [start_y[i], end_y[i]])
# plt.title('Powerlaw length fracture network')
plt.title('Powerlaw length fracture network')
plt.xlabel('x [m]')
plt.ylabel('y [m]')

segments = [((start_x[i], start_y[i]), (end_x[i], end_y[i])) for i in range(num_fractures)]

# Convert to LineString objects
lines = [LineString(seg) for seg in segments]

# Boundary to polygon object
boundary = Polygon([(domain_size[0], domain_size[1]), (domain_size[2], domain_size[1]), (domain_size[2], domain_size[3]), (domain_size[0], domain_size[3]), (domain_size[0], domain_size[1])])

# Find the intersections of a fracture with other fractures
intersections = set()
for i in range(len(lines)):
    for j in range(i + 1, len(lines)):
        if lines[i].intersects(lines[j]):
            inter = lines[i].intersection(lines[j])
            if inter.geom_type == 'Point':  # Ignore collinear overlaps
                intersections.add((inter.x, inter.y))

# # Find the intersections with the boundaries of the domain
# for line in lines: # Find lines that touch the boundary
#     if line.touches(boundary):
#         for point in line.boundary:  # Get the start and end points of the line
#             if boundary.contains(point):  # Check if the point is on the boundary
#                 intersections.add((point.x, point.y))

for line in lines:
    if line.intersects(boundary.exterior):
        intersection = line.intersection(boundary)  # Compute intersection
        # Handle different intersection types
        if intersection.geom_type == "Point":
            intersections.add((intersection.x, intersection.y))
        elif intersection.geom_type == "MultiPoint":
            for point in intersection.geoms:
                intersections.add((point.x, point.y))
        elif intersection.geom_type == "LineString":  # Edge case if collinear
            intersections.update(line.coords)

# Break segments at intersection points
new_segments = []
for line in lines:
    points = list(line.coords)  # Start with original endpoints
    for inter in intersections:
        inter_point = Point(inter)
        if line.distance(inter_point) < 1e-9:  # Check if intersection is on segment
            points.append(inter)
    points = sorted(points)  # Sort along the segment
    new_segments.extend([(points[i], points[i + 1]) for i in range(len(points) - 1)])

# Plot results
fig, ax = plt.subplots(figsize=(8, 8))
# Plot the Polygon
x, y = boundary.exterior.xy
ax.plot(x, y, 'b-', linewidth=2)  # 'b-' makes a blue outline
ax.fill(x, y, color='lightblue', alpha=0.5)  # Fill with transparency
# Plot original segments (dashed lines for reference)
for seg in segments:
    x, y = zip(*seg)
    ax.plot(x, y, 'k--', alpha=0.5)

fig, ax = plt.subplots(figsize=(8, 8))
# Plot new segments
for seg in new_segments:
    x, y = zip(*seg)
    ax.plot(x, y, 'b', linewidth=2)
# Plot intersection points
for inter in intersections:
    ax.scatter(*inter, color='red', zorder=3, s=100, edgecolor='black')
ax.set_title("Segment Intersection and Splitting")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.grid(True)
plt.show()






# # Create an OpenPNM network
# net = op.network.GenericNetwork(project=proj)
# net.update({'pore.coords': pore_coords, 'throat.conns': throat_conns})
# 
# # Visualize the network
# fig, ax = plt.subplots()
# ax.plot([start_x, end_x], [start_y, end_y], 'k-', alpha=0.5)
# ax.set_aspect('equal')
# plt.show()
