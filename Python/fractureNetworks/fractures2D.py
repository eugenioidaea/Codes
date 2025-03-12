import openpnm as op
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as spst
from shapely.geometry import LineString, Point, Polygon

# Create a project
proj = op.Project()

# Number of fractures
num_fractures = 10

# Define domain size
domain_size = [0.0, 0.0, 1.0, 1.0] # [Xmin, Ymin, Xmax, Ymax]
# Domain boundaries to polygon object
boundary = Polygon([(domain_size[0], domain_size[1]), (domain_size[2], domain_size[1]), (domain_size[2], domain_size[3]), (domain_size[0], domain_size[3]), (domain_size[0], domain_size[1])])

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

segments_unclipped = [((start_x[i], start_y[i]), (end_x[i], end_y[i])) for i in range(num_fractures)]

# Processed list to store clipped segments
lines = []

# Clip each segment to the polygon
for seg in segments_unclipped:
    line = LineString([seg[0], seg[1]])  # Create a LineString
    clipped_line = line.intersection(boundary)  # Clip to bounding box

    # If the intersection is a LineString, add it to results
    if clipped_line.geom_type == "LineString":
        lines.append(LineString(clipped_line.coords))

# # Clip end points to stay within the domain
# end_x = np.clip(end_x, domain_size[0], domain_size[2])
# end_y = np.clip(end_y, domain_size[1], domain_size[3])
# 
# segments = [((start_x[i], start_y[i]), (end_x[i], end_y[i])) for i in range(num_fractures)]
# 
# # Convert to LineString objects
# lines = [LineString(seg) for seg in segments]

# Find the intersections of a fracture with other fractures
intersections = set()
for i in range(len(lines)):
    for j in range(i + 1, len(lines)):
        if lines[i].intersects(lines[j]):
            inter = lines[i].intersection(lines[j])
            if inter.geom_type == 'Point':  # Ignore collinear overlaps
                intersections.add((inter.x, inter.y))

# Find the intersections of a fracture with the polygon boundary
for line in lines:
    if line.intersects(boundary.exterior):
        intersection = line.intersection(boundary.exterior)  # Compute intersection
        # Handle different intersection types
        if intersection.geom_type == "Point":
            intersections.add((intersection.x, intersection.y))
#         elif intersection.geom_type == "MultiPoint":
#             for point in intersection.geoms:
#                 intersections.add((point.x, point.y))
#         elif intersection.geom_type == "LineString":  # Edge case if collinear
#             intersections.update(line.coords)

# Break segments at intersection points
new_segments = []
for line in lines:
    points = list(line.coords)  # Start with original endpoints
    for inter in intersections:
        inter_point = Point(inter)
        # if line.distance(inter_point) < 1e-9 and all(elem not in domain_size for elem in tuple(inter)):
        if line.distance(inter_point) < 1e-9 and all(inter_point.distance(Point(p)) > 1e-9 for p in points):  # Check if intersection is on segment AND if the intersection is not on the boundary (to avoid the generation of 0 length segments)
        # if inter_point.within(line) and not any(inter_point.equals(Point(p)) for p in points):
            points.append(inter)  # Add the intersection coordinates to the original endpoints
    points = sorted(points)  # Sort along the segment
    new_segments.extend([(points[i], points[i + 1]) for i in range(len(points) - 1)])

# Filter the segments and remove the dead ends
filtered_segments = [seg for seg in new_segments if seg[0] in intersections and seg[1] in intersections]

# Plot results
fig, ax = plt.subplots(figsize=(8, 8))
# Plot the Polygon
x, y = boundary.exterior.xy
ax.plot(x, y, 'b-', linewidth=2)  # 'b-' makes a blue outline
ax.fill(x, y, color='lightblue', alpha=0.5)  # Fill with transparency
# Plot original segments (dashed lines for reference)
for seg in segments_unclipped:
    x, y = zip(*seg)
    ax.plot(x, y, 'k--', alpha=0.5)
# ax.set_xlim(domain_size[0], domain_size[2])
# ax.set_ylim(domain_size[1], domain_size[3])

fig, ax = plt.subplots(figsize=(8, 8))
# Plot new segments
for seg in new_segments:
    x, y = zip(*seg)
    ax.plot(x, y, 'b', linewidth=2)
# Plot intersection points
for inter in intersections:
    ax.scatter(*inter, color='red', zorder=3, s=20, edgecolor='black')
ax.set_title("Segment Intersection and Splitting")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_xlim(domain_size[0], domain_size[2])
ax.set_ylim(domain_size[1], domain_size[3])
ax.grid(True)

fig, ax = plt.subplots(figsize=(8, 8))
for (x1, y1), (x2, y2) in filtered_segments:
    ax.plot([x1, x2], [y1, y2], marker='o', linestyle='-', color='b')
ax.set_title("Segment Intersection Filtered")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_xlim(domain_size[0], domain_size[2])
ax.set_ylim(domain_size[1], domain_size[3])
ax.grid(True)




# # Create an OpenPNM network
# net = op.network.GenericNetwork(project=proj)
# net.update({'pore.coords': pore_coords, 'throat.conns': throat_conns})
# 
# # Visualize the network
# fig, ax = plt.subplots()
# ax.plot([start_x, end_x], [start_y, end_y], 'k-', alpha=0.5)
# ax.set_aspect('equal')
# plt.show()
