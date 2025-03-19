import openpnm as op
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as spst
from shapely.geometry import LineString, Point, Polygon

# Create a project
proj = op.Project()

# Number of fractures
num_fractures = 100

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




def create_openpnm_network(throat_list):
    # Flatten the throat list into a set of unique points
    all_points = np.array([point for throat in throat_list for point in throat])
    unique_points, unique_indices = np.unique(all_points, axis=0, return_inverse=True)
    
    # Map the original points to unique indices
    throat_conns = unique_indices.reshape(-1, 2)  # Reshape into (N,2) pairs
    
    # Create OpenPNM network dictionary
    network_dict = {
        "pore.coords": np.column_stack((unique_points, np.zeros(len(unique_points)))),  # Add z=0
        "throat.conns": throat_conns
    }
    
    return network_dict

# Convert to OpenPNM format
network = create_openpnm_network(filtered_segments)

# Convert to OpenPNM network
pn = op.network.Network(conns=network['throat.conns'], coords=network['pore.coords'])
pn.regenerate_models() # Compute geometric properties such as pore volume

boundaryLayer = 0.01
pn['pore.left']=pn['pore.coords'][:, 0]<boundaryLayer
pn['pore.right']=pn['pore.coords'][:, 0]>max(pn['pore.coords'][:, 0])-boundaryLayer

print(pn)

####################################################################################
# SIM SETUP
####################################################################################
Dmol = 1e-4 # Molecular Diffusion
endSim = (domain_size[2])**2/Dmol
simTime = (0, endSim) # Simulation starting and ending times
Cin = 10
Cout = 0
s = 0.5 # Conductance: variance of the diameters of the throats
concTimePlot = 1 # Plot the spatial map of the concentration between start (0) or end (1) of the simulation

liquid = op.phase.Phase(network=pn) # Phase dictionary initialisation

# Define Euclidean length model manually
def euclidean_throat_length(target):
    conns = target["throat.conns"]
    coords = target["pore.coords"]
    return np.linalg.norm(coords[conns[:, 0]] - coords[conns[:, 1]], axis=1)

# Add the throat length model
pn.add_model(propname="throat.length", model=euclidean_throat_length)

# Compute throat lengths
throatLength = pn["throat.length"] # l_max/10

# BIG QUESTION: WHEN THROATS WITH DIFFERENT DIAMETERS CONVERGE TO THE SAME PORE, WHAT IS THE DIAMETER OF THE PORE?
poreDiameter = l_min/2

# throatDiameter = np.ones(pn.Nt)*poreDiameter/2 # Constant throat diameters
throatDiameter = spst.lognorm.rvs(s, loc=0, scale=poreDiameter/2, size=pn.Nt) # Lognormal throat diameter

pn['throat.diameter'] = throatDiameter
Athroat = throatDiameter**2*np.pi/4
diffCond = Dmol*Athroat/throatLength

# CHECK ON SMALL/BIG THROATS: AFTER CLIPPING, THE LENGTH OF SOME SEGMENTS MAY BE VERY SMALL AND CONDUCTANCE VERY HIGH
diffCond[diffCond>1e-5] = 1e-5
diffCond[diffCond<1e-8] = 1e-8

liquid['throat.diffusive_conductance'] = diffCond
pn['pore.diameter'] = poreDiameter
pn['pore.volume'] = 4/3*np.pi*poreDiameter**3/8

tfd = op.algorithms.TransientFickianDiffusion(network=pn, phase=liquid) # TransientFickianDiffusion dictionary initialisation

# Boundary conditions
tfd.set_value_BC(pores=pn.pores(['pore.left']), values=Cin) # Inlet: fixed concentration
tfd.set_value_BC(pores=pn.pores(['pore.right']), values=Cout) # Outlet: fixed concentration
# tfd.set_rate_BC(pores=inlet, rates=Qin) # Inlet: fixed rate
# tfd.set_rate_BC(pores=outlet, rates=Qout) # Outlet: fixed rate

# Initial conditions
ic = np.concatenate((np.ones(sum(pn['pore.left']))*Cin, np.ones(len(pn['pore.coords'])-sum(pn['pore.left']))*Cout)) # Initial Concentration

tfd.run(x0=ic, tspan=simTime)

pc = tfd.soln['pore.concentration'](endSim*concTimePlot)
d = pn['pore.diameter']
ms = 100 # Markersize
fig, ax = plt.subplots(figsize=[8, 8])
op.visualization.plot_coordinates(network=pn, color_by=pc, size_by=d, markersize=ms, ax=ax)
op.visualization.plot_connections(network=pn, size_by=throatDiameter, linewidth=3, ax=ax)