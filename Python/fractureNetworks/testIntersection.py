import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point

# Example segment set
segments = [
    ((0, 0), (4, 4)),  # Diagonal segment
    ((0, 4), (4, 0)),  # Crosses first segment
    ((2, -1), (2, 5))  # Vertical segment crossing both
]

# Convert to LineString objects
lines = [LineString(seg) for seg in segments]

# Find intersections
intersections = set()
for i in range(len(lines)):
    for j in range(i + 1, len(lines)):
        if lines[i].intersects(lines[j]):
            inter = lines[i].intersection(lines[j])
            if inter.geom_type == 'Point':  # Ignore collinear overlaps
                intersections.add((inter.x, inter.y))

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

# 📌 Plot results
fig, ax = plt.subplots(figsize=(6, 6))

# Plot original segments (dashed lines for reference)
for seg in segments:
    x, y = zip(*seg)
    ax.plot(x, y, 'k--', alpha=0.5)

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