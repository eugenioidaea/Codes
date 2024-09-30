import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats

# Parameters
num_steps = 1000 # Number of steps
D = 1.0  # Diffusion constant
noise_strength = np.sqrt(2 * D)  # Strength of the noise term
mean = 0
std = 1
num_particles = 1000
uby = 10 # Vertical Upper Boundary
lby = -10 # Vertical Lower Boundary
lbx = 0 # Horizontal Left Boundary
rbx = 50 # Horizontal Right Boundary
init_shift = 1 # It aggregates the initial positions of the particles around the centre of the domain
reflectedInward = 90
crossOut = scipy.stats.norm.ppf(reflectedInward/100)
reflectedOutward = 30
crossIn = scipy.stats.norm.ppf(reflectedOutward/100)
arrival = np.zeros(num_particles)

bouncesBackIn = 0
bouncesBackOut = 0
crossInToOut = 0
crossOutToIn = 0

# Initialize arrays to store position data
x = [np.zeros(num_steps) for _ in range(num_particles)]
y0 = np.linspace(lby+init_shift, uby-init_shift, num_particles)
y = [np.zeros(num_steps) for _ in range(num_particles)]
for index, array in enumerate(y):
    array[0] = y0[index]

# Simulate Langevin dynamics
for n, position in enumerate(x):
    cross = False # After the particle crosses the fracture's walls once, it can freely move from fracture to matric and viceversa
    for i in range(1, num_steps):
        # Generate random forces (Gaussian white noise)
        eta_x = np.random.normal(mean, std)
        eta_y = np.random.normal(mean, std)
        
        x[n][i] = x[n][i-1] + noise_strength*eta_x
        y[n][i] = y[n][i-1] + noise_strength*eta_y

        if x[n][i] < lbx:
            x[n][i] = x[n][i-1] - noise_strength*eta_x
        # The following condition compares the particle time step i with a sample vector which stores the reflecting probabilities randomly arranged
        # The particle gets reflected until it crosses the fracture's wall. Once it crossed it moves freely between fractures' boundaries
        if cross == False:
            # The particle bounces back in against the fracture's wall
            if (y[n][i] > uby or y[n][i] < lby) and np.random.normal(mean, std) < crossOut:
                y[n][i] = y[n][i-1] - noise_strength*eta_y
                bouncesBackIn = bouncesBackIn+1
            # The particle leaves the fracture
            if (y[n][i] > uby or y[n][i] < lby) and np.random.normal(mean, std) > crossOut:
                cross = True
                crossInToOut = crossInToOut+1
        if cross == True:
            # The particle bounces against the fracture's wall
            if (y[n][i] < uby or y[n][i] > lby) and np.random.normal(mean, std) < crossIn:
                y[n][i] = y[n][i-1] - noise_strength*eta_y
                bouncesBackOut = bouncesBackOut+1
            # The particle enters the fracture
            if (y[n][i] < uby or y[n][i] > lby) and np.random.normal(mean, std) > crossIn:
                cross = False
                crossOutToIn = crossOutToIn+1
        if x[n][i] > rbx:
            x[n] = x[n][:i]
            y[n] = y[n][:i]
            if y[n][i-1] < uby or y[n][i-1] > lby:
                arrival[n] = arrival[n]+1
            break

bc_time = [len(value)/num_steps for index, value in enumerate(x) if len(value)<num_steps]
bc_time.sort()
cum_part = [index/num_particles for index, value in enumerate(bc_time)]

print("Effective bounce-in fraction: ", 100*bouncesBackIn/(bouncesBackIn+crossInToOut))
print("Effective bounce-out fraction: ", 100*bouncesBackOut/(bouncesBackOut+crossInToOut))

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