import numpy as np
import math
import scipy.stats
import time

# Features ###################################################################
plotCharts = True # It controls graphical features (disable when run on HPC)
recordVideo = False # It slows down the script
recordTrajectories = True # It uses up memory

if plotCharts:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

# Parameters #################################################################
num_steps = int(1e4) # Number of steps
Dm = 0.1  # Diffusion for particles moving in the porous matrix
Df = 0.1  # Diffusion for particles moving in the fracture
dt = 1 # Time step
meanEta = 0 # Spatial jump distribution paramenter
stdEta = 1 # Spatial jump distribution paramenter
meanCross = 0 # Crossing probability parameter
stdCross = 1 # Crossing probability parameter
num_particles = int(1e3) # Number of particles in the simulation
uby = 1 # Vertical Upper Boundary
lby = -1 # Vertical Lower Boundary
lbx = 0 # Horizontal Left Boundary
rbx = 10 # Horizontal Right Boundary
init_shift = 0 # It aggregates the initial positions of the particles around the centre of the domain
reflectedInward = 90 # Percentage of impacts from the fracture reflected again into the fracture
reflectedOutward = 30 # Percentage of impacts from the porous matrix reflected again into the porous matrix
animatedParticle = 0 # Index of the particle whose trajectory will be animated
fTstp = 0 # First time step to be recorded in the video
lTstp = 90 # Final time step to appear in the video
bins = 100 # Number of bins for the logarithmic plot

noiseMatrix = np.sqrt(2*Dm*dt)  # Strength of the noise term for particle in the porous matrix
noiseFracture = np.sqrt(2*Df*dt)  # Strength of the noise term for particle in the fracture
crossOut = scipy.stats.norm.ppf(reflectedInward/100)
crossIn = scipy.stats.norm.ppf(reflectedOutward/100)
inFraRbx = [False for _ in range(num_particles)]
bc_time = []

bouncesBackIn = 0
bouncesBackOut = 0
crossInToOut = 0
crossOutToIn = 0
staysIn = 0
staysOut = 0
outsideFractureUp = False
outsideFractureDown = False

t = 0
pdf_part = []
x = np.zeros(num_particles) # Horizontal initial positions
y = np.linspace(lby+init_shift, uby-init_shift, num_particles) # Vertical initial positions
# xPath = [[] for _ in range(num_particles)] # List for storing the trajectories
# yPath = [[] for _ in range(num_particles)] # List for storing the trajectories
xPath = np.zeros((num_particles, num_steps))  # Matrix for storing x trajectories
yPath = np.zeros((num_particles, num_steps))  # Matrix for storing y trajectories

start_time = time.time() # Start timing the while loop

# Time loop ###########################################################################

while t<num_steps*dt:

    isIn = x<rbx # Get the positions of the particles that are still in the domain

    # Store the positions of each particle for all the time steps 
    if recordTrajectories:
        # [xPath[particleN].append(float(xAtTimeT)) for particleN, xAtTimeT in enumerate(x) if x[particleN]<rbx]
        # [yPath[particleN].append(float(yAtTimeT)) for particleN, yAtTimeT in enumerate(y) if x[particleN]<rbx]
        xPath[:, t] = x  # Store x positions for the current time step
        yPath[:, t] = y  # Store y positions for the current time step        

    # Update ALL the particles' position for time step t
    x[isIn] = x[isIn] + np.sqrt(2*Dm*dt)*np.random.normal(meanEta, stdEta, np.sum(isIn))
    y[isIn] = y[isIn] + np.sqrt(2*Dm*dt)*np.random.normal(meanEta, stdEta, np.sum(isIn))

    # Elastic reflection for the particles which hit the fracture's walls
    x = np.where(x<lbx, -x+2*lbx, x)
    y = np.where(y>uby, -y+2*uby, y)
    y = np.where(y<lby, -y+2*lby, y)

    # Slightly less efficient implementation of the reflection boundary condition
    '''x[x<lbx] = -x[x<lbx] + 2*lbx
    y[y<lby] = -y[y<lby] + 2*lby
    y[y>uby] = -y[y>uby] + 2*lby'''

    pdf_part.append(sum(x[isIn]>rbx)) # Count the particle which exit the right boundary at each time step

    t = t + dt

end_time = time.time() # Stop timing the while loop
execution_time = end_time - start_time

# Plot section #########################################################################
if plotCharts and recordTrajectories:
    plt.figure(figsize=(8, 8))
    for i in range(num_particles):
        plt.plot(xPath[i], yPath[i], lw=0.5)
    plt.axhline(y=uby, color='r', linestyle='--', linewidth=2)
    plt.axhline(y=lby, color='r', linestyle='--', linewidth=2)
    plt.axvline(x=lbx, color='black', linestyle='-', linewidth=2)
    plt.title("2D Diffusion Process (Langevin Equation)")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True)
    plt.show()

Time = np.linspace(dt, t, len(pdf_part))
pdf_part = np.asarray(pdf_part)
plt.figure(figsize=(8, 8))
plt.plot(Time, pdf_part/num_particles)
plt.figure(figsize=(8, 8))
plt.plot(Time, np.cumsum(pdf_part)/num_particles)

i = 0
particlesTstep=np.zeros(num_particles)
# Retrieve the number of steps for each particle from the pdf of the breakthrough curve
for index, value in enumerate(pdf_part):
    particlesTstep[i:i+value] = index
    i = i+value
# Logarithmic plot
timeLinSpaced = np.linspace(dt, t, bins) # Linearly spaced bins
timeLogSpaced = np.logspace(np.log10(dt), np.log10(t), bins) # Logarithmic spaced bins
countsLin, binEdgesLin = np.histogram(particlesTstep, timeLinSpaced)
countsLog, binEdgesLog = np.histogram(particlesTstep, timeLogSpaced)
countsNormLin = (countsLin/num_particles)/np.diff(binEdgesLin)
countsNormLog = (countsLog/num_particles)/np.diff(binEdgesLog)
plt.figure(figsize=(8, 8))
plt.plot(binEdgesLin[1:], countsNormLin)
plt.plot(binEdgesLog[1:], countsNormLog)
print(f"Execution time: {execution_time:.6f} seconds")