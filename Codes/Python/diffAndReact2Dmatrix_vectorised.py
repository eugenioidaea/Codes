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
num_steps = 5000 # Number of steps
Dm = 0.1  # Diffusion for particles moving in the porous matrix
Df = 0.1  # Diffusion for particles moving in the fracture
dt = 1 # Time step
meanEta = 0 # Spatial jump distribution paramenter
stdEta = 1 # Spatial jump distribution paramenter
meanCross = 0 # Crossing probability parameter
stdCross = 1 # Crossing probability parameter
num_particles = int(1e4) # Number of particles in the simulation
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

noiseMatrix = np.sqrt(2*Dm*dt)  # Strength of the noise term for particle in the porous matrix
noiseFracture = np.sqrt(2*Df*dt)  # Strength of the noise term for particle in the fracture
crossOut = scipy.stats.norm.ppf(reflectedInward/100)
crossIn = scipy.stats.norm.ppf(reflectedOutward/100)
# inFraRbx = [False for _ in range(num_particles)]
bc_time = []

bouncesBackIn = 0
bouncesBackOut = 0
crossInToOut = 0
crossOutToIn = 0
staysIn = 0
staysOut = 0
outsideFractureUp = False
outsideFractureDown = False

start_time = time.time()
# If recording trajectories is enabled #############################################
# Initialize arrays to store position data
x = np.zeros(num_particles)
y = np.linspace(lby+init_shift, uby-init_shift, num_particles)

# Simulate Langevin dynamics
t = 0
pdf_part = []
while t<num_steps*dt:

   t = t + dt

   isIn = x<rbx

   x[isIn] = x[isIn] + np.sqrt(2*Dm*dt)*np.random.normal(meanEta, stdEta, np.sum(isIn))
   y[isIn] = y[isIn] + np.sqrt(2*Dm*dt)*np.random.normal(meanEta, stdEta, np.sum(isIn))

   x = np.where(x<lbx, -x+2*lbx, x)
   y = np.where(y>uby, -y+2*uby, y)
   y = np.where(y<lby, -y+2*lby, y)

   '''x[x<lbx] = -x[x<lbx] + 2*lbx
   y[y<lby] = -y[y<lby] + 2*lby
   y[y>uby] = -y[y>uby] + 2*lby'''
   
   pdf_part.append(sum(x[isIn]>rbx))

end_time = time.time()
execution_time = end_time - start_time

Time = np.linspace(dt, t, len(pdf_part))

pdf_part = np.asarray(pdf_part)
plt.plot(Time, pdf_part/num_particles)
plt.plot(Time, np.cumsum(pdf_part)/num_particles)

#plt.hist(counts, bin_edges)
i = 0
particlesTstep=np.zeros(num_particles)
for index, value in enumerate(pdf_part):
    particlesTstep[i:i+value] = index
    i = i+value
bins = 100
timeLinSpaced = np.linspace(dt, t, bins)
timeLogSpaced = np.logspace(np.log10(dt), np.log10(t), bins)
counts, bin_edges = np.histogram(particlesTstep, timeLinSpaced)
plt.plot(bin_edges[1:], counts)

print(f"Execution time: {execution_time:.6f} seconds")