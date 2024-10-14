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
num_steps = int(1e3) # Number of steps
Dm = 0.1  # Diffusion for particles moving in the porous matrix
Df = 1  # Diffusion for particles moving in the fracture
dt = 1 # Time step
meanEta = 0 # Spatial jump distribution paramenter
stdEta = 1 # Spatial jump distribution paramenter
num_particles = int(1e4) # Number of particles in the simulation
uby = 1 # Vertical Upper Boundary
lby = -1 # Vertical Lower Boundary
lbx = 0 # Horizontal Left Boundary
rbx = 10 # Horizontal Right Boundary
init_shift = 0 # It aggregates the initial positions of the particles around the centre of the domain
reflectedInward = 90 # Percentage of impacts from the fracture reflected again into the fracture
reflectedOutward = 20 # Percentage of impacts from the porous matrix reflected again into the porous matrix
animatedParticle = 0 # Index of the particle whose trajectory will be animated
fTstp = 0 # First time step to be recorded in the video
lTstp = 90 # Final time step to appear in the video
bins = 100 # Number of bins for the logarithmic plot

bouncesBackIn = 0
bouncesBackOut = 0
crossInToOut = 0
crossOutToIn = 0
staysIn = 0
staysOut = 0

t = 0
pdf_part = []
x = np.zeros(num_particles) # Horizontal initial positions
y = np.linspace(lby+init_shift, uby-init_shift, num_particles) # Vertical initial positions
xPath = np.zeros((num_particles, num_steps))  # Matrix for storing x trajectories
yPath = np.zeros((num_particles, num_steps))  # Matrix for storing y trajectories
inside = [True for _ in range(num_particles)]
outsideAbove = [False for _ in range(num_particles)]
outsideBelow = [False for _ in range(num_particles)]
bc_time = []

start_time = time.time() # Start timing the while loop

# Time loop ###########################################################################

while t<num_steps*dt:

    # Store the positions of each particle for all the time steps 
    if recordTrajectories:
        xPath[:, t] = x  # Store x positions for the current time step
        yPath[:, t] = y  # Store y positions for the current time step        

    isIn = x<rbx # Get the positions of the particles that are still in the domain (wheter inside or outside the fracture)
    fracture = isIn & inside
    matrix = isIn & np.logical_or(outsideAbove, outsideBelow)

    # Update ALL the particles' position for time step t
    x[fracture] = x[fracture] + np.sqrt(2*Df*dt)*np.random.normal(meanEta, stdEta, np.sum(fracture))
    y[fracture] = y[fracture] + np.sqrt(2*Df*dt)*np.random.normal(meanEta, stdEta, np.sum(fracture))
    x[matrix] = x[matrix] + np.sqrt(2*Dm*dt)*np.random.normal(meanEta, stdEta, np.sum(matrix))
    y[matrix] = y[matrix] + np.sqrt(2*Dm*dt)*np.random.normal(meanEta, stdEta, np.sum(matrix))

    # Elastic reflection for the particles which hit the fracture's walls and some of them can diffuse into the matrix
    x = np.where(x<lbx, -x+2*lbx, x)

    crossOutAbove = inside & (y>uby)
    crossOutBelow = inside & (y<lby)
    crossInAbove = outsideAbove  & (y > lby) & (y < uby)
    crossInBelow = outsideBelow & (y>lby) & (y<uby)

    crossInToOut = np.random.rand(num_particles) < reflectedInward/100
    crossOutToIn = np.random.rand(num_particles) > reflectedOutward/100

    crossInToOutAbove = crossOutAbove & crossInToOut # Above upper boundary y
    crossInToOutBelow = crossOutBelow & crossInToOut # Below lower boundary y
    crossOutToInAbove = crossInAbove & crossOutToIn # Above upper boundary y
    crossOutToInBelow = crossInBelow & crossOutToIn # Below lower boundary y

    y = np.where(crossInToOutAbove, -y+2*uby, y)
    y = np.where(crossInToOutBelow, -y+2*lby, y)
    y = np.where(crossOutToInAbove, -y+2*uby, y)
    y = np.where(crossOutToInBelow, -y+2*lby, y)

    inside = (y<uby) & (y>lby) # Particles inside the fracture
    outsideAbove = y>uby # Particles in the porous matrix above the fracture
    outsideBelow = y<lby #Particles in the porous matrix below the fracture

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