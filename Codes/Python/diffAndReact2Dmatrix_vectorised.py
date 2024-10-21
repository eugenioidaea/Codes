import numpy as np
import math
import scipy.stats
import time

# Features ###################################################################
plotCharts = True # It controls graphical features (disable when run on HPC)
recordVideo = False # It slows down the script
recordTrajectories = True # It uses up memory
rbxOn = True # It controls the right boundary condition

if plotCharts:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

# Parameters #################################################################
num_steps = int(1e3) # Number of steps
Dm = 1  # Diffusion for particles moving in the porous matrix
Df = 1  # Diffusion for particles moving in the fracture
dt = 1 # Time step
meanEta = 0 # Spatial jump distribution paramenter
stdEta = 1 # Spatial jump distribution paramenter
num_particles = int(1e2) # Number of particles in the simulation
uby = 1 # Vertical Upper Boundary
lby = -1 # Vertical Lower Boundary
lbx = 0 # Horizontal Left Boundary
rbx = 50 # Horizontal Right Boundary
init_shift = 0 # It aggregates the initial positions of the particles around the centre of the domain
reflectedInward = 100 # Percentage of impacts from the fracture reflected again into the fracture
reflectedOutward = 20 # Percentage of impacts from the porous matrix reflected again into the porous matrix
animatedParticle = 0 # Index of the particle whose trajectory will be animated
fTstp = 0 # First time step to be recorded in the video
lTstp = 90 # Final time step to appear in the video
binsTime = 50 # Number of temporal bins for the logarithmic plot
binsSpace = 80 # Number of spatial bins for the concentration profile
recordSpatialConc = int(1e2) # Concentration profile recorded time
stopBTC = 10 # % of particles that need to pass the control plane before the simulation is ended

# Initialisation ####################################################################

t = 0 # Time
i = 0 # Index for converting Eulerian pdf to Lagrangian pdf
cdf = 0
pdf_part = np.zeros(num_steps)
x = np.zeros(num_particles) # Horizontal initial positions
y = np.linspace(lby+init_shift, uby-init_shift, num_particles) # Vertical initial positions
if recordTrajectories:
    xPath = np.zeros((num_particles, num_steps))  # Matrix for storing x trajectories
    yPath = np.zeros((num_particles, num_steps))  # Matrix for storing y trajectories
inside = [True for _ in range(num_particles)]
outsideAbove = [False for _ in range(num_particles)]
outsideBelow = [False for _ in range(num_particles)]
particleRT = np.zeros(num_particles) # Array which stores each particles' number of steps
Time = np.linspace(dt, num_steps*dt, num_steps) # Array that stores time steps
timeLinSpaced = np.linspace(dt, dt*num_steps, binsTime) # Linearly spaced bins
timeLogSpaced = np.logspace(np.log10(dt), np.log10(dt*num_steps), binsTime) # Logarithmic spaced bins
xBins = np.linspace(0, rbx, binsSpace) # Linearly spaced bins

# Functions ##########################################################################

def update_positions(x, y, fracture, matrix, Df, Dm, dt, meanEta, stdEta):
    x[fracture] += np.sqrt(2*Df*dt)*np.random.normal(meanEta, stdEta, np.sum(fracture))
    y[fracture] += np.sqrt(2*Df*dt)*np.random.normal(meanEta, stdEta, np.sum(fracture))
    x[matrix] += np.sqrt(2*Dm*dt)*np.random.normal(meanEta, stdEta, np.sum(matrix))
    y[matrix] += np.sqrt(2*Dm*dt)*np.random.normal(meanEta, stdEta, np.sum(matrix))
    return x, y

def apply_reflection(x, y, crossInToOutAbove, crossInToOutBelow,  crossOutToInAbove, crossOutToInBelow, 
                     crossOutAbove, crossOutBelow, crossInAbove, crossInBelow, uby, lby, rbxOn):
    if rbxOn:
        x = np.where(x<lbx, -x+2*lbx, x)
    y[np.where(crossOutAbove)[0]] = np.where(crossInToOutAbove, y[np.where(crossOutAbove)[0]], -y[np.where(crossOutAbove)[0]]+2*uby)
    y[np.where(crossOutBelow)[0]] = np.where(crossInToOutBelow, y[np.where(crossOutBelow)[0]], -y[np.where(crossOutBelow)[0]]+2*lby)
    y[np.where(crossInAbove)[0]]  = np.where(crossOutToInAbove, y[np.where(crossInAbove)[0]], -y[np.where(crossInAbove)[0]]+2*uby)
    y[np.where(crossInBelow)[0]] = np.where(crossOutToInBelow, y[np.where(crossInBelow)[0]], -y[np.where(crossInBelow)[0]]+2*lby)
    while np.any(y[np.where(crossOutAbove)[0]]<lby) | np.any(y[np.where(crossOutBelow)[0]]>uby): # Avoid jumps across the whole height of the fracture
        yyb=np.where(crossOutBelow)[0] # Particles which hit the lower limit of the fracture
        yyyb=yyb[np.where(y[yyb.flatten()]>uby)[0]] # Get the indeces of those that would be reflected above the uby
        y[yyyb.flatten()] = -y[yyyb.flatten()] + 2*uby # Reflect them back
        yya=np.where(crossOutAbove)[0]
        yyya=yya[np.where(y[yya.flatten()]<lby)[0]]
        y[yyya.flatten()] = -y[yyya.flatten()] + 2*lby
    return x, y

# Time loop ###########################################################################

start_time = time.time() # Start timing the while loop

while (cdf<stopBTC/100*num_particles) & (t<num_steps*dt):

    # Store the positions of each particle for all the time steps 
    if recordTrajectories:
        xPath[:, t] = x  # Store x positions for the current time step
        yPath[:, t] = y  # Store y positions for the current time step        

    if rbxOn:
        isIn = x<rbx # Get the positions of the particles that are in the domain (wheter inside or outside the fracture)
    else:
        isIn = np.ones(num_particles, dtype=bool)
    fracture = isIn & inside # Particles in the fracture
    outside = np.array(outsideAbove) | np.array(outsideBelow) # Particles outside the fracture
    matrix = isIn & outside # Particles in the domain and outside the fracture

    # Update the position of all the particles at a given time steps according to the Langevin dynamics
    x, y = update_positions(x, y, fracture, matrix, Df, Dm, dt, meanEta, stdEta)

    # Particles which in principles would cross the fractures' walls
    crossOutAbove = fracture & (y>uby)
    crossOutBelow = fracture & (y<lby)
    crossInAbove = outsideAbove  & (y > lby) & (y < uby)
    crossInBelow = outsideBelow & (y>lby) & (y<uby)

    # Decide the number of impacts that will cross the fracture's walls
    probCrossOutAbove = np.random.rand(np.sum(crossOutAbove)) > reflectedInward/100
    probCrossOutBelow = np.random.rand(np.sum(crossOutBelow)) > reflectedInward/100
    probCrossInAbove = np.random.rand(np.sum(crossInAbove)) > reflectedOutward/100
    probCrossInBelow = np.random.rand(np.sum(crossInBelow)) > reflectedOutward/100

    # Successfull crossing based on uniform probability distribution
    crossInToOutAbove = probCrossOutAbove & (crossOutAbove[np.where(crossOutAbove)[0]])
    crossInToOutBelow = probCrossOutBelow & (crossOutBelow[np.where(crossOutBelow)[0]])
    crossOutToInAbove = probCrossInAbove & (crossInAbove[np.where(crossInAbove)[0]])
    crossOutToInBelow = probCrossInBelow & (crossInBelow[np.where(crossInBelow)[0]])
    
    # Update the reflected particles' positions according to an elastic reflection dynamic
    x, y = apply_reflection(x, y, crossInToOutAbove, crossInToOutBelow,  crossOutToInAbove, crossOutToInBelow,
                            crossOutAbove, crossOutBelow, crossInAbove, crossInBelow, uby, lby, rbxOn)

    inside = (y<uby) & (y>lby) # Particles inside the fracture
    outsideAbove = y>uby # Particles in the porous matrix above the fracture
    outsideBelow = y<lby # Particles in the porous matrix below the fracture

    pdf_part[int(t/dt)] = sum(abs(x[isIn])>rbx) # Count the particle which exit the right boundary at each time step

    if t == recordSpatialConc:
        countsSpace, binEdgeSpace = np.histogram(x, xBins, density=True)

    cdf = sum(pdf_part)
    t += dt    

if recordTrajectories:
    xPath = xPath[:, :t]
    yPath = yPath[:, :t]

end_time = time.time() # Stop timing the while loop
execution_time = end_time - start_time

# Retrieve the number of steps for each particle from the pdf of the breakthrough curve
for index, value in enumerate(pdf_part):
    particleRT[int(i):int(i+value)] = index*dt
    i = i+value

meanTstep = particleRT.mean()
stdTstep = particleRT.std()

# Plot section #########################################################################

# Trajectories
if plotCharts and recordTrajectories:
    plt.figure(figsize=(8, 8))
    for i in range(num_particles):
        plt.plot(xPath[i], yPath[i], lw=0.5)
    plt.axhline(y=uby, color='r', linestyle='--', linewidth=2)
    plt.axhline(y=lby, color='r', linestyle='--', linewidth=2)
    if rbxOn:
        plt.axvline(x=lbx, color='black', linestyle='-', linewidth=2)
    plt.title("2D Diffusion Process (Langevin Equation)")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True)
    plt.show()

# PDF
plt.figure(figsize=(8, 8))
plt.plot(Time, pdf_part/num_particles)
plt.xscale('log')

# CDF
plt.figure(figsize=(8, 8))
plt.plot(Time, np.cumsum(pdf_part)/num_particles)
plt.xscale('log')

# 1-CDF
plt.figure(figsize=(8, 8))
plt.plot(Time, 1-np.cumsum(pdf_part)/num_particles)
plt.xscale('log')
plt.yscale('log')

# Binning for plotting the pdf from a Lagrangian vector
countsLog, binEdgesLog = np.histogram(particleRT, timeLogSpaced, density=True)
plt.figure(figsize=(8, 8))
plt.plot(binEdgesLog[1:][countsLog!=0], countsLog[countsLog!=0], 'r*')
plt.xscale('log')
plt.yscale('log')

# Spatial concentration profile at 'recordSpatialConc' time
plt.figure(figsize=(8, 8))
plt.plot(binEdgeSpace[1:][countsSpace!=0], countsSpace[countsSpace!=0], 'b-')
if rbxOn:
    plt.axvline(x=lbx, color='black', linestyle='-', linewidth=2)

# Statistichs
print(f"Execution time: {execution_time:.6f} seconds")
print(f"<t>: {meanTstep:.6f} s")
print(f"sigmat: {stdTstep:.6f} s")