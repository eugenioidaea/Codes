import numpy as np
import math
import scipy.stats
import time

# Features ###################################################################
plotCharts = True # It controls graphical features (disable when run on HPC)
recordVideo = False # It slows down the script
recordTrajectories = False # It uses up memory

if plotCharts:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

# Parameters #################################################################
num_steps = int(1e4) # Number of steps
Dm = 1  # Diffusion for particles moving in the porous matrix
Df = 1  # Diffusion for particles moving in the fracture
dt = 1 # Time step
meanEta = 0 # Spatial jump distribution paramenter
stdEta = 1 # Spatial jump distribution paramenter
num_particles = int(1e4) # Number of particles in the simulation
uby = 1 # Vertical Upper Boundary
lby = -1 # Vertical Lower Boundary
lbx = 0 # Horizontal Left Boundary
rbx = 30 # Horizontal Right Boundary
init_shift = 0 # It aggregates the initial positions of the particles around the centre of the domain
reflectedInward = 100 # Percentage of impacts from the fracture reflected again into the fracture
reflectedOutward = 20 # Percentage of impacts from the porous matrix reflected again into the porous matrix
animatedParticle = 0 # Index of the particle whose trajectory will be animated
fTstp = 0 # First time step to be recorded in the video
lTstp = 90 # Final time step to appear in the video
bins = 50 # Number of bins for the logarithmic plot
stopBTC = 100 # % of particles that need to pass the control plane before the simulation is ended

# Initialisation ####################################################################

t = 0 # Time
i = 0 # Index for converting Eulerian pdf to Lagrangian pdf
cdf = 0
pdf_part = np.zeros(num_steps)
x = np.zeros(num_particles) # Horizontal initial positions
y = np.linspace(lby+init_shift, uby-init_shift, num_particles) # Vertical initial positions
xPath = np.zeros((num_particles, num_steps))  # Matrix for storing x trajectories
yPath = np.zeros((num_particles, num_steps))  # Matrix for storing y trajectories
inside = [True for _ in range(num_particles)]
outsideAbove = [False for _ in range(num_particles)]
outsideBelow = [False for _ in range(num_particles)]
particleRT = np.zeros(num_particles) # Array which stores each particles' number of steps
Time = np.linspace(dt, num_steps*dt, num_steps) # Array that stores time steps
timeLinSpaced = np.linspace(dt, dt*num_steps, bins) # Linearly spaced bins
timeLogSpaced = np.logspace(np.log10(dt), np.log10(dt*num_steps), bins) # Logarithmic spaced bins

# Functions ##########################################################################

def apply_reflection(x, y, crossInToOutAbove, crossInToOutBelow,  crossOutToInAbove, crossOutToInBelow, uby, lby):
    x = np.where(x<lbx, -x+2*lbx, x)
    y = np.where(crossInToOutAbove, -y+2*uby, y)
    y = np.where(crossInToOutBelow, -y+2*lby, y)
    y = np.where(crossOutToInAbove, -y+2*uby, y)
    y = np.where(crossOutToInBelow, -y+2*lby, y)
    return x, y

def update_positions(x, y, fracture, matrix, Df, Dm, dt, meanEta, stdEta):
    x[fracture] += np.sqrt(2*Df*dt)*np.random.normal(meanEta, stdEta, np.sum(fracture))
    y[fracture] += np.sqrt(2*Df*dt)*np.random.normal(meanEta, stdEta, np.sum(fracture))
    x[matrix] += np.sqrt(2*Dm*dt)*np.random.normal(meanEta, stdEta, np.sum(matrix))
    y[matrix] += np.sqrt(2*Dm*dt)*np.random.normal(meanEta, stdEta, np.sum(matrix))
    return x, y

# Time loop ###########################################################################

start_time = time.time() # Start timing the while loop

while (cdf<stopBTC/100*num_particles) & (t<num_steps*dt):

    # Store the positions of each particle for all the time steps 
    if recordTrajectories:
        xPath[:, t] = x  # Store x positions for the current time step
        yPath[:, t] = y  # Store y positions for the current time step        

    isIn = x<rbx # Get the positions of the particles that are in the domain (wheter inside or outside the fracture)
    fracture = isIn & inside # Particles in the fracture
    outside = np.array(outsideAbove) | np.array(outsideBelow) # Particles outside the fracture
    matrix = isIn & outside # Particles in the domain and outside the fracture

    # Update the position of all the particles at a given time steps according to the Langevin dynamics
    x, y = update_positions(x, y, fracture, matrix, Df, Dm, dt, meanEta, stdEta)

    # Particles which in principles would cross the fractures' walls
    crossOutAbove = inside & (y>uby)
    crossOutBelow = inside & (y<lby)
    crossInAbove = outsideAbove  & (y > lby) & (y < uby)
    crossInBelow = outsideBelow & (y>lby) & (y<uby)

    # Successfull crossing based on uniform probability distribution
    crossInToOut = np.random.rand(num_particles) < reflectedInward/100
    crossOutToIn = np.random.rand(num_particles) > reflectedOutward/100

    # Find the indeces of those particles that did not successfully corss the fracture
    crossInToOutAbove = crossOutAbove & crossInToOut # Above upper boundary y
    crossInToOutBelow = crossOutBelow & crossInToOut # Below lower boundary y
    crossOutToInAbove = crossInAbove & crossOutToIn # Above upper boundary y
    crossOutToInBelow = crossInBelow & crossOutToIn # Below lower boundary y
    
    # Update the reflected particles' positions according to an elastic reflection dynamic
    x, y = apply_reflection(x, y, crossInToOutAbove, crossInToOutBelow,  crossOutToInAbove, crossOutToInBelow, uby, lby)

    inside = (y<uby) & (y>lby) # Particles inside the fracture
    outsideAbove = y>uby # Particles in the porous matrix above the fracture
    outsideBelow = y<lby #Particles in the porous matrix below the fracture

    pdf_part[int(t/dt)] = sum(x[isIn]>rbx) # Count the particle which exit the right boundary at each time step

    cdf = sum(pdf_part)
    t += dt    

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


# Statistichs
print(f"Execution time: {execution_time:.6f} seconds")
print(f"<t>: {meanTstep:.6f} s")
print(f"sigmat: {stdTstep:.6f} s")