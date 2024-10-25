import numpy as np
import math
import scipy.stats
import time

# Features ###################################################################
plotCharts = True # It controls graphical features (disable when run on HPC)
# recordVideo = False # It slows down the script
recordTrajectories = True # It uses up memory
lbxOn = True # It controls the left boundary condition
degradation = False # Switch for the degradation of the particles
reflection = False # Switch between reflection and adsorption

if plotCharts:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

# Parameters #################################################################
sim_time = int(1e3)
dt = 0.1 # Time step
num_steps = int(sim_time/dt) # Number of steps
Dm = 0.1  # Diffusion for particles moving in the porous matrix
Df = 0.1  # Diffusion for particles moving in the fracture
meanEta = 0 # Spatial jump distribution paramenter
stdEta = 1 # Spatial jump distribution paramenter
num_particles = int(1e2) # Number of particles in the simulation
uby = 1 # Vertical Upper Boundary
lby = -1 # Vertical Lower Boundary
rbx = 10 # Horizontal Right Boundary
if lbxOn:
    lbx = 0 # Horizontal Left Boundary
else:
    lbx = -rbx
init_shift = 0 # It aggregates the initial positions of the particles around the centre of the domain
reflectedInward = 100 # Percentage of impacts from the fracture reflected again into the fracture
reflectedOutward = 20 # Percentage of impacts from the porous matrix reflected again into the porous matrix
animatedParticle = 0 # Index of the particle whose trajectory will be animated
fTstp = 0 # First time step to be recorded in the video
lTstp = 90 # Final time step to appear in the video
binsTime = 50 # Number of temporal bins for the logarithmic plot
binsSpace = 50 # Number of spatial bins for the concentration profile
recordSpatialConc = int(1e2) # Concentration profile recorded time
stopBTC = 100 # % of particles that need to pass the control plane before the simulation is ended
k_deg = 0.01 # Degradation kinetic constant
k_ads = 0.1 # Adsorption constant
ap = 0.4 # Adsorption probability

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
crossOutLeft = [False for _ in range(num_particles)]
outsideAbove = [False for _ in range(num_particles)]
outsideBelow = [False for _ in range(num_particles)]
particleRT = np.zeros(num_particles) # Array which stores each particles' number of steps
Time = np.linspace(dt, sim_time, num_steps) # Array that stores time steps
timeLinSpaced = np.linspace(dt, dt*num_steps, binsTime) # Linearly spaced bins
timeLogSpaced = np.logspace(np.log10(dt), np.log10(dt*num_steps), binsTime) # Logarithmically spaced bins
xBins = np.linspace(lbx, rbx, binsSpace) # Linearly spaced bins
xLogBins = np.logspace(np.log10(1e-10), np.log10(rbx), binsSpace) # Logarithmically spaced bins

# Functions ##########################################################################
def update_positions(x, y, fracture, matrix, Df, Dm, dt, meanEta, stdEta):
    x[fracture] += np.sqrt(2*Df*dt)*np.random.normal(meanEta, stdEta, np.sum(fracture))
    y[fracture] += np.sqrt(2*Df*dt)*np.random.normal(meanEta, stdEta, np.sum(fracture))
    x[matrix] += np.sqrt(2*Dm*dt)*np.random.normal(meanEta, stdEta, np.sum(matrix))
    y[matrix] += np.sqrt(2*Dm*dt)*np.random.normal(meanEta, stdEta, np.sum(matrix))
    return x, y

def apply_reflection(x, y, crossInToOutAbove, crossInToOutBelow,  crossOutToInAbove, crossOutToInBelow, 
                     crossOutAbove, crossOutBelow, crossInAbove, crossInBelow, uby, lby, lbxOn):
    if lbxOn:
        x = np.where(x<lbx, -x+2*lbx, x)
    y[crossOutAbove] = np.where(crossInToOutAbove[crossOutAbove], y[crossOutAbove], -y[crossOutAbove]+2*uby)
    y[crossOutBelow] = np.where(crossInToOutBelow[crossOutBelow], y[crossOutBelow], -y[crossOutBelow]+2*lby)
    y[crossInAbove] = np.where(crossOutToInAbove[crossInAbove], y[crossInAbove], -y[crossInAbove]+2*uby)
    y[crossInBelow] = np.where(crossOutToInBelow[crossInBelow], y[crossInBelow], -y[crossInBelow]+2*lby)
    yoa = np.where(crossOutAbove)[0]
    yca = np.where(crossInToOutAbove)[0]
    yob = np.where(crossOutBelow)[0]
    ycb = np.where(crossInToOutBelow)[0]
    reflected = np.concatenate((np.setdiff1d(yoa, yca), np.setdiff1d(yob, ycb))) # Particles that are reflected as the difference between those that hit the wall and the ones that cross it
    while np.any((y[reflected]<lby) | (y[reflected]>uby)): # Check on the positions of all the particles that should not cross
        y[reflected[y[reflected]>uby]] = -y[reflected[y[reflected]>uby]] + 2*uby # Reflect them back
        y[reflected[y[reflected]<lby]] = -y[reflected[y[reflected]<lby]] + 2*lby # Reflect them back
    return x, y

def apply_adsorption(x, y, crossOutAbove, crossOutBelow, crossOutLeft, adsDist):
    x = np.where(crossOutLeft & (adsDist<=ap), lbx, x)
    y = np.where(crossOutAbove & (adsDist<=ap), uby, y)  # Store x positions for the current time step
    y = np.where(crossOutBelow & (adsDist<=ap), lby, y)  # Store y positions for the current time step
    x = np.where(crossOutLeft & (adsDist>ap), -x+2*lbx, x)
    y = np.where(crossOutAbove & (adsDist>ap), -y+2*uby, y)  # Store x positions for the current time step
    y = np.where(crossOutBelow & (adsDist>ap), -y+2*lby, y)  # Store y positions for the current time step        
    return x, y

def analytical_seminf(x, t, D):
    y = x*np.exp(-x**2/(4*D*t))/(np.sqrt(4*np.pi*D*t**3))
    return y

def analytical_inf(x, t, D):
    y = np.exp(-x**2/(4*D*t))/(np.sqrt(4*np.pi*D*t))
    return y

def degradation_dist(num_steps, k_deg, num_particles):
    t_steps = np.linspace(0, sim_time, num_steps)
    exp_prob = k_deg*np.exp(-k_deg*t_steps)
    exp_prob /= exp_prob.sum()
    valueRange = np.linspace(0, sim_time, num_steps)
    survivalTimeDist = np.random.choice(valueRange, size=num_particles, p=exp_prob)
    return survivalTimeDist, exp_prob

def adsorption_dist(k_ads):
    points = np.linspace(0, 1, 1000)
    expProbAds = np.exp(-k_ads*points)
    expProbAds /= expProbAds.sum()
    valueRange = np.linspace(0, 1, 1000)
    adsDist = np.random.choice(valueRange, size=num_particles, p=expProbAds)
    return adsDist

# Time loop ###########################################################################
start_time = time.time() # Start timing the while loop

# Chemical degradation times
if degradation:
    survivalTimeDist, exp_prob = degradation_dist(num_steps, k_deg, num_particles)
else:
    survivalTimeDist = np.ones(num_particles)*sim_time

while (cdf<stopBTC/100) & (t<sim_time):

    liveParticle = survivalTimeDist>t # Particles which are degradeted

    # Store the positions of each particle for all the time steps 
    if recordTrajectories:
        xPath[:, int(t/dt)] = np.where(liveParticle, x, 0)  # Store x positions for the current time step
        yPath[:, int(t/dt)] = np.where(liveParticle, y, 0)  # Store y positions for the current time step

    isIn = abs(x)<rbx # Get the positions of the particles that are in the domain (wheter inside or outside the fracture)
    fracture = isIn & inside & liveParticle # Particles in the domain and inside the fracture and not degradeted yet
    outside = np.array(outsideAbove) | np.array(outsideBelow) # Particles outside the fracture
    matrix = isIn & outside & liveParticle # Particles in the domain and outside the fracture and not degradeted yet
    if lbxOn:
        fracture = fracture & np.logical_not(crossOutLeft)
        matrix = matrix & np.logical_not(crossOutLeft)

    # Update the position of all the particles at a given time steps according to the Langevin dynamics
    x, y = update_positions(x, y, fracture, matrix, Df, Dm, dt, meanEta, stdEta)

    # Particles which in principles would cross the fractures' walls
    crossOutLeft = (x<lbx)
    crossOutAbove = fracture & (y>uby)
    crossOutBelow = fracture & (y<lby)
    crossInAbove = outsideAbove  & (y > lby) & (y < uby)
    crossInBelow = outsideBelow & (y>lby) & (y<uby)

    # Decide the number of impacts that will cross the fracture's walls
    probCrossOutAbove = np.full(num_particles, False)
    probCrossOutBelow = np.full(num_particles, False)
    probCrossInAbove = np.full(num_particles, False)
    probCrossInBelow = np.full(num_particles, False)
    probCrossOutAbove[np.where(crossOutAbove)[0]] = np.random.rand(np.sum(crossOutAbove)) > reflectedInward/100
    probCrossOutBelow[np.where(crossOutBelow)[0]] = np.random.rand(np.sum(crossOutBelow)) > reflectedInward/100
    probCrossInAbove[np.where(crossInAbove)[0]] = np.random.rand(np.sum(crossInAbove)) > reflectedOutward/100
    probCrossInBelow[np.where(crossInBelow)[0]] = np.random.rand(np.sum(crossInBelow)) > reflectedOutward/100

    # Successfull crossing based on uniform probability distribution
    crossInToOutAbove = probCrossOutAbove & crossOutAbove
    crossInToOutBelow = probCrossOutBelow & crossOutBelow
    crossOutToInAbove = probCrossInAbove & crossInAbove
    crossOutToInBelow = probCrossInBelow & crossInBelow
    
    if reflection:
        # Update the reflected particles' positions according to an elastic reflection dynamic
        x, y = apply_reflection(x, y, crossInToOutAbove, crossInToOutBelow,  crossOutToInAbove, crossOutToInBelow,
                                crossOutAbove, crossOutBelow, crossInAbove, crossInBelow, uby, lby, lbxOn)
    else:
        adsDist = adsorption_dist(k_ads)
        x, y = apply_adsorption(x, y, crossOutAbove, crossOutBelow, crossOutLeft, adsDist)

    crossOutLeft = (x==lbx)
    inside = (y<uby) & (y>lby) # Particles inside the fracture
    outsideAbove = y>uby # Particles in the porous matrix above the fracture
    outsideBelow = y<lby # Particles in the porous matrix below the fracture

    pdf_part[int(t/dt)] = sum(abs(x[isIn])>rbx) # Count the particle which exit the right boundary at each time step

    if (t <= recordSpatialConc) & (recordSpatialConc < t+dt):
        if lbxOn:
            countsSpaceLog, binEdgeSpaceLog = np.histogram(x, xLogBins, density=True)
        else:
            countsSpace, binEdgeSpace = np.histogram(x, xBins, density=True)

    cdf = sum(pdf_part)/num_particles
    t += dt    

end_time = time.time() # Stop timing the while loop
execution_time = end_time - start_time

# Retrieve the number of steps for each particle from the pdf of the breakthrough curve
for index, value in enumerate(pdf_part):
    particleRT[int(i):int(i+value)] = index*dt
    i = i+value

# Compute simulation statistichs
meanTstep = particleRT.mean()
stdTstep = particleRT.std()
if (dt*10>(uby-lby)**2/Df):
    print("WARNING! Time step dt should be reduced to avoid jumps across the fracture width")

# Verificaiton of the code
if lbxOn:
    yAnalytical = analytical_seminf(xLogBins, recordSpatialConc, Df)
else:
    yAnalytical = analytical_inf(xBins, recordSpatialConc, Df)

# Plot section #########################################################################
# Trajectories
if plotCharts and recordTrajectories:
    plt.figure(figsize=(8, 8))
    for i in range(num_particles):
        plt.plot(xPath[i][:][xPath[i][:]!=0], yPath[i][:][xPath[i][:]!=0], lw=0.5)
    plt.axhline(y=uby, color='r', linestyle='--', linewidth=2)
    plt.axhline(y=lby, color='r', linestyle='--', linewidth=2)
    if lbxOn:
        plt.axvline(x=lbx, color='black', linestyle='-', linewidth=2)
    plt.title("2D Diffusion Process (Langevin Equation)")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True)
    plt.show()

if plotCharts:
    # PDF
    plt.figure(figsize=(8, 8))
    plt.plot(Time, pdf_part/num_particles)
    plt.xscale('log')
    plt.title("PDF")

    # CDF
    plt.figure(figsize=(8, 8))
    plt.plot(Time, np.cumsum(pdf_part)/num_particles)
    plt.xscale('log')
    plt.title("CDF")

    if cdf>0:
        # 1-CDF
        plt.figure(figsize=(8, 8))
        plt.plot(Time, 1-np.cumsum(pdf_part)/num_particles)
        plt.xscale('log')
        plt.yscale('log')
        plt.title("1-CDF")

        # Binning for plotting the pdf from a Lagrangian vector
        countsLog, binEdgesLog = np.histogram(particleRT, timeLogSpaced, density=True)
        plt.figure(figsize=(8, 8))
        plt.plot(binEdgesLog[:-1][countsLog!=0], countsLog[countsLog!=0], 'r*')
        plt.xscale('log')
        plt.yscale('log')

    # Spatial concentration profile at 'recordSpatialConc' time
    plt.figure(figsize=(8, 8))
    if lbxOn:
        plt.plot(binEdgeSpaceLog[:-1][countsSpaceLog!=0], countsSpaceLog[countsSpaceLog!=0], 'b-')    
        plt.plot(xLogBins, yAnalytical, 'k-')
        plt.axvline(x=lbx, color='black', linestyle='-', linewidth=2)
    else:
        plt.plot(binEdgeSpace[:-1][countsSpace!=0], countsSpace[countsSpace!=0], 'b-')
        plt.plot(xBins, yAnalytical, 'k-')
    plt.title("Empirical vs analytical solution")

    if degradation:
        # Distribution of survival times for particles
        plt.figure(figsize=(8, 8))
        if recordTrajectories:
            effTstepNum = np.array([np.count_nonzero(row)*dt for row in xPath])
            plt.plot(np.arange(0, num_particles, 1), np.sort(effTstepNum)[::-1], 'b*')
        plt.plot(np.arange(0, num_particles, 1), np.sort(survivalTimeDist)[::-1], 'k-')
        plt.title("Survival time distribution")

# Statistichs
print(f"Execution time: {execution_time:.6f} seconds")
print(f"<t>: {meanTstep:.6f} s")
print(f"sigmat: {stdTstep:.6f} s")