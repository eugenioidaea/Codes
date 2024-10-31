
from IPython import get_ipython
get_ipython().run_line_magic('reset', '-f')
import numpy as np
import time

# Features ###################################################################
plotCharts = True # It controls graphical features (disable when run on HPC)
# recordVideo = False # It slows down the script
recordTrajectories = True # It uses up memory
lbxOn = True # It controls the position of the left boundary
lbxAdsorption = True # It controls whether the particles get adsorpted or reflected on the left boundary 
degradation = False # Switch for the degradation of the particles
reflection = True # It defines the upper and lower fracture's walls behaviour, wheather particles are reflected or adsorpted
stopOnCDF = False # Simulation is terminated when CDF reaches the stopBTC value

if plotCharts:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

# Parameters #################################################################
sim_time = int(1e3)
dt = 1 # Time step
num_steps = int(sim_time/dt) # Number of steps
x0 = 2 # Initial horizontal position of the particles
Dm = 0.1  # Diffusion for particles moving in the porous matrix
Df = 0.1  # Diffusion for particles moving in the fracture
meanEta = 0 # Spatial jump distribution paramenter
stdEta = 1 # Spatial jump distribution paramenter
num_particles = int(1e3) # Number of particles in the simulation
uby = 1 # Upper Boundary
lby = -1 # Lower Boundary
cpx = 10 # Vertical Control Plane
if lbxOn:
    lbx = 0 # Left Boundary
init_shift = 0 # It aggregates the initial positions of the particles around the centre of the domain
reflectedInward = 100 # Percentage of impacts from the fracture reflected again into the fracture
reflectedOutward = 20 # Percentage of impacts from the porous matrix reflected again into the porous matrix
animatedParticle = 0 # Index of the particle whose trajectory will be animated
fTstp = 0 # First time step to be recorded in the video
lTstp = 90 # Final time step to appear in the video
binsTime = 20 # Number of temporal bins for the logarithmic plot
binsSpace = 50 # Number of spatial bins for the concentration profile
recordSpatialConc = int(1e2) # Concentration profile recorded time
stopBTC = 100 # % of particles that need to pass the control plane before the simulation is ended
k_deg = 0.01 # Degradation kinetic constant
k_ads = 0.1 # Adsorption constant
ap = 1 # Adsorption probability

# Initialisation ####################################################################
t = 0 # Time
i = 0 # Index for converting Eulerian pdf to Lagrangian pdf
cdf = 0
pdf_part = np.zeros(num_steps)
pdf_lbxOn = np.zeros(num_steps)
x = np.ones(num_particles)*x0 # Horizontal initial positions
y = np.linspace(lby+init_shift, uby-init_shift, num_particles) # Vertical initial positions
if recordTrajectories:
    xPath = np.zeros((num_particles, num_steps))  # Matrix for storing x trajectories
    yPath = np.zeros((num_particles, num_steps))  # Matrix for storing y trajectories
inside = [True for _ in range(num_particles)]
crossOutLeft = [False for _ in range(num_particles)]
outsideAbove = [False for _ in range(num_particles)]
outsideBelow = [False for _ in range(num_particles)]
particleRT = np.zeros(num_particles) # Array which stores each particles' number of steps
particleSemiInfRT = np.zeros(num_particles)
Time = np.linspace(dt, sim_time, num_steps) # Array that stores time steps
timeLinSpaced = np.linspace(dt, sim_time, binsTime) # Linearly spaced bins
timeLogSpaced = np.logspace(np.log10(dt), np.log10(sim_time), binsTime) # Logarithmically spaced bins
xBins = np.linspace(-cpx, cpx, binsSpace) # Linearly spaced bins
xLogBins = np.logspace(np.log10(1e-10), np.log10(cpx), binsSpace) # Logarithmically spaced bins
probCrossOutAbove = np.full(num_particles, False) # Probability of crossing the fracture's walls
probCrossOutBelow = np.full(num_particles, False) # Probability of crossing the fracture's walls
probCrossInAbove = np.full(num_particles, False) # Probability of crossing the fracture's walls
probCrossInBelow = np.full(num_particles, False) # Probability of crossing the fracture's walls

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
        if lbxAdsorption:
            x = np.where(crossOutLeft, lbx, x)
        else:
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
    if lbxOn:
        x = np.where(crossOutLeft & (adsDist<=ap), lbx, x)
        x = np.where(crossOutLeft & (adsDist>ap), -x+2*lbx, x)
    y = np.where(crossOutAbove & (adsDist<=ap), uby, y)  # Store x positions for the current time step
    y = np.where(crossOutBelow & (adsDist<=ap), lby, y)  # Store y positions for the current time step
    y = np.where(crossOutAbove & (adsDist>ap), -y+2*uby, y)  # Store x positions for the current time step
    y = np.where(crossOutBelow & (adsDist>ap), -y+2*lby, y)  # Store y positions for the current time step        
    return x, y

def analytical_seminf(x0, t, D):
    y = x0*np.exp(-x0**2/(4*D*t))/(np.sqrt(4*np.pi*D*t**3))
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

while t<sim_time:

    if stopOnCDF & (cdf>stopBTC/100):
        break

    liveParticle = survivalTimeDist>t # Particles which are degradeted

    # Store the positions of each particle for all the time steps 
    if recordTrajectories:
        xPath[:, int(t/dt)] = np.where(liveParticle, x, 0)  # Store x positions for the current time step
        yPath[:, int(t/dt)] = np.where(liveParticle, y, 0)  # Store y positions for the current time step

    isIn = abs(x)<cpx # Get the positions of the particles that are located within the control planes (wheter inside or outside the fracture)
    fracture = inside & liveParticle # Particles in the domain and inside the fracture and not degradeted yet
    outside = np.array(outsideAbove) | np.array(outsideBelow) # Particles outside the fracture
    matrix = isIn & outside & liveParticle # Particles in the domain and outside the fracture and not degradeted yet
    if lbxOn:
        fracture = fracture & np.logical_not(crossOutLeft)
        matrix = matrix & np.logical_not(crossOutLeft)

    # Update the position of all the particles at a given time steps according to the Langevin dynamics
    x, y = update_positions(x, y, fracture, matrix, Df, Dm, dt, meanEta, stdEta)

    # Particles which in principles would cross the fractures' walls
    crossOutAbove = fracture & (y>uby)
    crossOutBelow = fracture & (y<lby)
    crossInAbove = outsideAbove  & (y > lby) & (y < uby)
    crossInBelow = outsideBelow & (y>lby) & (y<uby)

    # Decide the number of impacts that will cross the fracture's walls
    probCrossOutAbove[np.where(crossOutAbove)[0]] = np.random.rand(np.sum(crossOutAbove)) > reflectedInward/100
    probCrossOutBelow[np.where(crossOutBelow)[0]] = np.random.rand(np.sum(crossOutBelow)) > reflectedInward/100
    probCrossInAbove[np.where(crossInAbove)[0]] = np.random.rand(np.sum(crossInAbove)) > reflectedOutward/100
    probCrossInBelow[np.where(crossInBelow)[0]] = np.random.rand(np.sum(crossInBelow)) > reflectedOutward/100

    # Successfull crossing based on uniform probability distribution
    crossInToOutAbove = probCrossOutAbove & crossOutAbove
    crossInToOutBelow = probCrossOutBelow & crossOutBelow
    crossOutToInAbove = probCrossInAbove & crossInAbove
    crossOutToInBelow = probCrossInBelow & crossInBelow

    # Particles hitting the left control plane
    if lbxOn:
        crossOutLeft = (x<lbx)
        pdf_lbxOn[int(t/dt)] = sum(crossOutLeft)

    # Decide what happens to the particles which hit the fracture's walls: all get reflected, some get reflected some manage to escape, all get absorbed by the fracture's walls
    if reflection:
        # Update the reflected particles' positions according to an elastic reflection dynamic
        x, y = apply_reflection(x, y, crossInToOutAbove, crossInToOutBelow,  crossOutToInAbove, crossOutToInBelow,
                                crossOutAbove, crossOutBelow, crossInAbove, crossInBelow, uby, lby, lbxOn)
    else:
        adsDist = adsorption_dist(k_ads)
        x, y = apply_adsorption(x, y, crossOutAbove, crossOutBelow, crossOutLeft, adsDist)

    # Record the pdf of the btc on the left control panel        
    if lbxOn:
        crossOutLeft = (x==lbx)

    # Find the particle's locations with relative to the fracture's walls
    inside = (y<uby) & (y>lby) # Particles inside the fracture
    outsideAbove = y>uby # Particles in the porous matrix above the fracture
    outsideBelow = y<lby # Particles in the porous matrix below the fracture

    pdf_part[int(t/dt)] = sum(abs(x[isIn])>cpx) # Count the particle which exit the control planes at each time step

    # Record the spatial distribution of the particles at a given time, e.g.: 'recordSpatialConc'
    if (t >= recordSpatialConc) & (recordSpatialConc < t+dt):
        countsSpace, binEdgeSpace = np.histogram(x, xBins, density=True)
        binCenterSpace = (binEdgeSpace[:-1] + binEdgeSpace[1:]) / 2

    # Compute the CDF and increase the time
    cdf = sum(pdf_part)/num_particles
    t += dt

end_time = time.time() # Stop timing the while loop
execution_time = end_time - start_time

# Post processing ######################################################################
iPart = 0
iLbxOn = 0
# Retrieve the number of steps for each particle from the pdf of the breakthrough curve
for index, (valPdfPart, valLbxOn) in enumerate(zip(pdf_part, pdf_lbxOn)):
    particleRT[int(iPart):int(iPart+valPdfPart)] = index*dt
    particleSemiInfRT[int(iLbxOn):int(iLbxOn+valLbxOn)] = index*dt
    iPart = iPart+valPdfPart
    iLbxOn = iLbxOn+valLbxOn

# Compute simulation statistichs
meanTstep = particleRT.mean()
stdTstep = particleRT.std()
if (dt*10>(uby-lby)**2/Df):
    print("WARNING! Time step dt should be reduced to avoid jumps across the fracture width")

# Verificaiton of the code
if lbxOn:
    countsSemiInfLog, binSemiInfLog = np.histogram(particleSemiInfRT, timeLogSpaced, density=True)
    timeBinsLog = (binSemiInfLog[:-1] + binSemiInfLog[1:]) / 2
    analPdfSemiInf = analytical_seminf(x0, timeBinsLog, Df)
else:
    yAnalytical = analytical_inf(binCenterSpace, recordSpatialConc, Df)

# Plot section #########################################################################
if plotCharts and recordTrajectories:
    # Trajectories
    plt.figure(figsize=(8, 8))
    for i in range(num_particles):
        plt.plot(xPath[i][:][xPath[i][:]!=0], yPath[i][:][xPath[i][:]!=0], lw=0.5)
    plt.axhline(y=uby, color='r', linestyle='--', linewidth=2)
    plt.axhline(y=lby, color='r', linestyle='--', linewidth=2)
    plt.axvline(x=cpx, color='b', linestyle='--', linewidth=2)
    plt.axvline(x=-cpx, color='b', linestyle='--', linewidth=2)
    plt.axvline(x=x0, color='yellow', linestyle='--', linewidth=2)
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

if plotCharts and cdf>0:
    # 1-CDF
    plt.figure(figsize=(8, 8))
    plt.plot(Time, 1-np.cumsum(pdf_part)/num_particles)
    plt.xscale('log')
    plt.yscale('log')
    plt.title("1-CDF")

    # Binning for plotting the pdf from a Lagrangian vector
    countsLog, binEdgesLog = np.histogram(particleRT, timeLogSpaced, density=True)
    binCentersLog = (binEdgesLog[:-1] + binEdgesLog[1:]) / 2
    plt.figure(figsize=(8, 8))
    plt.plot(binCentersLog[countsLog!=0], countsLog[countsLog!=0], 'r*')
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Lagrangian PDF")

if plotCharts and lbxOn:
    plt.figure(figsize=(8, 8))
    plt.plot(timeBinsLog, countsSemiInfLog, 'b*')
    plt.plot(timeBinsLog, analPdfSemiInf, 'r-')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('PDF of BTC - Simulated vs analytical solution')
    plt.xlabel('Time step')
    plt.ylabel('Normalised number of particles')

if plotCharts and degradation:
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
if lbxOn:
    print(f"sum(|empiricalPdf-analyticalPdf|)= {sum(np.abs(countsSemiInfLog-analPdfSemiInf))}")
else:
    print(f"sum(|yEmpirical-yAnalytical|)= {sum(np.abs(countsSpace-yAnalytical))}")