debug = False
if not debug:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
import numpy as np
import time

# Features ###################################################################
plotCharts = True # It controls graphical features (disable when run on HPC)
recordTrajectories = False # It uses up memory
degradation = True # Switch for the degradation of the particles
reflection = True # It defines the upper and lower fracture's walls behaviour, wheather particles are reflected or adsorpted
lbxOn = False # It controls the position of the left boundary
lbxAdsorption = False # It controls whether the particles get adsorpted or reflected on the left boundary 
stopOnCDF = False # Simulation is terminated when CDF reaches the stopBTC value
vcpOn = False # It regulates the visualisation of the vertical control plane
matrixDiffVerification = False # It activates the matrix-diffusion verification testcase
# recordVideo = False # It slows down the script

if plotCharts:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

# Parameters #################################################################
num_particles = int(1e6) # Number of particles in the simulation
sim_time = int(500)
dt = 1 # Time step
num_steps = int(sim_time/dt) # Number of steps
Df = 0.1  # Diffusion for particles moving in the fracture
Dm = 0.001  # Diffusion for particles moving in the porous matrix
x0 = 0 # Initial horizontal position of the particles
uby = 1 # Upper Boundary
lby = -1 # Lower Boundary
vcp = 10 # Vertical Control Plane
if lbxOn:
    lbx = 0 # Vertical control plane
recordSpatialConc = int(1e2) # Concentration profile recorded time
stopBTC = 100 # % of particles that need to pass the control plane before the simulation is ended
k_deg = 0.05 # Degradation kinetic constant
k_ads = 0.1 # Adsorption constant
ap = 1 # Adsorption probability
binsXinterval = 10 # Extension of the region where spatial concentration is recorded
binsTime = 40 # Number of temporal bins for the logarithmic plot
binsSpace = 50 # Number of spatial bins for the concentration profile
reflectedInward = 1.0 # Probability of impacts from the fracture reflected again into the fracture
# reflectedInward = np.sqrt(Df)/(np.sqrt(Df)+np.sqrt(Dm))
reflectedOutward = 1.0 # Probability of impacts from the porous matrix reflected again into the porous matrix
# reflectedOutward = np.sqrt(Dm)/(np.sqrt(Df)+np.sqrt(Dm))
init_shift = 0 # It aggregates the initial positions of the particles around the centre of the domain
meanEta = 0 # Spatial jump distribution paramenter
stdEta = 1 # Spatial jump distribution paramenter
animatedParticle = 0 # Index of the particle whose trajectory will be animated
fTstp = 0 # First time step to be recorded in the video
lTstp = 90 # Final time step to appear in the video
if matrixDiffVerification:
    lbx = 4
    rbx = 8
    cbx = 6

# Initialisation ####################################################################
t = 0 # Time
i = 0 # Index for converting Eulerian pdf to Lagrangian pdf
cdf = 0
pdf_part = np.zeros(num_steps)
pdf_lbxOn = np.zeros(num_steps)
x = np.ones(num_particles)*x0 # Horizontal initial positions
y = np.linspace(lby+init_shift, uby-init_shift, num_particles) # Vertical initial positions
if recordTrajectories:
    xPath = np.zeros((num_particles, num_steps+1))  # Matrix for storing x trajectories
    yPath = np.zeros((num_particles, num_steps+1))  # Matrix for storing y trajectories
inside = np.ones(num_particles, dtype=bool)
crossOutLeft = [False for _ in range(num_particles)]
crossOutRight = [False for _ in range(num_particles)]
outsideAbove = [False for _ in range(num_particles)]
outsideBelow = [False for _ in range(num_particles)]
liveParticle = np.array([True for _ in range(num_particles)])
particleRT = []
particleSemiInfRT = []
Time = np.linspace(dt, sim_time, num_steps) # Array that stores time steps
# timeLinSpaced = np.linspace(dt, sim_time, binsTime) # Linearly spaced bins
timeLogSpaced = np.logspace(np.log10(dt), np.log10(sim_time), binsTime) # Logarithmically spaced bins
xBins = np.linspace(-binsXinterval, binsXinterval, binsSpace) # Linearly spaced bins
yBins = np.linspace(-binsXinterval, binsXinterval, binsSpace) # Linearly spaced bins
probCrossOutAbove = np.full(num_particles, False) # Probability of crossing the fracture's walls
probCrossOutBelow = np.full(num_particles, False) # Probability of crossing the fracture's walls
probCrossInAbove = np.full(num_particles, False) # Probability of crossing the fracture's walls
probCrossInBelow = np.full(num_particles, False) # Probability of crossing the fracture's walls
particleSteps = np.zeros(num_particles)
impacts = 0

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
    if matrixDiffVerification:
        x[crossOutLeft] = -x[crossOutLeft]+2*lbx
        x[crossOutRight] = -x[crossOutRight]+2*rbx
    return x, y

def apply_adsorption(x, y, crossOutAbove, crossOutBelow, crossOutLeft, adsDist, impacts):
    if lbxOn:
        x = np.where(crossOutLeft & (adsDist<=ap), lbx, x)
        x = np.where(crossOutLeft & (adsDist>ap), -x+2*lbx, x)
    y[crossOutAbove] = np.where(adsDist[crossOutAbove]<=ap, uby, -y[crossOutAbove]+2*uby)
    y[crossOutBelow] = np.where(adsDist[crossOutBelow]<=ap, lby, -y[crossOutBelow]+2*lby)
    impacts = impacts+np.count_nonzero(crossOutAbove | crossOutBelow)
    # y = np.where(crossOutAbove & (adsDist<=ap), uby, y)  # Particles get adsorbed with probability 'ap'
    # y = np.where(crossOutBelow & (adsDist<=ap), lby, y)  # Particles get adsorbed with probability 'ap'
    # y = np.where(crossOutAbove & (adsDist>ap), -y+2*uby, y)  # Particles are reflected with probability '1-ap'
    # y = np.where(crossOutBelow & (adsDist>ap), -y+2*lby, y)  # Particles are reflected with probability '1-ap'
    return x, y, impacts

def analytical_seminf(x0, t, D):
    y = x0*np.exp(-x0**2/(4*D*t))/(np.sqrt(4*np.pi*D*t**3))
    return y

def analytical_inf(x, t, D):
    y = np.exp(-x**2/(4*D*t))/(np.sqrt(4*np.pi*D*t))
    return y

def degradation_dist(num_steps, k_deg, num_particles):
    t_steps = np.linspace(1, sim_time, num_steps)
    exp_prob = k_deg*np.exp(-k_deg*t_steps)
    exp_prob = exp_prob/exp_prob.sum()
    survivalTimeDist = np.random.choice(t_steps, size=num_particles, p=exp_prob)
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

while t<sim_time and bool(liveParticle.any()) and bool(((y!=-1) & (y!=1)).any()):

    liveParticle = np.array(survivalTimeDist>t) # Particles which are not degradeted

    if stopOnCDF & (cdf>stopBTC/100):
        break

    isIn = abs(x)<vcp # Get the positions of the particles that are located within the control planes (wheter inside or outside the fracture)
    fracture = inside & liveParticle # Particles in the domain and inside the fracture and not degradeted yet
    outside = np.array(outsideAbove) | np.array(outsideBelow) # Particles outside the fracture
    matrix = outside & liveParticle # Particles in the domain and outside the fracture and not degradeted yet
    if lbxOn:
        fracture = fracture & np.logical_not(crossOutLeft)
        matrix = matrix & np.logical_not(crossOutLeft)
    particleSteps[fracture | matrix] += 1 # It keeps track of the number of steps of each particle (no degradeted nor adsorbed are moving)
    # particleSteps[survivalTimeDist>t] = particleSteps[survivalTimeDist>t] + 1 # It keeps track of the number of steps of each particle

    # Update the position of all the particles at a given time steps according to the Langevin dynamics
    x, y = update_positions(x, y, fracture, matrix, Df, Dm, dt, meanEta, stdEta)

    # Particles which in principles would cross the fractures' walls
    crossOutAbove = fracture & (y>uby)
    crossOutBelow = fracture & (y<lby)
    crossInAbove = outsideAbove  & (y > lby) & (y < uby)
    crossInBelow = outsideBelow & (y>lby) & (y<uby)
    if matrixDiffVerification:
        crossOutLeft = x<lbx
        crossOutRight = x>rbx
        crossLeftToRight = x>cbx
        crossRightToLeft = x<cbx

    # Decide the number of impacts that will cross the fracture's walls
    probCrossOutAbove[crossOutAbove] = np.random.rand(np.sum(crossOutAbove)) > reflectedInward
    probCrossOutBelow[crossOutBelow] = np.random.rand(np.sum(crossOutBelow)) > reflectedInward
    probCrossInAbove[crossInAbove] = np.random.rand(np.sum(crossInAbove)) > reflectedOutward
    probCrossInBelow[crossInBelow] = np.random.rand(np.sum(crossInBelow)) > reflectedOutward
    # probCrossCenterWall[np.where(crossLeftToRight)[0]] = np.random.rand(np.sum(crossLeftToRight)) > reflectedLeft
    # probCrossCenterWall[np.where(crossRightToLeft)[0]] = np.random.rand(np.sum(crossLeftToRight)) > reflectedRight

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
        # adsDist = adsorption_dist(k_ads) # Exponential distribution
        adsDist = np.random.uniform(0, 1, num_particles) # Uniform distribution
        x, y, impacts = apply_adsorption(x, y, crossOutAbove, crossOutBelow, crossOutLeft, adsDist, impacts)

    # Record the pdf of the btc on the left control panel        
    if lbxOn:
        crossOutLeft = (x==lbx)

    # Find the particle's locations with respect to the fracture's walls and exclude those that get adsorbed on the walls from the live particles
    inside = (y<uby) & (y>lby) # Particles inside the fracture
    outsideAbove = y>uby # Particles in the porous matrix above the fracture
    outsideBelow = y<lby # Particles in the porous matrix below the fracture

    # Count the particle which exit the control planes at each time step
    pdf_part[int(t/dt)] = sum(abs(x[isIn])>vcp)
    # Compute the CDF and increase the time
    cdf = sum(pdf_part)/num_particles
    # Move forward time step
    t += dt

    binCenterSpace = (xBins[:-1] + xBins[1:]) / 2
    # Record the spatial distribution of the particles at a given time, e.g.: 'recordSpatialConc'
    if (t <= recordSpatialConc) & (recordSpatialConc < t+dt):
        countsSpace, binEdgeSpace = np.histogram(x, xBins, density=True)

    # Store the positions of each particle for all the time steps 
    if recordTrajectories:
        xPath[:, int(t/dt)] = np.where(liveParticle, x, 0)  # Store x positions for the current time step
        yPath[:, int(t/dt)] = np.where(liveParticle, y, 0)  # Store y positions for the current time step

end_time = time.time() # Stop timing the while loop
execution_time = end_time - start_time

# Post processing ######################################################################
iPart = 0
iLbxOn = 0
# Retrieve the number of steps for each particle from the pdf of the breakthrough curve
for index, (valPdfPart, valLbxOn) in enumerate(zip(pdf_part, pdf_lbxOn)):
    particleRT.extend([index*dt+1]*int(valPdfPart))
    particleSemiInfRT.extend([index*dt+1]*int(valLbxOn))
    iPart = iPart+valPdfPart
    iLbxOn = iLbxOn+valLbxOn

particleRT = np.array(particleRT)
particleSemiInfRT = np.array(particleSemiInfRT)

# Compute simulation statistichs
meanTstep = np.array(particleRT).mean()
stdTstep = np.array(particleRT).std()
if (dt*10>(uby-lby)**2/Df):
    print("WARNING! Time step dt should be reduced to avoid jumps across the fracture width")

# Verificaiton of the code
if lbxOn:
    countsSemiInfLog, binSemiInfLog = np.histogram(particleSemiInfRT, timeLogSpaced, density=True)
    timeBinsLog = (binSemiInfLog[:-1] + binSemiInfLog[1:]) / 2
    analPdfSemiInf = analytical_seminf(x0, timeBinsLog, Df)
else:
    yAnalytical = analytical_inf(binCenterSpace, recordSpatialConc, Df)

# Compute the number of particles at a given time over the whole domain extension
if np.logical_not(reflection):
    # Average concentration in X and Y
    recordTdist = int(Time[-2]) # Final time step
    vInterval = np.array([x0-0.1, x0+0.1])
    hInterval = np.array([(lby+uby)/2-0.1, (lby+uby)/2+0.1])
    if recordTrajectories:
        yRecordTdist = yPath[:, recordTdist]
    else:
        yRecordTdist = y
    yDistAll = yRecordTdist[(yRecordTdist != lby) & (yRecordTdist != uby)]
    vDistAll, vBinsAll = np.histogram(yDistAll, np.linspace(lby, uby, binsSpace))
    if recordTrajectories:
        xRecordTdist = xPath[:, recordTdist]
    else:
        xRecordTdist = x
    xDistAll = xRecordTdist[(yRecordTdist != lby) & (yRecordTdist != uby)]
    hDistAll, hBinsAll = np.histogram(xDistAll, np.linspace(-binsXinterval, binsXinterval, binsSpace))

# Compute the number of particles at a given time within a vertical and a horizontal stripe
if recordTrajectories and np.logical_not(reflection):
    # Average concentration in X and Y
    recordTdist = int(Time[-2])
    vInterval = np.array([x0-0.1, x0+0.1])
    hInterval = np.array([(lby+uby)/2-0.1, (lby+uby)/2+0.1])
    yDist = yPath[(xPath[:, recordTdist]>vInterval[0]) & (xPath[:, recordTdist]<vInterval[1]), recordTdist]
    yDist = yDist[(yDist != lby) & (yDist != uby)]
    vDist, vBins = np.histogram(yDist, np.linspace(lby, uby, binsSpace))
    xDist = xPath[(yPath[:, recordTdist]>hInterval[0]) & (yPath[:, recordTdist]<hInterval[1]), recordTdist]
    hDist, hBins = np.histogram(xDist, np.linspace(-binsXinterval, binsXinterval, binsSpace))

# Survival time distribution
liveParticlesInTime = np.sum(particleSteps[:, None] > Time, axis=0)
liveParticlesInTimeNorm = liveParticlesInTime/liveParticlesInTime.sum()

# Statistichs ########################################################################
print(f"Execution time: {execution_time:.6f} seconds")
print(f"<t>: {meanTstep:.6f} s")
print(f"sigmat: {stdTstep:.6f} s")
if np.logical_not(reflection):
    print(f"# adsorbed particles/# impacts: {num_particles/impacts}")
if recordSpatialConc>t:
    print("WARNING! The simulation time is smaller than the specified time for recording the spatial distribution of the concentration")
elif lbxOn & (recordSpatialConc<t):
    print(f"sum(|empiricalPdf-analyticalPdf|)= {sum(np.abs(countsSemiInfLog-analPdfSemiInf))}")
else:
    print(f"sum(|yEmpirical-yAnalytical|)= {sum(np.abs(countsSpace-yAnalytical))}")

# Save and export variables for plotting #############################################
# Filter the variables we want to save by type
variablesToSave = {name: value for name, value in globals().items() if isinstance(value, (np.ndarray, int, float, bool))}
# Save all the variables to an .npz file
# np.savez('infiniteDomain1e6.npz', **variablesToSave)
# np.savez('semiInfiniteDomain1e3.npz', **variablesToSave)
# np.savez('degradation_3.npz', **variablesToSave)
# np.savez('totalAdsorption_3.npz', **variablesToSave)
# np.savez('finalPositions1e5.npz', **variablesToSave)
# np.savez('testSemra.npz', **variablesToSave)
# np.savez('matrixDiffusionVerification.npz', **variablesToSave)
# np.savez('partialAdsorption.npz', **variablesToSave)