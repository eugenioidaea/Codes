debug = False
if not debug:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
import numpy as np
import time

# Features ###################################################################
plotCharts =                True # It controls graphical features (disable when run on HPC)
matrixDecay =               False # It activates the radioactive decay only in the porous matrix
domainDecay =               True # Switch for the radioactive (exponential) decay of the particles in the whole domain
diffuseIntoMatrix =         False # Depending on the value of the boundary conditions (Semra 1993), particles can be reflected or partially diffuse into the porou matrix
adsorptionProbability =     True # Particles' adsorption probability (ap) sets the fraction of impacts that are adsorbed on average at every time step
matrixDiffVerification =    False # It activates the matrix-diffusion verification testcase
lbxOn =                     False # It controls the position of the left boundary
lbxAdsorption =             False # It controls whether the particles get adsorpted or reflected on the left boundary 
stopOnCDF =                 False # Simulation is terminated when CDF reaches the stopBTC value
vcpOn =                     False # It regulates the visualisation of the vertical control plane
recordTrajectories =        False # It uses up memory
recordVideo =               False # Only implemented in the non-vectorised script

if plotCharts:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

# Parameters #################################################################
num_particles = int(1e6) # Number of particles in the simulation
sim_time = int(8e2)
dt = 0.1 # Time step
num_steps = int(sim_time/dt) # Number of steps
Df = 0.004 # Diffusion for particles moving in the fracture
Dm = 0.001  # Diffusion for particles moving in the porous matrix
ap = 1 # Adsorption probability
kDecay = 0.05 # Degradation kinetic constant
xInit = 0 # Initial horizontal position of the particles
uby = 1 # Upper Boundary
lby = -1 # Lower Boundary
vcp = 10 # Vertical Control Plane
if lbxOn:
    lbx = 0 # Left Boundary X
if matrixDiffVerification:
    Dl = 0.01 # Diffusion left side of the domain (matrixDiffVerification only)
    Dr = 0.001 # Diffusion right side of the domain (matrixDiffVerification only)
    probReflectedLeft = 0.0 # Particles being reflected while crossing left to right the central wall
    # probReflectedLeft = np.sqrt(Dl)/(np.sqrt(Dl)+np.sqrt(Dr))
    probReflectedRight = 0.0 # Particles being reflected while crossing right to left the central wall
    # probReflectedRight = np.sqrt(Dr)/(np.sqrt(Dl)+np.sqrt(Dr))
# probReflectedInward = 1.0 # Probability of impacts from the fracture reflected again into the fracture
probReflectedInward = np.sqrt(Df)/(np.sqrt(Df)+np.sqrt(Dm))
# probReflectedOutward = 1.0 # Probability of impacts from the porous matrix reflected again into the porous matrix
probReflectedOutward = np.sqrt(Dm)/(np.sqrt(Df)+np.sqrt(Dm))
recordSpatialConc = int(1e2) # Concentration profile recorded time
stopBTC = 100 # % of particles that need to pass the control plane before the simulation is ended
k_ads = 0.1 # Adsorption constant (currently not used since the probability of adsorption is a random variable from a UNIFORM distribution and not EXPONENTIAL)
binsXinterval = 10 # Extension of the region where spatial concentration is recorded
binsTime = int(num_steps) # Number of temporal bins for the logarithmic plot
binsSpace = 50 # Number of spatial bins for the concentration profile
init_shift = 0 # It aggregates the initial positions of the particles around the centre of the domain
meanEta = 0 # Spatial jump distribution paramenter
stdEta = 1 # Spatial jump distribution paramenter
animatedParticle = 0 # Index of the particle whose trajectory will be animated
fTstp = 0 # First time step to be recorded in the video
lTstp = 90 # Final time step to appear in the video
if matrixDiffVerification: # Boundaries of the test case for the verification of the matrix diffusion conditions
    lbx = 4
    rbx = 8
    cbx = 6
if matrixDiffVerification: # Initial positions of the particles
    noc = 10 # Number of columns
    nor = int(num_particles/noc) # Number of rows
    x0 = np.linspace(lbx+((uby-lby)*0.01), cbx-((uby-lby)*0.01), noc) # Particles initial positions: left
    # x0 = np.linspace(cbx, rbx, noc) # Particles initial positions: right
    y0 = np.linspace(lby+((uby-lby)*0.01), uby-((uby-lby)*0.01), nor) # Small shift is needed otherwise particles tend to escape during first/second step
    x0 = np.tile(x0, nor)
    y0 = np.repeat(y0, noc)
else:
    x0 = np.ones(num_particles)*xInit # Horizontal initial positions
    y0 = np.linspace(lby+init_shift, uby-init_shift, num_particles) # Vertical initial positions

# Initialisation ####################################################################
t = 0 # Time
i = 0 # Index for converting Eulerian pdf to Lagrangian pdf
cdf = 0
pdf_part = np.zeros(num_steps)
pdf_lbxOn = np.zeros(num_steps)
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
timeStep = np.linspace(dt, sim_time, num_steps) # Array that stores time steps
timeLinSpaced = np.linspace(0, sim_time, binsTime+1) # Linearly spaced bins
timeLogSpaced = np.logspace(np.log10(dt), np.log10(sim_time), binsTime) # Logarithmically spaced bins
variableWidth = abs(timeLogSpaced-timeLogSpaced[::-1])/max(abs(timeLogSpaced-timeLogSpaced[::-1]))
timeTwoLogSpaced = np.cumsum(sim_time/sum(variableWidth)*variableWidth)
xBins = np.linspace(-binsXinterval, binsXinterval, binsSpace) # Linearly spaced bins
yBins = np.linspace(-binsXinterval, binsXinterval, binsSpace) # Linearly spaced bins
probCrossOutAbove = np.full(num_particles, False) # Probability of crossing the fracture's walls
probCrossOutBelow = np.full(num_particles, False) # Probability of crossing the fracture's walls
probCrossInAbove = np.full(num_particles, False) # Probability of crossing the fracture's walls
probCrossInBelow = np.full(num_particles, False) # Probability of crossing the fracture's walls
probCrossLeftToRight = np.full(num_particles, False)
probCrossRightToLeft = np.full(num_particles, False)
particleSteps = np.zeros(num_particles)
timeInMatrix = np.zeros(num_particles)
impacts = 0
numOfLivePart = []
Time = []
x = x0.copy()
y = y0.copy()

# Functions ##########################################################################
def update_positions(x, y, fracture, matrix, Df, Dm, dt, meanEta, stdEta):
    if matrixDiffVerification:
        x[left] += np.sqrt(2*Dl*dt)*np.random.normal(meanEta, stdEta, np.sum(left))
        y[left] += np.sqrt(2*Dl*dt)*np.random.normal(meanEta, stdEta, np.sum(left))
        x[right] += np.sqrt(2*Dr*dt)*np.random.normal(meanEta, stdEta, np.sum(right))
        y[right] += np.sqrt(2*Dr*dt)*np.random.normal(meanEta, stdEta, np.sum(right))
    else:
        x[fracture] += np.sqrt(2*Df*dt)*np.random.normal(meanEta, stdEta, np.sum(fracture))
        y[fracture] += np.sqrt(2*Df*dt)*np.random.normal(meanEta, stdEta, np.sum(fracture))
        x[matrix] += np.sqrt(2*Dm*dt)*np.random.normal(meanEta, stdEta, np.sum(matrix))
        y[matrix] += np.sqrt(2*Dm*dt)*np.random.normal(meanEta, stdEta, np.sum(matrix))
    return x, y

def apply_reflection(x, y, crossOutAbove, crossOutBelow, crossInAbove, crossInBelow, uby, lby, lbxOn):
    if lbxOn:
        if lbxAdsorption:
            x = np.where(crossOutLeft, lbx, x)
        else:
            x = np.where(x<lbx, -x+2*lbx, x)
    y[crossOutAbove] = np.where(probCrossOutAbove<probReflectedInward, -y[crossOutAbove]+2*uby, uby+(y[crossOutAbove]-uby)*(np.sqrt(Dm)/np.sqrt(Df)))
    y[crossOutBelow] = np.where(probCrossOutBelow<probReflectedInward, -y[crossOutBelow]+2*lby, lby+(y[crossOutBelow]-lby)*(np.sqrt(Dm)/np.sqrt(Df)))
    y[crossInAbove] = np.where(probCrossInAbove<probReflectedOutward, -y[crossInAbove]+2*uby, uby-(uby-y[crossInAbove])*(np.sqrt(Df)/np.sqrt(Dm)))
    y[crossInBelow] = np.where(probCrossInBelow<probReflectedOutward, -y[crossInBelow]+2*lby, lby-(lby-y[crossInBelow])*(np.sqrt(Df)/np.sqrt(Dm)))
    if matrixDiffVerification:
        x[crossOutLeft] = -x[crossOutLeft]+2*lbx
        x[crossOutRight] = -x[crossOutRight]+2*rbx
        x[crossLeftToRight] = np.where(probCrossLeftToRight<probReflectedLeft, -x[crossLeftToRight]+2*cbx, cbx+(x[crossLeftToRight]-cbx)*(np.sqrt(Dr)/np.sqrt(Dl)))
        x[crossRightToLeft] = np.where(probCrossRightToLeft<probReflectedRight, -x[crossRightToLeft]+2*cbx, cbx-(cbx-x[crossRightToLeft])*(np.sqrt(Dl)/np.sqrt(Dr)))
    return x, y

def apply_adsorption(x, y, crossOutAbove, crossOutBelow, crossOutLeft, adsDist, impacts):
    if lbxOn:
        x = np.where(crossOutLeft & (adsDist<=ap), lbx, x)
        x = np.where(crossOutLeft & (adsDist>ap), -x+2*lbx, x)
    y[crossOutAbove] = np.where(adsDist[crossOutAbove]<=ap, uby, -y[crossOutAbove]+2*uby)
    y[crossOutBelow] = np.where(adsDist[crossOutBelow]<=ap, lby, -y[crossOutBelow]+2*lby)
    impacts = impacts+np.count_nonzero(crossOutAbove | crossOutBelow)
    return x, y, impacts

def analytical_seminf(xInit, t, D):
    y = xInit*np.exp(-xInit**2/(4*D*t))/(np.sqrt(4*np.pi*D*t**3))
    return y

def analytical_inf(x, t, D):
    y = np.exp(-x**2/(4*D*t))/(np.sqrt(4*np.pi*D*t))
    return y

def degradation_dist(num_steps, kDecay, num_particles):
    t_steps = np.linspace(dt, sim_time, num_steps)
    exp_prob = kDecay*np.exp(-kDecay*t_steps)
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

survivalTimeDist = np.ones(num_particles)*sim_time
if domainDecay:
    survivalTimeDist, exp_prob = degradation_dist(num_steps, kDecay, num_particles)
if matrixDecay:
    survivalTimeDist, exp_prob = degradation_dist(num_steps, kDecay, num_particles)

# Time loop ###########################################################################
start_time = time.time() # Start timing the while loop

while t<sim_time and (not(numOfLivePart) or numOfLivePart[-1]>0) and bool(liveParticle.any()) and bool(((y!=lby) & (y!=uby)).any()):

    liveParticle = np.array(survivalTimeDist>t) # Particles which are not degradeted
    if matrixDecay:
        liveParticle = np.array(survivalTimeDist>timeInMatrix) # Radioactive decay in the porous matrix

    if stopOnCDF & (cdf>stopBTC/100):
        break

    isIn = abs(x)<vcp # Get the positions of the particles that are located within the control planes (wheter inside or outside the fracture)
    fracture = inside & liveParticle # Particles in the domain and inside the fracture and not degradeted yet
    outside = np.array(outsideAbove) | np.array(outsideBelow) # Particles outside the fracture
    matrix = outside & liveParticle # Particles in the domain and outside the fracture and not degradeted yet
    if lbxOn:
        fracture = fracture & np.logical_not(crossOutLeft)
        matrix = matrix & np.logical_not(crossOutLeft)
    # particleSteps[fracture | matrix] += 1 # It keeps track of the number of steps of each particle (no degradeted nor adsorbed are moving)
    # particleSteps[survivalTimeDist>t] = particleSteps[survivalTimeDist>t] + 1 # It keeps track of the number of steps of each particle
    if matrixDiffVerification:
        left = x<cbx
        right = x>cbx
    numOfLivePart.extend([fracture.sum()+matrix.sum()])
    Time.extend([t])

    # Update the position of all the particles at a given time steps according to the Langevin dynamics
    x, y = update_positions(x, y, fracture, matrix, Df, Dm, dt, meanEta, stdEta)

    # Particles which would cross the fractures' walls if no condition is applied at the boundaries
    crossOutAbove = fracture & (y>uby)
    crossOutBelow = fracture & (y<lby)
    crossInAbove = outsideAbove  & (y>lby) & (y<uby)
    crossInBelow = outsideBelow & (y>lby) & (y<uby)
    if matrixDiffVerification:
        crossOutLeft = x<lbx
        crossOutRight = x>rbx
        crossLeftToRight = left & (x>cbx)
        crossRightToLeft = right & (x<cbx)

    # Generate a vector populated with random variables from a uniform distribution with as many elements as the number of particles that would cross the boundaries
    probCrossOutAbove = np.random.rand(np.sum(crossOutAbove))
    probCrossOutBelow = np.random.rand(np.sum(crossOutBelow))
    probCrossInAbove = np.random.rand(np.sum(crossInAbove))
    probCrossInBelow = np.random.rand(np.sum(crossInBelow))
    if matrixDiffVerification:
        probCrossLeftToRight = np.random.rand(np.sum(crossLeftToRight))
        probCrossRightToLeft = np.random.rand(np.sum(crossRightToLeft))

    # Particles hitting the left control plane
    if lbxOn:
        crossOutLeft = (x<lbx)
        pdf_lbxOn[int(t/dt)] = sum(crossOutLeft)

    # Decide what happens to the particles which hit the fracture's walls: all get reflected, some get reflected some manage to escape, all get absorbed by the fracture's walls
    if diffuseIntoMatrix:
        # Update the reflected particles' positions according to an elastic reflection dynamic
        x, y = apply_reflection(x, y, crossOutAbove, crossOutBelow, crossInAbove, crossInBelow, uby, lby, lbxOn)
    if adsorptionProbability:
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

    pdf_part[int(t/dt)] = sum(abs(x[isIn])>vcp) # Count the particle which exit the control planes at each time step
    cdf = sum(pdf_part)/num_particles # Compute the CDF and increase the time

    timeInMatrix[matrix] += dt
    t += dt # Move forward time step

    # Record the spatial distribution of the particles at a given time, e.g.: 'recordSpatialConc'
    if (t <= recordSpatialConc) & (recordSpatialConc < t+dt):
        countsSpace, binEdgeSpace = np.histogram(x, xBins, density=True)

    # Store the positions of each particle for all the time steps 
    if recordTrajectories:
        xPath[:, int(t/dt)] = np.where(liveParticle, x, 0)  # Store x positions for the current time step
        yPath[:, int(t/dt)] = np.where(liveParticle, y, 0)  # Store y positions for the current time step

numOfLivePart = np.array(numOfLivePart)
Time = np.array(Time)

end_time = time.time() # Stop timing the while loop
execution_time = end_time - start_time

# Post processing ######################################################################
tau = (uby-lby)**2/Df

iPart = 0
iLbxOn = 0
# Retrieve the number of steps for each particle from the pdf of the breakthrough curve
for index, (valPdfPart, valLbxOn) in enumerate(zip(pdf_part, pdf_lbxOn)):
    particleRT.extend([index+1]*int(valPdfPart))
    particleSemiInfRT.extend([index+1]*int(valLbxOn))
    iPart = iPart+valPdfPart
    iLbxOn = iLbxOn+valLbxOn

particleRT = np.array(particleRT)
particleSemiInfRT = np.array(particleSemiInfRT)

# Compute simulation statistichs
meanTstep = np.array(particleRT).mean()
stdTstep = np.array(particleRT).std()
if (dt*10>(uby-lby)**2/Df):
    print("WARNING! Time step dt should be reduced to avoid jumps across the fracture width")
if (dt*10>(uby-lby)**2/Dm):
    print("WARNING! Time step dt should be reduced to avoid jumps across the fracture width")

# Verificaiton of the code
if lbxOn:
    countsSemiInfLog, binSemiInfLog = np.histogram(particleSemiInfRT, timeLogSpaced, density=True)
    timeBinsLog = (binSemiInfLog[:-1] + binSemiInfLog[1:]) / 2
    analPdfSemiInf = analytical_seminf(xInit, timeBinsLog, Df)
else:
    binCenterSpace = (xBins[:-1] + xBins[1:]) / 2
    yAnalytical = analytical_inf(binCenterSpace, recordSpatialConc, Df)

# Compute the number of particles at a given time over the whole domain extension
if adsorptionProbability:
    # Average concentration in X and Y
    recordTdist = int(timeStep[-2]) # Final time step
    vInterval = np.array([xInit-0.1, xInit+0.1])
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
if recordTrajectories and adsorptionProbability:
    # Average concentration in X and Y
    recordTdist = int(timeStep[-2])
    vInterval = np.array([xInit-0.1, xInit+0.1])
    hInterval = np.array([(lby+uby)/2-0.1, (lby+uby)/2+0.1])
    yDist = yPath[(xPath[:, recordTdist]>vInterval[0]) & (xPath[:, recordTdist]<vInterval[1]), recordTdist]
    yDist = yDist[(yDist != lby) & (yDist != uby)]
    vDist, vBins = np.histogram(yDist, np.linspace(lby, uby, binsSpace))
    xDist = xPath[(yPath[:, recordTdist]>hInterval[0]) & (yPath[:, recordTdist]<hInterval[1]), recordTdist]
    hDist, hBins = np.histogram(xDist, np.linspace(-binsXinterval, binsXinterval, binsSpace))

# Survival time distribution
# liveParticlesInTime = np.sum(particleSteps[:, None] > timeLinSpaced, axis=0)
# liveParticlesInTimeNorm = liveParticlesInTime/num_particles
# liveParticlesInTimePDF = liveParticlesInTime/sum(liveParticlesInTime*np.diff(np.insert(timeLinSpaced, 0, 0)))
# liveParticlesInLogTime = np.sum(particleSteps[:, None] > timeLogSpaced, axis=0)
# liveParticlesInLogTimeNorm = liveParticlesInLogTime/num_particles
# liveParticlesInLogTimePDF = liveParticlesInLogTime/sum(liveParticlesInLogTime*np.diff(np.insert(timeLogSpaced, 0, 0)))
# liveParticlesInTwoLogTime = np.sum(particleSteps[:, None] > timeTwoLogSpaced, axis=0)
# liveParticlesInTwoLogTime = liveParticlesInTwoLogTime/num_particles
# liveParticlesInTwoLogTimePDF = liveParticlesInTwoLogTime/sum(liveParticlesInTwoLogTime*np.diff(np.insert(timeTwoLogSpaced, 0, 0)))

# Statistichs ########################################################################
print(f"Execution time: {execution_time:.6f} seconds")
print(f"<t>: {meanTstep:.6f} s")
print(f"sigmat: {stdTstep:.6f} s")
if adsorptionProbability:
    print(f"# adsorbed particles/# impacts: {num_particles/impacts}")
if recordSpatialConc>t:
    print("WARNING! The simulation time is smaller than the specified time for recording the spatial distribution of the concentration")
elif lbxOn & (recordSpatialConc<t):
    print(f"sum(|empiricalPdf-analyticalPdf|)= {sum(np.abs(countsSemiInfLog-analPdfSemiInf))}")
else:
    print(f"sum(|yEmpirical-yAnalytical|)= {sum(np.abs(countsSpace-yAnalytical))}")

# Save and export variables to a .npz file #############################################
save = input("Do you want to save? Results may be overwritten. [Y][N]")
if save.upper()=="Y":
    variablesToSave = {name: value for name, value in globals().items() if isinstance(value, (np.ndarray, int, float, bool))} # Filter the variables we want to save by type
    # np.savez('infiniteDomain1e6.npz', **variablesToSave)
    # np.savez('semiInfiniteDomain1e3.npz', **variablesToSave)
    # np.savez('degradation_3.npz', **variablesToSave)
    # np.savez('totalAdsorption_3.npz', **variablesToSave)
    # np.savez('finalPositions1e5.npz', **variablesToSave)
    # np.savez('testSemra.npz', **variablesToSave)
    # np.savez('matrixDiffusionVerification.npz', **variablesToSave)
    # np.savez('partialAdsorption.npz', **variablesToSave)
    # if matrixDiffVerification:
        # np.savez('Dl01Dr01Rl0Rr0.npz', **variablesToSave)
        # np.savez('Dl01Dr001Rl0Rr0.npz', **variablesToSave)
        # np.savez('Dl01Dr01RlPlRrPr.npz', **variablesToSave)
        # np.savez('Dl01Dr001RlPlRrPr.npz', **variablesToSave)
    # np.savez('compareAdsD1.npz', **variablesToSave)
    # np.savez('compareAdsD01.npz', **variablesToSave)
    # np.savez('compareAdsD001.npz', **variablesToSave)
    # np.savez('compareAdsD0001.npz', **variablesToSave)
    # np.savez('compareAp2.npz', **variablesToSave)
    # np.savez('compareAp4.npz', **variablesToSave)
    # np.savez('compareAp6.npz', **variablesToSave)
    # np.savez('compareAdsP90.npz', **variablesToSave)
    # np.savez('compareAdsP80.npz', **variablesToSave)
    # np.savez('compareAdsP60.npz', **variablesToSave)
    # np.savez('compareAdsP40.npz', **variablesToSave)
    # np.savez('compareAdsP20.npz', **variablesToSave)
    # np.savez('compareAdsP10.npz', **variablesToSave)
    # np.savez('compareTau4.npz', **variablesToSave)
    # np.savez('compareTau40.npz', **variablesToSave)
    # np.savez('compareTau100.npz', **variablesToSave)
    # np.savez('compareTau400.npz', **variablesToSave)
    np.savez('compareTau1000.npz', **variablesToSave)
    # np.savez('compareTau4000.npz', **variablesToSave)
    # np.savez('compareP80.npz', **variablesToSave)
    # np.savez('compareP60.npz', **variablesToSave)
    # np.savez('compareP40.npz', **variablesToSave)
    # np.savez('matrixDecayK01.npz', **variablesToSave)
    # np.savez('matrixDecayK001.npz', **variablesToSave)
    # np.savez('matrixDecayK004.npz', **variablesToSave)
    # np.savez('matrixDecayK007.npz', **variablesToSave)
    # np.savez('matrixDecayK0001.npz', **variablesToSave)
    # np.savez('domainDecayK01.npz', **variablesToSave)
    # np.savez('domainDecayK001.npz', **variablesToSave)
    # np.savez('domainDecayK0001.npz', **variablesToSave)
    # np.savez('Dm1e-2matrixK1e-2.npz', **variablesToSave)
    # np.savez('Dm5e-4matrixK1e-2.npz', **variablesToSave)
    # np.savez('Dm1e-4matrixK1e-2.npz', **variablesToSave)
    # np.savez('Dm1e-5matrixK1e-2.npz', **variablesToSave)
    if variablesToSave:
        print("\n RESULTS SAVED")
else:
    print("\n RESULTS NOT SAVED.")