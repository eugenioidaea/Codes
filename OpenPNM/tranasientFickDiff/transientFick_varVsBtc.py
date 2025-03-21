import openpnm as op
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as spst
op.visualization.set_mpl_style()
np.set_printoptions(precision=5)
from sklearn.linear_model import LinearRegression
import scipy.optimize as opt
import scipy.special as spsp
import time
from lmfit import Model
from mpl_toolkits.mplot3d import Axes3D

# Sim inputs ###################################################################
numSim = 3
shape = [10, 3, 3]
spacing = 1e-3 # It is the distance between pores that it does not necessarily correspond to the length of the throats because of the tortuosity
poreDiameter = spacing/10
Adomain = (shape[1] * shape[2])*(spacing**2) # Should the diameters of the pores be considered?
Ldomain = (shape[0]-1)*spacing+shape[0]*poreDiameter
Dmol = 1e-5 # Molecular Diffusion

cs = 0.9 # BTC relative control section location (0 is beginning and 1 is the end)

s = np.linspace(0.4, 1.2, numSim) # Conductance: variance of the diameters of the throats

# Boundary & Initial conditions ################################################
Cout = 0
Cin = 1
# Cin = (Ldomain-Cout*cs*Ldomain)/(Ldomain-cs*Ldomain) # This condition forces the max conc at the control section to be 1
Qin = 0
Qout = 0
endSim = ((shape[0]-1)*spacing)**2/Dmol
simTime = (0, endSim) # Simulation starting and ending times

concTimePlot = 1 # Plot the spatial map of the concentration between start (0) or end (1) of the simulation

# Initialisation ################################################################
net = op.network.Cubic(shape=shape, spacing=spacing) # Shape of the elementary cell of the network: cubic
# geo = op.models.collections.geometry.spheres_and_cylinders # Shape of the pore and throats
# net.add_model_collection(geo, domain='all') # Assign the shape of pores and throats to the network
net.regenerate_models() # Compute geometric properties such as pore volume

net['throat.length'] = spacing
net['pore.diameter'] = poreDiameter
net['pore.volume'] = 4/3*np.pi*poreDiameter**3/8

cAvg = []
tAvg = []

# print(net)

liquid = op.phase.Phase(network=net) # Phase dictionary initialisation

for i in range(numSim):
    # OPTION 1: CONSTANT DIAMETER
    # throatDiameter = np.ones(net.Nt)*poreDiameter/2

    # OPTION 2: LOGNORMAL DIST DIAMETERS WITH FIXED SEED
    # np.random.seed(42)
    # throatDiameter = np.random.lognormal(mean=np.log(poreDiameter/2), sigma=s, size=net.Nt)

    # OPTION 3: LOGNORMAL DIST DIAMETERS WITH RANDOM SEED
    throatDiameter = spst.lognorm.rvs(s[i], loc=0, scale=poreDiameter/2, size=net.Nt)

    net['throat.diameter'] = throatDiameter
    Athroat = throatDiameter**2*np.pi/4
    diffCond = Dmol*Athroat/spacing # Dmol[m2/s]*Athroat[m2]/spacing[m] = diffCond[m3/s]
    liquid['throat.diffusive_conductance'] = diffCond
    net['throat.volume'] = Athroat*spacing

    tfd = op.algorithms.TransientFickianDiffusion(network=net, phase=liquid) # TransientFickianDiffusion dictionary initialisation

    inlet = net.pores(['left'])
    outlet = net.pores(['right'])
    csBtc = np.arange(int(np.floor((shape[0]*shape[1]*shape[2])*cs-shape[1]*shape[2])), int(np.ceil(shape[0]*shape[1]*shape[2]*cs)), 1) # Nodes for recording the BTC at Control Section cs
    # csBtc = np.arange(int(shape[0]*shape[1]*cs), int(shape[0]*shape[1]*cs+shape[1]), 1) # Nodes for recording the BTC at Control Section cs

    # Boundary conditions
    tfd.set_value_BC(pores=inlet, values=Cin) # Inlet: fixed concentration
    tfd.set_value_BC(pores=outlet, values=Cout) # Outlet: fixed concentration
    # tfd.set_rate_BC(pores=inlet, rates=Qin) # Inlet: fixed rate
    # tfd.set_rate_BC(pores=outlet, rates=Qout) # Outlet: fixed rate

    # Initial conditions
    ic = np.concatenate((np.ones(shape[1]*shape[2])*Cin, np.ones((shape[0]-1)*shape[1]*shape[2])*Cout)) # Initial Concentration: shape[1] represents the first column of pores and (shape[0]-1)*shape[1] are all the rest of the pores in the domain

    # Algorithm settings
    # tfd.setup(t_scheme='cranknicolson', t_final=100, t_output=5, t_step=1, t_tolerance=1e-12)
    print(tfd.settings)

    start_time = time.time()

    # Run the simulation
    # solSetting = op.integrators.ScipyRK45(atol=1e-06, rtol=1e-06, verbose=False, linsolver=None)
    # tfd.run(x0=ic, tspan=simTime, saveat=endSim/2, integrator=solSetting)
    tfd.run(x0=ic, tspan=simTime)

    end_time = time.time()
    elapsed_time = end_time - start_time # Compute elapsed time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

    # liquid.update(tfd.soln)
    times = tfd.soln['pore.concentration'].t # Store the time steps

    cAvg1sim = np.empty(len(times))
    # Get the flux-averaged concentration at the outlet for every time step
    q_front = diffCond[csBtc]
    # q_front = tfd.rate(throats=csBtc, mode='single')*Athroat[csBtc] # [outlet]
    # pore_throats = net.find_neighbor_throats(pores=[csBtc]) # Find the indexes of the throats connected to the pores
    for j, ti in enumerate(times):
        c_front = tfd.soln['pore.concentration'](ti)[csBtc] # [outlet]
        cAvg1sim[int(j)] = (q_front*c_front).sum() / q_front.sum()
        # cAvg = np.append(cAvg, c_front.sum())
    cAvg.append(cAvg1sim)
    tAvg.append(times)
    # btcScalefactor = max(tfd.soln['pore.concentration'](endSim)[csBtc]) # NORMALISATION FACTOR ???
    # cAvg = cAvg / btcScalefactor

BtcVsVar = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
interval = 10 # Print solution every N times
for i in range(numSim):
    plt.plot(tAvg[i], cAvg[i], '*-', markerfacecolor='none', label=f"s = {s[i]:.2f}")
    # plt.plot(tAvg[i][::interval], cAvg[i][::interval], '*-', markerfacecolor='none', label=f"s = {s[i]:.2f}")
plt.title('Breakthrough curves')
plt.xlabel('time [s]')
plt.ylabel('concentration [-]')
plt.xscale('log')
# plt.yscale('log')
plt.legend(loc='best')


highlightCoords = net['pore.coords'][csBtc]
pc = tfd.soln['pore.concentration'](endSim*concTimePlot)
# tc = tfd.interpolate_data(propname='throat.concentration')
# tc = tfd.soln['pore.concentration'](1)[throat.all]
d = net['pore.diameter']
ms = 50 # Markersize
if shape[2]==1:
    fig, ax = plt.subplots(figsize=[8, 8])
    op.visualization.plot_coordinates(network=net, color_by=pc, size_by=d, markersize=ms, ax=ax)
    op.visualization.plot_connections(network=net, size_by=throatDiameter, linewidth=3, ax=ax)
    ax.plot([(csBtc[0]/shape[1]+0.5)*spacing, (csBtc[0]/shape[1]+0.5)*spacing], [-shape[1]*spacing*ms/100, shape[1]*spacing*ms/100], linewidth=3)
    ax.text((csBtc[0]/shape[1]+0.5)*spacing, shape[1]*spacing*ms/100, "Control section 1", ha='right', va='bottom')
else:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    op.visualization.plot_coordinates(network=net, color_by=pc, size_by=d, markersize=10, ax=ax)
    op.visualization.plot_connections(network=net, size_by=throatDiameter, linewidth=3, ax=ax)
    ax.scatter(highlightCoords[:, 0], highlightCoords[:, 1], highlightCoords[:, 2], color='black', s=ms, label="Highlighted Pores")
    ycs = np.linspace(0, shape[1]*spacing, 10)
    zcs = np.linspace(0, shape[2]*spacing, 10)
    Ycs, Zcs = np.meshgrid(ycs, zcs)
    Xcs = np.ones(len(ycs))*highlightCoords[0, 0]
    ax.plot_surface(Xcs, Ycs, Zcs)
    ax.text(Xcs[-1], Ycs[-1][-1], Zcs[-1][-1], "Control plane 1")
    # ax.set_ylim(-0.01, 0.06)
    # ax.set_zlim(-0.01, 0.06)
    plt.tight_layout()