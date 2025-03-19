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
shape = [20, 4, 4]
spacing = 1e-3 # It is the distance between pores that it does not necessarily correspond to the length of the throats because of the tortuosity
poreDiameter = spacing/10
Adomain = (shape[1] * shape[2])*(spacing**2) # Should the diameters of the pores be considered?
Ldomain = (shape[0]-1)*spacing+shape[0]*poreDiameter
Dmol = 1e-6 # Molecular Diffusion

cs = 0.5 # BTC relative control section location (0 is beginning and 1 is the end)

# Boundary & Initial conditions ################################################
Cout = 0
Cin = (Ldomain-Cout*cs*Ldomain)/(Ldomain-cs*Ldomain)
Qin = 0
Qout = 0
endSim = ((shape[0]-1)*spacing)**2/Dmol
simTime = (0, endSim) # Simulation starting and ending times

D0 = 1e-7 # Initial guess for diffusion coefficient during optimisation

concTimePlot = 1 # Plot the spatial map of the concentration between start (0) or end (1) of the simulation

# Initialisation ################################################################
net = op.network.Cubic(shape=shape, spacing=spacing) # Shape of the elementary cell of the network: cubic
# geo = op.models.collections.geometry.spheres_and_cylinders # Shape of the pore and throats
# net.add_model_collection(geo, domain='all') # Assign the shape of pores and throats to the network
net.regenerate_models() # Compute geometric properties such as pore volume

net['throat.length'] = spacing
net['pore.diameter'] = poreDiameter
net['pore.volume'] = 4/3*np.pi*poreDiameter**3/8

# print(net)

liquid = op.phase.Phase(network=net) # Phase dictionary initialisation

# Conductance
s = 0.8 # Variance of the throat diameters

# OPTION 1: CONSTANT DIAMETER
# throatDiameter = np.ones(net.Nt)*poreDiameter/2

# OPTION 2: LOGNORMAL DIST DIAMETERS WITH FIXED SEED
# np.random.seed(42)
# throatDiameter = np.random.lognormal(mean=np.log(poreDiameter/2), sigma=s, size=net.Nt)

# OPTION 3: LOGNORMAL DIST DIAMETERS WITH RANDOM SEED
throatDiameter = spst.lognorm.rvs(s, loc=0, scale=poreDiameter/2, size=net.Nt)

net['throat.diameter'] = throatDiameter
Athroat = throatDiameter**2*np.pi/4
diffCond = Dmol*Athroat/spacing
liquid['throat.diffusive_conductance'] = diffCond
net['throat.volume'] = Athroat*spacing

tfd = op.algorithms.TransientFickianDiffusion(network=net, phase=liquid) # TransientFickianDiffusion dictionary initialisation

inlet = net.pores(['left'])
outlet = net.pores(['right'])
csBtc = np.arange(int(np.floor((shape[0]*shape[1])*cs-shape[1])), int(np.ceil(shape[0]*shape[1]*cs)), 1) # Nodes for recording the BTC at Control Section cs
# csBtc = np.arange(int(shape[0]*shape[1]*cs), int(shape[0]*shape[1]*cs+shape[1]), 1) # Nodes for recording the BTC at Control Section cs

# Boundary conditions
tfd.set_value_BC(pores=inlet, values=Cin) # Inlet: fixed concentration
tfd.set_value_BC(pores=outlet, values=Cout) # Inlet: fixed concentration
# tfd.set_rate_BC(pores=inlet, rates=Qin) # Outlet: fixed rate
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

# Get the flux-averaged concentration at the outlet for every time step
cAvg = np.array([])
for ti in times:
    c_front = tfd.soln['pore.concentration'](ti)[csBtc] # [outlet]
    q_front = tfd.rate(throats=net.Ts, mode='single')[csBtc]*Athroat[csBtc] # [outlet]
    cAvg = np.append(cAvg, (q_front*c_front).sum() / q_front.sum())
    # cAvg = np.append(cAvg, c_front.sum())
# btcScalefactor = max(tfd.soln['pore.concentration'](endSim)[csBtc]) # NORMALISATION FACTOR ???
# cAvg = cAvg / btcScalefactor

# METRICS FOR STEADY STATE #####################################################################
# rate_inlet = -tfd.rate(pores=outlet)[0] # Fluxes leaving the pores are negative
# print(f'Flow rate: {rate_inlet:.5e} m3/s')

# KdOpenPNM = rate_inlet/(Cin-cAvg[-1])
# print(f'Diffusive conductance from OpenPNM (Qd/deltaC)', "{0:.6E}".format(KdOpenPNM))
# KdGmean = spst.gmean(diffCond)
# print(f'The geometric mean of the diffusive conductances is Kd =', "{0:.6E}".format(KdGmean))

# DeffQ = rate_inlet * Ldomain / (Adomain * (Cin - cAvg[-1]))
# print(f'Effective diffusivity DeffQ [m2/s]', "{0:.6E}".format(DeffQ))

# V_p = net['pore.volume'].sum()
# V_t = net['throat.volume'].sum()
# V_bulk = np.prod(shape)*(spacing**3)
# e = (V_p + V_t) / V_bulk
# print('The porosity is: ', "{0:.6E}".format(e))
# tau = e * Dmol / DeffQ
# print('The tortuosity is:', "{0:.6E}".format(tau))
#################################################################################################
# Normalisation
def minMaxNorm(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
    # return data / np.max(data)

# Analytical solution for semi-infinite domain and continuous injection
def cdfBTC(t, D):
    C = (1-spsp.erf(Ldomain*cs/(2*np.sqrt(D*t)))) # / btcScalefactor
    return C

# Error function to be minismied
def errFunc(D, times, cAvg):
    Cpred = cdfBTC(times, D)
    return np.sum((cAvg-Cpred)**2)

# Synthetic data for testing
# Dtest = 1e-4
# cAvg = cdfBTC(times, Dtest)

# times = minMaxNorm(times)
# cAvgNorm = minMaxNorm(cAvg)

# Initial guess
C0 = cdfBTC(times, D0)
err0 = np.sum((cAvg-C0)**2)
# Mminimisation constraints
bounds = [(1e-10, 1e-1)]  # Example bound: D should be between 1e-6 and 10

# OPTIMISATION
fitting = opt.minimize(errFunc, D0, args=(times, cAvg), bounds=bounds, method='Nelder-Mead')
# fitting = opt.minimize(errFunc, D0, args=(times, cAvgNorm), bounds=bounds, method='Powell')
# fitting = opt.minimize(errFunc, D0, args=(times, cAvgNorm), bounds=bounds, method='CG')
# fitting = opt.minimize(errFunc, D0, args=(times, cAvgNorm), bounds=bounds, method='BFGS')
# fitting = opt.minimize(errFunc, D0, args=(times, cAvgNorm), bounds=bounds, method='L-BFGS-B')
# fitting = opt.minimize(errFunc, D0, args=(times, cAvgNorm), bounds=bounds, method='TNC')
# fitting = opt.minimize(errFunc, D0, args=(times, cAvgNorm), bounds=bounds, method='COBYLA')
# fitting = opt.minimize(errFunc, D0, args=(times, cAvgNorm), bounds=bounds, method='SLSQP')
# fitting = opt.minimize(errFunc, D0, args=(times, cAvgNorm), bounds=bounds, method='trust-constr')
DeffOPT = fitting.x[0]  # Fitted parameter
Copt = cdfBTC(times, DeffOPT)
errOpt=np.sum((cAvg-Copt)**2)

# CURVE FITTING
DeffFIT, covariance = opt.curve_fit(cdfBTC, times, cAvg, p0=[D0], bounds=bounds[0])
DeffFIT = DeffFIT[0]
Cfit = cdfBTC(times, DeffFIT)
errFit=np.sum((cAvg-Cfit)**2)

# LEAST SQUARE
cdfModel = Model(cdfBTC)
params = cdfModel.make_params(D=D0)
params['D'].set(min=bounds[0][0], max=bounds[0][1])
# result = cdfModel.fit(cAvgNorm, params, t=times, method='leastsq')
# result = cdfModel.fit(cAvgNorm, params, t=times, method='least_squares')
# result = cdfModel.fit(cAvgNorm, params, t=times, method='differential_evolution')
result = cdfModel.fit(cAvg, params, t=times, method='basinhopping')
print(result.fit_report())
DeffLSQ = result.params['D'].value
Clsq = cdfBTC(times, DeffLSQ)
errLsq=np.sum((cAvg-Clsq)**2)

# Metrics ##########################################################
print(f'Number of nodes: {shape}')
print(f'Variance of the underlying diameter dist: {s:.2e}')
print(f'Average outlet final conc: {np.mean(cAvg):.5e}')
print(f"Molecular diff Dmol: ", "{0:.6E}".format(Dmol))
print(f"BTC initial error: ", "{0:.6E}".format(err0))
print(f"BTC optimised error: ", "{0:.6E}".format(errOpt))
print(f"BTC fitted error: ", "{0:.6E}".format(errFit))
print(f"BTC least square error = {errLsq:.4f}")

# Cumulative Inverse Gaussian ######################################

# Analytical solution for semi-infinite domain and continuous injection
def cdfINVGAU(t, mu, lam):
    cInvGau = spst.norm.cdf(np.sqrt(lam/t)*(t/mu-1))+np.exp(2*lam/mu)*spst.norm.cdf(-np.sqrt(lam/t)*(t/mu+1))
    return cInvGau

# Error function to be minismied
def errInvGau(params, times, cAvg):
    mu, lam = params
    Cpred = cdfINVGAU(times, mu, lam)
    return np.sum((cAvg-Cpred)**2)

# Initial guess
mulam0 = [1, 1]

# Minimisation
minInvGau = opt.minimize(errInvGau, mulam0, args=(times, cAvg)) #, bounds=bounds)

mu, lam = minInvGau.x
Cinvgau = cdfINVGAU(times, mu, lam)

varBtc = mu**3/lam

# Plot #############################################################
# networkLabels = plt.figure(figsize=(8, 8))
# networkLabels = op.visualization.plot_tutorial(net, font_size=6)

poreNetwork = plt.figure(figsize=(8, 8))
poreNetwork = op.visualization.plot_coordinates(net)
poreNetwork = op.visualization.plot_connections(net, size_by=liquid['throat.diffusive_conductance'], ax=poreNetwork)

lognormDist = plt.figure(figsize=(8, 8))
if 'diffCond' in globals():
    plt.hist(diffCond, edgecolor='k')
    plt.title('Lognormal distribution of diffusive conductances')
    plt.xlabel(r'$k_D [m^3/s]$')
    plt.ylabel(r'$Number of throats [-]$')

breakthrough = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
plt.plot(times, cAvg)
# plt.plot(times, Copt)
plt.title("Conc in time at the outlet")
plt.xlabel(r'$Time [s]$')
plt.ylabel(r'$Concentration [-]$')
# plt.xscale('log')
# plt.yscale('log')
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='black')
plt.grid(True, which="minor", linestyle=':', linewidth=0.5, color='gray')
plt.legend(loc='best')
plt.tight_layout()

btcTail = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
plt.plot(times, Cin-np.array(cAvg))
plt.title("Conc in time at the outlet")
plt.xlabel(r'$Time [s]$')
plt.ylabel(r'$C_{in}-Concentration [-]$')
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='black')
plt.grid(True, which="minor", linestyle=':', linewidth=0.5, color='gray')
plt.legend(loc='best')
plt.tight_layout()

btcInterp = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
plt.plot(times, Cin-cAvg, linewidth='5')

tReshaped = (times[-100:-1]).reshape(-1, 1)
linRegCavg = LinearRegression().fit(tReshaped, np.log(Cin-cAvg[-100:-1]))
interpCavg = np.exp(linRegCavg.intercept_+linRegCavg.coef_*times)
plt.plot(times, interpCavg, color='black', linewidth='2')
plt.text(times[int(len(times)/2)], interpCavg[int(len(interpCavg)/2)], r"$C_{in}-Conc = e^{" + f"{linRegCavg.intercept_:.5f} {linRegCavg.coef_[0]:.5f} * t" + "}$", fontsize=18, ha='left', va='bottom')

plt.title("Concentration in time at the outlet")
plt.xlabel(r'$Time [s]$')
plt.ylabel(r'$C_{in}-Concentration [-]$')
# plt.xscale('log')
plt.yscale('log')
# plt.ylim([0.1, max(cAvg)])
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='black')
plt.grid(True, which="minor", linestyle=':', linewidth=0.5, color='gray')
plt.legend(loc='best')
plt.tight_layout()

NormBTC = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
interval=int(endSim//10)
plt.plot(times[::interval], cAvg[::interval], '*-', markerfacecolor='none', label=r"$D_{mol}=$" + f"{Dmol:.4E}")
plt.plot(times[::interval], C0[::interval], 'o-', markerfacecolor='none', label=r"$D_0=$" + f"{D0:.4E}")
plt.plot(times[::interval], Copt[::interval], 's-', markerfacecolor='none', label=r"$D_{opt}=$" + f"{DeffOPT:.4E}")
plt.plot(times[::interval], Cfit[::interval], 'd-', markerfacecolor='none', label=r"$D_{fit}=$" + f"{DeffFIT:.4E}")
plt.plot(times[::interval], Clsq[::interval], 'p-', markerfacecolor='none', label=r"$D_{lsq}=$" + f"{DeffLSQ:.4E}")
# plt.plot(times[::interval], Cinvgau[::interval], '^-', markerfacecolor='none', label=f"mu={mu:.4f}, lam={lam:.4f}")
plt.title('Breakthrough curves')
plt.xlabel('time [s]')
plt.ylabel('concentration [-]')
plt.legend(loc='best')

BTCs = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
plt.plot(times, cAvg, label='CopenPNM')
# plt.plot(times, C0, label='C0norm')
# plt.plot(times, Copt, label='CoptNorm')
# plt.plot(times, Cfit, label='CoptFit')
# plt.plot(times, Clsq, label='ClsqNorm')
# plt.legend(loc='best')

residuals = plt.figure(figsize=(8, 8))
plt.plot(times, result.residual, 'o')  # Check if residuals are randomly distributed
plt.title('Residuals of least square fitting')
plt.xlabel('t norm [-]')
plt.ylabel('lsq residual value [m2/s]')
plt.show()

pc = tfd.soln['pore.concentration'](endSim*concTimePlot)
# tc = tfd.interpolate_data(propname='throat.concentration')
# tc = tfd.soln['pore.concentration'](1)[throat.all]
d = net['pore.diameter']
ms = 100 # Markersize
if shape[2]==1:
    fig, ax = plt.subplots(figsize=[8, 8])
    op.visualization.plot_coordinates(network=net, color_by=pc, size_by=d, markersize=ms, ax=ax)
    # op.visualization.plot_connections(network=net, color_by=tc, linewidth=3, ax=ax)
    ax.plot([(csBtc[0]/shape[1]+0.5)*spacing, (csBtc[0]/shape[1]+0.5)*spacing], [-shape[1]*spacing*ms/100, shape[1]*spacing*ms/100], linewidth=3)
    ax.text((csBtc[0]/shape[1]+0.5)*spacing, shape[1]*spacing*ms/100, "Control section 1", ha='right', va='bottom')
else:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    op.visualization.plot_coordinates(network=net, color_by=pc, size_by=d, markersize=ms, ax=ax)
    offset = Ldomain*0.1
    ycs = np.linspace(0-offset, shape[1]*spacing+offset, 10)
    zcs = np.linspace(0-offset, shape[2]*spacing+offset, 10)
    Ycs, Zcs = np.meshgrid(ycs, zcs)
    Xcs = np.ones(len(ycs))*cs*Ldomain
    ax.plot_surface(Xcs, Ycs, Zcs)
    ax.text(Xcs[-1], Ycs[-1][-1], Zcs[-1][-1], "Control plane 1")
#_ = plt.axis('off')