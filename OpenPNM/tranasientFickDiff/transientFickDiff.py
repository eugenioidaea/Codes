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

# Sim inputs ###################################################################
shape = [20, 10, 1]
spacing = 1e-3 # It is the distance between pores that it does not necessarily correspond to the length of the throats because of the tortuosity
# throatDiameter = spacing/10
poreDiameter = spacing/10
Dmol = 1e-6 # Molecular Diffusion
Cin = 5
endSim = ((shape[0]-1)*spacing)**2/Dmol
simTime = (0, endSim) # Simulation starting and ending times

# Initialisation ################################################################
net = op.network.Cubic(shape=shape, spacing=spacing) # Shape of the elementary cell of the network: cubic
# geo = op.models.collections.geometry.spheres_and_cylinders # Shape of the pore and throats
# net.add_model_collection(geo, domain='all') # Assign the shape of pores and throats to the network
net.regenerate_models() # Compute geometric properties such as pore volume

net['throat.length'] = spacing
net['pore.diameter'] = poreDiameter
net['pore.volume'] = 4/3*np.pi*poreDiameter**3/8

Adomain = (shape[1] * shape[2])*(spacing**2)
Ldomain = (shape[1]-1)*spacing

# print(net)

liquid = op.phase.Phase(network=net) # Phase dictionary initialisation

# Conductance
s = 0.5 # Variance of the conductance
throatDiameter = np.ones(net.Nt)*poreDiameter/2
# throatDiameter = spst.lognorm.rvs(s, loc=0, scale=poreDiameter/2, size=net.Nt) # Diameter lognormal distribution
net['throat.diameter'] = throatDiameter
Athroat = throatDiameter**2*np.pi/4
diffCond = Dmol*Athroat/spacing
liquid['throat.diffusive_conductance'] = diffCond
net['throat.volume'] = Athroat*spacing

tfd = op.algorithms.TransientFickianDiffusion(network=net, phase=liquid) # TransientFickianDiffusion dictionary initialisation

inlet = net.pores(['left'])
outlet = net.pores(['right'])

# Boundary conditions
tfd.set_value_BC(pores=inlet, values=Cin) # Inlet: fixed concentration
tfd.set_rate_BC(pores=outlet, rates=0) # Outlet: fixed rate

# Initial conditions
ic = np.concatenate((np.ones(shape[1])*Cin, np.zeros((shape[0]-1)*shape[1]))) # Initial Concentration: shape[1] represents the first column of pores and (shape[0]-1)*shape[1] are all the rest of the pores in the domain

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
cAvg = []
for ti in times:
    c_front = tfd.soln['pore.concentration'](ti)[outlet]
    q_front = tfd.rate(throats=net.Ts, mode='single')[outlet]
    cAvg.append((q_front*c_front).sum() / q_front.sum())
cAvg = np.array(cAvg)

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
    # C = -(np.sqrt(D)*t**(3/2)*spsp.erf(Ldomain/(2*np.sqrt(D*t))))/(np.sqrt(D*t**3))
    C = 1-spsp.erf(Ldomain/(2*np.sqrt(D*t)))
    return C

# Error function to be minismied
def errFunc(D, tNorm, cAvgNorm):
    Cpred = cdfBTC(tNorm, D)
    Cpred = minMaxNorm(Cpred)
    return np.sum((cAvgNorm-Cpred)**2)

# Synthetic data for testing
# Dtest = 1e-4
# cAvg = cdfBTC(times, Dtest)

tNorm = minMaxNorm(times)
cAvgNorm = minMaxNorm(cAvg)

# Initial guess
D0 = 1e-5
C0 = cdfBTC(tNorm, D0)
C0norm = minMaxNorm(C0)

# OPTIMISATION
bounds = [(1e-7, 1)]  # Example bound: D should be between 1e-6 and 10
fitting = opt.minimize(errFunc, D0, args=(tNorm, cAvgNorm), bounds=bounds, method='Nelder-Mead')
# fitting = opt.minimize(errFunc, D0, args=(tNorm, cAvgNorm), bounds=bounds, method='Powell')
# fitting = opt.minimize(errFunc, D0, args=(tNorm, cAvgNorm), bounds=bounds, method='CG')
# fitting = opt.minimize(errFunc, D0, args=(tNorm, cAvgNorm), bounds=bounds, method='BFGS')
# fitting = opt.minimize(errFunc, D0, args=(tNorm, cAvgNorm), bounds=bounds, method='L-BFGS-B')
# fitting = opt.minimize(errFunc, D0, args=(tNorm, cAvgNorm), bounds=bounds, method='TNC')
# fitting = opt.minimize(errFunc, D0, args=(tNorm, cAvgNorm), bounds=bounds, method='COBYLA')
# fitting = opt.minimize(errFunc, D0, args=(tNorm, cAvgNorm), bounds=bounds, method='SLSQP')
# fitting = opt.minimize(errFunc, D0, args=(tNorm, cAvgNorm), bounds=bounds, method='trust-constr')
DeffBTC = fitting.x[0]  # Fitted parameter

Copt = cdfBTC(tNorm, DeffBTC)
CoptNorm = minMaxNorm(Copt)

# CURVE FITTING
DeffFIT, covariance = opt.curve_fit(cdfBTC, tNorm, cAvgNorm, p0=[D0], bounds=bounds[0])
DeffFIT = DeffFIT[0]
Cfit = cdfBTC(tNorm, DeffFIT)
CfitNorm = minMaxNorm(Cfit)

# LEAST SQUARE
cdfModel = Model(cdfBTC)
params = cdfModel.make_params(D=D0)
params['D'].set(min=bounds[0][0], max=bounds[0][1])
result = cdfModel.fit(cAvgNorm, params, t=tNorm)
print(result.fit_report())
DeffLSQ = result.params['D'].value
Clsq = cdfBTC(tNorm, DeffLSQ)
ClsqNorm = minMaxNorm(Clsq)

# Metrics ##########################################################
print(f'Number of nodes: {shape}')
print(f'Variance of the underlying diameter dist: {s:.2e}')
print(f'Average outlet final conc: {np.mean(cAvg):.5e}')
print(f"Molecular diff Dmol: ", "{0:.6E}".format(Dmol))
print(f"Initial guess Deff0: ", "{0:.6E}".format(D0))
print(f"BTC optimised Deff: ", "{0:.6E}".format(DeffBTC))
print(f"BTC fitted Deff: ", "{0:.6E}".format(DeffFIT))
print(f"BTC least square Deff = {DeffLSQ:.4f}")

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
plt.plot(tNorm, cAvgNorm, 'o', label='CopenPNM')
plt.plot(tNorm, C0norm, '--', label='C0norm')
plt.plot(tNorm, CoptNorm, label='CoptNorm')
plt.plot(tNorm, CfitNorm, label='CfitNorm')
# plt.plot(tNorm, ClsqNorm, label='ClsqNorm')
plt.legend(loc='best')

BTCs = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
plt.plot(times, cAvg, label='CopenPNM')
plt.plot(times, C0, label='C0norm')
plt.plot(times, Copt, label='CoptNorm')
plt.plot(times, Cfit, label='CoptFit')
plt.plot(times, Clsq, label='ClsqNorm')
plt.legend(loc='best')

pc = tfd.soln['pore.concentration'](0.5*endSim)
# tc = tfd.interpolate_data(propname='throat.concentration')
# tc = tfd.soln['pore.concentration'](1)[throat.all]
d = net['pore.diameter']
fig, ax = plt.subplots(figsize=[5, 5])
op.visualization.plot_coordinates(network=net, color_by=pc, size_by=d, markersize=400, ax=ax)
# op.visualization.plot_connections(network=net, color_by=tc, linewidth=3, ax=ax)
_ = plt.axis('off')