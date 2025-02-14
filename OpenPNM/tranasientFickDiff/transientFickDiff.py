import openpnm as op
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as spst
op.visualization.set_mpl_style()
np.set_printoptions(precision=5)
from sklearn.linear_model import LinearRegression
import scipy.optimize as opt
import scipy.special as spsp

spacing = 1e-3 # It is the distance between pores that it does not necessarily correspond to the length of the throats because of the tortuosity
# throatDiameter = spacing/10
poreDiameter = spacing/10
Dmol = 1e-3 # Molecular Diffusion

# Pore network #####################################################
shape = [10, 10, 1]
net = op.network.Cubic(shape=shape, spacing=spacing) # Shape of the elementary cell of the network: cubic
# geo = op.models.collections.geometry.spheres_and_cylinders # Shape of the pore and throats
# net.add_model_collection(geo, domain='all') # Assign the shape of pores and throats to the network
net.regenerate_models() # Compute geometric properties such as pore volume

net['throat.length'] = spacing
net['pore.diameter'] = poreDiameter
net['pore.volume'] = 4/3*np.pi*poreDiameter**3/8

# print(net)

liquid = op.phase.Phase(network=net)

# Lognormal diffusive conductance ###############################################
throatDiameter = spst.lognorm.rvs(0.5, loc=0, scale=poreDiameter/2, size=net.Nt) # Conductance lognormal distribution
net['throat.diameter'] = throatDiameter
Athroat = throatDiameter**2*np.pi/4
diffCond = Dmol*Athroat/spacing
liquid['throat.diffusive_conductance'] = diffCond
net['throat.volume'] = Athroat*spacing

tfd = op.algorithms.TransientFickianDiffusion(network=net, phase=liquid)

inlet = net.pores(['left'])
outlet = net.pores(['right'])
Cin = 1
tfd.set_value_BC(pores=inlet, values=Cin)
tfd.set_rate_BC(pores=outlet, rates=0) # Outlet BC: fixed rate
ic = np.concatenate((np.ones(shape[1])*Cin, np.zeros((shape[0]-1)*shape[1]))) # Initial Concentration: shape[1] represents the first column of pores and (shape[0]-1)*shape[1] are all the rest of the pores in the domain
simTime = (0, 10) # Simulation starting and ending times




tfd.run(x0=ic, tspan=simTime)
# liquid.update(tfd.soln)
times = tfd.soln['pore.concentration'].t



cAvg = []
for ti in times:
    c_front = tfd.soln['pore.concentration'](ti)[outlet]
    q_front = tfd.rate(throats=net.Ts, mode='single')[outlet]
    cAvg.append((q_front*c_front).sum() / q_front.sum())
cAvg = np.array(cAvg)



rate_inlet = -tfd.rate(pores=outlet)[0] # Fluxes leaving the pores are negative
print(f'Flow rate: {rate_inlet:.5e} m3/s')
print(f'Average outlet final conc: {np.mean(cAvg):.5e}')

Adomain = (shape[1] * shape[2])*(spacing**2)
Ldomain = (shape[1]-1)*spacing
# D_eff_fracture = rate_inlet * spacing / (shape[0] * Athroat * (Cin - C_out))
D_eff = rate_inlet * spst.hmean(spacing) / (spst.hmean(Athroat) * (Cin - cAvg[-1])/net.Nt)
D_eff_fracture = rate_inlet * spacing / (np.mean(Athroat) * (Cin - cAvg[-1]))
D_eff_domain = rate_inlet * Ldomain / (Adomain * (Cin - cAvg[-1]))
print(f'Effective diffusivity [m2/s]', "{0:.6E}".format(D_eff))
print(f'Effective diffusivity (throat dimensions) [m2/s]', "{0:.6E}".format(D_eff_fracture))
print(f'Effective diffusivity (domain dimensions) [m2/s]', "{0:.6E}".format(D_eff_domain))
KdOpenPNM = rate_inlet/(Cin-cAvg[-1])
print(f'Diffusive conductance from OpenPNM (Qd/deltaC)', "{0:.6E}".format(KdOpenPNM))
KdGmean = spst.gmean(diffCond)
print(f'The geometric mean of the diffusive conductances is Kd =', "{0:.6E}".format(KdGmean))

V_p = net['pore.volume'].sum()
V_t = net['throat.volume'].sum()
V_bulk = np.prod(shape)*(spacing**3)
e = (V_p + V_t) / V_bulk
print('The porosity is: ', "{0:.6E}".format(e))

tau = e * Dmol / D_eff
print('The tortuosity is:', "{0:.6E}".format(tau))


def minMaxNorm(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def cdfBTC(t, D):
    C = -(np.sqrt(D)*t**(3/2)*spsp.erf(Ldomain/(2*np.sqrt(D*t))))/(np.sqrt(D*t**3)) #+ 1 # CONSTANT=1. WHY? IS IT BECAUSE LIMIT(C(t)) FOR t->0 IS -1?
    # C = np.nan_to_num(C, nan=0)
    return C

def errFunc(D, tNorm, cAvgNorm):
    Cpred = cdfBTC(tNorm, D)
    CpredNorm = minMaxNorm(Cpred)
    return np.sum((cAvgNorm-CpredNorm)**2)
    # return np.sum((cAvg-Cpred)**2)

# non0times = times[1:]     
tNorm = minMaxNorm(times)[1:]
cAvgNorm = minMaxNorm(cAvg)[1:]

D0 = 1e-4
C0 = cdfBTC(tNorm, D0)
C0norm = minMaxNorm(C0)

bounds = [(1e-8, 1)]  # Example bound: D should be between 1e-6 and 10
fitting = opt.minimize(errFunc, D0, args=(tNorm, cAvgNorm), bounds=bounds, method='Nelder-Mead', tol=1e-9)
# fitting = opt.minimize(errFunc, D0, args=(tNorm, cAvgNorm), bounds=bounds, method='Powell')
# fitting = opt.minimize(errFunc, D0, args=(tNorm, cAvgNorm), bounds=bounds, method='CG')
# fitting = opt.minimize(errFunc, D0, args=(tNorm, cAvgNorm), bounds=bounds, method='BFGS')
# fitting = opt.minimize(errFunc, D0, args=(tNorm, cAvgNorm), bounds=bounds, method='L-BFGS-B')
# fitting = opt.minimize(errFunc, D0, args=(tNorm, cAvgNorm), bounds=bounds, method='TNC')
# fitting = opt.minimize(errFunc, D0, args=(tNorm, cAvgNorm), bounds=bounds, method='COBYLA')
# fitting = opt.minimize(errFunc, D0, args=(tNorm, cAvgNorm), bounds=bounds, method='SLSQP')
# fitting = opt.minimize(errFunc, D0, args=(tNorm, cAvgNorm), bounds=bounds, method='trust-constr')
Deff = fitting.x[0]  # Fitted parameter

Cfit = cdfBTC(tNorm, Deff)
CfitNorm = minMaxNorm(Cfit)

breakthrough = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
plt.plot(tNorm, cAvgNorm, label='CopenPNM')
plt.plot(tNorm, C0norm, label='C0norm')
plt.plot(tNorm, CfitNorm, label='CfitNorm')
plt.legend(loc='best')

print(f"Fitted D: {Deff:.4f}")

# Plot #############################################################
networkLabels = plt.figure(figsize=(8, 8))
networkLabels = op.visualization.plot_tutorial(net, font_size=6)

poreNetwork = plt.figure(figsize=(8, 8))
poreNetwork = op.visualization.plot_coordinates(net)
poreNetwork = op.visualization.plot_connections(net, size_by=liquid['throat.diffusive_conductance'], ax=poreNetwork)

lognormDist = plt.figure(figsize=(8, 8))
if 'diffCond' in globals():
    plt.hist(diffCond, edgecolor='k')
    plt.title('Lognormal distribution of diffusive conductances')
    plt.xlabel(r'$k_D [m^3/s]$')
    plt.ylabel(r'$Number of throats [-]$')

pc = tfd.soln['pore.concentration'](3)
# tc = tfd.interpolate_data(propname='throat.concentration')
# tc = tfd.soln['pore.concentration'](1)[throat.all]
d = net['pore.diameter']
fig, ax = plt.subplots(figsize=[5, 5])
op.visualization.plot_coordinates(network=net, color_by=pc, size_by=d, markersize=400, ax=ax)
# op.visualization.plot_connections(network=net, color_by=tc, linewidth=3, ax=ax)
_ = plt.axis('off')

breakthrough = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
plt.plot(times, cAvg)
# plt.plot(times, Cfit)
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
plt.plot(times, 1-np.array(cAvg))
plt.title("Conc in time at the outlet")
plt.xlabel(r'$Time [s]$')
plt.ylabel(r'$1-Concentration [-]$')
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='black')
plt.grid(True, which="minor", linestyle=':', linewidth=0.5, color='gray')
plt.legend(loc='best')
plt.tight_layout()

btcInterp = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
plt.plot(times, 1-cAvg, linewidth='5')

tReshaped = (times[-100:-1]).reshape(-1, 1)
linRegCavg = LinearRegression().fit(tReshaped, np.log(Cin-cAvg[-100:-1]))
interpCavg = np.exp(linRegCavg.intercept_+linRegCavg.coef_*times)
plt.plot(times, interpCavg, color='black', linewidth='2')
plt.text(times[1000], interpCavg[1000], f"y={linRegCavg.coef_[0]:.5f}x+{linRegCavg.intercept_:.5f}", fontsize=18, ha='left', va='bottom')

plt.title("Conc in time at the outlet")
plt.xlabel(r'$Time [s]$')
plt.ylabel(r'$1-Concentration [-]$')
# plt.xscale('log')
plt.yscale('log')
# plt.ylim([0.1, max(cAvg)])
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='black')
plt.grid(True, which="minor", linestyle=':', linewidth=0.5, color='gray')
plt.legend(loc='best')
plt.tight_layout()