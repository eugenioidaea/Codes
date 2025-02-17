import openpnm as op
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as spst
op.visualization.set_mpl_style()
np.set_printoptions(precision=5)
from sklearn.linear_model import LinearRegression

spacing = 1e-3 # It is the distance between pores that it does not necessarily correspond to the length of the throats because of the tortuosity
# throatDiameter = spacing/10
poreDiameter = spacing/10
Dmol = 1e-5 # Molecular Diffusion

# Pore network #####################################################
shape = [100, 10, 10]
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
Deff = []
throatVariance = np.linspace(0.1, 6, 100)
for i in range(len(throatVariance)):
    throatDiameter = spst.lognorm.rvs(throatVariance[i], loc=0, scale=poreDiameter/2, size=net.Nt) # Conductance lognormal distribution
    net['throat.diameter'] = throatDiameter
    Athroat = throatDiameter**2*np.pi/4
    diffCond = Dmol*Athroat/spacing
    liquid['throat.diffusive_conductance'] = diffCond
    net['throat.volume'] = Athroat*spacing

    fd = op.algorithms.FickianDiffusion(network=net, phase=liquid)

    inlet = net.pores('left')
    outlet = net.pores('right')
    C_in, C_out = [10, 5]
    fd.set_value_BC(pores=inlet, values=C_in)
    fd.set_value_BC(pores=outlet, values=C_out)

    fd.run()

    rate_inlet = fd.rate(pores=inlet)[0]
    print(f'Flow rate: {rate_inlet:.5e} m3/s')

    Adomain = ((shape[1]-1) * (shape[2]-1))*(spacing**2)
    Ldomain = net.Nt*spacing
    Deff.extend([rate_inlet * Ldomain / (Adomain * (C_in - C_out))])
    print(f'Effective diffusivity (domain dimensions) [m2/s]', "{0:.6E}".format(Deff[i]))
    KdOpenPNM = rate_inlet/(C_in-C_out)
    print(f'Diffusive conductance from OpenPNM (Qd/deltaC)', "{0:.6E}".format(KdOpenPNM))
    KdGmean = spst.gmean(diffCond)
    print(f'The geometric mean of the diffusive conductances is Kd =', "{0:.6E}".format(KdGmean))

    V_p = net['pore.volume'].sum()
    V_t = net['throat.volume'].sum()
    V_bulk = np.prod(shape)*(spacing**3)
    e = (V_p + V_t) / V_bulk
    print('The porosity is: ', "{0:.6E}".format(e))

    tau = e * Dmol / Deff[i]
    print('The tortuosity is:', "{0:.6E}".format(tau))

# Plot #############################################################
lognormDist = plt.figure(figsize=(8, 8))
if 'diffCond' in globals():
    log_bins = np.logspace(np.log10(min(diffCond)), np.log10(max(diffCond)), num=20)
    hist, edges = np.histogram(diffCond, bins=log_bins)
    plt.hist(diffCond, bins=log_bins, edgecolor='k')
    # plt.hist(diffCond, edgecolor='k')
    plt.title('Lognormal distribution of diffusive conductances')
    plt.xlabel(r'$k_D [m^3/s]$')
    plt.ylabel(r'$Number of throats [-]$')
    plt.xscale('log')

pc = fd['pore.concentration']
tc = fd.interpolate_data(propname='throat.concentration')
d = net['pore.diameter']

poreNetwork = plt.figure(figsize=(8, 8))
poreNetwork = op.visualization.plot_coordinates(net, size_by=d, markersize=0.1, ax=poreNetwork)
poreNetwork = op.visualization.plot_connections(net, size_by=liquid['throat.diffusive_conductance'], linewidth=20, ax=poreNetwork)

fig, ax = plt.subplots(figsize=[8, 8])
op.visualization.plot_coordinates(network=net, color_by=pc, size_by=d, markersize=4, ax=ax)
op.visualization.plot_connections(network=net, color_by=tc, linewidth=1, ax=ax)
# _ = plt.axis('off')

effDiffVsCondVar = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
plt.plot(throatVariance, Deff, 'o', markerfacecolor='blue', markeredgecolor='blue', markersize=5)
plt.plot(throatVariance[-1], Deff[-1], 'o', markerfacecolor='blue', markeredgecolor='red', markersize=10, markeredgewidth=3)
plt.title("Effective diffusion vs conductance dist var")
plt.xlabel(r'$s$')
plt.ylabel(r'$D_{eff}$')

throatVarianceReshaped = (throatVariance).reshape(-1, 1)
linRegDeff = LinearRegression().fit(throatVarianceReshaped, np.log(Deff))
interpDeff = np.exp(linRegDeff.intercept_+linRegDeff.coef_*throatVariance)
plt.plot(throatVariance, interpDeff, color='black', linewidth='4')
plt.text(throatVariance[len(throatVariance)//2], interpDeff[len(interpDeff)//2], r"$D_{eff} = e^{" + f"{linRegDeff.intercept_:.5f} + {linRegDeff.coef_[0]:.5f} * s" + "}$", fontsize=18, ha='right', va='bottom')

# plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='black')
plt.grid(True, which="minor", linestyle=':', linewidth=0.5, color='gray')
plt.legend(loc='best')
plt.tight_layout()