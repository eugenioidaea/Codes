import openpnm as op
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as spst
op.visualization.set_mpl_style()
np.set_printoptions(precision=5)

spacing = 1e-3 # It is the distance between pores that it does not necessarily correspond to the length of the throats because of the tortuosity
# throatDiameter = spacing/10
poreDiameter = spacing/10
Dmol = 1e-5 # Molecular Diffusion

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

inlet = net.pores('left')
outlet = net.pores('right')
C_in, C_out = [1, 0]
tfd.set_value_BC(pores=inlet, values=C_in)
tfd.set_value_BC(pores=outlet, values=C_out)





# tfd.settings['t_initial'] = 0
# tfd.settings['t_final'] = 10  # Total simulation time [s]
# tfd.settings['t_output'] = 1  # Time step for output
# tfd.settings['t_step'] = 0.1  # Internal time step
# tfd.setup(t_scheme='cranknicolson', t_final=100, t_output=5, t_step=1, t_tolerance=1e-12)





tfd.run(np.concatenate((np.ones(shape[1]), np.zeros((shape[0]-1)*shape[1]))), (0, 10))

rate_inlet = -tfd.rate(pores=outlet)[0] # Fluxes leaving the pores are negative
print(f'Flow rate: {rate_inlet:.5e} m3/s')

Adomain = (shape[1] * shape[2])*(spacing**2)
Ldomain = net.Nt*spacing
# D_eff_fracture = rate_inlet * spacing / (shape[0] * Athroat * (C_in - C_out))
D_eff = rate_inlet * spst.hmean(spacing) / (spst.hmean(Athroat) * (C_in - C_out)/net.Nt)
D_eff_fracture = rate_inlet * spacing / (np.mean(Athroat) * (C_in - C_out))
D_eff_domain = rate_inlet * Ldomain / (Adomain * (C_in - C_out))
print(f'Effective diffusivity [m2/s]', "{0:.6E}".format(D_eff))
print(f'Effective diffusivity (throat dimensions) [m2/s]', "{0:.6E}".format(D_eff_fracture))
print(f'Effective diffusivity (domain dimensions) [m2/s]', "{0:.6E}".format(D_eff_domain))
KdOpenPNM = rate_inlet/(C_in-C_out)
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

pc = tfd['pore.concentration']
tc = tfd.interpolate_data(propname='throat.concentration')
d = net['pore.diameter']
fig, ax = plt.subplots(figsize=[5, 5])
op.visualization.plot_coordinates(network=net, color_by=pc, size_by=d, markersize=400, ax=ax)
op.visualization.plot_connections(network=net, color_by=tc, linewidth=3, ax=ax)
_ = plt.axis('off')