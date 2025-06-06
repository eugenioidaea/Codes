import openpnm as op
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as spst
op.visualization.set_mpl_style()
np.set_printoptions(precision=5)

spacing = 1e-3 # It is the distance between pores that it does not necessarily correspond to the length of the throats because of the tortuosity
throatDiameter = spacing/10
poreDiameter = throatDiameter*2
Dmol = 1e-3 # Molecular Diffusion

# Pore network #####################################################
shape = [2, 1, 1]
net = op.network.Cubic(shape=shape, spacing=spacing) # Shape of the elementary cell of the network: cubic
# geo = op.models.collections.geometry.spheres_and_cylinders # Shape of the pore and throats
# net.add_model_collection(geo, domain='all') # Assign the shape of pores and throats to the network
net.regenerate_models() # Compute geometric properties such as pore volume

net['throat.length'] = spacing
net['throat.diameter'] = throatDiameter
net['pore.diameter'] = poreDiameter
net['pore.volume'] = 4/3*np.pi*poreDiameter**3/8
Athroat = throatDiameter**2*np.pi/4
net['throat.volume'] = Athroat*spacing

print(net)

liquid = op.phase.Phase(network=net)

# Uniform diffusive conductance #################################################
unifCond = np.full(net.Nt, Dmol*Athroat/spacing)
liquid['throat.diffusive_conductance'] = unifCond

# Lognormal diffusive conductance ###############################################
# cld = spst.lognorm.rvs(0.5, loc=0, scale=Dmol, size=net.Nt) # Conductance lognormal distribution
# liquid['throat.diffusive_conductance'] = cld

fd = op.algorithms.FickianDiffusion(network=net, phase=liquid)

inlet = net.pores('left')
outlet = net.pores('right')
C_in, C_out = [10, 5]
fd.set_value_BC(pores=inlet, values=C_in)
fd.set_value_BC(pores=outlet, values=C_out)

fd.run()

rate_inlet = fd.rate(pores=inlet)[0]
print(f'Flow rate: {rate_inlet:.5e} m3/s')

Adomain = throatDiameter**2
Ldomain = (shape[0]-1)*spacing
D_eff_fracture = rate_inlet * Ldomain / (Athroat * (C_in - C_out))
D_eff = rate_inlet * Ldomain / (Adomain * (C_in - C_out))
print(f'Effective diffusivity (throat dimensions) [m2/s]', "{0:.6E}".format(D_eff_fracture))
print(f'Effective diffusivity (domain dimensions) [m2/s]', "{0:.6E}".format(D_eff))

V_p = net['pore.volume'].sum()
V_t = net['throat.volume'].sum()
V_bulk = np.prod(shape)*(spacing**3)
e = (V_p + V_t) / V_bulk
print('The porosity is: ', "{0:.6E}".format(e))

tau = e * Dmol / D_eff
print('The tortuosity is:', "{0:.6E}".format(tau))

# Plot #############################################################
poreNetwork = plt.figure(figsize=(8, 8))
poreNetwork = op.visualization.plot_coordinates(net)
poreNetwork = op.visualization.plot_connections(net, size_by=liquid['throat.diffusive_conductance'], ax=poreNetwork)

lognormDist = plt.figure(figsize=(8, 8))
if 'cld' in globals():
    plt.hist(cld, edgecolor='k')
    plt.title('Lognormal distribution of diffusive conductances')
    plt.xlabel(r'$k_D [m^3/s]$')
    plt.ylabel(r'$Number of throats [-]$')

pc = fd['pore.concentration']
tc = fd.interpolate_data(propname='throat.concentration')
d = net['pore.diameter']
fig, ax = plt.subplots(figsize=[5, 5])
op.visualization.plot_coordinates(network=net, color_by=pc, size_by=d, markersize=400, ax=ax)
op.visualization.plot_connections(network=net, color_by=tc, linewidth=3, ax=ax)
_ = plt.axis('off')