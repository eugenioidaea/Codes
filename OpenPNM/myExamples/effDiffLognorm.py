import openpnm as op
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as spst
op.visualization.set_mpl_style()
np.set_printoptions(precision=5)

shape = [10, 10, 1]
spacing = 1e-5
net = op.network.Cubic(shape=shape, spacing=spacing)

geo = op.models.collections.geometry.spheres_and_cylinders
net.add_model_collection(geo, domain='all')
net.regenerate_models()

print(net)

tld = spst.lognorm.rvs(0.5, loc=0, scale=1e-11, size=net.Nt) # Throats lognormal distribution
liquid = op.phase.Phase(network=net)
liquid['throat.diffusive_conductance'] = tld # In this test, the molecular diffusion values are assigned to the hydralic conductance property

fd = op.algorithms.FickianDiffusion(network=net, phase=liquid)

inlet = net.pores('left')
outlet = net.pores('right')
C_in, C_out = [10, 5]
fd.set_value_BC(pores=inlet, values=C_in)
fd.set_value_BC(pores=outlet, values=C_out)

fd.run()

rate_inlet = fd.rate(pores=inlet)[0]
print(f'Molar flow rate: {rate_inlet:.5e} mol/s')

A = (shape[1] * shape[2])*(spacing**2)
L = shape[0]*spacing
D_eff = rate_inlet * L / (A * (C_in - C_out))
print(f'Effective diffusivity: {D_eff:.5e} m2/s')

# Plot #############################################################
poreNetwork = plt.figure(figsize=(8, 8))
poreNetwork = op.visualization.plot_coordinates(net)
poreNetwork = op.visualization.plot_connections(net, size_by=liquid['throat.diffusive_conductance'], ax=poreNetwork)

lognormDist = plt.figure(figsize=(8, 8))
plt.hist(tld, edgecolor='k')
plt.title('Distribution of throats diffusivities')
plt.xlabel('Diffusive conductance [m2/s]')
plt.ylabel('Number of throats [-]')

pc = fd['pore.concentration']
tc = fd.interpolate_data(propname='throat.concentration')
d = net['pore.diameter']
fig, ax = plt.subplots(figsize=[5, 5])
op.visualization.plot_coordinates(network=net, color_by=pc, size_by=d, markersize=400, ax=ax)
op.visualization.plot_connections(network=net, color_by=tc, linewidth=3, ax=ax)
_ = plt.axis('off')