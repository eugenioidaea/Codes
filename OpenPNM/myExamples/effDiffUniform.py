import openpnm as op
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as spst
op.visualization.set_mpl_style()
np.set_printoptions(precision=5)


throatLength = 1e-3
throatDiameter = throatLength/10
poreDiameter = throatDiameter*2
md = 1e-3 # Molecular Diffusion

# Pore network #####################################################
shape = [10, 10, 1]
net = op.network.Cubic(shape=shape, spacing=throatLength)

tl = [throatLength]*net.Nt
td = [throatDiameter]*net.Nt
pd = [poreDiameter]*net.Np
Athroat = throatDiameter**2*np.pi/4

net['throat.length'] = tl
net['throat.diameter'] = td
net['pore.diameter'] = pd

print(net)

liquid = op.phase.Phase(network=net)

# Uniform diffusive conductance #################################################
# unifCond = np.full(net.Nt, md*Athroat/throatLength)
# liquid['throat.diffusive_conductance'] = unifCond

# Lognormal diffusive conductance ###############################################
cld = spst.lognorm.rvs(0.5, loc=0, scale=md, size=net.Nt) # Conductance lognormal distribution
liquid['throat.diffusive_conductance'] = cld

fd = op.algorithms.FickianDiffusion(network=net, phase=liquid)

inlet = net.pores('left')
outlet = net.pores('right')
C_in, C_out = [10, 5]
fd.set_value_BC(pores=inlet, values=C_in)
fd.set_value_BC(pores=outlet, values=C_out)

fd.run()

rate_inlet = fd.rate(pores=inlet)[0]
print(f'Flow rate: {rate_inlet:.5e} m3/s')

Adomain = (shape[1] * shape[2])*(throatLength**2)
Ldomain = shape[0]*throatLength
D_eff_fracture = rate_inlet * throatLength / (Athroat * (C_in - C_out))
D_eff_domain = rate_inlet * Ldomain / (Adomain * (C_in - C_out))
print(f'Effective diffusivity (fracture dimensions): {D_eff_fracture:.5e} m2/s')
print(f'Effective diffusivity (domain dimensions): {D_eff_domain:.5e} m2/s')

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