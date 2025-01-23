import openpnm as op
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as spst
op.visualization.set_mpl_style()
np.random.seed(0)

# Pore network #####################################################
pn = op.network.Cubic(shape=[10, 5, 1], spacing=1e-5)

tl = [1e-4]*pn.Nt
td = [1e-6]*pn.Nt
pd = [1e-5]*pn.Np

pn['throat.length'] = tl
pn['throat.diameter'] = td
pn['pore.diameter'] = pd

# Liquid properties #################################################
tld = spst.lognorm.rvs(0.5, loc=0, scale=1, size=pn.Nt) # Throats lognormal distribution

liquid = op.phase.Phase(network=pn)
liquid['throat.hydraulic_conductance'] = tld

# Stokes flow #######################################################
sf = op.algorithms.StokesFlow(network=pn, phase=liquid)

sf.set_value_BC(pores=pn.pores('left'), values=2)
sf.set_rate_BC(pores=pn.pores('right'), rates=-1) # Outlet BC: fixed rate
# sf.set_value_BC(pores=pn.pores('right'), values=1) # Outlet BC: fixed pressure

soln = sf.run()
print(sf)
sf['pore.pressure'][pn.pores('right')]

# Diffusion #########################################################
liquid['throat.diffusive_conductance'] = tld # In this test, the molecular diffusion values are assigned to the hydralic conductance property

fd = op.algorithms.FickianDiffusion(network=pn, phase=liquid)

Cl = 1
Cr = 0
fd.set_value_BC(pores=pn.pores('left'), values=Cl)
fd.set_value_BC(pores=pn.pores('right'), values=Cr)

fd.run()

Q = sf.rate(throats=pn.pores('right'), mode='group')[0]
kEff = Q/(Cl-Cr)
A = (pn['throat.diameter'][pn.pores('right')]**2*np.pi/4).sum()
dEff = kEff/A

print(f'Effective diffusion Deff: {dEff}')

# Plot ##############################################################
poreNetwork = plt.figure(figsize=(8, 8))
poreNetwork = op.visualization.plot_coordinates(pn)
poreNetwork = op.visualization.plot_connections(pn, size_by=liquid['throat.diffusive_conductance'], ax=poreNetwork)

lognormDist = plt.figure(figsize=(8, 8))
lognormDist = plt.hist(tld, edgecolor='k')

pc = fd['pore.concentration']
tc = fd.interpolate_data(propname='throat.concentration')
d = pn['pore.diameter']
fig, ax = plt.subplots(figsize=[5, 5])
op.visualization.plot_coordinates(network=pn, color_by=pc, size_by=d, markersize=400, ax=ax)
op.visualization.plot_connections(network=pn, color_by=tc, linewidth=3, ax=ax)
_ = plt.axis('off')