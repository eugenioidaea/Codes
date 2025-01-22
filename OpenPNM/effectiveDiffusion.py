import openpnm as op
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as spst
op.visualization.set_mpl_style()
np.random.seed(0)

# Pore network ####################################################
pn = op.network.Cubic(shape=[10, 5, 1], spacing=1e-5)

tl = [1e-4]*pn.Nt
td = [1e-6]*pn.Nt
pd = [1e-5]*pn.Np

pn['throat.length'] = tl
pn['throat.diameter'] = td
pn['pore.diameter'] = pd

# Liquid properties ################################################
tld = spst.lognorm.rvs(0.5, loc=0, scale=1, size=pn.Nt) # Throats lognormal distribution

liquid = op.phase.Phase(network=pn)
liquid['throat.hydraulic_conductance'] = tld # In this test, the molecular diffusion values are assigned to the hydralic conductance property

sf = op.algorithms.StokesFlow(network=pn, phase=liquid)

sf.set_value_BC(pores=pn.pores('left'), values=100_000)
sf.set_rate_BC(pores=pn.pores('right'), rates=1e-10)

soln = sf.run()
print(sf)
sf['pore.pressure'][pn.pores('right')]

# Plot ##############################################################
poreNetwork = plt.figure(figsize=(8, 8))
poreNetwork = op.visualization.plot_coordinates(pn)
poreNetwork = op.visualization.plot_connections(pn, size_by=liquid['throat.hydraulic_conductance'], ax=poreNetwork)

lognormDist = plt.figure(figsize=(8, 8))
lognormDist = plt.hist(tld, edgecolor='k')