import openpnm as op
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as spst
op.visualization.set_mpl_style()
np.random.seed(0)

throatLength = 1e-3
throatDiameter = throatLength/10
poreDiameter = throatDiameter*2

# Pore network #####################################################
pn = op.network.Cubic(shape=[10, 5, 1], spacing=throatLength)

tl = [throatLength]*pn.Nt
td = [throatDiameter]*pn.Nt
pd = [poreDiameter]*pn.Np

pn['throat.length'] = tl
pn['throat.diameter'] = td
pn['pore.diameter'] = pd

# Liquid properties #################################################
tld = spst.lognorm.rvs(0.5, loc=0, scale=1, size=pn.Nt) # Throats lognormal distribution

liquid = op.phase.Phase(network=pn)
liquid['throat.hydraulic_conductance'] = tld

# Stokes flow #######################################################
sf = op.algorithms.StokesFlow(network=pn, phase=liquid)

pl = 1e-3
pr = 1e-4
sf.set_value_BC(pores=pn.pores('left'), values=pl)
# sf.set_rate_BC(pores=pn.pores('right'), rates=-1e-8) # Outlet BC: fixed rate
sf.set_value_BC(pores=pn.pores('right'), values=pr) # Outlet BC: fixed pressure

soln = sf.run()
print(sf)
sf['pore.pressure'][pn.pores('right')]

# Diffusion #########################################################
liquid['throat.diffusive_conductance'] = tld # In this test, the molecular diffusion values are assigned to the hydralic conductance property

fd = op.algorithms.FickianDiffusion(network=pn, phase=liquid)

Cl = 1e-3
Cr = 0
fd.set_value_BC(pores=pn.pores('left'), values=Cl)
fd.set_value_BC(pores=pn.pores('right'), values=Cr)

fd.run()

Q = fd.rate(throats=pn.pores('right'), mode='group')[0]
kEff = Q/(Cl-Cr)
A = (pn['throat.diameter'][pn.pores('right')]**2*np.pi/4).sum()
dEff = kEff/A # *L ??

print(f'Total flux rate Qtot [m3/s]: {Q}')
print(f'Effective diffusion Deff: {dEff}')

mu = 1 # Viscosity model??!
qManual = np.pi*(throatDiameter/2)**4/(8*mu*throatLength)*(pl-pr)

print('Manual verification:\n'
      f'qManual [m3/s]: {qManual}')
# Plot ##############################################################
poreNetwork = plt.figure(figsize=(8, 8))
poreNetwork = op.visualization.plot_coordinates(pn)
poreNetwork = op.visualization.plot_connections(pn, size_by=liquid['throat.diffusive_conductance'], ax=poreNetwork)

lognormDist = plt.figure(figsize=(8, 8))
plt.hist(tld, edgecolor='k')
plt.title('Distribution of throats diffusivities')
plt.xlabel('Diffusive flux [m3/s]')
plt.ylabel('Number of throats [-]')

pc = fd['pore.concentration']
tc = fd.interpolate_data(propname='throat.concentration')
d = pn['pore.diameter']
fig, ax = plt.subplots(figsize=[5, 5])
op.visualization.plot_coordinates(network=pn, color_by=pc, size_by=d, markersize=400, ax=ax)
op.visualization.plot_connections(network=pn, color_by=tc, linewidth=3, ax=ax)
_ = plt.axis('off')