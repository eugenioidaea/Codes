import numpy as np
import openpnm as op
import matplotlib.pyplot as plt
op.visualization.set_mpl_style()

pn = op.network.Demo(shape=[5, 5, 1], spacing=5e-5)
print(pn)

water = op.phase.Phase(network=pn)

# .add_model is a new attribute of the water dict
# propname is the key
# model is the value
water.add_model(propname='pore.viscosity',
                model=op.models.phase.viscosity.water_correlation)
print(water)
print(water['pore.viscosity'])

R = pn['throat.diameter']/2
L = pn['throat.length']
mu = water['throat.viscosity']  # See ProTip below
water['throat.hydraulic_conductance'] = np.pi*R**4/(8*mu*L)
print(water['throat.hydraulic_conductance'])

sf = op.algorithms.StokesFlow(network=pn, phase=water)
print(sf)

sf.set_value_BC(pores=pn.pores('left'), values=100_000)
sf.set_rate_BC(pores=pn.pores('right'), rates=1e-10)
print(sf)

soln = sf.run()
water.update(sf.soln)

sf['pore.pressure'][pn.pores('right')]

fig, ax = plt.subplots(figsize=(5.5, 4))
ax = op.visualization.plot_coordinates(pn)
ax = op.visualization.plot_connections(pn, ax=ax)

fig, ax = plt.subplots(figsize=(5.5, 4))
ax = op.visualization.plot_coordinates(pn, ax=ax, color_by=water['pore.viscosity'])

fig, ax = plt.subplots(figsize=(5.5, 4))
ax = op.visualization.plot_connections(pn, ax=ax, color_by=water['throat.hydraulic_conductance'])