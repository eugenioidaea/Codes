# Hagen-Poiseuille verification code for a single conduit (adapted from https://openpnm.org/examples/tutorials/08_simulating_transport.html)
import openpnm as op
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
op.visualization.set_mpl_style()

coords = [[0, 0, 0], # coordinates for pore 0
          [1, 0, 0]] # coordinates for pore 1

conns = [[0, 1]] # throat 0 connects pores 0 and 1

Dpore = [0.1, # diameter for pore 0
         0.1] # diameter for pore 1

Dthroat = [0.4] # diameter for throat 0

Lthroat = coords[1][0]-coords[0][0] # length for throat 0

pn = op.network.Network(coords=coords, conns=conns)
ax = op.visualization.plot_connections(pn)
ax = op.visualization.plot_coordinates(pn, ax=ax)

# Aissgn properties
pn['pore.diameter'] = Dpore
pn['throat.diameter'] = Dthroat
pn['throat.length'] = Lthroat

# Define the left and right labels for pores
pn['pore.left'] = False
#pn['pore.left'][[0]] = True
pn.set_label(label='left', pores=pn[[0]])
pn['pore.right'] = False
# pn['pore.right'][[1]] = True
pn.set_label(label='right', pores=pn[[1]])

water = op.phase.Phase(network=pn) # empty phase object

water.add_model(propname='pore.viscosity',
                model=op.models.phase.viscosity.water_correlation) # pore-scale model for computing the viscosity of water
# print(water)
# print(water['pore.viscosity'])

R = pn['throat.diameter']/2
L = pn['throat.length']
mu = water['throat.viscosity']  # See ProTip below
water['throat.hydraulic_conductance'] = np.pi*R**4/(8*mu*L) # manual calculation of conductance

sf = op.algorithms.StokesFlow(network=pn, phase=water)

sf.set_value_BC(pores=pn.pores('left'), values=2)
sf.set_rate_BC(pores=pn.pores('right'), rates=-1) # Outlet BC: fixed rate
# sf.set_value_BC(pores=pn.pores('right'), values=1) # Outlet BC: fixed pressure

soln = sf.run()
water.update(sf.soln)
Q = sf.rate(pores=pn.pores('left'), mode='group')[0]

print(sf)
print(f'Inlet cooridnates: {pn['pore.coords@left'][0]}')
print(f'Outlet cooridnates: {pn['pore.coords@right'][0]}')
print('Pressure left:', sf['pore.pressure'][pn.pores('left')][0])
print('Pressure right:', sf['pore.pressure'][pn.pores('right')][0])
print(f'Flow rate: {Q}')
print('Conductance:', water['throat.hydraulic_conductance'][0])

pore_data_sheet = pd.DataFrame({k: pn[k] for k in pn.props(element='pore') if pn[k].ndim == 1})