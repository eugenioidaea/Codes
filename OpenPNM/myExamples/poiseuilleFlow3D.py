# Hagen-Poiseuille verification code for 3D cubic network with homogeneous pore and throat geometries
# Adapted from https://openpnm.org/examples/tutorials/08_simulating_transport.html and https://openpnm.org/examples/applications/absolute_permeability.html
import numpy as np
import openpnm as op
op.visualization.set_mpl_style()
np.random.seed(10)
np.set_printoptions(precision=5)

pn = op.network.Cubic(shape=[15, 15, 15], spacing=1e-2)
# pn.add_model_collection(op.models.collections.geometry.spheres_and_cylinders)
# pn.regenerate_models()
Lthroat = [1e-2]*pn.Nt
Dthroat = [1e-4]*pn.Nt
Dpore = [1e-3]*pn.Np

pn['throat.length'] = Lthroat
pn['pore.diameter'] = Dpore
pn['throat.diameter'] = Dthroat

water = op.phase.Phase(network=pn)
water.add_model(propname='pore.viscosity',
                model=op.models.phase.viscosity.water_correlation)
print(water['pore.viscosity'])

R = pn['throat.diameter']/2
L = pn['throat.length']
mu = water['throat.viscosity']  # See ProTip below
water['throat.hydraulic_conductance'] = np.pi*R**4/(8*mu*L)
print(water['throat.hydraulic_conductance'])

sf = op.algorithms.StokesFlow(network=pn, phase=water)

Pleft = 2
Pright = 1
sf.set_value_BC(pores=pn.pores('left'), values=Pleft)
# sf.set_rate_BC(pores=pn.pores('right'), rates=-0.01) # Outlet BC: fixed rate
sf.set_value_BC(pores=pn.pores('right'), values=Pright) # Outlet BC: fixed pressure

soln = sf.run()
water.update(sf.soln)

print(sf)

ax = op.visualization.plot_connections(pn)
ax = op.visualization.plot_coordinates(pn, ax=ax, color_by=water['pore.pressure'])

Q = sf.rate(pores=pn.pores('left'), mode='group')[0]
A = op.topotools.get_domain_area(pn, inlets=pn.pores('left'), outlets=pn.pores('right'))
L = op.topotools.get_domain_length(pn, inlets=pn.pores('left'), outlets=pn.pores('right'))
K = Q * L * mu[0] / (A * (Pleft-Pright)) # mu and Delta_P were assumed to be 1.
print(f'The value of K is: {K}')