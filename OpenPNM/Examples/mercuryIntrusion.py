import openpnm as op
import numpy as np
import matplotlib.pyplot as plt
op.visualization.set_mpl_style()

# Creating a Cubic Network
Nx, Ny, Nz = 10, 10, 10
Lc = 1e-4
pn = op.network.Cubic([Nx, Ny, Nz], spacing=Lc)
pn.add_model_collection(op.models.collections.geometry.spheres_and_cylinders)
pn.regenerate_models()
print(pn)

# Defining a Phase
hg = op.phase.Mercury(network=pn)
hg.add_model(propname='throat.entry_pressure',
             model=op.models.physics.capillary_pressure.washburn)
hg.regenerate_models()
print(hg)

# Performing a Drainage Simulation
mip = op.algorithms.Drainage(network=pn, phase=hg)
mip.set_inlet_BC(pores=pn.pores(['left', 'right']))
mip.run(pressures=np.logspace(4, 6))

data = mip.pc_curve()
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.semilogx(data.pc, data.snwp, 'k-o')
ax.set_xlabel('capillary pressure [Pa]')
ax.set_ylabel('mercury saturation');

# Generate phase and physics
water = op.phase.Water(network=pn)
water.add_model(propname='throat.hydraulic_conductance',
                model=op.models.physics.hydraulic_conductance.generic_hydraulic)

# Create algorithm, set boundary conditions and run simulation
sf = op.algorithms.StokesFlow(network=pn, phase=water)
Pin, Pout = (200_000, 101_325)
sf.set_value_BC(pores=pn.pores('left'), values=Pin)
sf.set_value_BC(pores=pn.pores('right'), values=Pout)
sf.run()

Q = sf.rate(pores=pn.pores('left'))
A = Ny*Nz*Lc**2
L = Nx*Lc
mu = water['pore.viscosity'].mean()
K = Q*mu*L/(A*(Pin-Pout))
print(K)

import openpnm.models.geometry as gmods

pn['pore.old_diameter'] = pn.pop('pore.diameter')
pn.add_model(propname='pore.diameter',
             model=gmods.pore_size.weibull,
             shape=0.5, loc=0, scale=1e-5)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.hist(pn['pore.diameter'], edgecolor='k', label='new diameter')
ax.hist(pn['pore.old_diameter'], edgecolor='k', label='old_diameter', bins=20)
ax.set_xlabel('diameter [m]')
ax.set_ylabel('frequency')
ax.legend();