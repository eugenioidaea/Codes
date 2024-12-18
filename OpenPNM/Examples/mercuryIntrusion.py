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