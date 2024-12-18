import openpnm as op
import numpy as np
import matplotlib.pyplot as plt
op.visualization.set_mpl_style()

pn = op.network.Cubic(shape=[4, 4, 4], spacing=1e-5)
ax = op.visualization.plot_coordinates(pn)
ax = op.visualization.plot_connections(pn, ax=ax)

pn = op.network.Cubic(shape=[8, 4, 2], spacing=[10e-5, 5e-5, 4e-5])
ax = op.visualization.plot_coordinates(pn)
ax = op.visualization.plot_connections(pn, ax=ax)

pn = op.network.Cubic(shape=[4, 4, 4], connectivity=26)
ax = op.visualization.plot_coordinates(pn)
ax = op.visualization.plot_connections(pn, ax=ax)

pn = op.network.Cubic(shape=[4, 4, 4], connectivity=26)
np.random.seed(0)
drop = np.random.randint(0, pn.Nt, 500)
op.topotools.trim(network=pn, throats=drop)
ax = op.visualization.plot_coordinates(pn)
ax = op.visualization.plot_connections(pn, ax=ax)

print(pn)

Ps = pn.pores('back')
print("The following pores are labelled 'back':", Ps)
Ps = pn.pores(['back', 'left'])
print("The following pores are labelled 'back' OR 'left':", Ps)

im = op.topotools.template_cylinder_annulus(z=3, r_outer=8, r_inner=3)
pn = op.network.CubicTemplate(template=im)
ax = op.visualization.plot_coordinates(pn)
ax = op.visualization.plot_connections(pn, ax=ax)

fcc = op.network.FaceCenteredCubic(shape=[4, 4, 4], spacing=1e-5)
ax = op.visualization.plot_connections(fcc)
op.visualization.plot_coordinates(fcc, ax=ax)
bcc = op.network.BodyCenteredCubic(shape=[4, 4, 4], spacing=1e-5)
ax = op.visualization.plot_connections(bcc)
ax = op.visualization.plot_coordinates(bcc, ax=ax);

print(bcc)

ax = op.visualization.plot_connections(bcc, throats=bcc.throats('body_to_body'))
ax = op.visualization.plot_coordinates(bcc, pores=bcc.pores('body'), ax=ax)

ax = op.visualization.plot_connections(bcc, throats=bcc.throats('body_to_body'))
ax = op.visualization.plot_coordinates(bcc, pores=bcc.pores('body'), ax=ax)
ax = op.visualization.plot_connections(bcc, throats=bcc.throats('corner_to_body'), c='g', ax=ax)
ax = op.visualization.plot_coordinates(bcc, pores=bcc.pores('corner'), ax=ax, c='orange')