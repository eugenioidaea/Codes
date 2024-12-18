import numpy as np
import openpnm as op
op.visualization.set_mpl_style()

pn = op.network.Demo(shape=[5, 5, 1], spacing=5e-5)
print(pn)