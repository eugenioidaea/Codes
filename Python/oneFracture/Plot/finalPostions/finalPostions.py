debug = False
if not debug:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from sklearn.linear_model import LinearRegression

save = False

loadFinalPositions = np.load('compareAdsP100.npz')
for name, value in (loadFinalPositions.items()):
    globals()[name] = value

finalPositionMatrixDecay = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
plt.plot(x, y, 'b*')
# plt.plot(xK001, yK001, 'r*')
# plt.plot(xK01, yK01, 'b*')
plt.plot([xInit, xInit], [lby, uby], color='yellow', linewidth=3)
plt.axhline(y=uby, color='r', linestyle='--', linewidth=3)
plt.axhline(y=lby, color='r', linestyle='--', linewidth=3)