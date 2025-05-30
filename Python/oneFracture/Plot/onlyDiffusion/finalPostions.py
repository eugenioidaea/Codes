debug = False
if not debug:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from sklearn.linear_model import LinearRegression

save = False

loadFinalPositions = np.load('onlyDiff.npz', allow_pickle=True)
for name, value in (loadFinalPositions.items()):
    globals()[name] = value

finalPositionMatrixDecay = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
for i in range(len(xRT)):
    finalPositionMatrixDecay = plt.figure(figsize=(8, 8))
    plt.plot(xRT[i], yRT[i], 'b*')
    plt.plot([xInit, xInit], [lby, uby], color='yellow', linestyle='--', linewidth=3)
    plt.axhline(y=uby, color='r', linewidth=3)
    plt.axhline(y=lby, color='r', linewidth=3)
    plt.xlim(np.min(np.min(xRT)), np.max(np.max(xRT)))
# plt.plot(xRT[0], yRT[0], 'b*')
# plt.plot(xRT[1], yRT[1], 'r*')
# plt.plot(xRT[2], yRT[2], 'g*')
# plt.plot(xRT[3], yRT[3], 'y*')
# plt.plot([xInit, xInit], [lby, uby], color='yellow', linestyle='--', linewidth=3)
# plt.axhline(y=uby, color='r', linewidth=3)
# plt.axhline(y=lby, color='r', linewidth=3)