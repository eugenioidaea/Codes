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
    plt.gca().set_yticks([])
    plt.gca().set_yticklabels([])

# plt.plot(xRT[0], yRT[0], 'b*')
# plt.plot(xRT[1], yRT[1], 'r*')
# plt.plot(xRT[2], yRT[2], 'g*')
# plt.plot(xRT[3], yRT[3], 'y*')
# plt.plot([xInit, xInit], [lby, uby], color='yellow', linestyle='--', linewidth=3)
# plt.axhline(y=uby, color='r', linewidth=3)
# plt.axhline(y=lby, color='r', linewidth=3)

fig, axes = plt.subplots(1, 4, figsize=(15, 5))
for i in range(len(xRT)):
    if i == 0:
        axes[i].plot(xRT[i], yRT[i], 'b*')
    else:
        livePmask = xRT[i] != xRT[i-1]
        axes[i].plot(xRT[i][livePmask], yRT[i][livePmask], 'b*')
    axes[i].plot([xInit, xInit], [lby, uby], color='orange', linestyle='--', linewidth=3)
    axes[i].axhline(y=uby, color='brown', linewidth=3)
    axes[i].axhline(y=lby, color='brown', linewidth=3)
    axes[i].set_xlim(np.min(np.min(xRT)), np.max(np.max(xRT)))
    if i != 0:
        axes[i].set_yticks([])  # Remove y-axis ticks
        axes[i].set_yticklabels([])  # Remove y-axis labels