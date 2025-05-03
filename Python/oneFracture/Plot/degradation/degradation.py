debug = False
if not debug:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from sklearn.linear_model import LinearRegression

save = False

# loadFinalPositions = np.load('degradation.npz')
# for name, value in (loadFinalPositions.items()):
#     globals()[name] = value

loadFinalPositions = np.load('onlyDecayTau400.npz')
for name, value in (loadFinalPositions.items()):
    globals()[name] = value

# Distribution of survival times for particles
plt.figure(figsize=(8, 8))
# plt.plot(np.arange(0, num_particles, 1), np.sort(particleStepsDeg)[::-1], 'b*')
plt.plot(np.arange(0, num_particles, 1), np.sort(survivalTimeDist)[::-1], 'k-')
plt.title("Survival time distribution")

# Distribution of live particles in time
survivalTimeDistribution = plt.figure(figsize=(8, 8), dpi=300)
plt.rcParams.update({'font.size': 20})
plt.plot(timeStep, exp_prob, 'r-', label='Exp pdf')
plt.plot(Time[::10], numOfLivePart[::10]/numOfLivePart.sum(), 'o', markerfacecolor='none', markeredgecolor='blue', markersize='5', label='Direct simulation')
# plt.title("Live particle distribution in time")
plt.xscale('log')
plt.yscale('log')
plt.xlim(0, t)
plt.ylim(min(numOfLivePart[:-2]/numOfLivePart.sum()), max(numOfLivePart/numOfLivePart.sum())+0.1*max(numOfLivePart/numOfLivePart.sum()))
plt.xlabel(r'$Time \quad [s]$')
plt.ylabel(r'$p_s \quad [-]$')
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='black')
plt.grid(True, which="minor", linestyle=':', linewidth=0.5, color='gray')
plt.legend(loc='best')
plt.tight_layout()