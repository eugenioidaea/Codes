debug = False
if not debug:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Choose what should be plotted #############################################################

plotMatrixDecay = True

save = True

# Load simulation results from .npz files ###################################################
if plotMatrixDecay:
    loadMatrixDecayK005 = np.load('matrixDecayK005.npz')
    for name, value in (loadMatrixDecayK005.items()):
        globals()[name] = value
    numOfLivePartK005 = numOfLivePart.copy()
    timeK005 = Time.copy()
    xK005 = x.copy()
    yK005 = y.copy()

    loadMatrixDecayK010 = np.load('matrixDecayK010.npz')
    for name, value in (loadMatrixDecayK010.items()):
        globals()[name] = value
    numOfLivePartK010 = numOfLivePart.copy()
    timeK010 = Time.copy()
    xK010 = x.copy()
    yK010 = y.copy()

# Plot section #########################################################################
finalPositionMatrixDecay = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
plt.plot(xK005, yK005, 'b*')
plt.plot(xK010, yK010, 'r*')
plt.plot([xInit, xInit], [lby, uby], color='yellow', linewidth=2)
plt.axhline(y=uby, color='r', linestyle='--', linewidth=1)
plt.axhline(y=lby, color='r', linestyle='--', linewidth=1)

survTimeDistMatrixDecay = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
plt.plot(timeK005, numOfLivePartK005/num_particles, 'o', markerfacecolor='none', markeredgecolor='blue', markersize='5', label=r'$k_{decay} = 0.05$')
plt.plot(timeK010, numOfLivePartK010/num_particles, 'o', markerfacecolor='none', markeredgecolor='red', markersize='5', label=r'$k_{decay} = 0.10$')
# plt.plot(Time60, numOfLivePartP60/num_particles, 'o', markerfacecolor='none', markeredgecolor='red', markersize='5', label=r'$p = 0.6$')
# plt.plot(Time40, numOfLivePartP40/num_particles, 'o', markerfacecolor='none', markeredgecolor='green', markersize='5', label=r'$p = 0.4$')
plt.title("Survival time distributions")
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$t$')
plt.ylabel(r'$N/N_0$')
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='black')
plt.grid(True, which="minor", linestyle=':', linewidth=0.5, color='gray')
# plt.xlim(0, 200)
plt.legend(loc='best')
plt.tight_layout()
# logBins = np.logspace(np.log10(dt), np.log10(timeK005.max()), len(timeK005))
# binIndeces = np.digitize(timeK005, logBins)
# numOfLivePartLog = np.array([numOfLivePartK005[binIndeces == i].mean() for i in range(0, len(timeK005))])
# plt.rcParams.update({'font.size': 20})
# plt.plot(logBins, numOfLivePartLog/num_particles, 'o', markerfacecolor='none', markeredgecolor='red', markersize='5', label=r'$\tau_d = 40$')

derStep = 10
ratesMatrixDecay = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
midTimesK005 = ((timeK005[::derStep])[:-1] + (timeK005[::derStep])[1:]) / 2
dLivedtK005 = -np.diff(np.log(numOfLivePartK005[::derStep]))/np.diff(timeK005[::derStep])
midTimesK010 = ((timeK010[::derStep])[:-1] + (timeK010[::derStep])[1:]) / 2
dLivedtK010 = -np.diff(np.log(numOfLivePartK010[::derStep]))/np.diff(timeK010[::derStep])
plt.plot(midTimesK005, dLivedtK005, 'o', markerfacecolor='none', markeredgecolor='blue', markersize='5', label=r'$k_{decay}=0.05$')
plt.plot(midTimesK010, dLivedtK010, 'o', markerfacecolor='none', markeredgecolor='red', markersize='5', label=r'$k_{decay}=0.10$')
plt.title("Reaction rates")
plt.xlabel(r'$t$')
plt.ylabel('k(t)')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(0, 20)
# plt.ylim(0, 0.1)
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='gray')
plt.legend(loc='best')
plt.tight_layout()
sliceDecayK005 = slice(50, 100)
sliceDecayK010 = slice(10, 50)
timeReshapedK005 = (midTimesK005[sliceDecayK005]).reshape(-1, 1)
timeReshapedK010 = (midTimesK010[sliceDecayK010]).reshape(-1, 1)
interpK005 = LinearRegression().fit(timeReshapedK005, dLivedtK005[sliceDecayK005])
interpK010 = LinearRegression().fit(timeReshapedK010, dLivedtK010[sliceDecayK010])
kInterpLinK005 = interpK005.intercept_+interpK005.coef_*midTimesK005[sliceDecayK005]
plt.plot(midTimesK005[sliceDecayK005], kInterpLinK005, color='black', linewidth='2')
plt.text(midTimesK005[sliceDecayK005][0], kInterpLinK005[0], f"k={interpK005.intercept_:.5f}", fontsize=18, ha='right', va='top')
kInterpLinK010 = interpK010.intercept_+interpK010.coef_*midTimesK010[sliceDecayK010]
plt.plot(midTimesK010[sliceDecayK010], kInterpLinK010, color='black', linewidth='2')
plt.text(midTimesK010[sliceDecayK010][0], kInterpLinK010[0], f"k={interpK010.intercept_:.5f}", fontsize=18, ha='right', va='top')

survTimeDistSemilogMatrixdecay = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
plt.plot(timeK005, numOfLivePartK005/num_particles, 'o', markerfacecolor='none', markeredgecolor='blue', markersize='5', label=r'$k_{decay} = 0.05$')
plt.plot(timeK010, numOfLivePartK010/num_particles, 'o', markerfacecolor='none', markeredgecolor='red', markersize='5', label=r'$k_{decay} = 0.10$')
plt.title("Survival time distributions")
plt.yscale('log')
plt.xlabel(r'$t$')
plt.ylabel(r'$N/N_0$')
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='black')
plt.grid(True, which="minor", linestyle=':', linewidth=0.5, color='gray')
plt.legend(loc='best')
plt.tight_layout()
sliceSemilogK005 = slice(200, 400)
sliceSemilogK010 = slice(200, 400)
timeReshapedSemilogK005 = (timeK005[sliceSemilogK005]).reshape(-1, 1)
timeReshapedSemilogK010 = (timeK005[sliceSemilogK010]).reshape(-1, 1)
interpSemilogK005 = LinearRegression().fit(timeReshapedSemilogK005, np.log(numOfLivePartK005[sliceSemilogK005]/num_particles))
interpSemilogK010 = LinearRegression().fit(timeReshapedSemilogK010, np.log(numOfLivePartK010[sliceSemilogK010]/num_particles))
kInterpSemilogK005 = np.exp(interpSemilogK005.intercept_+interpSemilogK005.coef_*timeReshapedSemilogK005)
plt.plot(timeReshapedSemilogK005, kInterpSemilogK005, color='black', linewidth='2')
plt.text(timeReshapedSemilogK005[-1], kInterpSemilogK005[-1], f"k={interpSemilogK005.coef_[0]:.5f}", fontsize=18, ha='right', va='top')
kInterpSemilogK010 = np.exp(interpSemilogK010.intercept_+interpSemilogK010.coef_*timeReshapedSemilogK010)
plt.plot(timeReshapedSemilogK010, kInterpSemilogK010, color='black', linewidth='2')
plt.text(timeReshapedSemilogK010[-1], kInterpSemilogK010[-1], f"k={interpSemilogK010.coef_[0]:.5f}", fontsize=18, ha='right', va='top')

if plotMatrixDecay & save:
    finalPositionMatrixDecay.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/finalPositionMatrixDecay.png", format="png", bbox_inches="tight")
    survTimeDistMatrixDecay.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/survTimeDistMatrixDecay.png", format="png", bbox_inches="tight")
    ratesMatrixDecay.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/ratesMatrixDecay.png", format="png", bbox_inches="tight")
    survTimeDistSemilogMatrixdecay.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/survTimeDistSemilogMatrixdecay.png", format="png", bbox_inches="tight")