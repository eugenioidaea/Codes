debug = False
if not debug:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.interpolate import CubicSpline

# Choose what should be plotted #############################################################

plotMatrixDecay = True

save = False

# Load simulation results from .npz files ###################################################
if plotMatrixDecay:
    loadMatrixDecayK01 = np.load('matrixDecayK01.npz')
    for name, value in (loadMatrixDecayK01.items()):
        globals()[name] = value
    numOfLivePartK01 = numOfLivePart.copy()
    timeK01 = Time.copy()
    xK01 = x.copy()
    yK01 = y.copy()

    loadMatrixDecayK001 = np.load('matrixDecayK001.npz')
    for name, value in (loadMatrixDecayK001.items()):
        globals()[name] = value
    numOfLivePartK001 = numOfLivePart.copy()
    timeK001 = Time.copy()
    xK001 = x.copy()
    yK001 = y.copy()

    loadMatrixDecayK0001 = np.load('matrixDecayK0001.npz')
    for name, value in (loadMatrixDecayK0001.items()):
        globals()[name] = value
    numOfLivePartK0001 = numOfLivePart.copy()
    timeK0001 = Time.copy()
    xK0001 = x.copy()
    yK0001 = y.copy()

    loadDomainDecayK01 = np.load('domainDecayK01.npz')
    for name, value in (loadDomainDecayK01.items()):
        globals()[name] = value
    numOfLivePartDomainK01 = numOfLivePart.copy()
    timeDomainK01 = Time.copy()

    loadDomainDecayK001 = np.load('domainDecayK001.npz')
    for name, value in (loadDomainDecayK001.items()):
        globals()[name] = value
    numOfLivePartDomainK001 = numOfLivePart.copy()
    timeDomainK001 = Time.copy()

    loadDomainDecayK0001 = np.load('domainDecayK0001.npz')
    for name, value in (loadDomainDecayK0001.items()):
        globals()[name] = value
    numOfLivePartDomainK0001 = numOfLivePart.copy()
    timeDomainK0001 = Time.copy()

# Plot section #########################################################################
finalPositionMatrixDecay = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
plt.plot(xK0001, yK0001, 'g*')
plt.plot(xK001, yK001, 'r*')
plt.plot(xK01, yK01, 'b*')
plt.plot([xInit, xInit], [lby, uby], color='yellow', linewidth=2)
plt.axhline(y=uby, color='r', linestyle='--', linewidth=1)
plt.axhline(y=lby, color='r', linestyle='--', linewidth=1)

survTimeDistMatrixDecay = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
plt.plot(timeK01, numOfLivePartK01/num_particles, 'o', markerfacecolor='none', markeredgecolor='blue', markersize='5', label=r'$k_{matrixDecay} = 0.1$')
plt.plot(timeK001, numOfLivePartK001/num_particles, 'o', markerfacecolor='none', markeredgecolor='red', markersize='5', label=r'$k_{matrixDecay} = 0.01$')
plt.plot(timeK0001, numOfLivePartK0001/num_particles, 'o', markerfacecolor='none', markeredgecolor='green', markersize='5', label=r'$k_{matrixDecay} = 0.001$')
plt.plot(timeDomainK01, numOfLivePartDomainK01/num_particles, color='blue', linestyle='-', label=r'$k_{domainDecay}=0.1$')
plt.plot(timeDomainK001, numOfLivePartDomainK001/num_particles, color='red', linestyle='-', label=r'$k_{domainDecay}=0.01$')
plt.plot(timeDomainK0001, numOfLivePartDomainK0001/num_particles, color='green', linestyle='-', label=r'$k_{domainDecay}=0.001$')
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
# logBins = np.logspace(np.log10(dt), np.log10(timeK01.max()), len(timeK01))
# binIndeces = np.digitize(timeK01, logBins)
# numOfLivePartLog = np.array([numOfLivePartK01[binIndeces == i].mean() for i in range(0, len(timeK01))])
# plt.rcParams.update({'font.size': 20})
# plt.plot(logBins, numOfLivePartLog/num_particles, 'o', markerfacecolor='none', markeredgecolor='red', markersize='5', label=r'$\tau_d = 40$')

derStep = 10
ratesMatrixDecay = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
midTimesK01 = ((timeK01[::derStep])[:-1] + (timeK01[::derStep])[1:]) / 2
dLivedtK01 = -np.diff(np.log(numOfLivePartK01[::derStep]))/np.diff(timeK01[::derStep])
midTimesK001 = ((timeK001[::derStep])[:-1] + (timeK001[::derStep])[1:]) / 2
dLivedtK001 = -np.diff(np.log(numOfLivePartK001[::derStep]))/np.diff(timeK001[::derStep])
midTimesK0001 = ((timeK0001[::derStep])[:-1] + (timeK0001[::derStep])[1:]) / 2
dLivedtK0001 = -np.diff(np.log(numOfLivePartK0001[::derStep]))/np.diff(timeK0001[::derStep])
plt.plot(midTimesK01, dLivedtK01, 'o', markerfacecolor='none', markeredgecolor='blue', markersize='5', label=r'$k_{matrixDecay}=0.1$')
plt.plot(midTimesK001, dLivedtK001, 'o', markerfacecolor='none', markeredgecolor='red', markersize='5', label=r'$k_{matrixDecay}=0.01$')
plt.plot(midTimesK0001, dLivedtK0001, 'o', markerfacecolor='none', markeredgecolor='green', markersize='5', label=r'$k_{matrixDecay}=0.001$')
plt.title("Reaction rates from 10-spaced derivative")
plt.xlabel(r'$t$')
plt.ylabel(r'$k(t)$')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(0, 20)
# plt.ylim(0, 0.1)
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='gray')
plt.legend(loc='best')
plt.tight_layout()
sliceDecayK01 = slice(50, 100)
sliceDecayK001 = slice(100, 200)
sliceDecayK0001 = slice(200, 300)
timeReshapedK01 = (midTimesK01[sliceDecayK01]).reshape(-1, 1)
timeReshapedK001 = (midTimesK001[sliceDecayK001]).reshape(-1, 1)
timeReshapedK0001 = (midTimesK0001[sliceDecayK0001]).reshape(-1, 1)
interpK01 = LinearRegression().fit(timeReshapedK01, dLivedtK01[sliceDecayK01])
interpK001 = LinearRegression().fit(timeReshapedK001, dLivedtK001[sliceDecayK001])
interpK0001 = LinearRegression().fit(timeReshapedK0001, dLivedtK0001[sliceDecayK0001])
kInterpLinK01 = interpK01.intercept_+interpK01.coef_*midTimesK01[sliceDecayK01]
plt.plot(midTimesK01[sliceDecayK01], kInterpLinK01, color='black', linewidth='2')
plt.text(midTimesK01[sliceDecayK01][0], kInterpLinK01[0], f"k={interpK01.intercept_:.5f}", fontsize=18, ha='left', va='bottom')
kInterpLinK001 = interpK001.intercept_+interpK001.coef_*midTimesK001[sliceDecayK001]
plt.plot(midTimesK001[sliceDecayK001], kInterpLinK001, color='black', linewidth='2')
plt.text(midTimesK001[sliceDecayK001][0], kInterpLinK001[0], f"k={interpK001.intercept_:.5f}", fontsize=18, ha='left', va='bottom')
kInterpLinK0001 = interpK0001.intercept_+interpK0001.coef_*midTimesK0001[sliceDecayK0001]
plt.plot(midTimesK0001[sliceDecayK0001], kInterpLinK0001, color='black', linewidth='2')
plt.text(midTimesK0001[sliceDecayK0001][0], kInterpLinK0001[0], f"k={interpK0001.intercept_:.5f}", fontsize=18, ha='left', va='bottom')

survTimeDistSemilogMatrixdecay = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
plt.plot(timeK01, numOfLivePartK01/num_particles, 'o', markerfacecolor='none', markeredgecolor='blue', markersize='5', label=r'$k_{matrixDecay} = 0.1$')
plt.plot(timeK001, numOfLivePartK001/num_particles, 'o', markerfacecolor='none', markeredgecolor='red', markersize='5', label=r'$k_{matrixDecay} = 0.01$')
plt.plot(timeK0001, numOfLivePartK0001/num_particles, 'o', markerfacecolor='none', markeredgecolor='green', markersize='5', label=r'$k_{matrixDecay} = 0.001$')
plt.title("Survival time distributions")
plt.yscale('log')
plt.xlabel(r'$t$')
plt.ylabel(r'$N/N_0$')
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='black')
plt.grid(True, which="minor", linestyle=':', linewidth=0.5, color='gray')
plt.legend(loc='best')
plt.tight_layout()
sliceSemilogK01 = slice(500, 1000)
sliceSemilogK001 = slice(1000, 2000)
sliceSemilogK0001 = slice(2000, 3000)
timeReshapedSemilogK01 = (timeK01[sliceSemilogK01]).reshape(-1, 1)
timeReshapedSemilogK001 = (timeK001[sliceSemilogK001]).reshape(-1, 1)
timeReshapedSemilogK0001 = (timeK0001[sliceSemilogK0001]).reshape(-1, 1)
interpSemilogK01 = LinearRegression().fit(timeReshapedSemilogK01, np.log(numOfLivePartK01[sliceSemilogK01]/num_particles))
interpSemilogK001 = LinearRegression().fit(timeReshapedSemilogK001, np.log(numOfLivePartK001[sliceSemilogK001]/num_particles))
interpSemilogK0001 = LinearRegression().fit(timeReshapedSemilogK0001, np.log(numOfLivePartK0001[sliceSemilogK0001]/num_particles))
kInterpSemilogK01 = np.exp(interpSemilogK01.intercept_+interpSemilogK01.coef_*timeReshapedSemilogK01)
plt.plot(timeReshapedSemilogK01, kInterpSemilogK01, color='black', linewidth='4')
plt.text(timeReshapedSemilogK01[-1], kInterpSemilogK01[-1], f"k={interpSemilogK01.coef_[0]:.5f}", fontsize=18, ha='left', va='top')
kInterpSemilogK001 = np.exp(interpSemilogK001.intercept_+interpSemilogK001.coef_*timeReshapedSemilogK001)
plt.plot(timeReshapedSemilogK001, kInterpSemilogK001, color='black', linewidth='4')
plt.text(timeReshapedSemilogK001[-1], kInterpSemilogK001[-1], f"k={interpSemilogK001.coef_[0]:.5f}", fontsize=18, ha='left', va='top')
kInterpSemilogK0001 = np.exp(interpSemilogK0001.intercept_+interpSemilogK0001.coef_*timeReshapedSemilogK0001)
plt.plot(timeReshapedSemilogK0001, kInterpSemilogK0001, color='black', linewidth='4')
plt.text(timeReshapedSemilogK0001[-1], kInterpSemilogK0001[-1], f"k={interpSemilogK0001.coef_[0]:.5f}", fontsize=18, ha='left', va='top')

spStep = 10
splineK01 = CubicSpline(timeK01, numOfLivePartK01/num_particles)
xSplineK01 = np.linspace(timeK01.min(), timeK01[-2], spStep)
ySplineK01 = splineK01(xSplineK01)
plt.plot(xSplineK01, ySplineK01, color='yellow')
splineK001 = CubicSpline(timeK001, numOfLivePartK001/num_particles)
xSplineK001 = np.linspace(timeK001.min(), timeK001[-2], spStep)
ySplineK001 = splineK001(xSplineK001)
plt.plot(xSplineK001, ySplineK001, color='yellow')
splineK0001 = CubicSpline(timeK0001, numOfLivePartK0001/num_particles)
xSplineK0001 = np.linspace(timeK0001.min(), timeK0001[-2], spStep)
ySplineK0001 = splineK0001(xSplineK0001)
plt.plot(xSplineK0001, ySplineK0001, color='yellow')

derStep = 1
ratesMatrixDecaySpline = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
midTimesK01 = ((xSplineK01[::derStep])[:-1] + (xSplineK01[::derStep])[1:]) / 2
dLivedtK01 = -np.diff(np.log(ySplineK01[::derStep]))/np.diff(xSplineK01[::derStep])
midTimesK001 = ((xSplineK001[::derStep])[:-1] + (xSplineK001[::derStep])[1:]) / 2
dLivedtK001 = -np.diff(np.log(ySplineK001[::derStep]))/np.diff(xSplineK001[::derStep])
midTimesK0001 = ((xSplineK0001[::derStep])[:-1] + (xSplineK0001[::derStep])[1:]) / 2
dLivedtK0001 = -np.diff(np.log(ySplineK0001[::derStep]))/np.diff(xSplineK0001[::derStep])
plt.plot(midTimesK01, dLivedtK01, '-', markerfacecolor='none', markeredgecolor='blue', markersize='5', label=r'$k_{matrixDecay}=0.1$')
plt.plot(midTimesK001, dLivedtK001, '-', markerfacecolor='none', markeredgecolor='red', markersize='5', label=r'$k_{matrixDecay}=0.01$')
plt.plot(midTimesK0001, dLivedtK0001, '-', markerfacecolor='none', markeredgecolor='green', markersize='5', label=r'$k_{matrixDecay}=0.001$')
plt.title("Reaction rates from spline derivation")
plt.xlabel(r'$t$')
plt.ylabel(r'$k(t)$')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(0, 20)
# plt.ylim(0, 0.1)
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='gray')
plt.legend(loc='best')
plt.tight_layout()

logBins = 50
survTimeDistLogBinsMatrixdecay = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
timeLogSpacedK01 = np.logspace(timeK01.min(), np.log10(timeK01.max()), logBins)-1 # Logarithmically spaced bins
timeLogSpacedK001 = np.logspace(timeK001.min(), np.log10(timeK001.max()), logBins)-1 # Logarithmically spaced bins
timeLogSpacedK0001 = np.logspace(timeK0001.min(), np.log10(timeK0001.max()), logBins)-1 # Logarithmically spaced bins
timeIndexK01 = np.array([np.sum(timeK01[i]>timeLogSpacedK01) for i in range(len(timeK01))])
timeIndexK001 = np.array([np.sum(timeK001[i]>timeLogSpacedK001) for i in range(len(timeK001))])
timeIndexK0001 = np.array([np.sum(timeK0001[i]>timeLogSpacedK0001) for i in range(len(timeK0001))])
# timeIndex = np.digitize(timeK01, timeLogSpaced)
timeLogMeanK01 = np.array([timeK01[timeIndexK01 == i].mean() for i in range(len(timeLogSpacedK01))])
timeLogMeanK001 = np.array([timeK001[timeIndexK001 == i].mean() for i in range(len(timeLogSpacedK001))])
timeLogMeanK0001 = np.array([timeK0001[timeIndexK0001 == i].mean() for i in range(len(timeLogSpacedK0001))])
numPartLogMeanK01 = np.array([numOfLivePartK01[timeIndexK01 == i].mean() for i in range(len(timeLogSpacedK01))])
numPartLogMeanK001 = np.array([numOfLivePartK001[timeIndexK001 == i].mean() for i in range(len(timeLogSpacedK001))])
numPartLogMeanK0001 = np.array([numOfLivePartK0001[timeIndexK0001 == i].mean() for i in range(len(timeLogSpacedK0001))])
plt.plot(timeLogMeanK01, numPartLogMeanK01/num_particles, 'o', markerfacecolor='none', markeredgecolor='blue', markersize='5', label=r'$k_{matrixDecay} = 0.1$')
plt.plot(timeLogMeanK001, numPartLogMeanK001/num_particles, 'o', markerfacecolor='none', markeredgecolor='red', markersize='5', label=r'$k_{matrixDecay} = 0.01$')
plt.plot(timeLogMeanK0001, numPartLogMeanK0001/num_particles, 'o', markerfacecolor='none', markeredgecolor='green', markersize='5', label=r'$k_{matrixDecay} = 0.001$')
for i, (xi, yi) in enumerate(zip(timeLogMeanK01[-4:], (numPartLogMeanK01/num_particles)[-4:])):
    plt.text(xi, yi, f'({xi:.1f}; {yi:.6f})', fontsize=12, ha='left', va='bottom')
plt.yscale('log')
plt.title("Surv time dist with log-binning")
plt.xlabel(r'$t$')
plt.ylabel(r'$k(t)$')

derStep = 1
ratesMatrixDecayLog = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
midTimesK01 = ((timeLogMeanK01[::derStep])[:-1] + (timeLogMeanK01[::derStep])[1:]) / 2
dLivedtK01 = -np.diff(np.log(numPartLogMeanK01[::derStep]))/np.diff(timeLogMeanK01[::derStep])
midTimesK001 = ((timeLogMeanK001[::derStep])[:-1] + (timeLogMeanK001[::derStep])[1:]) / 2
dLivedtK001 = -np.diff(np.log(numPartLogMeanK001[::derStep]))/np.diff(timeLogMeanK001[::derStep])
midTimesK0001 = ((timeLogMeanK0001[::derStep])[:-1] + (timeLogMeanK0001[::derStep])[1:]) / 2
dLivedtK0001 = -np.diff(np.log(numPartLogMeanK0001[::derStep]))/np.diff(timeLogMeanK0001[::derStep])
plt.plot(midTimesK01, dLivedtK01, '-', markerfacecolor='none', markeredgecolor='blue', markersize='5', label=r'$k_{matrixDecay}=0.1$')
plt.plot(midTimesK001, dLivedtK001, '-', markerfacecolor='none', markeredgecolor='red', markersize='5', label=r'$k_{matrixDecay}=0.01$')
plt.plot(midTimesK0001, dLivedtK0001, '-', markerfacecolor='none', markeredgecolor='green', markersize='5', label=r'$k_{matrixDecay}=0.001$')
plt.title("Reaction rates from log-binning")
plt.xlabel(r'$t$')
plt.ylabel(r'$k(t)$')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(0, 20)
# plt.ylim(0, 0.1)
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='gray')
plt.legend(loc='best')
plt.tight_layout()

effVsDecayRates = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
plt.plot([0.1], interpSemilogK01.coef_[0], 'o', markerfacecolor='blue', markeredgecolor='blue', markersize='10')
plt.plot([0.01], interpSemilogK001.coef_[0], 'o', markerfacecolor='red', markeredgecolor='red', markersize='10')
plt.plot([0.001], interpSemilogK0001.coef_[0], 'o', markerfacecolor='green', markeredgecolor='green', markersize='10')
plt.title("Effective rates vs radioactive decays")
plt.xlabel(r'$k_{matrixDecay}$')
plt.ylabel(r'$k_{eff}$')
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='gray')
plt.legend(loc='best')
plt.tight_layout()

if plotMatrixDecay & save:
    finalPositionMatrixDecay.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/finalPositionMatrixDecay.png", format="png", bbox_inches="tight")
    survTimeDistMatrixDecay.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/survTimeDistMatrixDecay.png", format="png", bbox_inches="tight")
    ratesMatrixDecay.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/ratesMatrixDecay.png", format="png", bbox_inches="tight")
    survTimeDistSemilogMatrixdecay.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/survTimeDistSemilogMatrixdecay.png", format="png", bbox_inches="tight")
    ratesMatrixDecaySpline.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/ratesMatrixDecaySpline.png", format="png", bbox_inches="tight")
    survTimeDistLogBinsMatrixdecay.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/survTimeDistLogBinsMatrixdecay.png", format="png", bbox_inches="tight")
    ratesMatrixDecayLog.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/ratesMatrixDecayLog.png", format="png", bbox_inches="tight")
    effVsDecayRates.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/effVsDecayRates.png", format="png", bbox_inches="tight")