debug = False
if not debug:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from sklearn.linear_model import LinearRegression

save = False

# LOAD SECTION #################################################
loadCompareAdsD1 = np.load('compareAdsD1.npz')
for name, value in (loadCompareAdsD1.items()):
    globals()[name] = value
variableWidth = abs(timeLogSpaced-timeLogSpaced[::-1])/max(abs(timeLogSpaced-timeLogSpaced[::-1]))
timeTwoLogSpaced = np.cumsum(sim_time/sum(variableWidth)*variableWidth)
liveParticlesInTwoLogTime = np.sum(particleSteps[:, None] > timeTwoLogSpaced, axis=0)
liveParticlesInTwoLogTimeNorm = liveParticlesInTwoLogTime/sum(liveParticlesInTwoLogTime*np.diff(np.insert(timeTwoLogSpaced, 0, 0)))
# liveParticlesInTimeD1 = liveParticlesInTwoLogTime.copy()
# liveParticlesInTimeNormD1 = liveParticlesInTwoLogTimeNorm.copy()
liveParticlesInTimeD1 = liveParticlesInTime.copy()
liveParticlesInTimeNormD1 = liveParticlesInTimeNorm.copy()
# liveParticlesInTimeD1 = liveParticlesInTime.copy()
# liveParticlesInTimeNormD1 = liveParticlesInTimeNorm.copy()
tauD1 = (uby-lby)**2/Df

loadCompareAdsD01 = np.load('compareAdsD01.npz')
for name, value in (loadCompareAdsD01.items()):
    globals()[name] = value
variableWidth = abs(timeLogSpaced-timeLogSpaced[::-1])/max(abs(timeLogSpaced-timeLogSpaced[::-1]))
timeTwoLogSpaced = np.cumsum(sim_time/sum(variableWidth)*variableWidth)
liveParticlesInTwoLogTime = np.sum(particleSteps[:, None] > timeTwoLogSpaced, axis=0)
liveParticlesInTwoLogTimeNorm = liveParticlesInTwoLogTime/sum(liveParticlesInTwoLogTime*np.diff(np.insert(timeTwoLogSpaced, 0, 0)))
# liveParticlesInTimeD01 = liveParticlesInTwoLogTime.copy()
# liveParticlesInTimeNormD01 = liveParticlesInTwoLogTimeNorm.copy()
liveParticlesInTimeD01 = liveParticlesInTime.copy()
liveParticlesInTimeNormD01 = liveParticlesInTimeNorm.copy()
# liveParticlesInTimeD01 = liveParticlesInTime.copy()
# liveParticlesInTimeNormD01 = liveParticlesInTimeNorm.copy()
tauD01 = (uby-lby)**2/Df

loadCompareAdsD001 = np.load('compareAdsD001.npz')
for name, value in (loadCompareAdsD001.items()):
    globals()[name] = value
variableWidth = abs(timeLogSpaced-timeLogSpaced[::-1])/max(abs(timeLogSpaced-timeLogSpaced[::-1]))
timeTwoLogSpaced = np.cumsum(sim_time/sum(variableWidth)*variableWidth)
liveParticlesInTwoLogTime = np.sum(particleSteps[:, None] > timeTwoLogSpaced, axis=0)
liveParticlesInTwoLogTimeNorm = liveParticlesInTwoLogTime/sum(liveParticlesInTwoLogTime*np.diff(np.insert(timeTwoLogSpaced, 0, 0)))
# liveParticlesInTimeD001 = liveParticlesInTwoLogTime.copy()
# liveParticlesInTimeNormD001 = liveParticlesInTwoLogTimeNorm.copy()
liveParticlesInTimeD001 = liveParticlesInTime.copy()
liveParticlesInTimeNormD001 = liveParticlesInTimeNorm.copy()
# liveParticlesInTimeD001 = liveParticlesInTime.copy()
# liveParticlesInTimeNormD001 = liveParticlesInTimeNorm.copy()
tauD001 = (uby-lby)**2/Df

loadCompareAdsD0001 = np.load('compareAdsD0001.npz')
for name, value in (loadCompareAdsD0001.items()):
    globals()[name] = value
variableWidth = abs(timeLogSpaced-timeLogSpaced[::-1])/max(abs(timeLogSpaced-timeLogSpaced[::-1]))
timeTwoLogSpaced = np.cumsum(sim_time/sum(variableWidth)*variableWidth)
liveParticlesInTwoLogTime = np.sum(particleSteps[:, None] > timeTwoLogSpaced, axis=0)
liveParticlesInTwoLogTimeNorm = liveParticlesInTwoLogTime/sum(liveParticlesInTwoLogTime*np.diff(np.insert(timeTwoLogSpaced, 0, 0)))
# liveParticlesInTimeD0001 = liveParticlesInTwoLogTime.copy()
# liveParticlesInTimeNormD0001 = liveParticlesInTwoLogTimeNorm.copy()
liveParticlesInTimeD0001 = liveParticlesInTime.copy()
liveParticlesInTimeNormD0001 = liveParticlesInTimeNorm.copy()
# liveParticlesInTimeD0001 = liveParticlesInTime.copy()
# liveParticlesInTimeNormD0001 = liveParticlesInTimeNorm.copy()
tauD0001 = (uby-lby)**2/Df

# PLOT SECTION #################################################
# Distribution of live particles in time
survTimeDistCompareDiff = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
plt.plot(timeLinSpaced, liveParticlesInTimeD1/num_particles, label=r'$D_f = 1$', color='b', linestyle='-')
plt.plot(timeLinSpaced, liveParticlesInTimeD01/num_particles, label=r'$D_f = 0.1$', color='r', linestyle='-')
plt.plot(timeLinSpaced, liveParticlesInTimeD001/num_particles, label=r'$D_f = 0.01$', color='g', linestyle='-')
plt.plot(timeLinSpaced, liveParticlesInTimeD0001/num_particles, label=r'$D_f = 0.001$', color='purple', linestyle='-')
plt.title("Survival time distributions")
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$t$')
plt.ylabel(r'$N/N_0$')
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='black')
plt.grid(True, which="minor", linestyle=':', linewidth=0.5, color='gray')
plt.legend(loc='best')
plt.tight_layout()

# Normalised distribution of live particles in time and interpolation of the tail
survTimeDistCompareDiffNorm = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
plt.plot(timeLinSpaced, liveParticlesInTimeNormD1, 'o', markerfacecolor='none', markeredgecolor='blue', markersize='5', label=r'$D_f = 1$')
plt.plot(timeLinSpaced, liveParticlesInTimeNormD01, 'o', markerfacecolor='none', markeredgecolor='red', markersize='5', label=r'$D_f = 0.1$')
plt.plot(timeLinSpaced, liveParticlesInTimeNormD001, 'o', markerfacecolor='none', markeredgecolor='green', markersize='5', label=r'$D_f = 0.01$')
plt.plot(timeLinSpaced, liveParticlesInTimeNormD0001, 'o', markerfacecolor='none', markeredgecolor='purple', markersize='5', label=r'$D_f = 0.001$')
plt.title("Survival time distributions")
plt.yscale('log')
plt.xlabel(r'$t$')
plt.ylabel(r'$N/N_0$')
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='black')
plt.grid(True, which="minor", linestyle=':', linewidth=0.5, color='gray')
plt.legend(loc='best')
plt.tight_layout()
sliceDf1 = slice(5, 50)
sliceDf01 = slice(100, 300)
sliceDf001 = slice(300, 1000)
sliceDf0001 = slice(1000, 4000)
timeReshapedD1 = (timeLinSpaced[sliceDf1]).reshape(-1, 1)
timeReshapedD01 = (timeLinSpaced[sliceDf01]).reshape(-1, 1)
timeReshapedD001 = (timeLinSpaced[sliceDf001]).reshape(-1, 1)
timeReshapedD0001 = (timeLinSpaced[sliceDf0001]).reshape(-1, 1)
interpD1 = LinearRegression().fit(timeReshapedD1, np.log(liveParticlesInTimeNormD1[sliceDf1]))
interpD01 = LinearRegression().fit(timeReshapedD01, np.log(liveParticlesInTimeNormD01[sliceDf01]))
interpD001 = LinearRegression().fit(timeReshapedD001, np.log(liveParticlesInTimeNormD001[sliceDf001]))
interpD0001 = LinearRegression().fit(timeReshapedD0001, np.log(liveParticlesInTimeNormD0001[sliceDf0001]))
kInterpLinD1 = np.exp(interpD1.intercept_+interpD1.coef_*timeReshapedD1)
plt.plot(timeReshapedD1, kInterpLinD1, color='black', linewidth='2')
plt.text(timeReshapedD1[-1], kInterpLinD1[-1], f"k={interpD1.coef_[0]:.5f}", fontsize=18, ha='left', va='bottom')
kInterpLinD01 = np.exp(interpD01.intercept_+interpD01.coef_*timeReshapedD01)
plt.plot(timeReshapedD01, kInterpLinD01, color='black', linewidth='2')
plt.text(timeReshapedD01[-1], kInterpLinD01[-1], f"k={interpD01.coef_[0]:.5f}", fontsize=18, ha='left', va='bottom')
kInterpLinD001 = np.exp(interpD001.intercept_+interpD001.coef_*timeReshapedD001)
plt.plot(timeReshapedD001, kInterpLinD001, color='black', linewidth='2')
plt.text(timeReshapedD001[-1], kInterpLinD001[-1], f"k={interpD001.coef_[0]:.5f}", fontsize=18, ha='left', va='bottom')
kInterpLinD0001 = np.exp(interpD0001.intercept_+interpD0001.coef_*timeReshapedD0001)
plt.plot(timeReshapedD0001, kInterpLinD0001, color='black', linewidth='2')
plt.text(timeReshapedD0001[-1], kInterpLinD0001[-1], f"k={interpD0001.coef_[0]:.5f}", fontsize=18, ha='left', va='bottom')

# Rates of particles decay
derStep = 1
compareAdsRatesDiff = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
tDiff = np.diff(timeLinSpaced[::derStep])
dLivePartD1 = np.diff(np.log(liveParticlesInTimeD1[::derStep]))
dLivePartD01 = np.diff(np.log(liveParticlesInTimeD01[::derStep]))
dLivePartD001 = np.diff(np.log(liveParticlesInTimeD001[::derStep]))
dLivePartD0001 = np.diff(np.log(liveParticlesInTimeD0001[::derStep]))
dLivedtD1 = -dLivePartD1/tDiff
dLivedtD01 = -dLivePartD01/tDiff
dLivedtD001 = -dLivePartD001/tDiff
dLivedtD0001 = -dLivePartD0001/tDiff
midTimes = ((timeLinSpaced[::derStep])[:-1] + (timeLinSpaced[::derStep])[1:]) / 2
plt.plot(midTimes, dLivedtD1, 'o', markerfacecolor='none', markeredgecolor='blue', markersize='5', label=r'$D_f=1$') # , marker='+', linestyle='none', markersize='5')
plt.plot(midTimes, dLivedtD01, 'o', markerfacecolor='none', markeredgecolor='red', markersize='5', label=r'$D_f=0.1$') # , marker='+', linestyle='none', markersize='5')
plt.plot(midTimes, dLivedtD001, 'o', markerfacecolor='none', markeredgecolor='green', markersize='5', label=r'$D_f=0.01$') # , marker='+', linestyle='none', markersize='5')
plt.plot(midTimes, dLivedtD0001, 'o', markerfacecolor='none', markeredgecolor='purple', markersize='5', label=r'$D_f=0.001$') # , marker='+', linestyle='none', markersize='5')
plt.title("Effective reaction rates")
plt.xlabel(r'$t$')
plt.ylabel(r'$k(t)$')
plt.xscale('log')
plt.ylim(0, 0.2)
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='gray')
plt.legend(loc='best')
plt.tight_layout()
sliceDf1 = slice(5, 50)
sliceDf01 = slice(40, 200)
sliceDf001 = slice(100, 400)
sliceDf0001 = slice(200, 3000)
timeReshapedD1 = (midTimes[sliceDf1]).reshape(-1, 1)
timeReshapedD01 = (midTimes[sliceDf01]).reshape(-1, 1)
timeReshapedD001 = (midTimes[sliceDf001]).reshape(-1, 1)
timeReshapedD0001 = (midTimes[sliceDf0001]).reshape(-1, 1)
interpD1 = LinearRegression().fit(timeReshapedD1, dLivedtD1[sliceDf1])
interpD01 = LinearRegression().fit(timeReshapedD01, dLivedtD01[sliceDf01])
interpD001 = LinearRegression().fit(timeReshapedD001, dLivedtD001[sliceDf001])
interpD0001 = LinearRegression().fit(timeReshapedD0001, dLivedtD0001[sliceDf0001])
kInterpLinD1 = interpD1.intercept_+interpD1.coef_*midTimes[sliceDf1]
plt.plot(midTimes[sliceDf1], kInterpLinD1, color='black', linewidth='2')
plt.text(midTimes[sliceDf1][0], kInterpLinD1[0], f"k={interpD1.intercept_:.5f}", fontsize=18, ha='left', va='bottom')
kInterpLinD01 = interpD01.intercept_+interpD01.coef_*midTimes[sliceDf01]
plt.plot(midTimes[sliceDf01], kInterpLinD01, color='black', linewidth='2')
plt.text(midTimes[sliceDf01][0], kInterpLinD01[0], f"k={interpD01.intercept_:.5f}", fontsize=18, ha='left', va='bottom')
kInterpLinD001 = interpD001.intercept_+interpD001.coef_*midTimes[sliceDf001]
plt.plot(midTimes[sliceDf001], kInterpLinD001, color='black', linewidth='2')
plt.text(midTimes[sliceDf001][0], kInterpLinD001[0], f"k={interpD001.intercept_:.5f}", fontsize=18, ha='right', va='bottom')
kInterpLinD0001 = interpD0001.intercept_+interpD0001.coef_*midTimes[sliceDf0001]
plt.plot(midTimes[sliceDf0001], kInterpLinD0001, color='black', linewidth='2')
plt.text(midTimes[sliceDf0001][0], kInterpLinD0001[0], f"k={interpD0001.intercept_:.5f}", fontsize=18, ha='left', va='bottom')

# Rates of normalised particles decay
compareDiffNormAdsRates = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
# dLivedtD1 = np.diff(liveParticlesInTimeNormD1)/np.diff(timeLinSpaced/tauD1)
# dLivedtD01 = np.diff(liveParticlesInTimeNormD01)/np.diff(timeLinSpaced/tauD01)
# dLivedtD001 = np.diff(liveParticlesInTimeNormD001)/np.diff(timeLinSpaced/tauD001)
# dLivedtD0001 = np.diff(liveParticlesInTimeNormD0001)/np.diff(timeLinSpaced/tauD0001)
dLivedtD1 = -np.diff(np.log(liveParticlesInTimeNormD1)[::10])/np.diff(timeLinSpaced[::10]/tauD1)
dLivedtD01 = -np.diff(np.log(liveParticlesInTimeNormD01)[::10])/np.diff(timeLinSpaced[::10]/tauD01)
dLivedtD001 = -np.diff(np.log(liveParticlesInTimeNormD001)[::10])/np.diff(timeLinSpaced[::10]/tauD001)
dLivedtD0001 = -np.diff(np.log(liveParticlesInTimeNormD0001)[::10])/np.diff(timeLinSpaced[::10]/tauD0001)
midTimes = ((timeLinSpaced)[::10][:-1] + (timeLinSpaced)[::10][1:]) / 2
# maskD1 = dLivedtD1!=0
# maskD01 = dLivedtD01!=0
# maskD001 = dLivedtD001!=0
# maskD0001 = dLivedtD0001!=0
# validMask = np.isfinite(dLivedtD001)
# splineD001 = make_interp_spline(midTimes[validMask], dLivedtD001[validMask], k=3)
# dLivedtD001spline = splineD001(midTimes[::100])
plt.plot(midTimes/tauD1, dLivedtD1, 'o', markerfacecolor='none', markeredgecolor='blue', markersize='5', label=r'$\tau_d = 4$')
plt.plot(midTimes/tauD01, dLivedtD01, 'o', markerfacecolor='none', markeredgecolor='red', markersize='5', label=r'$\tau_d = 40$')
plt.plot(midTimes/tauD001, dLivedtD001, 'o', markerfacecolor='none', markeredgecolor='green', markersize='5', label=r'$\tau_d = 400$')
plt.plot(midTimes/tauD0001, dLivedtD0001, 'o', markerfacecolor='none', markeredgecolor='purple', markersize='5', label=r'$\tau_d = 4000$')
# plt.plot(midTimes[::100]/tauD001, dLivedtD001spline, color='k')
# plt.axhline(y=0.5, color='black', linestyle='-')
# plt.axhline(y=1.2, color='black', linestyle='-')
plt.title("Reaction rates from normalised surv time dist")
plt.xlabel(r'$Time/\tau_i$')
plt.ylabel('k(t)')
plt.xscale('log')
plt.yscale('log')
plt.xlim(0, 20)
plt.ylim(0, 5)
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='gray')
plt.legend(loc='best')
plt.tight_layout()

# SAVE SECTION #######################################################
if save:
    survTimeDistCompareDiff.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/survTimeDistCompareDiff.png", format="png", bbox_inches="tight")
    survTimeDistCompareDiffNorm.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/survTimeDistCompareDiffNorm.png", format="png", bbox_inches="tight")
    compareAdsRatesDiff.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/compareAdsRatesDiff.png", format="png", bbox_inches="tight")
    compareDiffNormAdsRates.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/compareDiffNormAdsRates.png", format="png", bbox_inches="tight")