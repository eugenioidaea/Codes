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
plt.plot(timeLinSpaced, liveParticlesInTimeD1, label=r'$D_f = 1$', color='b', linestyle='-')
plt.plot(timeLinSpaced, liveParticlesInTimeD01, label=r'$D_f = 0.1$', color='r', linestyle='-')
plt.plot(timeLinSpaced, liveParticlesInTimeD001, label=r'$D_f = 0.01$', color='g', linestyle='-')
plt.plot(timeLinSpaced, liveParticlesInTimeD0001, label=r'$D_f = 0.001$', color='purple', linestyle='-')
plt.title("Survival time distributions")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('t')
plt.ylabel('N')
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='black')
plt.grid(True, which="minor", linestyle=':', linewidth=0.5, color='gray')
plt.legend(loc='best')
plt.tight_layout()

# Normalised distribution of live particles in time and interpolation of the tail
survTimeDistCompareDiffNorm = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
# plt.plot(timeLinSpaced[::30]/tauD1, liveParticlesInTimeNormD1[::30], 'o', markerfacecolor='none', markeredgecolor='blue', markersize='5', label=r'$\tau_d = 4$')
# plt.plot(timeLinSpaced[::30]/tauD01, liveParticlesInTimeNormD01[::30], 'o', markerfacecolor='none', markeredgecolor='red', markersize='5', label=r'$\tau_d = 40$')
# plt.plot(timeLinSpaced[::30]/tauD001, liveParticlesInTimeNormD001[::30], 'o', markerfacecolor='none', markeredgecolor='green', markersize='5', label=r'$\tau_d = 400$')
# plt.plot(timeLinSpaced[::30]/tauD0001, liveParticlesInTimeNormD0001[::30], 'o', markerfacecolor='none', markeredgecolor='purple', markersize='5', label=r'$\tau_d = 4000$')
plt.plot(timeLinSpaced[::2], np.log(liveParticlesInTimeNormD1[::2]), 'o', markerfacecolor='none', markeredgecolor='blue', markersize='5', label=r'$\tau_d = 4$')
plt.plot(timeLinSpaced[::20], np.log(liveParticlesInTimeNormD01[::20]), 'o', markerfacecolor='none', markeredgecolor='red', markersize='5', label=r'$\tau_d = 40$')
plt.plot(timeLinSpaced[::100], np.log(liveParticlesInTimeNormD001[::100]), 'o', markerfacecolor='none', markeredgecolor='green', markersize='5', label=r'$\tau_d = 400$')
plt.plot(timeLinSpaced[::300], np.log(liveParticlesInTimeNormD0001[::300]), 'o', markerfacecolor='none', markeredgecolor='purple', markersize='5', label=r'$\tau_d = 4000$')
# yD1 = np.exp(-0.046-0.62*(timeLinSpaced))
yD1 = np.exp(-0.05-0.15*(timeLinSpaced))
plt.plot(timeLinSpaced, np.log(yD1), color='blue')
# yD01 = np.exp(-0.15-0.85*(timeLinSpaced))
yD01 = np.exp(-0.15-0.021*(timeLinSpaced))    
plt.plot(timeLinSpaced, np.log(yD01), color='red')
# yD001 = np.exp(-0.2-0.94*(timeLinSpaced))
yD001 = np.exp(-0.2-0.0023*(timeLinSpaced))
plt.plot(timeLinSpaced, np.log(yD001), color='green')
# yD0001 = np.exp(-0.21-0.97*(timeLinSpaced))
yD0001 = np.exp(-0.21-0.00024*(timeLinSpaced))
plt.plot(timeLinSpaced, np.log(yD0001), color='purple')
plt.title("Normalised survival time distribution")
# plt.xscale('log')
# plt.yscale('log')
plt.xlim(0, 8000)
plt.ylim(-12, 1)
plt.xlabel(r'$t$')
# plt.ylabel('Normalised number of live particles')
plt.ylabel(r'$ln(p_s)$')
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='black')
plt.grid(True, which="minor", linestyle=':', linewidth=0.5, color='gray')
plt.legend(loc='best')
plt.tight_layout()

timeReshapedD1 = (timeLinSpaced[30:80]).reshape(-1, 1)
model = LinearRegression()
model.fit(timeReshapedD1, np.log(liveParticlesInTimeNormD1[30:80]))
print(f"Coeff D1: {model.coef_}")
print(f"Intercept D1: {model.intercept_}")

timeReshapedD01 = (timeLinSpaced[200:300]).reshape(-1, 1)
model = LinearRegression()
model.fit(timeReshapedD01, np.log(liveParticlesInTimeNormD01[200:300]))
print(f"Coef D01: {model.coef_}")
print(f"Intercept D01: {model.intercept_}")

timeReshapedD001 = (timeLinSpaced[2500:3000]).reshape(-1, 1)
model = LinearRegression()
model.fit(timeReshapedD001, np.log(liveParticlesInTimeNormD001[2500:3000]))
print(f"Coeff D001: {model.coef_}")
print(f"Intercept D001: {model.intercept_}")

timeReshapedD0001 = (timeLinSpaced[5000:6000]).reshape(-1, 1)
model = LinearRegression()
model.fit(timeReshapedD0001, np.log(liveParticlesInTimeNormD0001[5000:6000]))
print(f"Coef D0001: {model.coef_}")
print(f"InterceptD0001: {model.intercept_}")

# Rates of particles decay
compareAdsRatesDiff = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
tDiff = np.diff(timeLogSpaced)
dLivePartD1 = np.diff(np.log(liveParticlesInTimeD1))
dLivePartD01 = np.diff(np.log(liveParticlesInTimeD01))
dLivePartD001 = np.diff(np.log(liveParticlesInTimeD001))
dLivePartD0001 = np.diff(np.log(liveParticlesInTimeD0001))
dLivedtD1 = dLivePartD1/tDiff
dLivedtD01 = dLivePartD01/tDiff
dLivedtD001 = dLivePartD001/tDiff
dLivedtD0001 = dLivePartD0001/tDiff
midTimes = ((timeLogSpaced)[:-1] + (timeLogSpaced)[1:]) / 2
maskD1 = dLivedtD1!=0
maskD01 = dLivedtD01!=0
maskD001 = dLivedtD001!=0
maskD0001 = dLivedtD0001!=0
plt.plot(midTimes[maskD1], dLivedtD1[maskD1], label='D=1', color='b') # , marker='+', linestyle='none', markersize='5')
plt.plot(midTimes[maskD01], dLivedtD01[maskD01], label='D=0.1', color='r') # , marker='+', linestyle='none', markersize='5')
plt.plot(midTimes[maskD001], dLivedtD001[maskD001], label='D=0.01', color='g') # , marker='+', linestyle='none', markersize='5')
plt.plot(midTimes[maskD0001], dLivedtD0001[maskD0001], label='D=0.001', color='purple') # , marker='+', linestyle='none', markersize='5')
plt.title("Effective reaction rate")
plt.xlabel('Time')
plt.ylabel('k(t)')
plt.xscale('log')
plt.ylim(-1, 0.1)
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='gray')
plt.legend(loc='best')
plt.tight_layout()

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