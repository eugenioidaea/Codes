debug = False
if not debug:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from sklearn.linear_model import LinearRegression

save = False

loadCompareAdsP80 = np.load('compareAdsP80.npz')
for name, value in (loadCompareAdsP80.items()):
    globals()[name] = value
liveParticlesInTimeP80 = liveParticlesInTime.copy()
liveParticlesInTimeNormP80 = liveParticlesInTimeNorm.copy()
tauP = (uby-lby)**2/Df

loadCompareAdsP60 = np.load('compareAdsP60.npz')
for name, value in (loadCompareAdsP60.items()):
    globals()[name] = value
liveParticlesInTimeP60 = liveParticlesInTime.copy()
liveParticlesInTimeNormP60 = liveParticlesInTimeNorm.copy()
tauP = (uby-lby)**2/Df

loadCompareAdsP40 = np.load('compareAdsP40.npz')
for name, value in (loadCompareAdsP40.items()):
    globals()[name] = value
liveParticlesInTimeP40 = liveParticlesInTime.copy()
liveParticlesInTimeNormP40 = liveParticlesInTimeNorm.copy()
tauP = (uby-lby)**2/Df

loadCompareAdsP20 = np.load('compareAdsP20.npz')
for name, value in (loadCompareAdsP20.items()):
    globals()[name] = value
liveParticlesInTimeP20 = liveParticlesInTime.copy()
liveParticlesInTimeNormP20 = liveParticlesInTimeNorm.copy()
tauP = (uby-lby)**2/Df

# Distribution of live particles in time
survTimeDistCompareProb = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
plt.plot(timeLinSpaced, liveParticlesInTimeP80, label=r'$p=80$', color='b', linestyle='-')
plt.plot(timeLinSpaced, liveParticlesInTimeP60, label=r'$p=60$', color='r', linestyle='-')
plt.plot(timeLinSpaced, liveParticlesInTimeP40, label=r'$p=40$', color='g', linestyle='-')
plt.title("Survival times")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Time')
plt.ylabel('Number of live particles')
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='black')
plt.grid(True, which="minor", linestyle=':', linewidth=0.5, color='gray')
plt.legend(loc='best')
plt.tight_layout()

# Normalised distribution of live particles in time
#    survTimeDistCompareProbNorm = plt.figure(figsize=(8, 8))
#    plt.rcParams.update({'font.size': 20})
#    plt.plot(timeLinSpaced/tauP, liveParticlesInTimeNormP80, label=r'$p=80$', color='b', linestyle='-')
#    plt.plot(timeLinSpaced/tauP, liveParticlesInTimeNormP60, label=r'$p=60$', color='r', linestyle='-')
#    plt.plot(timeLinSpaced/tauP, liveParticlesInTimeNormP40, label=r'$p=40$', color='g', linestyle='-')
#    plt.title("Survival time distribution")
#    plt.xscale('log')
#    plt.yscale('log')
#    plt.xlabel(r'$Time/tau_i$')
#    plt.ylabel('Normalised number of live particles')
#    plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='black')
#    plt.grid(True, which="minor", linestyle=':', linewidth=0.5, color='gray')
#    plt.legend(loc='best')
#    plt.tight_layout()

# Normalised distribution of live particles in time
survTimeDistCompareProbNorm = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
plt.plot(timeLogSpaced[::30]/tauP, liveParticlesInTimeNormP80[::30], 'o', markerfacecolor='none', markeredgecolor='blue', markersize='5', label=r'$p = 0.8$')
plt.plot(timeLogSpaced[::30]/tauP, liveParticlesInTimeNormP60[::30], 'o', markerfacecolor='none', markeredgecolor='red', markersize='5', label=r'$p = 0.6$')
plt.plot(timeLogSpaced[::30]/tauP, liveParticlesInTimeNormP40[::30], 'o', markerfacecolor='none', markeredgecolor='green', markersize='5', label=r'$p = 0.4$')
# plt.plot(timeLogSpaced[::30]/tauD0001, liveParticlesInTimeNormD0001[::30], 'o', markerfacecolor='none', markeredgecolor='purple', markersize='5', label=r'$\tau_d = 4000$')
# plt.plot(np.log(timeLinSpaced/tauD1), np.log(liveParticlesInTimeNormD1), label=r'$D_f = 1$', color='b', linestyle='-')
# plt.plot(np.log(timeLinSpaced/tauD01), np.log(liveParticlesInTimeNormD01), label=r'$D_f = 0.1$', color='r', linestyle='-')
# plt.plot(np.log(timeLinSpaced/tauD001), np.log(liveParticlesInTimeNormD001), label=r'$D_f = 0.01$', color='g', linestyle='-')
yP08 = np.exp(-0.5*(timeLogSpaced/tauP))
plt.plot(timeLogSpaced/tauP, yP08, color='black')
plt.text((timeLogSpaced/tauP)[5000], yP08[5000], "k(t)=-0.5", fontsize=12) 
yP04 = np.exp(-0.9*(timeLogSpaced/tauP))
plt.plot(timeLogSpaced/tauP, yP04, color='black')
plt.text((timeLogSpaced/tauP)[5000]*0.3, yP04[5000], "k(t)=-0.9", fontsize=12)
plt.title("Normalised survival time distribution")
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.01, 30)
plt.ylim(1e-6, 1)
plt.xlabel(r'$Time/\tau_i$')
plt.ylabel('Normalised number of live particles')
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='black')
plt.grid(True, which="minor", linestyle=':', linewidth=0.5, color='gray')
plt.legend(loc='best')
plt.tight_layout()

# Normalised distribution of live particles in time and interpolation of the tail
survTimeDistCompareDiffNorm = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
# plt.plot(timeLinSpaced[::20]/tauP, np.log(liveParticlesInTimeNormP80[::20]), 'o', markerfacecolor='none', markeredgecolor='blue', markersize='5', label=r'$P = 0.8$')
# plt.plot(timeLinSpaced[::20]/tauP, np.log(liveParticlesInTimeNormP60[::20]), 'o', markerfacecolor='none', markeredgecolor='red', markersize='5', label=r'$P = 0.6$')
# plt.plot(timeLinSpaced[::20]/tauP, np.log(liveParticlesInTimeNormP40[::20]), 'o', markerfacecolor='none', markeredgecolor='green', markersize='5', label=r'$P = 0.4$')
# plt.plot(timeLinSpaced[::20]/tauP, np.log(liveParticlesInTimeNormP20[::20]), 'o', markerfacecolor='none', markeredgecolor='purple', markersize='5', label=r'$P = 0.2$')
# yP80 = np.exp(-0.14-0.77*(timeLinSpaced/tauP))
# plt.plot(timeLinSpaced/tauP, np.log(yP80), color='blue')
# yP60 = np.exp(-0.11-0.69*(timeLinSpaced/tauP))
# plt.plot(timeLinSpaced/tauP, np.log(yP60), color='red')
# # plt.text((timeLinSpaced/tauD01)[5000], yD01[5000], "k(t)=-0.5", fontsize=12) 
# yP40 = np.exp(-0.078-0.55*(timeLinSpaced/tauP))
# plt.plot(timeLinSpaced/tauP, np.log(yP40), color='green')
# # plt.text((timeLinSpaced/tauD001)[7000]*0.3, yD001[7000], "k(t)=-1.2", fontsize=12)
# yP20 = np.exp(-0.016-0.35*(timeLinSpaced/tauP))
# plt.plot(timeLinSpaced/tauP, np.log(yP20), color='purple')
plt.plot(timeLinSpaced[::20], np.log(liveParticlesInTimeNormP80[::20]), 'o', markerfacecolor='none', markeredgecolor='blue', markersize='5', label=r'$P = 0.8$')
plt.plot(timeLinSpaced[::20], np.log(liveParticlesInTimeNormP60[::20]), 'o', markerfacecolor='none', markeredgecolor='red', markersize='5', label=r'$P = 0.6$')
plt.plot(timeLinSpaced[::20], np.log(liveParticlesInTimeNormP40[::20]), 'o', markerfacecolor='none', markeredgecolor='green', markersize='5', label=r'$P = 0.4$')
plt.plot(timeLinSpaced[::20], np.log(liveParticlesInTimeNormP20[::20]), 'o', markerfacecolor='none', markeredgecolor='purple', markersize='5', label=r'$P = 0.2$')
yP80 = np.exp(-0.14-0.019*(timeLinSpaced))
plt.plot(timeLinSpaced, np.log(yP80), color='blue')
yP60 = np.exp(-0.11-0.017*(timeLinSpaced))
plt.plot(timeLinSpaced, np.log(yP60), color='red')
# plt.text((timeLinSpaced/tauD01)[5000], yD01[5000], "k(t)=-0.5", fontsize=12) 
yP40 = np.exp(-0.078-0.014*(timeLinSpaced))
plt.plot(timeLinSpaced, np.log(yP40), color='green')
# plt.text((timeLinSpaced/tauD001)[7000]*0.3, yD001[7000], "k(t)=-1.2", fontsize=12)
yP20 = np.exp(-0.016-0.0087*(timeLinSpaced))
plt.plot(timeLinSpaced, np.log(yP20), color='purple')
plt.title("Normalised survival time distribution")
# plt.xscale('log')
# plt.yscale('log')
plt.xlim(0, 1500)
plt.ylim(-12, 1)
plt.xlabel(r'$t$')
# plt.ylabel('Normalised number of live particles')
plt.ylabel(r'$ln(p_s)$')
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='black')
plt.grid(True, which="minor", linestyle=':', linewidth=0.5, color='gray')
plt.legend(loc='best')
plt.tight_layout()

timeReshapedP80 = (timeLinSpaced[200:400]).reshape(-1, 1)
model = LinearRegression()
model.fit(timeReshapedP80, np.log(liveParticlesInTimeNormP80[200:400]))
print(f"Coeff P80: {model.coef_}")
print(f"Intercept P80: {model.intercept_}")

timeReshapedP60 = (timeLinSpaced[200:400]).reshape(-1, 1)
model = LinearRegression()
model.fit(timeReshapedP60, np.log(liveParticlesInTimeNormP60[200:400]))
print(f"Coef P60: {model.coef_}")
print(f"Intercept P60: {model.intercept_}")

timeReshapedP40 = (timeLinSpaced[200:400]).reshape(-1, 1)
model = LinearRegression()
model.fit(timeReshapedP40, np.log(liveParticlesInTimeNormP40[200:400]))
print(f"Coeff P40: {model.coef_}")
print(f"Intercept P40: {model.intercept_}")

timeReshapedP20 = (timeLinSpaced[200:400]).reshape(-1, 1)
model = LinearRegression()
model.fit(timeReshapedP20, np.log(liveParticlesInTimeNormP20[200:400]))
print(f"Coeff P20: {model.coef_}")
print(f"Intercept P20: {model.intercept_}")

# Rates of particles decay
compareAdsRatesProb = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
tDiff = np.diff(timeLinSpaced)
dLivedtP80 = np.diff(np.log(liveParticlesInTimeP80))/tDiff
dLivedtP60 = np.diff(np.log(liveParticlesInTimeP60))/tDiff
dLivedtP40 = np.diff(np.log(liveParticlesInTimeP40))/tDiff
midTimes = ((timeLinSpaced)[:-1] + (timeLinSpaced)[1:]) / 2
plt.plot(midTimes, dLivedtP80, label='p=80', color='b') # , marker='+', linestyle='none', markersize='5')
plt.plot(midTimes, dLivedtP60, label='p=60', color='r') # , marker='+', linestyle='none', markersize='5')
plt.plot(midTimes, dLivedtP40, label='p=40', color='g') # , marker='+', linestyle='none', markersize='5')
plt.title("Effective reaction rate")
plt.xlabel('Time')
plt.ylabel('k(t)')
plt.xscale('log')
plt.ylim(-0.2, 0.01)
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='gray')
plt.legend(loc='best')
plt.tight_layout()

#    compareProbNormAdsRates = plt.figure(figsize=(8, 8))
#    plt.rcParams.update({'font.size': 20})
#    dLivedtP80 = np.diff(np.log(liveParticlesInTimeNormP80))/np.diff(timeLinSpaced/tauP)
#    dLivedtP60 = np.diff(np.log(liveParticlesInTimeNormP60))/np.diff(timeLinSpaced/tauP)
#    dLivedtP40 = np.diff(np.log(liveParticlesInTimeNormP40))/np.diff(timeLinSpaced/tauP)
#    midTimes = ((timeLinSpaced)[:-1] + (timeLinSpaced)[1:]) / 2
#    plt.plot(midTimes/tauP, dLivedtP80, label='p=80', color='b') # , marker='+', linestyle='none', markersize='5')
#    plt.plot(midTimes/tauP, dLivedtP60, label='p=60', color='r') # , marker='+', linestyle='none', markersize='5')
#    plt.plot(midTimes/tauP, dLivedtP40, label='p=40', color='g') # , marker='+', linestyle='none', markersize='5')
#    plt.title("Reaction rates from normalised surv time dist")
#    plt.xlabel(r'$Time/tau_i$')
#    plt.ylabel('k(t)')
#    plt.xscale('log')
#    plt.ylim(-8, 0.01)
#    plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='gray')
#    plt.legend(loc='best')
#    plt.tight_layout()

# Rates of normalised particles decay
compareProbNormAdsRates = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
dLivedtP80 = -np.diff(np.log(liveParticlesInTimeNormP80))/np.diff(timeLogSpaced/tauP)
dLivedtP60 = -np.diff(np.log(liveParticlesInTimeNormP60))/np.diff(timeLogSpaced/tauP)
dLivedtP40 = -np.diff(np.log(liveParticlesInTimeNormP40))/np.diff(timeLogSpaced/tauP)
# dLivedtD0001 = np.diff(np.log(liveParticlesInTimeNormD0001))/np.diff(timeLogSpaced/tauP)
midTimes = ((timeLogSpaced)[:-1] + (timeLogSpaced)[1:]) / 2
maskP80 = dLivedtP80!=0
maskP60 = dLivedtP60!=0
maskP40 = dLivedtP40!=0
# maskD0001 = dLivedtD0001!=0
# validMask = np.isfinite(dLivedtD001)
# splineD001 = make_interp_spline(midTimes[validMask], dLivedtD001[validMask], k=3)
# dLivedtD001spline = splineD001(midTimes[::100])
plt.plot(midTimes[maskP80]/tauP, dLivedtP80[maskP80], 'o', markerfacecolor='none', markeredgecolor='blue', markersize='5', label=r'$p = 0.8$')
plt.plot(midTimes[maskP60]/tauP, dLivedtP60[maskP60], 'o', markerfacecolor='none', markeredgecolor='red', markersize='5', label=r'$p = 0.6$')
plt.plot(midTimes[maskP40]/tauP, dLivedtP40[maskP40], 'o', markerfacecolor='none', markeredgecolor='green', markersize='5', label=r'$p = 0.4$')
# plt.plot(midTimes[maskD0001]/tauD0001, dLivedtD0001[maskD0001], label=r'$\tau_d = 4000$', color='purple') # , marker='+', linestyle='none', markersize='5')
# plt.plot(midTimes[::100]/tauD001, dLivedtD001spline, color='k')
plt.axhline(y=0.5, color='black', linestyle='-')
plt.axhline(y=0.9, color='black', linestyle='-')
plt.title("Reaction rates from normalised surv time dist")
plt.xlabel(r'$Time/\tau_i$')
plt.ylabel('k(t)')
plt.xscale('log')
# plt.yscale('log')
plt.xlim(0.01, 10)
plt.ylim(0.01, 30)
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='gray')
plt.legend(loc='best')
plt.tight_layout()

reactionVsDifferentP = plt.figure(figsize=(8, 8))
plt.plot([0.8], [0.019], 'o', markerfacecolor='blue', markeredgecolor='blue', markersize='10') #, label=r'$tau_d = 4$')
plt.plot([0.6], [0.017], 'o', markerfacecolor='red', markeredgecolor='red', markersize='10') #, label=r'$tau_d = 40$')
plt.plot([0.4], [0.014], 'o', markerfacecolor='green', markeredgecolor='green', markersize='10') #, label=r'$tau_d = 400$')
plt.plot([0.2], [0.0087], 'o', markerfacecolor='purple', markeredgecolor='purple', markersize='10') #, label=r'$tau_d = 4000$')
plt.title("Reaction rates vs adsorption probability")
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(0, 20)
# plt.ylim(-10, 1)
plt.xlabel(r'$P_{ads}$')
# plt.ylabel('Normalised number of live particles')
plt.ylabel(r'$k(t)$')
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='black')
plt.grid(True, which="minor", linestyle=':', linewidth=0.5, color='gray')
plt.legend(loc='best')
plt.tight_layout()

if save:
    survTimeDistCompareProb.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/survTimeDistCompareProb.png", format="png", bbox_inches="tight")
    survTimeDistCompareProbNorm.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/survTimeDistCompareProbNorm.png", format="png", bbox_inches="tight")
    compareAdsRatesProb.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/compareAdsRatesProb.png", format="png", bbox_inches="tight")
    compareProbNormAdsRates.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/compareProbNormAdsRates.png", format="png", bbox_inches="tight")