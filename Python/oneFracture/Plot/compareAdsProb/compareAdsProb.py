debug = False
if not debug:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from sklearn.linear_model import LinearRegression

save = False

loadCompareAdsP90 = np.load('compareAdsP90.npz')
for name, value in (loadCompareAdsP90.items()):
    globals()[name] = value
liveParticlesInTimeNormP90 = np.sort(numOfLivePart/num_particles)[::-1].copy()
TimeP90 = Time

loadCompareAdsP80 = np.load('compareAdsP80.npz')
for name, value in (loadCompareAdsP80.items()):
    globals()[name] = value
liveParticlesInTimeNormP80 = np.sort(numOfLivePart/num_particles)[::-1].copy()
TimeP80 = Time

loadCompareAdsP60 = np.load('compareAdsP60.npz')
for name, value in (loadCompareAdsP60.items()):
    globals()[name] = value
liveParticlesInTimeNormP60 = np.sort(numOfLivePart/num_particles)[::-1].copy()
TimeP60 = Time

loadCompareAdsP40 = np.load('compareAdsP40.npz')
for name, value in (loadCompareAdsP40.items()):
    globals()[name] = value
liveParticlesInTimeNormP40 = np.sort(numOfLivePart/num_particles)[::-1].copy()
TimeP40 = Time

loadCompareAdsP20 = np.load('compareAdsP20.npz')
for name, value in (loadCompareAdsP20.items()):
    globals()[name] = value
liveParticlesInTimeNormP20 = np.sort(numOfLivePart/num_particles)[::-1].copy()
TimeP20 = Time

loadCompareAdsP10 = np.load('compareAdsP10.npz')
for name, value in (loadCompareAdsP10.items()):
    globals()[name] = value
liveParticlesInTimeNormP10 = np.sort(numOfLivePart/num_particles)[::-1].copy()
TimeP10 = Time

tauP = (uby-lby)**2/Df

# Distribution of live particles in time
# survTimeDistCompareProb = plt.figure(figsize=(8, 8))
# plt.rcParams.update({'font.size': 20})
# plt.plot(Time, liveParticlesInTimeP80, label=r'$p=80$', color='b', linestyle='-')
# plt.plot(Time, liveParticlesInTimeP60, label=r'$p=60$', color='r', linestyle='-')
# plt.plot(Time, liveParticlesInTimeP40, label=r'$p=40$', color='g', linestyle='-')
# plt.title("Survival times")
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Time')
# plt.ylabel('Number of live particles')
# plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='black')
# plt.grid(True, which="minor", linestyle=':', linewidth=0.5, color='gray')
# plt.legend(loc='best')
# plt.tight_layout()

# Normalised distribution of live particles in time
#    survTimeDistCompareProbNorm = plt.figure(figsize=(8, 8))
#    plt.rcParams.update({'font.size': 20})
#    plt.plot(Time/tauP, liveParticlesInTimeNormP80, label=r'$p=80$', color='b', linestyle='-')
#    plt.plot(Time/tauP, liveParticlesInTimeNormP60, label=r'$p=60$', color='r', linestyle='-')
#    plt.plot(Time/tauP, liveParticlesInTimeNormP40, label=r'$p=40$', color='g', linestyle='-')
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
# survTimeDistCompareProbNorm = plt.figure(figsize=(8, 8))
# plt.rcParams.update({'font.size': 20})
# plt.plot(timeLogSpaced[::30]/tauP, liveParticlesInTimeNormP80[::30], 'o', markerfacecolor='none', markeredgecolor='blue', markersize='5', label=r'$p = 0.8$')
# plt.plot(timeLogSpaced[::30]/tauP, liveParticlesInTimeNormP60[::30], 'o', markerfacecolor='none', markeredgecolor='red', markersize='5', label=r'$p = 0.6$')
# plt.plot(timeLogSpaced[::30]/tauP, liveParticlesInTimeNormP40[::30], 'o', markerfacecolor='none', markeredgecolor='green', markersize='5', label=r'$p = 0.4$')
# # plt.plot(timeLogSpaced[::30]/tauD0001, liveParticlesInTimeNormD0001[::30], 'o', markerfacecolor='none', markeredgecolor='purple', markersize='5', label=r'$\tau_d = 4000$')
# # plt.plot(np.log(Time/tauD1), np.log(liveParticlesInTimeNormD1), label=r'$D_f = 1$', color='b', linestyle='-')
# # plt.plot(np.log(Time/tauD01), np.log(liveParticlesInTimeNormD01), label=r'$D_f = 0.1$', color='r', linestyle='-')
# # plt.plot(np.log(Time/tauD001), np.log(liveParticlesInTimeNormD001), label=r'$D_f = 0.01$', color='g', linestyle='-')
# yP08 = np.exp(-0.5*(timeLogSpaced/tauP))
# plt.plot(timeLogSpaced/tauP, yP08, color='black')
# plt.text((timeLogSpaced/tauP)[5000], yP08[5000], "k(t)=-0.5", fontsize=12) 
# yP04 = np.exp(-0.9*(timeLogSpaced/tauP))
# plt.plot(timeLogSpaced/tauP, yP04, color='black')
# plt.text((timeLogSpaced/tauP)[5000]*0.3, yP04[5000], "k(t)=-0.9", fontsize=12)
# plt.title("Normalised survival time distribution")
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(0.01, 30)
# plt.ylim(1e-6, 1)
# plt.xlabel(r'$Time/\tau_i$')
# plt.ylabel('Normalised number of live particles')
# plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='black')
# plt.grid(True, which="minor", linestyle=':', linewidth=0.5, color='gray')
# plt.legend(loc='best')
# plt.tight_layout()

# Normalised distribution of live particles in time and interpolation of the tail
survTimeDistCompareDiffNorm = plt.figure(figsize=(8, 8), dpi=300)
plt.rcParams.update({'font.size': 20})
sliceP90 = slice(0, sum(np.log(liveParticlesInTimeNormP90)>-10))
timeReshapedP90 = (TimeP90[sliceP90]).reshape(-1, 1)
yP90fit = LinearRegression().fit(timeReshapedP90, np.log(liveParticlesInTimeNormP90[sliceP90]))
sliceP80 = slice(0, sum(np.log(liveParticlesInTimeNormP80)>-10))
timeReshapedP80 = (TimeP80[sliceP80]).reshape(-1, 1)
yP80fit = LinearRegression().fit(timeReshapedP80, np.log(liveParticlesInTimeNormP80[sliceP80]))
sliceP60 = slice(0, sum(np.log(liveParticlesInTimeNormP60)>-10))
timeReshapedP60 = (TimeP60[sliceP60]).reshape(-1, 1)
yP60fit = LinearRegression().fit(timeReshapedP60, np.log(liveParticlesInTimeNormP60[sliceP60]))
sliceP40 = slice(0, sum(np.log(liveParticlesInTimeNormP40)>-10))
timeReshapedP40 = (TimeP40[sliceP40]).reshape(-1, 1)
yP40fit = LinearRegression().fit(timeReshapedP40, np.log(liveParticlesInTimeNormP40[sliceP40]))
sliceP20 = slice(0, sum(np.log(liveParticlesInTimeNormP20)>-10))
timeReshapedP20 = (TimeP20[sliceP20]).reshape(-1, 1)
yP20fit = LinearRegression().fit(timeReshapedP20, np.log(liveParticlesInTimeNormP20[sliceP20]))
sliceP10 = slice(0, sum(np.log(liveParticlesInTimeNormP10)>-10))
timeReshapedP10 = (TimeP10[sliceP10]).reshape(-1, 1)
yP10fit = LinearRegression().fit(timeReshapedP10, np.log(liveParticlesInTimeNormP10[sliceP10]))
plt.plot(TimeP90[::10], np.log(liveParticlesInTimeNormP90)[::10], 'o', markerfacecolor='none', markeredgecolor='pink', markersize='5', label=r'$p_{ads} = 0.9$')
plt.plot(TimeP80[::10], np.log(liveParticlesInTimeNormP80)[::10], 'o', markerfacecolor='none', markeredgecolor='blue', markersize='5', label=r'$p_{ads} = 0.8$')
plt.plot(TimeP60[::10], np.log(liveParticlesInTimeNormP60)[::10], 'o', markerfacecolor='none', markeredgecolor='red', markersize='5', label=r'$p_{ads} = 0.6$')
plt.plot(TimeP40[::10], np.log(liveParticlesInTimeNormP40)[::10], 'o', markerfacecolor='none', markeredgecolor='green', markersize='5', label=r'$p_{ads} = 0.4$')
plt.plot(TimeP20[::10], np.log(liveParticlesInTimeNormP20)[::10], 'o', markerfacecolor='none', markeredgecolor='purple', markersize='5', label=r'$p_{ads} = 0.2$')
plt.plot(TimeP10[::10], np.log(liveParticlesInTimeNormP10)[::10], 'o', markerfacecolor='none', markeredgecolor='orange', markersize='5', label=r'$p_{ads} = 0.1$')
yP90 = np.exp(yP90fit.intercept_+yP90fit.coef_[0]*(TimeP90))
plt.plot(TimeP90, np.log(yP90), color='pink')
plt.text(TimeP90[len(TimeP90)//2], np.log(yP90)[len(yP90)//2], r'$p_s = %g e^{%g t}$' % (round(np.exp(-yP90fit.intercept_), 2), round(yP90fit.coef_[0], 3)), fontsize=18, ha='left', va='bottom')
yP80 = np.exp(yP80fit.intercept_+yP80fit.coef_[0]*(TimeP80))
plt.plot(TimeP80, np.log(yP80), color='blue')
plt.text(TimeP80[len(TimeP80)//2+10], np.log(yP80)[len(yP80)//2+10], r'$p_s = %g e^{%g t}$' % (round(np.exp(-yP80fit.intercept_), 2), round(yP80fit.coef_[0], 3)), fontsize=18, ha='left', va='bottom')
yP60 = np.exp(yP60fit.intercept_+yP60fit.coef_[0]*(TimeP60))
plt.plot(TimeP60, np.log(yP60), color='red')
plt.text(TimeP60[len(TimeP60)//2+20], np.log(yP60)[len(yP60)//2+20], r'$p_s = %g e^{%g t}$' % (round(np.exp(-yP60fit.intercept_), 2), round(yP60fit.coef_[0], 3)), fontsize=18, ha='left', va='bottom')
yP40 = np.exp(yP40fit.intercept_+yP40fit.coef_[0]*(TimeP40))
plt.plot(TimeP40, np.log(yP40), color='green')
plt.text(TimeP40[len(TimeP40)//2+40], np.log(yP40)[len(yP40)//2+40], r'$p_s = %g e^{%g t}$' % (round(np.exp(-yP40fit.intercept_), 2), round(yP40fit.coef_[0], 3)), fontsize=18, ha='left', va='bottom')
yP20 = np.exp(yP20fit.intercept_+yP20fit.coef_[0]*(TimeP20))
plt.plot(TimeP20, np.log(yP20), color='purple')
plt.text(TimeP20[len(TimeP20)//2+90], np.log(yP20)[len(yP20)//2+90], r'$p_s = %g e^{%g t}$' % (round(np.exp(-yP20fit.intercept_), 2), round(yP20fit.coef_[0], 3)), fontsize=18, ha='left', va='bottom')
yP10 = np.exp(yP10fit.intercept_+yP10fit.coef_[0]*(TimeP10))
plt.plot(TimeP10, np.log(yP10), color='orange')
plt.text(TimeP10[len(TimeP10)//2+120], np.log(yP10)[len(yP10)//2+120], r'$p_s = %g e^{%g t}$' % (round(np.exp(-yP80fit.intercept_), 2), round(yP10fit.coef_[0], 3)), fontsize=18, ha='left', va='bottom')
# plt.title("Normalised survival time distribution")
# plt.xscale('log')
# plt.yscale('log')
plt.xlim(0, max(np.max(arr) for arr in [TimeP10, TimeP20, TimeP40, TimeP60, TimeP80, TimeP90]))
# plt.xlim(0, 250)
plt.ylim(-10, 0.1)
# plt.ylim(min(np.min(arr[~np.isneginf(arr)]) for arr in [np.log(liveParticlesInTimeNormP10), np.log(liveParticlesInTimeNormP20), np.log(liveParticlesInTimeNormP40), np.log(liveParticlesInTimeNormP60), np.log(liveParticlesInTimeNormP80), np.log(liveParticlesInTimeNormP90)]), 0.1)
plt.xlabel(r'$t$')
# plt.ylabel('Normalised number of live particles')
plt.ylabel(r'$ln(p_s)$')
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='black')
plt.grid(True, which="minor", linestyle=':', linewidth=0.5, color='gray')
plt.legend(loc='best')
plt.tight_layout()

# # Rates of particles decay
# compareAdsRatesProb = plt.figure(figsize=(8, 8))
# plt.rcParams.update({'font.size': 20})
# tDiff = np.diff(Time)
# dLivedtP80 = np.diff(np.log(liveParticlesInTimeP80))/tDiff
# dLivedtP60 = np.diff(np.log(liveParticlesInTimeP60))/tDiff
# dLivedtP40 = np.diff(np.log(liveParticlesInTimeP40))/tDiff
# midTimes = ((Time)[:-1] + (Time)[1:]) / 2
# plt.plot(midTimes, dLivedtP80, label='p=80', color='b') # , marker='+', linestyle='none', markersize='5')
# plt.plot(midTimes, dLivedtP60, label='p=60', color='r') # , marker='+', linestyle='none', markersize='5')
# plt.plot(midTimes, dLivedtP40, label='p=40', color='g') # , marker='+', linestyle='none', markersize='5')
# plt.title("Effective reaction rate")
# plt.xlabel('Time')
# plt.ylabel('k(t)')
# plt.xscale('log')
# plt.ylim(-0.2, 0.01)
# plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='gray')
# plt.legend(loc='best')
# plt.tight_layout()

#    compareProbNormAdsRates = plt.figure(figsize=(8, 8))
#    plt.rcParams.update({'font.size': 20})
#    dLivedtP80 = np.diff(np.log(liveParticlesInTimeNormP80))/np.diff(Time/tauP)
#    dLivedtP60 = np.diff(np.log(liveParticlesInTimeNormP60))/np.diff(Time/tauP)
#    dLivedtP40 = np.diff(np.log(liveParticlesInTimeNormP40))/np.diff(Time/tauP)
#    midTimes = ((Time)[:-1] + (Time)[1:]) / 2
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

# # Rates of normalised particles decay
# compareProbNormAdsRates = plt.figure(figsize=(8, 8))
# plt.rcParams.update({'font.size': 20})
# dLivedtP80 = -np.diff(np.log(liveParticlesInTimeNormP80))/np.diff(timeLogSpaced/tauP)
# dLivedtP60 = -np.diff(np.log(liveParticlesInTimeNormP60))/np.diff(timeLogSpaced/tauP)
# dLivedtP40 = -np.diff(np.log(liveParticlesInTimeNormP40))/np.diff(timeLogSpaced/tauP)
# # dLivedtD0001 = np.diff(np.log(liveParticlesInTimeNormD0001))/np.diff(timeLogSpaced/tauP)
# midTimes = ((timeLogSpaced)[:-1] + (timeLogSpaced)[1:]) / 2
# maskP80 = dLivedtP80!=0
# maskP60 = dLivedtP60!=0
# maskP40 = dLivedtP40!=0
# # maskD0001 = dLivedtD0001!=0
# # validMask = np.isfinite(dLivedtD001)
# # splineD001 = make_interp_spline(midTimes[validMask], dLivedtD001[validMask], k=3)
# # dLivedtD001spline = splineD001(midTimes[::100])
# plt.plot(midTimes[maskP80]/tauP, dLivedtP80[maskP80], 'o', markerfacecolor='none', markeredgecolor='blue', markersize='5', label=r'$p = 0.8$')
# plt.plot(midTimes[maskP60]/tauP, dLivedtP60[maskP60], 'o', markerfacecolor='none', markeredgecolor='red', markersize='5', label=r'$p = 0.6$')
# plt.plot(midTimes[maskP40]/tauP, dLivedtP40[maskP40], 'o', markerfacecolor='none', markeredgecolor='green', markersize='5', label=r'$p = 0.4$')
# # plt.plot(midTimes[maskD0001]/tauD0001, dLivedtD0001[maskD0001], label=r'$\tau_d = 4000$', color='purple') # , marker='+', linestyle='none', markersize='5')
# # plt.plot(midTimes[::100]/tauD001, dLivedtD001spline, color='k')
# plt.axhline(y=0.5, color='black', linestyle='-')
# plt.axhline(y=0.9, color='black', linestyle='-')
# plt.title("Reaction rates from normalised surv time dist")
# plt.xlabel(r'$Time/\tau_i$')
# plt.ylabel('k(t)')
# plt.xscale('log')
# # plt.yscale('log')
# plt.xlim(0.01, 10)
# plt.ylim(0.01, 30)
# plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='gray')
# plt.legend(loc='best')
# plt.tight_layout()

kt = np.abs([yP10fit.coef_[0], yP20fit.coef_[0], yP40fit.coef_[0], yP60fit.coef_[0], yP80fit.coef_[0], yP90fit.coef_[0]])
reactionVsDifferentP = plt.figure(figsize=(8, 8), dpi=300)
plt.plot([0.9], kt[5], 'o', markerfacecolor='pink', markeredgecolor='pink', markersize='10', label=r'$k(p_{ads})=%g$' % (-yP90fit.coef_[0]))
plt.plot([0.8], kt[4], 'o', markerfacecolor='blue', markeredgecolor='blue', markersize='10', label=r'$k(p_{ads})=%g$' % (-yP80fit.coef_[0]))
plt.plot([0.6], kt[3], 'o', markerfacecolor='red', markeredgecolor='red', markersize='10', label=r'$k(p_{ads})=%g$' % (-yP60fit.coef_[0]))
plt.plot([0.4], kt[2], 'o', markerfacecolor='green', markeredgecolor='green', markersize='10', label=r'$k(p_{ads})=%g$' % (-yP40fit.coef_[0]))
plt.plot([0.2], kt[1], 'o', markerfacecolor='purple', markeredgecolor='purple', markersize='10', label=r'$k(p_{ads})=%g$' % (-yP20fit.coef_[0]))
plt.plot([0.1], kt[0], 'o', markerfacecolor='orange', markeredgecolor='orange', markersize='10', label=r'$k(p_{ads})=%g$' % (-yP10fit.coef_[0]))
# plt.title("Reaction rates vs adsorption probability")
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(0, 20)
# plt.ylim(-10, 1)
plt.xlabel(r'$p_{ads}$')
# plt.ylabel('Normalised number of live particles')
plt.ylabel(r'$k(t)$')
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='black')
plt.grid(True, which="minor", linestyle=':', linewidth=0.5, color='gray')
pAdsReshaped = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9]).reshape(-1, 1)
yKfit = LinearRegression().fit(pAdsReshaped, kt)
xInterp = np.linspace(min(pAdsReshaped), max(pAdsReshaped), 10)
yInterp = xInterp*yKfit.coef_[0]+yKfit.intercept_
plt.plot(xInterp, yInterp, color="black")
plt.text(xInterp[len(xInterp)//2], yInterp[len(yInterp)//2], f"y={yKfit.coef_[0]:.5f}x+{yKfit.intercept_:.5f}", fontsize=18, ha='left', va='top')

# coefficients = np.polyfit([0.1, 0.2, 0.4, 0.6, 0.8], kt, 2)
# poly = np.poly1d(coefficients)
# yInterp2 = poly(xInterp)
# plt.plot(xInterp, yInterp2, label='2nd order polynomial', color='blue')
# plt.legend(loc='best')
# plt.tight_layout()

finalPositionMatrixDecay = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
plt.plot(x, y, 'g*')
# plt.plot(xK001, yK001, 'r*')
# plt.plot(xK01, yK01, 'b*')
plt.plot([xInit, xInit], [lby, uby], color='yellow', linewidth=2)
plt.axhline(y=uby, color='r', linestyle='--', linewidth=1)
plt.axhline(y=lby, color='r', linestyle='--', linewidth=1)

if save:
    survTimeDistCompareProb.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/survTimeDistCompareProb.png", format="png", bbox_inches="tight")
    survTimeDistCompareProbNorm.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/survTimeDistCompareProbNorm.png", format="png", bbox_inches="tight")
    compareAdsRatesProb.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/compareAdsRatesProb.png", format="png", bbox_inches="tight")
    compareProbNormAdsRates.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/compareProbNormAdsRates.png", format="png", bbox_inches="tight")