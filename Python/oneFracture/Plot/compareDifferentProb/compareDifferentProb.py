debug = False
if not debug:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from sklearn.linear_model import LinearRegression

save = False

loadCompareP80 = np.load('compareP80.npz', allow_pickle=True)
for name, value in (loadCompareP80.items()):
    globals()[name] = value
numOfLivePartP80 = numOfLivePart.copy()
Time80 = Time.copy()
tau80 = (uby-lby)**2/Df

loadCompareP60 = np.load('compareP60.npz', allow_pickle=True)
for name, value in (loadCompareP60.items()):
    globals()[name] = value
numOfLivePartP60 = numOfLivePart.copy()
Time60 = Time.copy()
tau60 = (uby-lby)**2/Df

loadCompareP40 = np.load('compareP40.npz', allow_pickle=True)
for name, value in (loadCompareP40.items()):
    globals()[name] = value
numOfLivePartP40 = numOfLivePart.copy()
Time40 = Time.copy()
tau40 = (uby-lby)**2/Df

# Particles' survival distribution and reaction rate for different adsorption probability ###############################
survTimeDistCompareAdsProb = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
plt.plot(Time80, numOfLivePartP80/num_particles, 'o', markerfacecolor='none', markeredgecolor='blue', markersize='5', label=r'$p = 0.8$')
plt.plot(Time60, numOfLivePartP60/num_particles, 'o', markerfacecolor='none', markeredgecolor='red', markersize='5', label=r'$p = 0.6$')
plt.plot(Time40, numOfLivePartP40/num_particles, 'o', markerfacecolor='none', markeredgecolor='green', markersize='5', label=r'$p = 0.4$')
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
# logBins = np.logspace(np.log10(dt), np.log10(Time.max()), len(Time))
# binIndeces = np.digitize(Time, logBins)
# numOfLivePartLog = np.array([numOfLivePart[binIndeces == i].mean() for i in range(0, len(Time))])
# plt.rcParams.update({'font.size': 20})
# plt.plot(logBins, numOfLivePartLog/num_particles, 'o', markerfacecolor='none', markeredgecolor='red', markersize='5', label=r'$\tau_d = 40$')

ratesCompareProb = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
dLivedtP80 = -np.diff(np.log(numOfLivePartP80/num_particles))/np.diff(Time80)
dLivedtP60 = -np.diff(np.log(numOfLivePartP60/num_particles))/np.diff(Time60)
dLivedtP40 = -np.diff(np.log(numOfLivePartP40/num_particles))/np.diff(Time40)
mask0p80 = dLivedtP80!=0
mask0p60 = dLivedtP60!=0
mask0p40 = dLivedtP40!=0
midTimes80 = ((Time80)[:-1] + (Time80)[1:]) / 2
midTimes60 = ((Time60)[:-1] + (Time60)[1:]) / 2
midTimes40 = ((Time40)[:-1] + (Time40)[1:]) / 2
plt.plot(midTimes80[mask0p80], dLivedtP80[mask0p80], 'o', markerfacecolor='none', markeredgecolor='blue', markersize='5', label=r'$p = 0.8$')
plt.plot(midTimes60[mask0p60], dLivedtP60[mask0p60], 'o', markerfacecolor='none', markeredgecolor='red', markersize='5', label=r'$p = 0.6$')
plt.plot(midTimes40[mask0p40], dLivedtP40[mask0p40], 'o', markerfacecolor='none', markeredgecolor='green', markersize='5', label=r'$p = 0.4$')
plt.title("Reaction rates")
plt.xlabel(r'$t$')
plt.ylabel('k(t)')
plt.xscale('log')
plt.yscale('log')
# plt.xlim(0, 20)
# plt.ylim(0, 200)
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='gray')
plt.legend(loc='best')
plt.tight_layout()
# maskNaN = np.isfinite(numOfLivePartLog)
# dLiveLogdt = -np.diff(np.log(numOfLivePartLog[maskNaN]/num_particles))/np.diff(np.log(Time[maskNaN]))
# midLogTimes = ((logBins)[maskNaN][:-1] + (logBins)[maskNaN][1:]) / 2
# plt.plot(midLogTimes, dLiveLogdt, 'o', markerfacecolor='none', markeredgecolor='red', markersize='5', label=r'$\tau_d = 40$')
sliceP80 = slice(10, 100)
sliceP60 = slice(20, 200)
sliceP40 = slice(40, 400)
timeReshapedP80 = (midTimes80[mask0p80][sliceP80]).reshape(-1, 1)
timeReshapedP60 = (midTimes60[mask0p60][sliceP60]).reshape(-1, 1)
timeReshapedP40 = (midTimes40[mask0p40][sliceP40]).reshape(-1, 1)
interpP80 = LinearRegression().fit(timeReshapedP80, dLivedtP80[mask0p80][sliceP80])
interpP60 = LinearRegression().fit(timeReshapedP60, dLivedtP60[mask0p60][sliceP60])
interpP40 = LinearRegression().fit(timeReshapedP40, dLivedtP40[mask0p40][sliceP40])
# timeLogReshaped = (midLogTimes[:100]).reshape(-1, 1)
# modelLog = LinearRegression()
# modelLog.fit(timeLogReshaped, dLiveLogdt[:100])
# print(f"Coeff D1: {modelLog.coef_}")
# print(f"Intercept D1: {modelLog.intercept_}")
kInterpLinP80 = interpP80.intercept_+interpP80.coef_*midTimes80[mask0p80][sliceP80]
plt.plot(midTimes80[mask0p80][sliceP80], kInterpLinP80, color='black', linewidth='2')
plt.text(midTimes80[mask0p80][sliceP80][0], kInterpLinP80[0], f"k={interpP80.intercept_:.2f}", fontsize=18, ha='right', va='top')
kInterpLinP60 = interpP60.intercept_+interpP60.coef_*midTimes60[mask0p60][sliceP60]
plt.plot(midTimes60[mask0p60][sliceP60], kInterpLinP60, color='black', linewidth='2')
plt.text(midTimes60[mask0p60][sliceP60][0], kInterpLinP60[0], f"k={interpP60.intercept_:.2f}", fontsize=18, ha='right', va='top')
kInterpLinP40 = interpP40.intercept_+interpP40.coef_*midTimes40[mask0p40][sliceP40]
plt.plot(midTimes40[mask0p40][sliceP40], kInterpLinP40, color='black', linewidth='2')
plt.text(midTimes40[mask0p40][sliceP40][0], kInterpLinP40[0], f"k={interpP40.intercept_:.2f}", fontsize=18, ha='right', va='top')
# kInterpLog = modelLog.intercept_+modelLog.coef_*midLogTimes
# plt.plot(midLogTimes, kInterpLog, color='red')

survTimeDistSemilogProb = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
plt.plot(Time80, numOfLivePartP80/num_particles, 'o', markerfacecolor='none', markeredgecolor='blue', markersize='5', label=r'$p = 0.8$')
plt.plot(Time60, numOfLivePartP60/num_particles, 'o', markerfacecolor='none', markeredgecolor='red', markersize='5', label=r'$p = 0.6$')
plt.plot(Time40, numOfLivePartP40/num_particles, 'o', markerfacecolor='none', markeredgecolor='green', markersize='5', label=r'$p = 0.4$')
plt.title("Survival time distributions")
# plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$t$')
plt.ylabel(r'$N/N_0$')
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='black')
plt.grid(True, which="minor", linestyle=':', linewidth=0.5, color='gray')
# plt.xlim(0, 200)
plt.legend(loc='best')
plt.tight_layout()
sliceSemilogP80 = slice(200, 400)
sliceSemilogP60 = slice(400, 600)
sliceSemilogP40 = slice(500, 700)
timeReshapedSemilogP80 = (Time80[sliceSemilogP80]).reshape(-1, 1)
timeReshapedSemilogP60 = (Time60[sliceSemilogP60]).reshape(-1, 1)
timeReshapedSemilogP40 = (Time40[sliceSemilogP40]).reshape(-1, 1)
interpSemilogP80 = LinearRegression().fit(timeReshapedSemilogP80, np.log(numOfLivePartP80[sliceSemilogP80]/num_particles))
interpSemilogP60 = LinearRegression().fit(timeReshapedSemilogP60, np.log(numOfLivePartP60[sliceSemilogP60]/num_particles))
interpSemilogP40 = LinearRegression().fit(timeReshapedSemilogP40, np.log(numOfLivePartP40[sliceSemilogP40]/num_particles))
kInterpSemilogP80 = np.exp(interpSemilogP80.intercept_+interpSemilogP80.coef_*timeReshapedSemilogP80)
plt.plot(timeReshapedSemilogP80, kInterpSemilogP80, color='black', linewidth='2')
plt.text(timeReshapedSemilogP80[-1], kInterpSemilogP80[-1], f"k={interpSemilogP80.coef_[0]:.2f}", fontsize=18, ha='right', va='top')
kInterpSemilogP60 = np.exp(interpSemilogP60.intercept_+interpSemilogP60.coef_*timeReshapedSemilogP60)
plt.plot(timeReshapedSemilogP60, kInterpSemilogP60, color='black', linewidth='2')
plt.text(timeReshapedSemilogP60[-1], kInterpSemilogP60[-1], f"k={interpSemilogP60.coef_[0]:.2f}", fontsize=18, ha='left')
kInterpSemilogP40 = np.exp(interpSemilogP40.intercept_+interpSemilogP40.coef_*timeReshapedSemilogP40)
plt.plot(timeReshapedSemilogP40, kInterpSemilogP40, color='black', linewidth='2')
plt.text(timeReshapedSemilogP40[-1], kInterpSemilogP40[-1], f"k={interpSemilogP40.coef_[0]:.2f}", fontsize=18, ha='left')

reactionVsProb = plt.figure(figsize=(8, 8))
plt.plot([80], -interpSemilogP80.coef_[0], 'o', markerfacecolor='blue', markeredgecolor='blue', markersize='10') #, label=r'$tau_d = 4$')
plt.plot([60], -interpSemilogP60.coef_[0], 'o', markerfacecolor='red', markeredgecolor='red', markersize='10') #, label=r'$tau_d = 40$')
plt.plot([40], -interpSemilogP40.coef_[0], 'o', markerfacecolor='green', markeredgecolor='green', markersize='10') #, label=r'$tau_d = 400$')
plt.title("Reaction rates vs adsorption probability")
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(0, 20)
# plt.ylim(-10, 1)
plt.xlabel(r'$p_{ads}$')
# plt.ylabel('Normalised number of live particles')
plt.ylabel(r'$k(p_{ads})$')
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='black')
plt.grid(True, which="minor", linestyle=':', linewidth=0.5, color='gray')
plt.legend(loc='best')
plt.tight_layout()

if save:
    survTimeDistCompareAdsProb.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/survTimeDistCompareAdsProb.png", format="png", bbox_inches="tight")
    ratesCompareProb.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/ratesCompareProb.png", format="png", bbox_inches="tight")
    survTimeDistSemilogProb.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/survTimeDistSemilogProb.png", format="png", bbox_inches="tight")
    reactionVsProb.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/reactionVsProb.png", format="png", bbox_inches="tight")