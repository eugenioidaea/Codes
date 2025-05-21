debug = False
if not debug:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from sklearn.linear_model import LinearRegression

save = False

loadCompareTau4 = np.load('compareTau4.npz')
for name, value in (loadCompareTau4.items()):
    globals()[name] = value
numOfLivePartTau4 = numOfLivePart.copy()
Time4 = Time.copy()
tau4 = (uby-lby)**2/Df

loadCompareTau40 = np.load('compareTau40.npz')
for name, value in (loadCompareTau40.items()):
    globals()[name] = value
numOfLivePartTau40 = numOfLivePart.copy()
Time40 = Time.copy()
tau40 = (uby-lby)**2/Df

loadCompareTau100 = np.load('compareTau100.npz')
for name, value in (loadCompareTau100.items()):
    globals()[name] = value
numOfLivePartTau100 = numOfLivePart.copy()
Time100 = Time.copy()
tau100 = (uby-lby)**2/Df

loadCompareTau400 = np.load('compareTau400.npz')
for name, value in (loadCompareTau400.items()):
    globals()[name] = value
numOfLivePartTau400 = numOfLivePart.copy()
Time400 = Time.copy()
tau400 = (uby-lby)**2/Df

loadCompareTau1000 = np.load('compareTau1000.npz')
for name, value in (loadCompareTau1000.items()):
    globals()[name] = value
numOfLivePartTau1000 = numOfLivePart.copy()
Time1000 = Time.copy()
tau1000 = (uby-lby)**2/Df

loadCompareTau4000 = np.load('compareTau4000.npz')
for name, value in (loadCompareTau4000.items()):
    globals()[name] = value
numOfLivePartTau4000 = numOfLivePart.copy()
Time4000 = Time.copy()
tau4000 = (uby-lby)**2/Df

# Survival time distributions and reaction rates for different tau ##################################################
survTimeDistCompareTau = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
plt.plot(Time4, numOfLivePartTau4/num_particles, 'o', markerfacecolor='none', markeredgecolor='blue', markersize='5', label=r'$\tau_d = 4$')
plt.plot(Time40, numOfLivePartTau40/num_particles, 'o', markerfacecolor='none', markeredgecolor='red', markersize='5', label=r'$\tau_d = 40$')
plt.plot(Time400, numOfLivePartTau400/num_particles, 'o', markerfacecolor='none', markeredgecolor='green', markersize='5', label=r'$\tau_d = 400$')
plt.plot(Time4000, numOfLivePartTau4000/num_particles, 'o', markerfacecolor='none', markeredgecolor='purple', markersize='5', label=r'$\tau_d = 4000$')    
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

ratesCompareTau = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
dLivedtTau4 = -np.diff(np.log(numOfLivePartTau4/num_particles))/np.diff(Time4)
dLivedtTau40 = -np.diff(np.log(numOfLivePartTau40/num_particles))/np.diff(Time40)
dLivedtTau400 = -np.diff(np.log(numOfLivePartTau400/num_particles))/np.diff(Time400)
dLivedtTau4000 = -np.diff(np.log(numOfLivePartTau4000/num_particles))/np.diff(Time4000)
mask0tau4 = dLivedtTau4!=0
mask0tau40 = dLivedtTau40!=0
mask0tau400 = dLivedtTau400!=0
mask0tau4000 = dLivedtTau4000!=0
midTimes4 = ((Time4)[:-1] + (Time4)[1:]) / 2
midTimes40 = ((Time40)[:-1] + (Time40)[1:]) / 2
midTimes400 = ((Time400)[:-1] + (Time400)[1:]) / 2
midTimes4000 = ((Time4000)[:-1] + (Time4000)[1:]) / 2
plt.plot(midTimes4[mask0tau4], dLivedtTau4[mask0tau4], 'o', markerfacecolor='none', markeredgecolor='blue', markersize='5', label=r'$\tau_d = 4$')
plt.plot(midTimes40[mask0tau40], dLivedtTau40[mask0tau40], 'o', markerfacecolor='none', markeredgecolor='red', markersize='5', label=r'$\tau_d = 40$')
plt.plot(midTimes400[mask0tau400], dLivedtTau400[mask0tau400], 'o', markerfacecolor='none', markeredgecolor='green', markersize='5', label=r'$\tau_d = 400$')
plt.plot(midTimes4000[mask0tau4000], dLivedtTau4000[mask0tau4000], 'o', markerfacecolor='none', markeredgecolor='purple', markersize='5', label=r'$\tau_d = 4000$')
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
sliceTau4 = slice(5, 50)
sliceTau40 = slice(20, 200)
sliceTau400 = slice(100, 600)
sliceTau4000 = slice(1000, 1500)
timeReshapedTau4 = (midTimes4[mask0tau4][sliceTau4]).reshape(-1, 1)
timeReshapedTau40 = (midTimes40[mask0tau40][sliceTau40]).reshape(-1, 1)
timeReshapedTau400 = (midTimes400[mask0tau400][sliceTau400]).reshape(-1, 1)
timeReshapedTau4000 = (midTimes4000[mask0tau4000][sliceTau4000]).reshape(-1, 1)
interpTau4 = LinearRegression().fit(timeReshapedTau4, dLivedtTau4[mask0tau4][sliceTau4])
interpTau40 = LinearRegression().fit(timeReshapedTau40, dLivedtTau40[mask0tau40][sliceTau40])
interpTau400 = LinearRegression().fit(timeReshapedTau400, dLivedtTau400[mask0tau400][sliceTau400])
interpTau4000 = LinearRegression().fit(timeReshapedTau4000, dLivedtTau4000[mask0tau4000][sliceTau4000])
# timeLogReshaped = (midLogTimes[:100]).reshape(-1, 1)
# modelLog = LinearRegression()
# modelLog.fit(timeLogReshaped, dLiveLogdt[:100])
# print(f"Coeff D1: {modelLog.coef_}")
# print(f"Intercept D1: {modelLog.intercept_}")
kInterpLinTau4 = interpTau4.intercept_+interpTau4.coef_*midTimes4[mask0tau4][sliceTau4]
plt.plot(midTimes4[mask0tau4][sliceTau4], kInterpLinTau4, color='black', linewidth='2')
plt.text(midTimes4[mask0tau4][sliceTau4][-1], kInterpLinTau4[-1], f"k={interpTau4.intercept_:.2f}", fontsize=18, ha='right', va='bottom')
kInterpLinTau40 = interpTau40.intercept_+interpTau40.coef_*midTimes40[mask0tau40][sliceTau40]
plt.plot(midTimes40[mask0tau40][sliceTau40], kInterpLinTau40, color='black', linewidth='2')
plt.text(midTimes40[mask0tau40][sliceTau40][-1], kInterpLinTau40[-1], f"k={interpTau40.intercept_:.2f}", fontsize=18, ha='right', va='bottom')
kInterpLinTau400 = interpTau400.intercept_+interpTau400.coef_*midTimes400[mask0tau400][sliceTau400]
plt.plot(midTimes400[mask0tau400][sliceTau400], kInterpLinTau400, color='black', linewidth='2')
plt.text(midTimes400[mask0tau400][sliceTau400][-1], kInterpLinTau400[-1], f"k={interpTau400.intercept_:.2f}", fontsize=18, ha='right', va='bottom')
kInterpLinTau4000 = interpTau4000.intercept_+interpTau4000.coef_*midTimes4000[mask0tau4000][sliceTau4000]
plt.plot(midTimes4000[mask0tau4000][sliceTau4000], kInterpLinTau4000, color='black', linewidth='2')
plt.text(midTimes4000[mask0tau4000][sliceTau4000][-1], kInterpLinTau4000[-1], f"k={interpTau4000.intercept_:.2f}", fontsize=18, ha='right', va='bottom')
# kInterpLog = modelLog.intercept_+modelLog.coef_*midLogTimes
# plt.plot(midLogTimes, kInterpLog, color='red')

survTimeDistSemilogTau = plt.figure(figsize=(8, 8), dpi=300)
plt.rcParams.update({'font.size': 20})
plt.plot(Time4, np.log(numOfLivePartTau4/num_particles), 'o', markerfacecolor='none', markeredgecolor='blue', markersize='5', label=r'$\tau_d = 4$')
plt.plot(Time40[::10], np.log(numOfLivePartTau40[::10]/num_particles), 'o', markerfacecolor='none', markeredgecolor='red', markersize='5', label=r'$\tau_d = 40$')
plt.plot(Time100[::10], np.log(numOfLivePartTau100[::10]/num_particles), 'o', markerfacecolor='none', markeredgecolor='pink', markersize='5', label=r'$\tau_d = 100$')
plt.plot(Time400[::20], np.log(numOfLivePartTau400[::20]/num_particles), 'o', markerfacecolor='none', markeredgecolor='green', markersize='5', label=r'$\tau_d = 400$')
plt.plot(Time1000[::10], np.log(numOfLivePartTau1000[::10]/num_particles), 'o', markerfacecolor='none', markeredgecolor='orange', markersize='5', label=r'$\tau_d = 1000$')
plt.plot(Time4000[::20], np.log(numOfLivePartTau4000[::20]/num_particles), 'o', markerfacecolor='none', markeredgecolor='purple', markersize='5', label=r'$\tau_d = 4000$')
# plt.title("Survival time distributions")
# plt.xscale('log')
# plt.yscale('log')
plt.xlabel(r'$t$')
plt.ylabel(r'$ln(p_s)$')
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='black')
plt.grid(True, which="minor", linestyle=':', linewidth=0.5, color='gray')
# plt.xlim(0, 200)
plt.legend(loc='best')
plt.ylim(-12, 0.1)
plt.tight_layout()
sliceSemilogTau4 = slice(0, np.count_nonzero(numOfLivePartTau4)-1)
sliceSemilogTau40 = slice(0, np.count_nonzero(numOfLivePartTau40)-1)
sliceSemilogTau100 = slice(0, np.count_nonzero(numOfLivePartTau100)-1)
sliceSemilogTau400 = slice(0, np.count_nonzero(numOfLivePartTau400)-1)
sliceSemilogTau1000 = slice(0, np.count_nonzero(numOfLivePartTau1000)-1)
sliceSemilogTau4000 = slice(0, np.count_nonzero(numOfLivePartTau4000)-1)
timeReshapedSemilogTau4 = (Time4[sliceSemilogTau4]).reshape(-1, 1)
timeReshapedSemilogTau40 = (Time40[sliceSemilogTau40]).reshape(-1, 1)
timeReshapedSemilogTau100 = (Time100[sliceSemilogTau100]).reshape(-1, 1)
timeReshapedSemilogTau400 = (Time400[sliceSemilogTau400]).reshape(-1, 1)
timeReshapedSemilogTau1000 = (Time1000[sliceSemilogTau1000]).reshape(-1, 1)
timeReshapedSemilogTau4000 = (Time4000[sliceSemilogTau4000]).reshape(-1, 1)
interpSemilogTau4 = LinearRegression().fit(timeReshapedSemilogTau4, np.log(numOfLivePartTau4[sliceSemilogTau4]/num_particles))
interpSemilogTau40 = LinearRegression().fit(timeReshapedSemilogTau40, np.log(numOfLivePartTau40[sliceSemilogTau40]/num_particles))
interpSemilogTau100 = LinearRegression().fit(timeReshapedSemilogTau100, np.log(numOfLivePartTau100[sliceSemilogTau100]/num_particles))
interpSemilogTau400 = LinearRegression().fit(timeReshapedSemilogTau400, np.log(numOfLivePartTau400[sliceSemilogTau400]/num_particles))
interpSemilogTau1000 = LinearRegression().fit(timeReshapedSemilogTau1000, np.log(numOfLivePartTau1000[sliceSemilogTau1000]/num_particles))
interpSemilogTau4000 = LinearRegression().fit(timeReshapedSemilogTau4000, np.log(numOfLivePartTau4000[sliceSemilogTau4000]/num_particles))
kInterpSemilogTau4 = np.exp(interpSemilogTau4.intercept_+interpSemilogTau4.coef_*timeReshapedSemilogTau4)
plt.plot(timeReshapedSemilogTau4, np.log(kInterpSemilogTau4), color='blue', linewidth='2')
plt.text(timeReshapedSemilogTau4[len(timeReshapedSemilogTau4)//2], np.log(kInterpSemilogTau4)[len(kInterpSemilogTau4)//2], r'$p_s = %g e^{%g t}$' % (round(1/np.exp(-interpSemilogTau4.intercept_), 2), round(interpSemilogTau4.coef_[0], 3)), fontsize=18, ha='left', va='bottom')
kInterpSemilogTau40 = np.exp(interpSemilogTau40.intercept_+interpSemilogTau40.coef_*timeReshapedSemilogTau40)
plt.plot(timeReshapedSemilogTau40, np.log(kInterpSemilogTau40), color='red', linewidth='2')
plt.text(timeReshapedSemilogTau40[len(timeReshapedSemilogTau40)//2], np.log(kInterpSemilogTau40)[len(kInterpSemilogTau40)//2], r'$p_s = %g e^{%g t}$' % (round(1/np.exp(-interpSemilogTau40.intercept_), 2), round(interpSemilogTau40.coef_[0], 3)), fontsize=18, ha='left', va='bottom')
kInterpSemilogTau100 = np.exp(interpSemilogTau100.intercept_+interpSemilogTau100.coef_*timeReshapedSemilogTau100)
plt.plot(timeReshapedSemilogTau100, np.log(kInterpSemilogTau100), color='pink', linewidth='2')
plt.text(timeReshapedSemilogTau100[len(timeReshapedSemilogTau100)//2+300], np.log(kInterpSemilogTau100)[len(kInterpSemilogTau100)//2+300], r'$p_s = %g e^{%g t}$' % (round(1/np.exp(-interpSemilogTau100.intercept_), 2), round(interpSemilogTau100.coef_[0], 3)), fontsize=18, ha='left', va='bottom')
kInterpSemilogTau400 = np.exp(interpSemilogTau400.intercept_+interpSemilogTau400.coef_*timeReshapedSemilogTau400)
plt.plot(timeReshapedSemilogTau400, np.log(kInterpSemilogTau400), color='green', linewidth='2')
plt.text(timeReshapedSemilogTau400[len(timeReshapedSemilogTau400)//2], np.log(kInterpSemilogTau400)[len(kInterpSemilogTau400)//2], r'$p_s = %g e^{%g t}$' % (round(1/np.exp(-interpSemilogTau400.intercept_), 2), round(interpSemilogTau400.coef_[0], 3)), fontsize=18, ha='left', va='bottom')
kInterpSemilogTau1000 = np.exp(interpSemilogTau1000.intercept_+interpSemilogTau1000.coef_*timeReshapedSemilogTau1000)
plt.plot(timeReshapedSemilogTau1000, np.log(kInterpSemilogTau1000), color='orange', linewidth='2')
plt.text(timeReshapedSemilogTau1000[len(timeReshapedSemilogTau1000)//2+300], np.log(kInterpSemilogTau1000)[len(kInterpSemilogTau1000)//2+300], r'$p_s = %g e^{%g t}$' % (round(1/np.exp(-interpSemilogTau1000.intercept_), 2), round(interpSemilogTau1000.coef_[0], 3)), fontsize=18, ha='left', va='bottom')
kInterpSemilogTau4000 = np.exp(interpSemilogTau4000.intercept_+interpSemilogTau4000.coef_*timeReshapedSemilogTau4000)
plt.plot(timeReshapedSemilogTau4000, np.log(kInterpSemilogTau4000), color='purple', linewidth='2')
plt.text(timeReshapedSemilogTau4000[len(timeReshapedSemilogTau4000)//2], np.log(kInterpSemilogTau4000)[len(kInterpSemilogTau4000)//2], r'$p_s = %g e^{%g t}$' % (round(1/np.exp(-interpSemilogTau4000.intercept_), 2), round(interpSemilogTau4000.coef_[0], 3)), fontsize=18, ha='left', va='bottom')

reactionVsTau = plt.figure(figsize=(8, 8), dpi=300)
plt.plot(tau4, -interpSemilogTau4.coef_[0], 'o', markerfacecolor='blue', markeredgecolor='blue', markersize='10', label=r'$k(\tau_d)=%g$' % (round(-interpSemilogTau4.coef_[0], 3)))
plt.plot(tau40, -interpSemilogTau40.coef_[0], 'o', markerfacecolor='red', markeredgecolor='red', markersize='10', label=r'$k(\tau_d)=%g$' % (round(-interpSemilogTau40.coef_[0], 3)))
plt.plot(tau100, -interpSemilogTau100.coef_[0], 'o', markerfacecolor='pink', markeredgecolor='pink', markersize='10', label=r'$k(\tau_d)=%g$' % (round(-interpSemilogTau100.coef_[0], 3)))
plt.plot(tau400, -interpSemilogTau400.coef_[0], 'o', markerfacecolor='green', markeredgecolor='green', markersize='10', label=r'$k(\tau_d)=%g$' % (round(-interpSemilogTau400.coef_[0], 3)))
plt.plot(tau1000, -interpSemilogTau1000.coef_[0], 'o', markerfacecolor='orange', markeredgecolor='orange', markersize='10', label=r'$k(\tau_d)=%g$' % (round(-interpSemilogTau1000.coef_[0], 3)))
plt.plot(tau4000, -interpSemilogTau4000.coef_[0], 'o', markerfacecolor='purple', markeredgecolor='purple', markersize='10', label=r'$k(\tau_d)=%g$' % (round(-interpSemilogTau4000.coef_[0], 3)))
# plt.title("Reaction rates vs characteristic times")
plt.xscale('log')
plt.yscale('log')
# plt.xlim(0, 20)
# plt.ylim(-10, 1)
plt.xlabel(r'$\tau_d$')
# plt.ylabel('Normalised number of live particles')
plt.ylabel(r'$k(\tau_d)$')
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='black')
plt.grid(True, which="minor", linestyle=':', linewidth=0.5, color='gray')
plt.legend(loc='best')
plt.tight_layout()

if save:
    survTimeDistCompareTau.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/survTimeDistCompareTau.png", format="png", bbox_inches="tight")
    ratesCompareTau.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/ratesCompareTau.png", format="png", bbox_inches="tight")
    survTimeDistSemilogTau.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/survTimeDistSemilogTau.png", format="png", bbox_inches="tight")
    reactionVsTau.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/reactionVsTau.png", format="png", bbox_inches="tight")