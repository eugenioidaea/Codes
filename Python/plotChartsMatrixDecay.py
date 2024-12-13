debug = False
if not debug:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Choose what should be plotted #############################################################

plotMatrixDecay = False

save = False

# Load simulation results from .npz files ###################################################
if plotMatrixDecay:
    loadMatrixDecay = np.load('matrixDecay.npz')
    for name, value in (loadMatrixDecay.items()):
        globals()[name] = value

# Plot section #########################################################################
plt.plot(x, y, 'b*')
plt.plot([xInit, xInit], [lby, uby], color='yellow', linewidth=2)
plt.axhline(y=uby, color='r', linestyle='--', linewidth=1)
plt.axhline(y=lby, color='r', linestyle='--', linewidth=1)

survTimeDistCompareAdsProb = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
plt.plot(Time, numOfLivePart/num_particles, 'o', markerfacecolor='none', markeredgecolor='blue', markersize='5', label=r'$k_{decay} = 0.05$')
# plt.plot(Time80, numOfLivePartP80/num_particles, 'o', markerfacecolor='none', markeredgecolor='blue', markersize='5', label=r'$p = 0.8$')
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
# logBins = np.logspace(np.log10(dt), np.log10(Time.max()), len(Time))
# binIndeces = np.digitize(Time, logBins)
# numOfLivePartLog = np.array([numOfLivePart[binIndeces == i].mean() for i in range(0, len(Time))])
# plt.rcParams.update({'font.size': 20})
# plt.plot(logBins, numOfLivePartLog/num_particles, 'o', markerfacecolor='none', markeredgecolor='red', markersize='5', label=r'$\tau_d = 40$')

from sklearn.linear_model import LinearRegression
ratesCompareProb = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
dLivedt = -np.diff(np.log(numOfLivePart/num_particles))/np.diff(Time)
# dLivedtP80 = -np.diff(np.log(numOfLivePartP80/num_particles))/np.diff(Time80)
# dLivedtP60 = -np.diff(np.log(numOfLivePartP60/num_particles))/np.diff(Time60)
# dLivedtP40 = -np.diff(np.log(numOfLivePartP40/num_particles))/np.diff(Time40)
mask0 = dLivedt!=0
# mask0p80 = dLivedtP80!=0
# mask0p60 = dLivedtP60!=0
# mask0p40 = dLivedtP40!=0
midTimes = ((Time)[:-1] + (Time)[1:]) / 2
# midTimes80 = ((Time80)[:-1] + (Time80)[1:]) / 2
# midTimes60 = ((Time60)[:-1] + (Time60)[1:]) / 2
# midTimes40 = ((Time40)[:-1] + (Time40)[1:]) / 2
plt.plot(midTimes[mask0], dLivedt[mask0], 'o', markerfacecolor='none', markeredgecolor='blue', markersize='5', label=r'$k_{decay} = 0.05$')
# plt.plot(midTimes80[mask0p80], dLivedtP80[mask0p80], 'o', markerfacecolor='none', markeredgecolor='blue', markersize='5', label=r'$p = 0.8$')
# plt.plot(midTimes60[mask0p60], dLivedtP60[mask0p60], 'o', markerfacecolor='none', markeredgecolor='red', markersize='5', label=r'$p = 0.6$')
# plt.plot(midTimes40[mask0p40], dLivedtP40[mask0p40], 'o', markerfacecolor='none', markeredgecolor='green', markersize='5', label=r'$p = 0.4$')
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
sliceDecay = slice(10, 200)
# sliceP80 = slice(10, 100)
# sliceP60 = slice(20, 200)
# sliceP40 = slice(40, 400)
timeReshaped = (midTimes[mask0][sliceDecay]).reshape(-1, 1)
# timeReshapedP80 = (midTimes80[mask0p80][sliceP80]).reshape(-1, 1)
# timeReshapedP60 = (midTimes60[mask0p60][sliceP60]).reshape(-1, 1)
# timeReshapedP40 = (midTimes40[mask0p40][sliceP40]).reshape(-1, 1)
interp = LinearRegression().fit(timeReshaped, dLivedt[mask0][sliceDecay])
# interpP80 = LinearRegression().fit(timeReshapedP80, dLivedtP80[mask0p80][sliceP80])
# interpP60 = LinearRegression().fit(timeReshapedP60, dLivedtP60[mask0p60][sliceP60])
# interpP40 = LinearRegression().fit(timeReshapedP40, dLivedtP40[mask0p40][sliceP40])
# timeLogReshaped = (midLogTimes[:100]).reshape(-1, 1)
# modelLog = LinearRegression()
# modelLog.fit(timeLogReshaped, dLiveLogdt[:100])
# print(f"Coeff D1: {modelLog.coef_}")
# print(f"Intercept D1: {modelLog.intercept_}")
kInterpLin = interp.intercept_+interp.coef_*midTimes[mask0][sliceDecay]
plt.plot(midTimes[mask0][sliceDecay], kInterpLin, color='black', linewidth='2')
plt.text(midTimes[mask0][sliceDecay][0], kInterpLin[0], f"k={interp.intercept_:.3f}", fontsize=18, ha='right', va='top')
# kInterpLinP80 = interpP80.intercept_+interpP80.coef_*midTimes80[mask0p80][sliceP80]
# plt.plot(midTimes80[mask0p80][sliceP80], kInterpLinP80, color='black', linewidth='2')
# plt.text(midTimes80[mask0p80][sliceP80][0], kInterpLinP80[0], f"k={interpP80.intercept_:.2f}", fontsize=18, ha='right', va='top')
# kInterpLinP60 = interpP60.intercept_+interpP60.coef_*midTimes60[mask0p60][sliceP60]
# plt.plot(midTimes60[mask0p60][sliceP60], kInterpLinP60, color='black', linewidth='2')
# plt.text(midTimes60[mask0p60][sliceP60][0], kInterpLinP60[0], f"k={interpP60.intercept_:.2f}", fontsize=18, ha='right', va='top')
# kInterpLinP40 = interpP40.intercept_+interpP40.coef_*midTimes40[mask0p40][sliceP40]
# plt.plot(midTimes40[mask0p40][sliceP40], kInterpLinP40, color='black', linewidth='2')
# plt.text(midTimes40[mask0p40][sliceP40][0], kInterpLinP40[0], f"k={interpP40.intercept_:.2f}", fontsize=18, ha='right', va='top')
# kInterpLog = modelLog.intercept_+modelLog.coef_*midLogTimes
# plt.plot(midLogTimes, kInterpLog, color='red')