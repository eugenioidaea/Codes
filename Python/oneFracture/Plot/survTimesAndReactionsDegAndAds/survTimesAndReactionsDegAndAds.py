debug = False
if not debug:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from sklearn.linear_model import LinearRegression

save = False

loadAdsorption = np.load('totalAdsorption_3.npz', allow_pickle=True)
for name, value in (loadAdsorption.items()):
    globals()[name] = value
liveParticlesInTimeAds = liveParticlesInTime.copy()
liveParticlesInLogTimeAds = liveParticlesInLogTime.copy()
liveParticlesInTimeNormAds = liveParticlesInTimeNorm.copy()
liveParticlesInLogTimeNormAds = liveParticlesInLogTimeNorm.copy()

loadDegradation = np.load('degradation_3.npz', allow_pickle=True)
for name, value in (loadDegradation.items()):
    globals()[name] = value
liveParticlesInTimeDeg = liveParticlesInTime.copy()
liveParticlesInLogTimeDeg = liveParticlesInLogTime.copy()
liveParticlesInTimeNormDeg = liveParticlesInTimeNorm.copy()
liveParticlesInLogTimeNormDeg = liveParticlesInLogTimeNorm.copy()

# Well-mixed vs diffusion-limited survival time distributions ###########################################################
# Distribution of live particles in time
survTimeDistCompare = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})    
plt.plot(timeStep, liveParticlesInTimeDeg, label=r'$p_s(t)=ke^{-kt} \, lin \, bins$', color='blue')
plt.plot(timeStep, liveParticlesInTimeAds, label=r'$p_s(t)=ads \, bc \, lin \, bins$', color='green')
plt.plot(timeLogSpaced, liveParticlesInLogTimeDeg, label=r'$p_s(t)=ke^{-kt} \, log \, bins$', color='b', linestyle='--')
plt.plot(timeLogSpaced, liveParticlesInLogTimeAds, label=r'$p_s(t)=ads \, bc \, log \, bins$', color='g', linestyle='--')
plt.title("Survival time distributions")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Time')
plt.ylabel('Number of live particles')
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='black')
plt.grid(True, which="minor", linestyle=':', linewidth=0.5, color='gray')
plt.legend(loc='best')
plt.tight_layout()

# Rates of particles decay from linspaced surv time dist
compareDecayDegradationRates = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
tDiff = np.diff(timeStep)
dSurvPart = np.diff(np.log(liveParticlesInTimeDeg))
dNonAdsPart = np.diff(np.log(liveParticlesInTimeAds))
dSurvdt = dSurvPart/tDiff
dNonAdsdt = dNonAdsPart/tDiff
midTimes = ((timeStep)[:-1] + (timeStep)[1:]) / 2
plt.plot(midTimes, dSurvdt, label='Well mixed lin bins', color='b') # , marker='+', linestyle='none', markersize='5')
plt.plot(midTimes, dNonAdsdt, label='Diff limited lin bins', color='g') # , marker='x', linestyle='none', markersize='5')
tLogDiff = np.diff(timeLogSpaced)
dSurvPartLog = np.diff(np.log(liveParticlesInLogTimeDeg))
dNonAdsPartLog = np.diff(np.log(liveParticlesInLogTimeAds))
dSurvLogdt = dSurvPartLog/tLogDiff
dNonAdsLogdt = dNonAdsPartLog/tLogDiff
midTimesLog = (timeLogSpaced[:-1] + timeLogSpaced[1:]) / 2
plt.plot(midTimesLog, dSurvLogdt, label='Well mixed log bins', color='b', linestyle='--') # marker='p', linestyle='none', markersize='5')
plt.plot(midTimesLog, dNonAdsLogdt, label='Diff limited log bins', color='g', linestyle='--') # marker='*', linestyle='none', markersize='5')
plt.title("Effective reaction rate")
plt.xlabel('Time')
plt.ylabel('k(t)')
plt.xscale('log')
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='gray')
plt.legend(loc='best')
plt.tight_layout()

# Normalised distribution of live particles in time
survTimeDistCompareNorm = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})    
tau = (uby-lby)**2/Df
# exp_decay = np.exp(-Time/tau)
# plt.plot(Time[:-1], np.log(exp_decay[:-1]), label=f'p_s(t)=e^(-t/tau) where tau_d={tau}', color='r')
plt.plot(timeStep/tau, liveParticlesInTimeNormDeg, label=r'$p_s(t)=ke^{-kt} \, lin \, bins$', color='blue')
plt.plot(timeStep/tau, liveParticlesInTimeNormAds, label=r'$p_s(t)=ads \, bc \, lin \, bins$', color='green')
plt.plot(timeLogSpaced/tau, liveParticlesInLogTimeNormDeg, label=r'$p_s(t)=ke^{-kt} \, log \, bins$', color='b', linestyle='--')
plt.plot(timeLogSpaced/tau, liveParticlesInLogTimeNormAds, label=r'$p_s(t)=ads \, bc \, log \, bins$', color='g', linestyle='--')
plt.title("Survival time PDFs")
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-7, 1)
plt.xlabel('Time/tau')
plt.ylabel('Normalised number of live particles')
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='black')
plt.grid(True, which="minor", linestyle=':', linewidth=0.5, color='gray')
plt.legend(loc='best')
plt.tight_layout()

# Rates of particles decay from linspaced surv time dist
compareDecayDegradationRatesNorm = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
tDiff = np.diff(timeStep/tau)
dSurvPart = np.diff(np.log(liveParticlesInTimeNormDeg))
dNonAdsPart = np.diff(np.log(liveParticlesInTimeNormAds))
dSurvdt = dSurvPart/tDiff
dNonAdsdt = dNonAdsPart/tDiff
midTimes = ((timeStep)[:-1] + (timeStep)[1:]) / 2
plt.plot(midTimes/tau, dSurvdt, label='Well mixed lin bins', color='b') # , marker='+', linestyle='none', markersize='5')
plt.plot(midTimes/tau, dNonAdsdt, label='Diff limited lin bins', color='g') # , marker='x', linestyle='none', markersize='5')
tLogDiff = np.diff(timeLogSpaced/tau)
dSurvPartLog = np.diff(np.log(liveParticlesInLogTimeNormDeg))
dNonAdsPartLog = np.diff(np.log(liveParticlesInLogTimeNormAds))
dSurvLogdt = dSurvPartLog/tLogDiff
dNonAdsLogdt = dNonAdsPartLog/tLogDiff
midTimesLog = (timeLogSpaced[:-1] + timeLogSpaced[1:]) / 2
plt.plot(midTimesLog/tau, dSurvLogdt, label='Well mixed log bins', color='b', linestyle='--') # marker='p', linestyle='none', markersize='5')
plt.plot(midTimesLog/tau, dNonAdsLogdt, label='Diff limited log bins', color='g', linestyle='--') # marker='*', linestyle='none', markersize='5')
plt.title("Effective reaction rate")
plt.xlabel('Time/tau')
plt.ylabel('k(t)')
plt.xscale('log')
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='gray')
plt.legend(loc='best')
plt.tight_layout()


if save:
    survTimeDistCompare.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/survTimeDistCompare.png", format="png", bbox_inches="tight")
    compareDecayDegradationRates.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/compareDecayDegradationRates.png", format="png", bbox_inches="tight")
    survTimeDistCompareNorm.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/survTimeDistCompareNorm.png", format="png", bbox_inches="tight")
    compareDecayDegradationRatesNorm.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/compareDecayDegradationRatesNorm.png", format="png", bbox_inches="tight")