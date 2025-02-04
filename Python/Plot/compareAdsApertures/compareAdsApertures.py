debug = False
if not debug:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from sklearn.linear_model import LinearRegression

save = False

loadCompareAdsAp2 = np.load('compareAp2.npz')
for name, value in (loadCompareAdsAp2.items()):
    globals()[name] = value
liveParticlesInTimeAp2 = liveParticlesInTime.copy()
liveParticlesInTimeNormAp2 = liveParticlesInTimeNorm.copy()
tauAp2 = (uby-lby)**2/Df

loadCompareAdsAp4 = np.load('compareAp4.npz')
for name, value in (loadCompareAdsAp4.items()):
    globals()[name] = value
liveParticlesInTimeAp4 = liveParticlesInTime.copy()
liveParticlesInTimeNormAp4 = liveParticlesInTimeNorm.copy()
tauAp4 = (uby-lby)**2/Df

loadCompareAdsAp6 = np.load('compareAp6.npz')
for name, value in (loadCompareAdsAp6.items()):
    globals()[name] = value
liveParticlesInTimeAp6 = liveParticlesInTime.copy()
liveParticlesInTimeNormAp6 = liveParticlesInTimeNorm.copy()
tauAp6 = (uby-lby)**2/Df

# Distribution of live particles in time
survTimeDistCompareApe = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
plt.plot(timeLinSpaced, liveParticlesInTimeAp2, label=r'$Aperture=2$', color='b', linestyle='-')
plt.plot(timeLinSpaced, liveParticlesInTimeAp4, label=r'$Aperture=4$', color='r', linestyle='-')
plt.plot(timeLinSpaced, liveParticlesInTimeAp6, label=r'$Aperture=6$', color='g', linestyle='-')
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
survTimeDistCompareApeNorm = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
plt.plot(timeLinSpaced/tauAp2, liveParticlesInTimeNormAp2, label=r'$Aperture=2$', color='b', linestyle='-')
plt.plot(timeLinSpaced/tauAp4, liveParticlesInTimeNormAp4, label=r'$Aperture=4$', color='r', linestyle='-')
plt.plot(timeLinSpaced/tauAp6, liveParticlesInTimeNormAp6, label=r'$Aperture=6$', color='g', linestyle='-')
plt.title("Survival time distribution")
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$Time/tau_i$')
plt.ylabel('Normalised number of live particles')
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='black')
plt.grid(True, which="minor", linestyle=':', linewidth=0.5, color='gray')
plt.legend(loc='best')
plt.tight_layout()

# Rates of particles decay
compareAdsRatesApe = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
tDiff = np.diff(timeLinSpaced)
dLivePartAp2 = np.diff(np.log(liveParticlesInTimeAp2))
dLivePartAp4 = np.diff(np.log(liveParticlesInTimeAp4))
dLivePartAp6 = np.diff(np.log(liveParticlesInTimeAp6))
dLivedtAp2 = dLivePartAp2/tDiff
dLivedtAp4 = dLivePartAp4/tDiff
dLivedtAp6 = dLivePartAp6/tDiff
midTimes = ((timeLinSpaced)[:-1] + (timeLinSpaced)[1:]) / 2
plt.plot(midTimes, dLivedtAp2, label='Aperture=2', color='b') # , marker='+', linestyle='none', markersize='5')
plt.plot(midTimes, dLivedtAp4, label='Aperture=4', color='r') # , marker='+', linestyle='none', markersize='5')
plt.plot(midTimes, dLivedtAp6, label='Aperture=6', color='g') # , marker='+', linestyle='none', markersize='5')
plt.title("Effective reaction rate")
plt.xlabel('Time')
plt.ylabel('k(t)')
plt.xscale('log')
plt.ylim(-0.2, 0.01)
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='gray')
plt.legend(loc='best')
plt.tight_layout()

compareApeNormAdsRates = plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 20})
dLivedtD1 = np.diff(np.log(liveParticlesInTimeNormAp2))/np.diff(timeLinSpaced/tauAp2)
dLivedtD01 = np.diff(np.log(liveParticlesInTimeNormAp4))/np.diff(timeLinSpaced/tauAp4)
dLivedtD001 = np.diff(np.log(liveParticlesInTimeNormAp6))/np.diff(timeLinSpaced/tauAp6)
midTimes = ((timeLinSpaced)[:-1] + (timeLinSpaced)[1:]) / 2
plt.plot(midTimes/tauAp2, dLivedtD1, label='Aperture=2', color='b') # , marker='+', linestyle='none', markersize='5')
plt.plot(midTimes/tauAp4, dLivedtD01, label='Aperture=4', color='r') # , marker='+', linestyle='none', markersize='5')
plt.plot(midTimes/tauAp6, dLivedtD001, label='Aperture=6', color='g') # , marker='+', linestyle='none', markersize='5')
plt.title("Reaction rates from normalised surv time dist")
plt.xlabel(r'$Time/tau_i$')
plt.ylabel('k(t)')
plt.xscale('log')
plt.ylim(-8, 0.01)
plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='gray')
plt.legend(loc='best')
plt.tight_layout()

if save:
    survTimeDistCompareApe.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/survTimeDistCompareApe.png", format="png", bbox_inches="tight")
    survTimeDistCompareApeNorm.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/survTimeDistCompareApeNorm.png", format="png", bbox_inches="tight")
    compareAdsRatesApe.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/compareAdsRatesApe.png", format="png", bbox_inches="tight")
    compareApeNormAdsRates.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/compareApeNormAdsRates.png", format="png", bbox_inches="tight")