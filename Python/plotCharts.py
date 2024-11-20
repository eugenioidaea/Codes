debug = False
if not debug:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
import numpy as np
import matplotlib.pyplot as plt

# Choose what should be plotted #############################################################

plotTrajectories = False
plotEulerianPdfCdf = False
plotLagrangianPdf = False
plotBreakthroughCurveVerification = False
plotSpatialConcentration = False
plotDegradation = False
plotFinalPositions = False
plotSruvivalTimeDistOfNonAdsorbed = False
plotSurvivalTimeDistAndReactionRatesForDegradationAndAdsorption = True

compare = True

# Load simulation results from .npz files ###################################################
loadAdsorption = np.load('totalAdsorption_3.npz')
for name, value in (loadAdsorption.items()):
    globals()[name] = value
liveParticlesInTimeNormAds = liveParticlesInTimeNorm
liveParticlesInLogTimeNormAds = liveParticlesInLogTimeNorm

loadDegradation = np.load('degradation_3.npz')
for name, value in (loadDegradation.items()):
    globals()[name] = value
liveParticlesInTimeNormDeg = liveParticlesInTimeNorm
liveParticlesInLogTimeNormDeg = liveParticlesInLogTimeNorm

# loadInfiniteDomain = np.load('infiniteDomain1e6.npz')
# for name, value in (loadInfiniteDomain.items()):
#     globals()[name] = value

# loadSemiInfiniteDomain = np.load('semiInfiniteDomain1e3.npz')
# for name, value in (loadSemiInfiniteDomain.items()):
#     globals()[name] = value

# loadFinalPositions = np.load('finalPositions1e5.npz')
# for name, value in (loadFinalPositions.items()):
#     globals()[name] = value

# loadTestSemra = np.load('matrixDiffusionVerification.npz')
# for name, value in (loadTestSemra.items()):
#     globals()[name] = value

# loadFinalPositions = np.load('partialAdsorption.npz')
# for name, value in (loadFinalPositions.items()):
#     globals()[name] = value

# Plot section #########################################################################
if plotTrajectories:
    # Trajectories
    trajectories = plt.figure(figsize=(8, 8))
    plt.rcParams.update({'font.size': 20})
    for i in range(num_particles):
        plt.plot(xPath[i][:][xPath[i][:]!=0], yPath[i][:][xPath[i][:]!=0], lw=0.5)
    plt.axhline(y=uby, color='r', linestyle='--', linewidth=2)
    plt.axhline(y=lby, color='r', linestyle='--', linewidth=2)
    if lbxOn:
        plt.axvline(x=lbx, color='b', linestyle='--', linewidth=2)
        plt.axvline(x=-lbx, color='b', linestyle='--', linewidth=2)
    plt.axvline(x=x0, color='yellow', linestyle='--', linewidth=2)
    if vcpOn:
        plt.axvline(x=vcp, color='black', linestyle='-', linewidth=2)
    plt.title("2D Diffusion Process (Langevin Equation)")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True)
    plt.tight_layout()

if plotEulerianPdfCdf:
    # PDF
    plt.figure(figsize=(8, 8))
    plt.plot(timeStep*dt, pdf_part/num_particles)
    plt.xscale('log')
    plt.title("PDF")

    # CDF
    plt.figure(figsize=(8, 8))
    plt.plot(timeStep*dt, np.cumsum(pdf_part)/num_particles)
    plt.xscale('log')
    plt.title("CDF")

    # 1-CDF
    plt.figure(figsize=(8, 8))
    plt.plot(timeStep*dt, 1-np.cumsum(pdf_part)/num_particles)
    plt.xscale('log')
    plt.yscale('log')
    plt.title("1-CDF")

if plotLagrangianPdf:
    # Binning for plotting the pdf from a Lagrangian vector
    countsLog, binEdgesLog = np.histogram(particleRT, timeLogSpaced, density=True)
    binCentersLog = (binEdgesLog[:-1] + binEdgesLog[1:]) / 2
    plt.figure(figsize=(8, 8))
    plt.plot(binCentersLog[countsLog!=0], countsLog[countsLog!=0], 'r*')
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Lagrangian PDF of the BTC")

if plotBreakthroughCurveVerification:
    pdfOfBtc = plt.figure(figsize=(8, 8))
    plt.rcParams.update({'font.size': 20})
    plt.plot(timeBinsLog, countsSemiInfLog, 'b*')
    plt.plot(timeBinsLog, analPdfSemiInf, 'r-')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1e-5, 1e-2)
    plt.title('PDF of breakthrough curve')
    plt.xlabel('Time step')
    plt.ylabel('Normalised number of particles')
    plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='black')
    plt.grid(True, which="minor", linestyle=':', linewidth=0.5, color='gray')
    plt.tight_layout()

# Spatial concentration profile at 'recordSpatialConc' time
if plotSpatialConcentration:
    spatialConcentration = plt.figure(figsize=(8, 8))
    plt.rcParams.update({'font.size': 20})
    plt.plot(binCenterSpace, countsSpace, 'b*')
    plt.plot(binCenterSpace, yAnalytical, color='red', linestyle='-')
    plt.axvline(x=x0, color='yellow', linestyle='--', linewidth=2)
    if lbxOn:
        plt.axvline(x=lbx, color='b', linestyle='--', linewidth=2)
        plt.axvline(x=-lbx, color='b', linestyle='--', linewidth=2)
    plt.xlabel('X position')
    plt.ylabel('Normalised number of particles')
    plt.title("Spatial concentration at t=100")
    plt.tight_layout()

if plotDegradation:
    # Distribution of survival times for particles
    plt.figure(figsize=(8, 8))
    plt.plot(np.arange(0, num_particles, 1), np.sort(particleStepsDeg)[::-1], 'b*')
    plt.plot(np.arange(0, num_particles, 1), np.sort(survivalTimeDist)[::-1], 'k-')
    plt.title("Survival time distribution")

    # Distribution of live particles in time
    survivalTimeDistribution = plt.figure(figsize=(8, 8))
    plt.plot(timeStep*dt, exp_prob, 'r-')
    plt.plot(timeStep*dt, liveParticlesInTimeNormDeg, 'b*')
    plt.title("Live particle distribution in time")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Time')
    plt.ylabel('PDF of live particles')
    plt.tight_layout()

if plotFinalPositions:
    # Final particles's positions
    finalPositions = plt.figure(figsize=(8, 8))
    # plt.plot(xPath[:, -1], yPath[:, -1], 'b*')
    plt.plot(x, y, 'b*')
    plt.axvline(x=x0, color='yellow', linestyle='--', linewidth=2)
    plt.axhline(y=uby, color='r', linestyle='--', linewidth=1)
    plt.axhline(y=lby, color='r', linestyle='--', linewidth=1)
    # for val in vInterval:
    #     plt.axvline(x=val, color='black', linestyle='--', linewidth=2)
    # for val in hInterval:
    #     plt.axhline(y=val, color='black', linestyle='--', linewidth=2)
    plt.tight_layout()

    # Vertical distribution of all particles
    finalPositionVertAll = plt.figure(figsize=(8, 8))
    plt.bar(vBinsAll[:-1], vDistAll, width=np.diff(vBinsAll), edgecolor="black", align="edge")
    plt.title('Particles distribution at the end of the simulation')
    plt.xlabel('Distance along Y')
    plt.ylabel('Number of particles')
    plt.tight_layout()

    # Horizontal distribution of all particles
    finalPositionHorAll = plt.figure(figsize=(8, 8))
    plt.bar(hBinsAll[:-1], hDistAll, width=np.diff(hBinsAll), edgecolor="black", align="edge")
    plt.title('Particles distribution at the end of the simulation')
    plt.xlabel('Distance along X')
    plt.ylabel('Number of particles')
    plt.tight_layout()

    # Vertical distribution
    finalPositionVert = plt.figure(figsize=(8, 8))
    plt.bar(vBins[:-1], vDist, width=np.diff(vBins), edgecolor="black", align="edge")
    plt.title('Particles distribution at the end of the simulation')
    plt.xlabel('Distance along Y')
    plt.ylabel('Number of particles')
    plt.tight_layout()

    # Horizontal distribution
    finalPositionHor = plt.figure(figsize=(8, 8))
    plt.bar(hBins[:-1], hDist, width=np.diff(hBins), edgecolor="black", align="edge")
    plt.title('Particles distribution at the end of the simulation')
    plt.xlabel('Distance along X')
    plt.ylabel('Number of particles')
    plt.tight_layout()

if plotSruvivalTimeDistOfNonAdsorbed:
    # Normalised distribution of non-absorbed particles in time
    diffusionLimitedSurvTimeDistNorm = plt.figure(figsize=(8, 8))
    plt.plot(timeStep*dt, liveParticlesInTimeNormAds, 'b*')
    plt.title("Non-absorbed particle normalised in time")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Time')
    plt.ylabel('Non-absorbed particles normalised')
    plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='black')
    plt.grid(True, which="minor", linestyle=':', linewidth=0.5, color='gray')
    plt.tight_layout()

# Well-mixed vs diffusion-limited survival time distributions ###########################################################

if plotSurvivalTimeDistAndReactionRatesForDegradationAndAdsorption:
    # Distribution of live particles in time
    survTimeDistCompare = plt.figure(figsize=(8, 8))
    plt.rcParams.update({'font.size': 20})    
    tau = (uby-lby)**2/Df
    # exp_decay = np.exp(-Time/tau)
    # plt.plot(Time[:-1], np.log(exp_decay[:-1]), label=f'p_s(t)=e^(-t/tau) where tau_d={tau}', color='r')
    plt.plot(timeStep*dt/tau, liveParticlesInTimeNormDeg, label=r'$p_s(t)=ke^{-kt} \, lin \, bins$', color='blue')
    plt.plot(timeStep*dt/tau, liveParticlesInTimeNormAds, label=r'$p_s(t)=ads \, bc \, lin \, bins$', color='green')
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

    # Rates of particles decay
    compareDecayDegradationRatesLin = plt.figure(figsize=(8, 8))
    plt.rcParams.update({'font.size': 20})
    tDiff = np.diff(timeStep*dt)
    dSurvPart = np.diff(np.log(liveParticlesInTimeNormDeg))
    dNonAdsPart = np.diff(np.log(liveParticlesInTimeNormAds))
    dSurvdt = dSurvPart/tDiff
    dNonAdsdt = dNonAdsPart/tDiff
    midTimes = ((timeStep*dt)[:-1] + (timeStep*dt)[1:]) / 2
    plt.plot(midTimes/tau, dSurvdt, label='Well mixed lin bins', color='b') # , marker='+', linestyle='none', markersize='5')
    plt.plot(midTimes/tau, dNonAdsdt, label='Diff limited lin bins', color='g') # , marker='x', linestyle='none', markersize='5')
    plt.title("Effective reaction rate")
    plt.xlabel('Time/tau')
    plt.ylabel('k(t)')
    plt.xscale('log')
    plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='gray')
    plt.legend(loc='best')
    plt.tight_layout()
    compareDecayDegradationRatesLog = plt.figure(figsize=(8, 8))
    plt.rcParams.update({'font.size': 20})
    tLogDiff = np.diff(timeLogSpaced)
    dSurvPartLog = np.diff(np.log(liveParticlesInLogTimeNormDeg))
    dNonAdsPartLog = np.diff(np.log(liveParticlesInLogTimeNormAds))
    dSurvLogdt = dSurvPartLog/tLogDiff
    dNonAdsLogdt = dNonAdsPartLog/tLogDiff
    midTimesLog = (timeLogSpaced[:-1] + timeLogSpaced[1:]) / 2
    plt.plot(midTimesLog/tau, dSurvLogdt, label='Well mixed log bins', color='b', linestyle='--', linewidth='5') # marker='p', linestyle='none', markersize='5')
    plt.plot(midTimesLog/tau, dNonAdsLogdt, label='Diff limited log bins', color='g', linestyle='--') # marker='*', linestyle='none', markersize='5')
    plt.title("Effective reaction rate")
    plt.xlabel('Time/tau')
    plt.ylabel('k(t)')
    plt.xscale('log')
    plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='gray')
    plt.legend(loc='best')
    plt.tight_layout()

    # dExpProb = np.diff(np.log(exp_decay))
    #dExpProbdt = dExpProb/tLogDiff

    # window_size = 100
    # window = np.ones(window_size) / window_size  # Averaging window
    # dSurvdt_smoothed = np.convolve(dSurvdt, window, mode='same')
    # dNonAdsdt_smoothed = np.convolve(dNonAdsdt, window, mode='same')
    # plt.ylim(-0.2, 0)
    # plt.plot(midTimes[:-1], dExpProbdt[:-1], label='Analytical', color='r')

    # plt.plot(midTimes[100:-1], dSurvdt_smoothed[100:-1], color='black')
    # plt.plot(midTimes[100:-1], dNonAdsdt_smoothed[100:-1], color='black')

# trajectories.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/trajectoriesInfinite.png", format="png", bbox_inches="tight")
# trajectories.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/trajectoriesSemiInfinite.png", format="png", bbox_inches="tight")
# trajectories.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/trajectoriesDegradation.png", format="png", bbox_inches="tight")
# trajectories.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/trajectoriesMatrixDiffusion.png", format="png", bbox_inches="tight")

# spatialConcentration.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/verificationInfinite1e6.png", format="png", bbox_inches="tight")
# spatialConcentration.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/verificationInfinite1e5.png", format="png", bbox_inches="tight")
# spatialConcentration.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/verificationMatrixDiffusion.png", format="png", bbox_inches="tight")

# pdfOfBtc.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/verificationSemi-infinite1e3.png", format="png", bbox_inches="tight")

# survivalTimeDistribution.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/liveParticleInTime.png", format="png", bbox_inches="tight")

# finalPositions.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/finalPositionsMatrixDiffusion.png", format="png", bbox_inches="tight")

# finalPositionVertAll.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/verticalFinalDist.png", format="png", bbox_inches="tight")

# finalPositionHorAll.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/horizontalFinalDist.png", format="png", bbox_inches="tight")

# finalPositionVert.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/verticalFinalDist.png", format="png", bbox_inches="tight")

# finalPositionHor.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/horizontalFinalDist.png", format="png", bbox_inches="tight")

# diffusionLimitedSurvTimeDist.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/nonAbsParticles.png", format="png", bbox_inches="tight")

# diffusionLimitedSurvTimeDistNorm.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/nonAbsParticlesNorm.png", format="png", bbox_inches="tight")

# survTimeDistCompare.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/survTimeDistCompare.png", format="png", bbox_inches="tight")

# compareDecayDegradationRates.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/survTimeDistRateCompare.png", format="png", bbox_inches="tight")