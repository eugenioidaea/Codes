debug = False
if not debug:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from sklearn.linear_model import LinearRegression

# Choose what should be plotted #############################################################

plotTrajectories =                  False
plotEulerianPdfCdf =                False
plotLagrangianPdf =                 False
plotBreakthroughCurveVerification = False
plotSpatialConcentration =          False
plotDegradation =                   False
FinalPositions =                    False
FinalPositionVertAll =              False
FinalPositionHorAll =               False
FinalPositionVert =                 False
FinalPositionHor =                  False
plotSruvivalTimeDistOfNonAdsorbed = False
survTimesAndReactionsDegAndAds =    True
compareAdsDiff =                    False
compareAdsApertures =               False
compareAdsProb =                    False
reactionVsTauAndProb =              False
compareDifferentTau =               False
compareDifferentProb =              False

save =                              False

# Load simulation results from .npz files ###################################################
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

if FinalPositions:
#    loadFinalPositions = np.load('Dl01Dr01Rl0Rr0.npz')
#    for name, value in (loadFinalPositions.items()):
#        globals()[name] = value

#    loadFinalPositions = np.load('Dl01Dr001Rl0Rr0.npz')
#    for name, value in (loadFinalPositions.items()):
#        globals()[name] = value

#    loadFinalPositions = np.load('Dl01Dr01RlPlRrPr.npz')
#    for name, value in (loadFinalPositions.items()):
#        globals()[name] = value

    loadFinalPositions = np.load('Dl01Dr001RlPlRrPr.npz')
    for name, value in (loadFinalPositions.items()):
        globals()[name] = value

#    loadFinalPositions = np.load('Dl01Dr001RlPlRrPr1e5ts.npz')
#    for name, value in (loadFinalPositions.items()):
#        globals()[name] = value

if compareAdsApertures:
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

if compareAdsProb:
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

if compareDifferentTau:
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

    loadCompareTau400 = np.load('compareTau400.npz')
    for name, value in (loadCompareTau400.items()):
        globals()[name] = value
    numOfLivePartTau400 = numOfLivePart.copy()
    Time400 = Time.copy()
    tau400 = (uby-lby)**2/Df

    loadCompareTau4000 = np.load('compareTau4000.npz')
    for name, value in (loadCompareTau4000.items()):
        globals()[name] = value
    numOfLivePartTau4000 = numOfLivePart.copy()
    Time4000 = Time.copy()
    tau4000 = (uby-lby)**2/Df

if compareDifferentProb:
    loadCompareP80 = np.load('compareP80.npz')
    for name, value in (loadCompareP80.items()):
        globals()[name] = value
    numOfLivePartP80 = numOfLivePart.copy()
    Time80 = Time.copy()
    tau80 = (uby-lby)**2/Df

    loadCompareP60 = np.load('compareP60.npz')
    for name, value in (loadCompareP60.items()):
        globals()[name] = value
    numOfLivePartP60 = numOfLivePart.copy()
    Time60 = Time.copy()
    tau60 = (uby-lby)**2/Df

    loadCompareP40 = np.load('compareP40.npz')
    for name, value in (loadCompareP40.items()):
        globals()[name] = value
    numOfLivePartP40 = numOfLivePart.copy()
    Time40 = Time.copy()
    tau40 = (uby-lby)**2/Df

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
    plt.axvline(x=xInit, color='yellow', linestyle='--', linewidth=2)
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
    plt.plot(timeStep, pdf_part/num_particles)
    plt.xscale('log')
    plt.title("PDF")

    # CDF
    plt.figure(figsize=(8, 8))
    plt.plot(timeStep, np.cumsum(pdf_part)/num_particles)
    plt.xscale('log')
    plt.title("CDF")

    # 1-CDF
    plt.figure(figsize=(8, 8))
    plt.plot(timeStep, 1-np.cumsum(pdf_part)/num_particles)
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
    plt.axvline(x=xInit, color='yellow', linestyle='--', linewidth=2)
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
    plt.plot(timeStep, exp_prob, 'r-')
    plt.plot(timeStep, liveParticlesInTimeNormDeg, 'b*')
    plt.title("Live particle distribution in time")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Time')
    plt.ylabel('PDF of live particles')
    plt.tight_layout()

if FinalPositions:
# Final particles's positions
    finalPositions = plt.figure(figsize=(8, 8))
    if matrixDiffVerification:
        plt.plot(x, y, 'b*')
        plt.plot([lbx, rbx, rbx, lbx, lbx], [lby, lby, uby, uby, lby], color='black', linewidth=2)
        plt.scatter(x0, y0, s=2, c='purple', alpha=1, edgecolor='none', marker='o')
        if (reflectedLeft).any() & (reflectedRight).any():
            plt.plot([cbx, cbx], [lby, uby], color='orange', linewidth=3, linestyle='--')
        histoMatriDiff = plt.figure(figsize=(8, 8))
        hDist, hBins = np.histogram(x, np.linspace(lbx, rbx, 100), density=True)
        plt.bar(hBins[:-1], hDist, width=np.diff(hBins), edgecolor="black", align="edge")
        plt.axvline(x=cbx, color='orange', linestyle='-', linewidth=2)
    else:
        # plt.plot(xPath[:, -1], yPath[:, -1], 'b*')
        plt.plot(x, y, 'b*')
        plt.axvline(x=xInit, color='yellow', linestyle='--', linewidth=2)
        plt.axhline(y=uby, color='r', linestyle='--', linewidth=1)
        plt.axhline(y=lby, color='r', linestyle='--', linewidth=1)
        # for val in vInterval:
        #     plt.axvline(x=val, color='black', linestyle='--', linewidth=2)
        # for val in hInterval:
        #     plt.axhline(y=val, color='black', linestyle='--', linewidth=2)
    plt.tight_layout()

if FinalPositionVertAll:
    # Vertical distribution of all particles
    finalPositionVertAll = plt.figure(figsize=(8, 8))
    plt.bar(vBinsAll[:-1], vDistAll, width=np.diff(vBinsAll), edgecolor="black", align="edge")
    plt.title('Particles distribution at the end of the simulation')
    plt.xlabel('Distance along Y')
    plt.ylabel('Number of particles')
    plt.tight_layout()

if FinalPositionHorAll:
    # Horizontal distribution of all particles
    finalPositionHorAll = plt.figure(figsize=(8, 8))
    plt.bar(hBinsAll[:-1], hDistAll, width=np.diff(hBinsAll), edgecolor="black", align="edge")
    plt.title('Particles distribution at the end of the simulation')
    plt.xlabel('Distance along X')
    plt.ylabel('Number of particles')
    plt.tight_layout()

if FinalPositionVert:
    # Vertical distribution
    finalPositionVert = plt.figure(figsize=(8, 8))
    plt.bar(vBins[:-1], vDist, width=np.diff(vBins), edgecolor="black", align="edge")
    plt.title('Particles distribution at the end of the simulation')
    plt.xlabel('Distance along Y')
    plt.ylabel('Number of particles')
    plt.tight_layout()

if FinalPositionHor:
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
    plt.plot(timeStep, liveParticlesInTimeNormAds, 'b*')
    plt.title("Non-absorbed particle normalised in time")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Time')
    plt.ylabel('Non-absorbed particles normalised')
    plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='black')
    plt.grid(True, which="minor", linestyle=':', linewidth=0.5, color='gray')
    plt.tight_layout()

if compareAdsApertures:
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

if compareAdsProb:
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

# Survival time distributions and reaction rates for different tau ##################################################
if compareDifferentTau:
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
    sliceTau4000 = slice(2000, 5000)
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

    survTimeDistSemilogTau = plt.figure(figsize=(8, 8))
    plt.rcParams.update({'font.size': 20})
    plt.plot(Time4, numOfLivePartTau4/num_particles, 'o', markerfacecolor='none', markeredgecolor='blue', markersize='5', label=r'$\tau_d = 4$')
    plt.plot(Time40, numOfLivePartTau40/num_particles, 'o', markerfacecolor='none', markeredgecolor='red', markersize='5', label=r'$\tau_d = 40$')
    plt.plot(Time400, numOfLivePartTau400/num_particles, 'o', markerfacecolor='none', markeredgecolor='green', markersize='5', label=r'$\tau_d = 400$')
    plt.plot(Time4000, numOfLivePartTau4000/num_particles, 'o', markerfacecolor='none', markeredgecolor='purple', markersize='5', label=r'$\tau_d = 4000$')
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
    sliceSemilogTau4 = slice(30, 80)
    sliceSemilogTau40 = slice(300, 500)
    sliceSemilogTau400 = slice(2000, 4000)
    sliceSemilogTau4000 = slice(8000, 16000)
    timeReshapedSemilogTau4 = (Time4[sliceSemilogTau4]).reshape(-1, 1)
    timeReshapedSemilogTau40 = (Time40[sliceSemilogTau40]).reshape(-1, 1)
    timeReshapedSemilogTau400 = (Time400[sliceSemilogTau400]).reshape(-1, 1)
    timeReshapedSemilogTau4000 = (Time4000[sliceSemilogTau4000]).reshape(-1, 1)
    interpSemilogTau4 = LinearRegression().fit(timeReshapedSemilogTau4, np.log(numOfLivePartTau4[sliceSemilogTau4]/num_particles))
    interpSemilogTau40 = LinearRegression().fit(timeReshapedSemilogTau40, np.log(numOfLivePartTau40[sliceSemilogTau40]/num_particles))
    interpSemilogTau400 = LinearRegression().fit(timeReshapedSemilogTau400, np.log(numOfLivePartTau400[sliceSemilogTau400]/num_particles))
    interpSemilogTau4000 = LinearRegression().fit(timeReshapedSemilogTau4000, np.log(numOfLivePartTau4000[sliceSemilogTau4000]/num_particles))
    kInterpSemilogTau4 = np.exp(interpSemilogTau4.intercept_+interpSemilogTau4.coef_*timeReshapedSemilogTau4)
    plt.plot(timeReshapedSemilogTau4, kInterpSemilogTau4, color='black', linewidth='2')
    plt.text(timeReshapedSemilogTau4[-1], kInterpSemilogTau4[-1], f"k={interpSemilogTau4.coef_[0]:.2f}", fontsize=18, ha='left')
    kInterpSemilogTau40 = np.exp(interpSemilogTau40.intercept_+interpSemilogTau40.coef_*timeReshapedSemilogTau40)
    plt.plot(timeReshapedSemilogTau40, kInterpSemilogTau40, color='black', linewidth='2')
    plt.text(timeReshapedSemilogTau40[-1], kInterpSemilogTau40[-1], f"k={interpSemilogTau40.coef_[0]:.2f}", fontsize=18, ha='left')
    kInterpSemilogTau400 = np.exp(interpSemilogTau400.intercept_+interpSemilogTau400.coef_*timeReshapedSemilogTau400)
    plt.plot(timeReshapedSemilogTau400, kInterpSemilogTau400, color='black', linewidth='2')
    plt.text(timeReshapedSemilogTau400[-1], kInterpSemilogTau400[-1], f"k={interpSemilogTau400.coef_[0]:.2f}", fontsize=18, ha='left')
    kInterpSemilogTau4000 = np.exp(interpSemilogTau4000.intercept_+interpSemilogTau4000.coef_*timeReshapedSemilogTau4000)
    plt.plot(timeReshapedSemilogTau4000, kInterpSemilogTau4000, color='black', linewidth='2')
    plt.text(timeReshapedSemilogTau4000[-1], kInterpSemilogTau4000[-1], f"k={interpSemilogTau4000.coef_[0]:.2f}", fontsize=18, ha='left')

    reactionVsTau = plt.figure(figsize=(8, 8))
    plt.plot(tau4, -interpSemilogTau4.coef_[0], 'o', markerfacecolor='blue', markeredgecolor='blue', markersize='10') #, label=r'$tau_d = 4$')
    plt.plot(tau40, -interpSemilogTau40.coef_[0], 'o', markerfacecolor='red', markeredgecolor='red', markersize='10') #, label=r'$tau_d = 40$')
    plt.plot(tau400, -interpSemilogTau400.coef_[0], 'o', markerfacecolor='green', markeredgecolor='green', markersize='10') #, label=r'$tau_d = 400$')
    plt.plot(tau4000, -interpSemilogTau4000.coef_[0], 'o', markerfacecolor='purple', markeredgecolor='purple', markersize='10') #, label=r'$tau_d = 400$')
    plt.title("Reaction rates vs characteristic times")
    # plt.xscale('log')
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

# Particles' survival distribution and reaction rate for different adsorption probability ###############################
if compareDifferentProb:
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

# Reaction rates vs characteristic times ################################################################################
if reactionVsTauAndProb:
    reactionVsDifferentTau = plt.figure(figsize=(8, 8))
    plt.plot([4], [0.15], 'o', markerfacecolor='blue', markeredgecolor='blue', markersize='10') #, label=r'$tau_d = 4$')
    plt.plot([40], [0.021], 'o', markerfacecolor='red', markeredgecolor='red', markersize='10') #, label=r'$tau_d = 40$')
    plt.plot([400], [0.0023], 'o', markerfacecolor='green', markeredgecolor='green', markersize='10') #, label=r'$tau_d = 400$')
    plt.plot([4000], [0.00024], 'o', markerfacecolor='purple', markeredgecolor='purple', markersize='10') #, label=r'$tau_d = 4000$')
    plt.title("Reaction rates vs characteristic times")
    # plt.xscale('log')
    plt.yscale('log')
    # plt.xlim(0, 20)
    # plt.ylim(-10, 1)
    plt.xlabel(r'$\tau_d$')
    # plt.ylabel('Normalised number of live particles')
    plt.ylabel(r'$k(t)$')
    plt.grid(True, which="major", linestyle='-', linewidth=0.7, color='black')
    plt.grid(True, which="minor", linestyle=':', linewidth=0.5, color='gray')
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

# survTimeDistCompareNorm.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/survTimeDistCompareNorm.png", format="png", bbox_inches="tight")

if survTimesAndReactionsDegAndAds & save:
    survTimeDistCompare.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/survTimeDistCompare.png", format="png", bbox_inches="tight")
    compareDecayDegradationRates.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/compareDecayDegradationRates.png", format="png", bbox_inches="tight")
    survTimeDistCompareNorm.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/survTimeDistCompareNorm.png", format="png", bbox_inches="tight")
    compareDecayDegradationRatesNorm.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/compareDecayDegradationRatesNorm.png", format="png", bbox_inches="tight")

# if FinalPositions & save:
    # finalPositions.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/positionsDl01Dr01Rl0Rr0.png", format="png", bbox_inches="tight")
    # histoMatriDiff.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/histDl01Dr01Rl0Rr0.png", format="png", bbox_inches="tight")
    # finalPositions.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/positionsDl01Dr001Rl0Rr0.png", format="png", bbox_inches="tight")
    # histoMatriDiff.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/histDl01Dr001Rl0Rr0.png", format="png", bbox_inches="tight")
    # finalPositions.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/positionsDl01Dr01RlPlRrPr.png", format="png", bbox_inches="tight")
    # histoMatriDiff.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/histDl01Dr01RlPlRrPr.png", format="png", bbox_inches="tight")
    # finalPositions.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/positionsDl01Dr001RlPlRrPr.png", format="png", bbox_inches="tight")
    # histoMatriDiff.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/histDl01Dr001RlPlRrPr.png", format="png", bbox_inches="tight")
    # finalPositions.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/positionsDl01Dr001RlPlRrPr1e5ts.png", format="png", bbox_inches="tight")
    # histoMatriDiff.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/histDl01Dr001RlPlRrPr1e5ts.png", format="png", bbox_inches="tight")

if compareAdsApertures & save:
    survTimeDistCompareApe.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/survTimeDistCompareApe.png", format="png", bbox_inches="tight")
    survTimeDistCompareApeNorm.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/survTimeDistCompareApeNorm.png", format="png", bbox_inches="tight")
    compareAdsRatesApe.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/compareAdsRatesApe.png", format="png", bbox_inches="tight")
    compareApeNormAdsRates.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/compareApeNormAdsRates.png", format="png", bbox_inches="tight")

if compareAdsProb & save:
    survTimeDistCompareProb.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/survTimeDistCompareProb.png", format="png", bbox_inches="tight")
    survTimeDistCompareProbNorm.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/survTimeDistCompareProbNorm.png", format="png", bbox_inches="tight")
    compareAdsRatesProb.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/compareAdsRatesProb.png", format="png", bbox_inches="tight")
    compareProbNormAdsRates.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/compareProbNormAdsRates.png", format="png", bbox_inches="tight")

if compareDifferentTau & save:
    survTimeDistCompareTau.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/survTimeDistCompareTau.png", format="png", bbox_inches="tight")
    ratesCompareTau.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/ratesCompareTau.png", format="png", bbox_inches="tight")
    survTimeDistSemilogTau.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/survTimeDistSemilogTau.png", format="png", bbox_inches="tight")
    reactionVsTau.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/reactionVsTau.png", format="png", bbox_inches="tight")

if compareDifferentProb & save:
    survTimeDistCompareAdsProb.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/survTimeDistCompareAdsProb.png", format="png", bbox_inches="tight")
    ratesCompareProb.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/ratesCompareProb.png", format="png", bbox_inches="tight")
    survTimeDistSemilogProb.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/survTimeDistSemilogProb.png", format="png", bbox_inches="tight")
    reactionVsProb.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/reactionVsProb.png", format="png", bbox_inches="tight")