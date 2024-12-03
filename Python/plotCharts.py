debug = False
if not debug:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# Choose what should be plotted #############################################################

plotTrajectories = False
plotEulerianPdfCdf = False
plotLagrangianPdf = False
plotBreakthroughCurveVerification = False
plotSpatialConcentration = False
plotDegradation = False
FinalPositions = False
FinalPositionVertAll = False
FinalPositionHorAll = False
FinalPositionVert = False
FinalPositionHor = False
plotSruvivalTimeDistOfNonAdsorbed = False
plotSurvivalTimeDistAndReactionRatesForDegradationAndAdsorption = False
compareAdsDiff = False
compareAdsApertures = False
compareAdsProb = True

# Load simulation results from .npz files ###################################################
if plotSurvivalTimeDistAndReactionRatesForDegradationAndAdsorption:
    loadAdsorption = np.load('totalAdsorption_3.npz')
    for name, value in (loadAdsorption.items()):
        globals()[name] = value
    liveParticlesInTimeAds = liveParticlesInTime.copy()
    liveParticlesInLogTimeAds = liveParticlesInLogTime.copy()
    liveParticlesInTimeNormAds = liveParticlesInTimeNorm.copy()
    liveParticlesInLogTimeNormAds = liveParticlesInLogTimeNorm.copy()

    loadDegradation = np.load('degradation_3.npz')
    for name, value in (loadDegradation.items()):
        globals()[name] = value
    liveParticlesInTimeDeg = liveParticlesInTime.copy()
    liveParticlesInLogTimeDeg = liveParticlesInLogTime.copy()
    liveParticlesInTimeNormDeg = liveParticlesInTimeNorm.copy()
    liveParticlesInLogTimeNormDeg = liveParticlesInLogTimeNorm.copy()

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
#
#    loadFinalPositions = np.load('Dl01Dr01RlPlRrPr.npz')
#    for name, value in (loadFinalPositions.items()):
#        globals()[name] = value
#
#    loadFinalPositions = np.load('Dl01Dr001RlPlRrPr.npz')
#    for name, value in (loadFinalPositions.items()):
#        globals()[name] = value

    loadFinalPositions = np.load('Dl01Dr001RlPlRrPr1e5ts.npz')
    for name, value in (loadFinalPositions.items()):
        globals()[name] = value

if compareAdsDiff:
    loadCompareAdsD1 = np.load('compareAdsD1.npz')
    for name, value in (loadCompareAdsD1.items()):
        globals()[name] = value
    variableWidth = abs(timeLogSpaced-timeLogSpaced[::-1])/max(abs(timeLogSpaced-timeLogSpaced[::-1]))
    timeTwoLogSpaced = np.cumsum(sim_time/sum(variableWidth)*variableWidth)
    liveParticlesInTwoLogTime = np.sum(particleSteps[:, None] > timeTwoLogSpaced, axis=0)
    liveParticlesInTwoLogTimeNorm = liveParticlesInTwoLogTime/sum(liveParticlesInTwoLogTime*np.diff(np.insert(timeTwoLogSpaced, 0, 0)))
    # liveParticlesInTimeD1 = liveParticlesInTwoLogTime.copy()
    # liveParticlesInTimeNormD1 = liveParticlesInTwoLogTimeNorm.copy()
    liveParticlesInTimeD1 = liveParticlesInLogTime.copy()
    liveParticlesInTimeNormD1 = liveParticlesInLogTimeNorm.copy()
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
    liveParticlesInTimeD01 = liveParticlesInLogTime.copy()
    liveParticlesInTimeNormD01 = liveParticlesInLogTimeNorm.copy()
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
    liveParticlesInTimeD001 = liveParticlesInLogTime.copy()
    liveParticlesInTimeNormD001 = liveParticlesInLogTimeNorm.copy()
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
    liveParticlesInTimeD0001 = liveParticlesInLogTime.copy()
    liveParticlesInTimeNormD0001 = liveParticlesInLogTimeNorm.copy()
    # liveParticlesInTimeD0001 = liveParticlesInTime.copy()
    # liveParticlesInTimeNormD0001 = liveParticlesInTimeNorm.copy()
    tauD0001 = (uby-lby)**2/Df

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
    liveParticlesInTimeP80 = liveParticlesInLogTime.copy()
    liveParticlesInTimeNormP80 = liveParticlesInLogTimeNorm.copy()
    tauP = (uby-lby)**2/Df

    loadCompareAdsP60 = np.load('compareAdsP60.npz')
    for name, value in (loadCompareAdsP60.items()):
        globals()[name] = value
    liveParticlesInTimeP60 = liveParticlesInLogTime.copy()
    liveParticlesInTimeNormP60 = liveParticlesInLogTimeNorm.copy()
    tauP = (uby-lby)**2/Df

    loadCompareAdsP40 = np.load('compareAdsP40.npz')
    for name, value in (loadCompareAdsP40.items()):
        globals()[name] = value
    liveParticlesInTimeP40 = liveParticlesInLogTime.copy()
    liveParticlesInTimeNormP40 = liveParticlesInLogTimeNorm.copy()
    tauP = (uby-lby)**2/Df

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
        if (reflectedLeft!=0) & (reflectedRight!=0):
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

if compareAdsDiff:
    # Distribution of live particles in time
    survTimeDistCompareDiff = plt.figure(figsize=(8, 8))
    plt.rcParams.update({'font.size': 20})
    plt.plot(timeLogSpaced, liveParticlesInTimeD1, label=r'$D_f = 1$', color='b', linestyle='-')
    plt.plot(timeLogSpaced, liveParticlesInTimeD01, label=r'$D_f = 0.1$', color='r', linestyle='-')
    plt.plot(timeLogSpaced, liveParticlesInTimeD001, label=r'$D_f = 0.01$', color='g', linestyle='-')
    plt.plot(timeLogSpaced, liveParticlesInTimeD0001, label=r'$D_f = 0.001$', color='purple', linestyle='-')
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
    survTimeDistCompareDiffNorm = plt.figure(figsize=(8, 8))
    plt.rcParams.update({'font.size': 20})
    plt.plot(timeLogSpaced[::30]/tauD1, liveParticlesInTimeNormD1[::30], 'o', markerfacecolor='none', markeredgecolor='blue', markersize='5', label=r'$\tau_d = 4$')
    plt.plot(timeLogSpaced[::30]/tauD01, liveParticlesInTimeNormD01[::30], 'o', markerfacecolor='none', markeredgecolor='red', markersize='5', label=r'$\tau_d = 40$')
    plt.plot(timeLogSpaced[::30]/tauD001, liveParticlesInTimeNormD001[::30], 'o', markerfacecolor='none', markeredgecolor='green', markersize='5', label=r'$\tau_d = 400$')
    plt.plot(timeLogSpaced[::30]/tauD0001, liveParticlesInTimeNormD0001[::30], 'o', markerfacecolor='none', markeredgecolor='purple', markersize='5', label=r'$\tau_d = 4000$')
    # plt.plot(np.log(timeLinSpaced/tauD1), np.log(liveParticlesInTimeNormD1), label=r'$D_f = 1$', color='b', linestyle='-')
    # plt.plot(np.log(timeLinSpaced/tauD01), np.log(liveParticlesInTimeNormD01), label=r'$D_f = 0.1$', color='r', linestyle='-')
    # plt.plot(np.log(timeLinSpaced/tauD001), np.log(liveParticlesInTimeNormD001), label=r'$D_f = 0.01$', color='g', linestyle='-')
    yD01 = np.exp(-0.5*(timeLogSpaced/tauD01))
    plt.plot(timeLogSpaced/tauD01, yD01, color='black')
    plt.text((timeLogSpaced/tauD01)[5000], yD01[5000], "k(t)=-0.5", fontsize=12) 
    yD001 = np.exp(-1.2*(timeLogSpaced/tauD001))
    plt.plot(timeLogSpaced/tauD001, yD001, color='black')
    plt.text((timeLogSpaced/tauD001)[7000]*0.3, yD001[7000], "k(t)=-1.2", fontsize=12)
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
    dLivedtD1 = -np.diff(np.log(liveParticlesInTimeNormD1))/np.diff(timeLogSpaced/tauD1)
    dLivedtD01 = -np.diff(np.log(liveParticlesInTimeNormD01))/np.diff(timeLogSpaced/tauD01)
    dLivedtD001 = -np.diff(np.log(liveParticlesInTimeNormD001))/np.diff(timeLogSpaced/tauD001)
    dLivedtD0001 = -np.diff(np.log(liveParticlesInTimeNormD0001))/np.diff(timeLogSpaced/tauD0001)
    midTimes = ((timeLogSpaced)[:-1] + (timeLogSpaced)[1:]) / 2
    maskD1 = dLivedtD1!=0
    maskD01 = dLivedtD01!=0
    maskD001 = dLivedtD001!=0
    maskD0001 = dLivedtD0001!=0
    # validMask = np.isfinite(dLivedtD001)
    # splineD001 = make_interp_spline(midTimes[validMask], dLivedtD001[validMask], k=3)
    # dLivedtD001spline = splineD001(midTimes[::100])
    plt.plot(midTimes[maskD1]/tauD1, dLivedtD1[maskD1], 'o', markerfacecolor='none', markeredgecolor='blue', markersize='5', label=r'$\tau_d = 4$')
    plt.plot(midTimes[maskD01]/tauD01, dLivedtD01[maskD01], 'o', markerfacecolor='none', markeredgecolor='red', markersize='5', label=r'$\tau_d = 40$')
    plt.plot(midTimes[maskD001]/tauD001, dLivedtD001[maskD001], 'o', markerfacecolor='none', markeredgecolor='green', markersize='5', label=r'$\tau_d = 400$')
    plt.plot(midTimes[maskD0001]/tauD0001, dLivedtD0001[maskD0001], 'o', markerfacecolor='none', markeredgecolor='purple', markersize='5', label=r'$\tau_d = 4000$')
    # plt.plot(midTimes[::100]/tauD001, dLivedtD001spline, color='k')
    plt.axhline(y=0.5, color='black', linestyle='-')
    plt.axhline(y=1.2, color='black', linestyle='-')
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

# Well-mixed vs diffusion-limited survival time distributions ###########################################################
if plotSurvivalTimeDistAndReactionRatesForDegradationAndAdsorption:
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

if plotSurvivalTimeDistAndReactionRatesForDegradationAndAdsorption:
    survTimeDistCompare.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/survTimeDistCompare.png", format="png", bbox_inches="tight")
    compareDecayDegradationRates.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/compareDecayDegradationRates.png", format="png", bbox_inches="tight")
    survTimeDistCompareNorm.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/survTimeDistCompareNorm.png", format="png", bbox_inches="tight")
    compareDecayDegradationRatesNorm.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/compareDecayDegradationRatesNorm.png", format="png", bbox_inches="tight")

if FinalPositions:
    finalPositions.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/positionsDl01Dr01Rl0Rr0.png", format="png", bbox_inches="tight")
    histoMatriDiff.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/histDl01Dr01Rl0Rr0.png", format="png", bbox_inches="tight")
    finalPositions.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/positionsDl01Dr001Rl0Rr0.png", format="png", bbox_inches="tight")
    histoMatriDiff.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/histDl01Dr001Rl0Rr0.png", format="png", bbox_inches="tight")
    finalPositions.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/positionsDl01Dr01RlPlRrPr.png", format="png", bbox_inches="tight")
    histoMatriDiff.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/histDl01Dr01RlPlRrPr.png", format="png", bbox_inches="tight")
    finalPositions.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/positionsDl01Dr001RlPlRrPr.png", format="png", bbox_inches="tight")
    histoMatriDiff.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/histDl01Dr001RlPlRrPr.png", format="png", bbox_inches="tight")
    finalPositions.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/positionsDl01Dr001RlPlRrPr1e5ts.png", format="png", bbox_inches="tight")
    histoMatriDiff.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/histDl01Dr001RlPlRrPr1e5ts.png", format="png", bbox_inches="tight")

# if compareAdsDiff:
#     survTimeDistCompareDiff.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/survTimeDistCompareDiff.png", format="png", bbox_inches="tight")
#     survTimeDistCompareDiffNorm.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/survTimeDistCompareDiffNorm.png", format="png", bbox_inches="tight")
#     compareAdsRatesDiff.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/compareAdsRatesDiff.png", format="png", bbox_inches="tight")
#     compareDiffNormAdsRates.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/compareDiffNormAdsRates.png", format="png", bbox_inches="tight")

if compareAdsApertures:
    survTimeDistCompareApe.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/survTimeDistCompareApe.png", format="png", bbox_inches="tight")
    survTimeDistCompareApeNorm.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/survTimeDistCompareApeNorm.png", format="png", bbox_inches="tight")
    compareAdsRatesApe.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/compareAdsRatesApe.png", format="png", bbox_inches="tight")
    compareApeNormAdsRates.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/compareApeNormAdsRates.png", format="png", bbox_inches="tight")

if compareAdsProb:
    survTimeDistCompareProb.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/survTimeDistCompareProb.png", format="png", bbox_inches="tight")
    survTimeDistCompareProbNorm.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/survTimeDistCompareProbNorm.png", format="png", bbox_inches="tight")
    compareAdsRatesProb.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/compareAdsRatesProb.png", format="png", bbox_inches="tight")
    compareProbNormAdsRates.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/compareProbNormAdsRates.png", format="png", bbox_inches="tight")