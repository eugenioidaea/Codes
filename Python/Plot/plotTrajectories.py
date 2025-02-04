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
plotSpatialConcentration =          False
plotDegradation =                   False
FinalPositionVertAll =              False
FinalPositionHorAll =               False
FinalPositionVert =                 False
FinalPositionHor =                  False
plotSruvivalTimeDistOfNonAdsorbed = False

save =                              False

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


# Reaction rates vs characteristic times ################################################################################

# trajectories.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/trajectoriesInfinite.png", format="png", bbox_inches="tight")
# trajectories.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/trajectoriesSemiInfinite.png", format="png", bbox_inches="tight")
# trajectories.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/trajectoriesDegradation.png", format="png", bbox_inches="tight")
# trajectories.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/trajectoriesMatrixDiffusion.png", format="png", bbox_inches="tight")

# spatialConcentration.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/verificationMatrixDiffusion.png", format="png", bbox_inches="tight")

# survivalTimeDistribution.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/liveParticleInTime.png", format="png", bbox_inches="tight")

# finalPositions.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/finalPositionsMatrixDiffusion.png", format="png", bbox_inches="tight")

# diffusionLimitedSurvTimeDist.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/nonAbsParticles.png", format="png", bbox_inches="tight")

# diffusionLimitedSurvTimeDistNorm.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/nonAbsParticlesNorm.png", format="png", bbox_inches="tight")

# survTimeDistCompareNorm.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/survTimeDistCompareNorm.png", format="png", bbox_inches="tight")