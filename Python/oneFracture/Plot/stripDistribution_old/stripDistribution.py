debug = False
if not debug:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from sklearn.linear_model import LinearRegression

save = False

loadFinalPositions = np.load('stripDistribution.npz')
for name, value in (loadFinalPositions.items()):
    globals()[name] = value

# loadFinalPositions = np.load('stripDistribution1e5.npz')
# for name, value in (loadFinalPositions.items()):
#     globals()[name] = value

# Vertical distribution of all particles
finalPositionVertAll = plt.figure(figsize=(8, 8), dpi=300)
plt.bar(vBinsAll[:-1], vDistAll, width=np.diff(vBinsAll), edgecolor="black", align="edge")
plt.title('Particles distribution at the end of the simulation')
plt.xlabel('Distance along Y')
plt.ylabel('Number of particles')
plt.tight_layout()

# Horizontal distribution of all particles
finalPositionHorAll = plt.figure(figsize=(8, 8), dpi=300)
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

# if save:
    # finalPositionVertAll.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/verticalFinalDist.png", format="png", bbox_inches="tight")
    # finalPositionHorAll.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/horizontalFinalDist.png", format="png", bbox_inches="tight")
    # finalPositionVert.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/verticalFinalDist.png", format="png", bbox_inches="tight")
    # finalPositionHor.savefig("/home/eugenio/Github/IDAEA/Overleaf/WeeklyMeetingNotes/images/horizontalFinalDist.png", format="png", bbox_inches="tight")